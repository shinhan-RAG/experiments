#!/usr/bin/env python3
"""Run one approved MIRACL 110K taxonomy generation against local vLLM only.

This file is copied into the KT bundle's ``generator/`` clean-source root.
It has no provider SDK, credential handling, remote download, or fallback
path.  A model response is accepted only as a request-indexed JSON assignment
envelope that the tracked taxonomy contract can rejoin and audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GENERATOR_ROOT = Path(__file__).resolve().parents[2]
if str(GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATOR_ROOT))

from src.miracl_ko.kt_bundle import (  # noqa: E402
    atomic_write_json,
    load_bundle_controls,
    load_json_object,
    operation_receipt_path,
    order_successful_batch_records,
    parse_env_file,
    path_from_config,
    require_config,
    require_operation_start,
    source_artifact_path,
    utc_now,
    validate_execution_lock,
)
from src.miracl_ko.preparation import file_record, iter_jsonl, sha256_file, sha256_json  # noqa: E402
from src.miracl_ko.taxonomy_artifact import (  # noqa: E402
    TAXONOMY_ARTIFACT_SCHEMA_VERSION,
    TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION,
    approval_record_sha256,
    build_taxonomy_generator_run,
    generator_contract_sha256,
    rejoin_taxonomy_generator_outputs,
    rejoin_taxonomy_generator_run_outputs,
    validate_flat_l1_assignments,
    validate_taxonomy_artifact,
)


def _read_preflight(output_dir: Path, *, plan_sha256: str) -> None:
    report = load_json_object(output_dir / "environment" / "preflight.json", label="KT preflight result")
    if report.get("status") != "passed":
        raise ValueError("KT preflight did not pass; generation is blocked")
    lock = report.get("execution_lock")
    if not isinstance(lock, Mapping) or lock.get("generation_plan_sha256") != plan_sha256:
        raise ValueError("KT preflight does not bind the exact approved generation plan")


def _read_model_identity(output_dir: Path, *, plan_sha256: str) -> None:
    report = load_json_object(output_dir / "environment" / "model_identity.json", label="KT model identity result")
    if report.get("status") != "passed":
        raise ValueError("KT model identity verification did not pass; generation is blocked")
    if report.get("generation_plan_sha256") != plan_sha256:
        raise ValueError("KT model identity verification does not bind the exact approved generation plan")


def _replace_payload(template: Mapping[str, Any], semantic_payload: list[dict[str, str]]) -> dict[str, Any]:
    request_body = deepcopy(template)
    body = request_body.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise ValueError("approved vLLM request template is invalid")
    replacements = 0
    payload_json = json.dumps(semantic_payload, ensure_ascii=False, separators=(",", ":"))
    for message in body["messages"]:
        if isinstance(message, dict) and message.get("content") == "{{taxonomy_semantic_payload_json}}":
            message["content"] = payload_json
            replacements += 1
    if replacements != 1:
        raise ValueError("approved vLLM request template must have exactly one semantic payload placeholder")
    return request_body


def _post_local_vllm(url: str, body: Mapping[str, Any], *, timeout_seconds: int) -> tuple[bytes, dict[str, Any]]:
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: localhost URL is config-validated
            raw = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"local vLLM request failed: {error}") from error
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("local vLLM returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("local vLLM response must be a JSON object")
    return raw, decoded


def _extract_outputs(vllm_response: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        content = vllm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("local vLLM response lacks choices[0].message.content") from error
    if not isinstance(content, str):
        raise ValueError("local vLLM response content must be a JSON string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("local vLLM response content is not JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"outputs"} or not isinstance(payload["outputs"], list):
        raise ValueError("local vLLM response content must be exactly {outputs: [...]}")
    return payload["outputs"]


def _catalog_from_plan(plan: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    parameters = plan["generator"].get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("approved generator parameters are invalid")
    artifact_id = parameters.get("taxonomy_artifact_id")
    catalog = parameters.get("label_catalog")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("approved generator parameters require taxonomy_artifact_id")
    if not isinstance(catalog, list):
        raise ValueError("approved generator parameters require label_catalog")
    normalized = [dict(item) for item in catalog if isinstance(item, Mapping)]
    if len(normalized) != len(catalog):
        raise ValueError("approved label_catalog entries must be objects")
    return artifact_id, normalized


def _write_receipt(
    *, output_dir: Path, plan: Mapping[str, Any], approval: Mapping[str, Any], contract: Mapping[str, Any],
    run: Mapping[str, Any], successful: list[dict[str, Any]], status: str,
) -> dict[str, Any]:
    if status not in {"complete", "partial"}:
        raise ValueError("generation receipt status is invalid")
    responses = []
    for record in successful:
        path = output_dir / "generation" / "raw-responses" / record["relative_path"]
        responses.append(load_json_object(path, label="transport response"))
    assignments = rejoin_taxonomy_generator_run_outputs(run, responses) if status == "complete" else None
    receipt = {
        "schema_version": TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "generation_plan_sha256": sha256_json(plan),
        "approval_record_sha256": approval_record_sha256(approval),
        "generator_contract_sha256": generator_contract_sha256(contract),
        "generator_source_commit": plan["generator_source_commit"],
        "generation_run_sha256": run["generation_run_sha256"],
        "ordered_generation_request_sha256s": [batch["generation_request_sha256"] for batch in run["batches"]],
        "raw_response_files": successful,
        "assignment_canonical_sha256": sha256_json(assignments) if assignments is not None else None,
        "batch_summary": {
            "total": len(run["batches"]),
            "succeeded": len(successful),
            "failed": len(run["batches"]) - len(successful),
            "retries": sum(record["retry_count"] for record in successful),
        },
        "started_at": utc_now(),
        "ended_at": utc_now(),
        "runtime": deepcopy(plan["generator"]["runtime"]),
        "determinism_status": plan["determinism_mode"],
        "replay_required": plan["determinism_mode"] == "replay_required",
    }
    path = output_dir / "generation" / "generation_receipt.json"
    atomic_write_json(path, receipt)
    return receipt


def _write_raw_response_hash_manifest(
    *, generation_dir: Path, plan_sha256: str, run_sha256: str, receipt: Mapping[str, Any],
) -> None:
    """Hash every retained response byte, including successful resumed batches.

    The receipt binds normalized transport responses.  This companion manifest
    also preserves byte hashes for the unmodified local vLLM replies, so a
    later collection can prove it did not omit an already-completed batch.
    """
    records: list[dict[str, Any]] = []
    for label, directory in (("transport_response", "raw-responses"), ("vllm_response", "raw-vllm")):
        root = generation_dir / directory
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            records.append({"kind": label, **file_record(path, relative_to=generation_dir)})
    atomic_write_json(generation_dir / "raw_response_hash_manifest.json", {
        "generation_plan_sha256": plan_sha256,
        "generation_run_sha256": run_sha256,
        "response_files": records,
        "receipt_sha256": sha256_json(receipt),
    })


def _execute_missing_batch(
    *, batch: Mapping[str, Any], plan: Mapping[str, Any], endpoint: str, raw_dir: Path,
    vllm_raw_dir: Path, attempts: dict[str, int], operation: dict[str, Any],
    operation_path: Path, state_lock: threading.Lock,
) -> dict[str, Any]:
    """Issue one request-bound passage batch with bounded retries only.

    An exception never creates an ``Other`` or ``unknown`` assignment. The
    caller records a partial receipt and stops scheduling new work instead.
    """
    ordinal = batch["batch_ordinal"]
    transport_name = f"batch-{ordinal}.json"
    transport_path = raw_dir / transport_name
    max_retries = plan["run_controls"]["max_retries"]
    timeout_seconds = plan["run_controls"]["timeout_seconds"]
    with state_lock:
        previous_attempts = int(attempts.get(str(ordinal), 0))
    while previous_attempts <= max_retries:
        previous_attempts += 1
        with state_lock:
            attempts[str(ordinal)] = previous_attempts
            operation["batch_attempts"] = dict(attempts)
            atomic_write_json(operation_path, operation)
        try:
            request = _replace_payload(plan["generator"]["vllm_execution"]["request_body_template"], batch["semantic_payload"])
            raw_bytes, vllm_response = _post_local_vllm(
                endpoint, request["body"], timeout_seconds=timeout_seconds,
            )
            raw_path = vllm_raw_dir / f"batch-{ordinal}.attempt-{previous_attempts}.json"
            raw_path.write_bytes(raw_bytes)
            transport_response = {
                "generation_request_sha256": batch["generation_request_sha256"],
                "outputs": _extract_outputs(vllm_response),
            }
            assignments = rejoin_taxonomy_generator_outputs(batch, transport_response)
            validate_flat_l1_assignments(assignments, generator=plan["generator"])
            atomic_write_json(transport_path, transport_response)
            return {
                "batch_ordinal": ordinal,
                "generation_request_sha256": batch["generation_request_sha256"],
                "relative_path": transport_name,
                "byte_size": transport_path.stat().st_size,
                "sha256": sha256_file(transport_path),
                "attempt_count": previous_attempts,
                "retry_count": previous_attempts - 1,
            }
        except Exception:
            if previous_attempts > max_retries:
                break
            time.sleep(1)
    raise RuntimeError(f"batch {ordinal} exhausted the approved retry limit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    config = require_config(parse_env_file(args.config))
    control_dir = path_from_config(bundle_root, config["KT_CONTROL_DIR"], label="KT_CONTROL_DIR")
    output_dir = path_from_config(bundle_root, config["KT_OUTPUT_DIR"], label="KT_OUTPUT_DIR")
    plan, approval, contract = load_bundle_controls(control_dir)
    lock = validate_execution_lock(
        bundle_root=bundle_root, config=config, plan=plan, approval=approval, supplied_contract=contract,
    )
    _read_preflight(output_dir, plan_sha256=lock["generation_plan_sha256"])
    _read_model_identity(output_dir, plan_sha256=lock["generation_plan_sha256"])
    data_dir = Path(lock["data_dir"])
    corpus_path = source_artifact_path(data_dir)
    records = (
        {"corpus_id": row["corpus_id"], "title": row["title"], "text": row["text"]}
        for row in iter_jsonl(corpus_path)
    )
    run = build_taxonomy_generator_run(
        records,
        generation_plan_sha256=lock["generation_plan_sha256"],
        batch_size=plan["run_controls"]["batch_size"],
        run_controls=plan["run_controls"],
    )
    generation_dir = output_dir / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    run_path = generation_dir / "generation_run.json"
    if run_path.exists() and load_json_object(run_path, label="existing generation run") != run:
        raise ValueError("existing generation run differs from the approved exact plan")
    atomic_write_json(run_path, run)
    operation = require_operation_start(
        output_dir, plan_sha256=lock["generation_plan_sha256"], run_sha256=run["generation_run_sha256"],
    )
    attempts = dict(operation["batch_attempts"])
    raw_dir = generation_dir / "raw-responses"
    vllm_raw_dir = generation_dir / "raw-vllm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vllm_raw_dir.mkdir(parents=True, exist_ok=True)
    success_by_ordinal: dict[int, dict[str, Any]] = {}
    pending_batches: list[Mapping[str, Any]] = []
    endpoint = f"http://{config['KT_VLLM_HOST']}:{config['KT_VLLM_PORT']}/v1/chat/completions"
    max_retries = plan["run_controls"]["max_retries"]
    for batch in run["batches"]:
        ordinal = batch["batch_ordinal"]
        transport_name = f"batch-{ordinal}.json"
        transport_path = raw_dir / transport_name
        if transport_path.is_file():
            response = load_json_object(transport_path, label="existing transport response")
            existing_assignments = rejoin_taxonomy_generator_outputs(batch, response)
            validate_flat_l1_assignments(existing_assignments, generator=plan["generator"])
            count = int(attempts.get(str(ordinal), 1))
            if count < 1 or count - 1 > max_retries:
                raise ValueError("existing successful batch has an invalid approved retry count")
            success_by_ordinal[ordinal] = {
                "batch_ordinal": ordinal,
                "generation_request_sha256": batch["generation_request_sha256"],
                "relative_path": transport_name,
                "byte_size": transport_path.stat().st_size,
                "sha256": sha256_file(transport_path),
                "attempt_count": count,
                "retry_count": count - 1,
            }
            continue
        pending_batches.append(batch)

    # At most the approved concurrency is in flight. Completion order is never
    # used as artifact order: every success record is canonicalized by ordinal.
    state_lock = threading.Lock()
    pending_index = 0
    failure: Exception | None = None
    futures: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}
    max_workers = plan["run_controls"]["transport_max_concurrency"]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def submit_next() -> bool:
            nonlocal pending_index
            if pending_index >= len(pending_batches):
                return False
            batch = pending_batches[pending_index]
            pending_index += 1
            future = executor.submit(
                _execute_missing_batch,
                batch=batch,
                plan=plan,
                endpoint=endpoint,
                raw_dir=raw_dir,
                vllm_raw_dir=vllm_raw_dir,
                attempts=attempts,
                operation=operation,
                operation_path=operation_receipt_path(output_dir),
                state_lock=state_lock,
            )
            futures[future] = batch
            return True

        while len(futures) < max_workers and submit_next():
            pass
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                batch = futures.pop(future)
                try:
                    record = future.result()
                    success_by_ordinal[record["batch_ordinal"]] = record
                except Exception as error:
                    if failure is None:
                        failure = error
                if failure is None:
                    submit_next()

    success_records = order_successful_batch_records(list(success_by_ordinal.values()))
    if failure is not None:
        receipt = _write_receipt(
            output_dir=output_dir, plan=plan, approval=approval, contract=contract,
            run=run, successful=success_records, status="partial",
        )
        operation["status"] = "partial"
        operation["completed_at"] = utc_now()
        atomic_write_json(operation_receipt_path(output_dir), operation)
        _write_raw_response_hash_manifest(
            generation_dir=generation_dir,
            plan_sha256=lock["generation_plan_sha256"],
            run_sha256=run["generation_run_sha256"],
            receipt=receipt,
        )
        raise RuntimeError(str(failure)) from failure
    receipt = _write_receipt(
        output_dir=output_dir, plan=plan, approval=approval, contract=contract,
        run=run, successful=success_records, status="complete",
    )
    artifact_id, catalog = _catalog_from_plan(plan)
    responses = [load_json_object(raw_dir / record["relative_path"], label="transport response") for record in success_records]
    assignments = rejoin_taxonomy_generator_run_outputs(run, responses)
    subset_manifest = load_json_object(data_dir / "subsets" / "manifest.json", label="MIRACL subset manifest")
    artifact = {
        "schema_version": TAXONOMY_ARTIFACT_SCHEMA_VERSION,
        "taxonomy_artifact_id": artifact_id,
        "dataset": "MIRACL",
        "language": "ko",
        "source_revisions": subset_manifest["source_revisions"],
        "retrieval_unit": "passage",
        "source_scale": 110_000,
        "projection_scale": 110_000,
        "input_contract": plan["input_contract"],
        "provenance": {**plan["input_provenance"], "generator": plan["generator"]},
        "label_catalog": catalog,
        "assignments": assignments,
        "projection": {"method": "source_110k_identity_v1", "source_artifact_sha256": None},
    }
    validate_taxonomy_artifact(artifact)
    artifact_path = output_dir / "artifacts" / "110000.json"
    atomic_write_json(artifact_path, artifact)
    operation["status"] = "complete"
    operation["completed_at"] = utc_now()
    atomic_write_json(operation_receipt_path(output_dir), operation)
    _write_raw_response_hash_manifest(
        generation_dir=generation_dir,
        plan_sha256=lock["generation_plan_sha256"],
        run_sha256=run["generation_run_sha256"],
        receipt=receipt,
    )
    print(json.dumps({"status": "complete", "artifact": str(artifact_path), "receipt": str(generation_dir / "generation_receipt.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"KT taxonomy generation failed loudly: {error}", file=sys.stderr)
        raise
