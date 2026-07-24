"""Leakage-safe taxonomy artifact contracts for MIRACL Korean passages.

This is a preparation-only boundary.  It does not generate a real taxonomy,
call a model, invoke a retriever, or register MIRACL with the Part 1/2 runner.
It instead validates the immutable artifact that a separately approved
generator would produce from 110K passage title/text and projects it to the
smaller nested fixtures without changing any shared assignment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping
import unicodedata

from src.miracl_ko.preparation import RETRIEVAL_UNIT, SCALE_SIZES, sha256_file, sha256_json


TAXONOMY_ARTIFACT_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-artifact.v1"
TAXONOMY_MANIFEST_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-manifest.v2"
TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-generator-batch.v1"
TAXONOMY_GENERATOR_RUN_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-generator-run.v2"
TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-generation-plan.v5"
TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-generation-receipt.v2"
TAXONOMY_VLLM_REQUEST_BODY_TEMPLATE_SCHEMA_VERSION = "dr-dci.miracl-ko-vllm-request-body-template.v1"
TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION = "dr-dci.miracl-ko-flat-l1-adapter.v1"
TAXONOMY_GENERATOR_CODE_CONTRACT_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-generator-code-contract.v1"
TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-approval-record.v1"
TAXONOMY_SOURCE_SCALE = 110_000
TAXONOMY_SEMANTIC_INPUT_FIELDS = ("title", "text")
TAXONOMY_MAPPING_KEY = "corpus_id"
TAXONOMY_NONSEMANTIC_CORPUS_ID_USES = (
    "mapping_join",
    "duplicate_detection",
    "deterministic_output_order",
)
TAXONOMY_FORBIDDEN_INPUT_FIELDS = frozenset({
    "query",
    "queries",
    "qid",
    "qrel",
    "qrels",
    "relevance",
    "answer",
    "answers",
    "gold",
    "gold_id",
    "gold_ids",
    "evidence",
    "evidence_id",
    "evidence_ids",
    "positive_passages",
    "negative_passages",
    "ranked_passage_ids",
    "evaluation_result",
})
SOURCE_IDENTITY_METHOD = "source_110k_identity_v1"
FILTER_PROJECTION_METHOD = "filter_110k_mapping_by_corpus_id_v1"
GENERATOR_CODE_AGGREGATE_METHOD = "sha256-json-canonical-v1"
FLAT_L1_SCORE_HANDLING = "not_used_by_existing_soft_boost"
FLAT_L1_UNKNOWN_HANDLING = "excluded_from_boost_eligibility"
TAXONOMY_BATCH_GROUPING = "corpus_id_sorted_contiguous_v1"
TAXONOMY_BATCH_ORDER = "batch_ordinal_ascending_v1"
TAXONOMY_RESUME_POLICY = "reuse_verified_complete_batches_only_v1"
TAXONOMY_IDEMPOTENCY_MODE = "generation_request_sha256_response_file_v1"
TAXONOMY_MAX_RETRIES_SEMANTICS = "per_batch_successful_response_v1"
_VERIFIED_RECEIPT_TOKEN = object()


@dataclass(frozen=True)
class VerifiedTaxonomyGenerationReceipt:
    """Opaque result returned only after raw receipt verification succeeds."""

    receipt_sha256: str
    generation_plan_sha256: str
    approval_record_sha256: str
    generator_contract_sha256: str
    generator_code_sha256: str
    generator_spec_sha256: str
    generator_source_commit: str
    generation_run_sha256: str
    assignment_canonical_sha256: str
    assignments: tuple[dict[str, Any], ...]
    _verification_token: object = field(repr=False, compare=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_rfc3339_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ):
        raise ValueError(f"{label} must be an RFC3339 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include timezone")
    return value


def _validate_json_value(value: Any, *, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} must use string object keys")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    raise ValueError(f"{label} must be JSON-serializable")


def _validate_semantic_input_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise ValueError("taxonomy input_contract must be an object")
    if contract.get("mapping_key") != TAXONOMY_MAPPING_KEY:
        raise ValueError("taxonomy mapping_key must be corpus_id")
    if contract.get("semantic_generator_input_fields") != list(TAXONOMY_SEMANTIC_INPUT_FIELDS):
        raise ValueError("taxonomy semantic generator inputs must be title/text only")
    if contract.get("corpus_id_semantic_use") != "prohibited":
        raise ValueError("taxonomy corpus_id semantic use must be prohibited")
    if contract.get("corpus_id_nonsemantic_uses") != list(TAXONOMY_NONSEMANTIC_CORPUS_ID_USES):
        raise ValueError("taxonomy corpus_id non-semantic uses must be declared exactly")
    forbidden = contract.get("forbidden_input_fields")
    if not isinstance(forbidden, list) or not TAXONOMY_FORBIDDEN_INPUT_FIELDS.issubset(
        {str(value).casefold() for value in forbidden}
    ):
        raise ValueError("taxonomy input_contract is missing forbidden leakage fields")


def build_semantic_generator_inputs(corpus_records: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Deprecated compatibility extractor; never use it to rejoin outputs.

    It now has the batch builder's deterministic sort semantics, but returns
    only the model-facing payload.  Real generation must use
    :func:`build_taxonomy_generator_batch` and
    :func:`rejoin_taxonomy_generator_outputs` together.
    """
    return [
        {"title": title, "text": text}
        for _, title, text in _prepare_taxonomy_generator_transport(corpus_records)
    ]


def _validate_transport_record(record: Mapping[str, Any], *, index: int) -> tuple[str, str, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"taxonomy corpus record {index} must be an object")
    lowered_fields = {str(field).casefold() for field in record}
    forbidden = sorted(lowered_fields & TAXONOMY_FORBIDDEN_INPUT_FIELDS)
    if forbidden:
        raise ValueError(f"taxonomy generator input contains forbidden leakage field: {forbidden[0]}")
    allowed_transport_fields = {TAXONOMY_MAPPING_KEY, *TAXONOMY_SEMANTIC_INPUT_FIELDS}
    unsupported = set(record) - allowed_transport_fields
    if unsupported:
        raise ValueError(f"taxonomy generator input has unsupported fields: {sorted(unsupported)}")
    corpus_id = _require_nonempty_string(record.get(TAXONOMY_MAPPING_KEY), label="taxonomy corpus_id")
    title = record.get("title")
    text = record.get("text")
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("taxonomy title/text inputs must be strings")
    return corpus_id, title, text


def _prepare_taxonomy_generator_transport(
    corpus_records: Iterable[Mapping[str, Any]],
) -> list[tuple[str, str, str]]:
    """Validate and stably sort non-semantic transport records once."""
    transport: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(corpus_records):
        corpus_id, title, text = _validate_transport_record(record, index=index)
        if corpus_id in seen_ids:
            raise ValueError(f"taxonomy generator input has duplicate corpus_id: {corpus_id}")
        seen_ids.add(corpus_id)
        transport.append((corpus_id, title, text))
    if not transport:
        raise ValueError("taxonomy generator input requires at least one passage")
    transport.sort(key=lambda row: row[0])
    return transport


def _generation_request_sha256(batch: Mapping[str, Any]) -> str:
    return sha256_json({
        "generation_plan_sha256": batch["generation_plan_sha256"],
        "mapping_envelope_sha256": batch["mapping_envelope_sha256"],
        "semantic_payload_sha256": batch["semantic_payload_sha256"],
        "batch_schema_version": batch["schema_version"],
        "batch_ordinal": batch["batch_ordinal"],
    })


def _build_taxonomy_generator_batch(
    transport: list[tuple[str, str, str]], *, generation_plan_sha256: str,
    batch_ordinal: int, batch_start: int, run_total_records: int,
) -> dict[str, Any]:
    if not transport:
        raise ValueError("taxonomy generator batch requires at least one passage")
    mapping_envelope = [
        {"request_index": index, "corpus_id": corpus_id}
        for index, (corpus_id, _, _) in enumerate(transport)
    ]
    semantic_payload = [
        {"title": title, "text": text}
        for _, title, text in transport
    ]
    batch = {
        "schema_version": TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION,
        "generation_plan_sha256": generation_plan_sha256,
        "batch_ordinal": batch_ordinal,
        "batch_start": batch_start,
        "batch_end": batch_start + len(transport),
        "run_total_records": run_total_records,
        "mapping_envelope": mapping_envelope,
        "semantic_payload": semantic_payload,
        "mapping_envelope_sha256": sha256_json(mapping_envelope),
        "semantic_payload_sha256": sha256_json(semantic_payload),
    }
    batch["generation_request_sha256"] = _generation_request_sha256(batch)
    validate_taxonomy_generator_batch(batch)
    return batch


def build_taxonomy_generator_batch(
    corpus_records: Iterable[Mapping[str, Any]], *, generation_plan_sha256: str,
    batch_ordinal: int = 0, batch_start: int = 0, run_total_records: int | None = None,
) -> dict[str, Any]:
    """Build one request-bound batch for a semantic generator.

    ``generation_request_sha256`` is transport metadata. It never enters a
    prompt or semantic payload, but every response must return it verbatim.
    Multi-batch 110K generation must use :func:`build_taxonomy_generator_run`
    so ordinal/range completeness is also checked.
    """
    transport = _prepare_taxonomy_generator_transport(corpus_records)
    if run_total_records is None:
        run_total_records = len(transport)
    return _build_taxonomy_generator_batch(
        transport,
        generation_plan_sha256=generation_plan_sha256,
        batch_ordinal=batch_ordinal,
        batch_start=batch_start,
        run_total_records=run_total_records,
    )


def validate_taxonomy_generator_batch(batch: Mapping[str, Any]) -> None:
    """Validate a non-semantic mapping envelope before output rejoining."""
    if not isinstance(batch, Mapping) or set(batch) != {
        "schema_version", "generation_plan_sha256", "batch_ordinal", "batch_start", "batch_end",
        "run_total_records", "mapping_envelope", "semantic_payload",
        "mapping_envelope_sha256", "semantic_payload_sha256", "generation_request_sha256",
    }:
        raise ValueError("taxonomy generator batch has missing or unsupported fields")
    if batch.get("schema_version") != TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION:
        raise ValueError("taxonomy generator batch schema_version is invalid")
    _require_sha256(batch.get("generation_plan_sha256"), label="taxonomy generator batch plan")
    for key in ("batch_ordinal", "batch_start", "batch_end", "run_total_records"):
        if type(batch.get(key)) is not int:
            raise ValueError(f"taxonomy generator batch {key} must be an integer")
    if batch["batch_ordinal"] < 0 or batch["batch_start"] < 0:
        raise ValueError("taxonomy generator batch ordinal and start must be non-negative")
    if batch["run_total_records"] <= 0 or batch["batch_end"] <= batch["batch_start"]:
        raise ValueError("taxonomy generator batch range is invalid")
    if batch["batch_end"] > batch["run_total_records"]:
        raise ValueError("taxonomy generator batch range exceeds run total")
    envelope = batch.get("mapping_envelope")
    payload = batch.get("semantic_payload")
    if not isinstance(envelope, list) or not envelope:
        raise ValueError("taxonomy generator mapping envelope must be a non-empty array")
    if not isinstance(payload, list) or len(payload) != len(envelope):
        raise ValueError("taxonomy generator semantic payload count must equal mapping envelope count")
    if batch["batch_end"] - batch["batch_start"] != len(envelope):
        raise ValueError("taxonomy generator batch range does not match payload count")
    previous_id: str | None = None
    seen_ids: set[str] = set()
    for index, row in enumerate(envelope):
        if not isinstance(row, dict) or set(row) != {"request_index", "corpus_id"}:
            raise ValueError("taxonomy generator mapping envelope has unsupported fields")
        if type(row.get("request_index")) is not int or row["request_index"] != index:
            raise ValueError("taxonomy generator mapping envelope request indexes must be sequential")
        corpus_id = _require_nonempty_string(row.get("corpus_id"), label="taxonomy generator envelope corpus_id")
        if corpus_id in seen_ids or (previous_id is not None and corpus_id < previous_id):
            raise ValueError("taxonomy generator mapping envelope must be corpus_id-sorted and unique")
        previous_id = corpus_id
        seen_ids.add(corpus_id)
    for row in payload:
        if not isinstance(row, dict) or set(row) != set(TAXONOMY_SEMANTIC_INPUT_FIELDS):
            raise ValueError("taxonomy semantic payload must contain title/text only")
        if not isinstance(row.get("title"), str) or not isinstance(row.get("text"), str):
            raise ValueError("taxonomy semantic payload title/text must be strings")
    _require_sha256(batch.get("mapping_envelope_sha256"), label="taxonomy generator mapping envelope")
    _require_sha256(batch.get("semantic_payload_sha256"), label="taxonomy generator semantic payload")
    if batch["mapping_envelope_sha256"] != sha256_json(envelope):
        raise ValueError("taxonomy generator mapping envelope hash does not match")
    if batch["semantic_payload_sha256"] != sha256_json(payload):
        raise ValueError("taxonomy generator semantic payload hash does not match")
    _require_sha256(batch.get("generation_request_sha256"), label="taxonomy generation request")
    if batch["generation_request_sha256"] != _generation_request_sha256(batch):
        raise ValueError("taxonomy generator generation request hash does not match")


def _default_run_controls(batch_size: int) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "batch_grouping": TAXONOMY_BATCH_GROUPING,
        "batch_order": TAXONOMY_BATCH_ORDER,
        "timeout_seconds": 120,
        "max_retries": 0,
        "max_retries_semantics": TAXONOMY_MAX_RETRIES_SEMANTICS,
        "resume_policy": TAXONOMY_RESUME_POLICY,
        "idempotency_mode": TAXONOMY_IDEMPOTENCY_MODE,
    }


def _validate_run_controls(controls: Any) -> None:
    required = {
        "batch_size", "batch_grouping", "batch_order", "timeout_seconds", "max_retries",
        "max_retries_semantics",
        "resume_policy", "idempotency_mode",
    }
    if not isinstance(controls, Mapping) or set(controls) != required:
        raise ValueError("taxonomy generator run controls have missing or unsupported fields")
    if type(controls.get("batch_size")) is not int or controls["batch_size"] <= 0:
        raise ValueError("taxonomy generator run batch_size must be a positive integer")
    if type(controls.get("timeout_seconds")) is not int or controls["timeout_seconds"] <= 0:
        raise ValueError("taxonomy generator run timeout_seconds must be a positive integer")
    if type(controls.get("max_retries")) is not int or controls["max_retries"] < 0:
        raise ValueError("taxonomy generator run max_retries must be a non-negative integer")
    if controls.get("max_retries_semantics") != TAXONOMY_MAX_RETRIES_SEMANTICS:
        raise ValueError("taxonomy generator run max_retries semantics is invalid")
    if controls.get("batch_grouping") != TAXONOMY_BATCH_GROUPING:
        raise ValueError("taxonomy generator run batch_grouping is invalid")
    if controls.get("batch_order") != TAXONOMY_BATCH_ORDER:
        raise ValueError("taxonomy generator run batch_order is invalid")
    if controls.get("resume_policy") != TAXONOMY_RESUME_POLICY:
        raise ValueError("taxonomy generator run resume_policy is invalid")
    if controls.get("idempotency_mode") != TAXONOMY_IDEMPOTENCY_MODE:
        raise ValueError("taxonomy generator run idempotency_mode is invalid")


def build_taxonomy_generator_run(
    corpus_records: Iterable[Mapping[str, Any]], *, generation_plan_sha256: str, batch_size: int,
    run_controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Split one 110K-scale run into complete, request-bound transport batches."""
    _require_sha256(generation_plan_sha256, label="taxonomy generator run plan")
    controls = dict(run_controls) if run_controls is not None else _default_run_controls(batch_size)
    _validate_run_controls(controls)
    if controls["batch_size"] != batch_size:
        raise ValueError("taxonomy generator run batch_size does not match run controls")
    transport = _prepare_taxonomy_generator_transport(corpus_records)
    total = len(transport)
    batches = [
        _build_taxonomy_generator_batch(
            transport[start:start + batch_size],
            generation_plan_sha256=generation_plan_sha256,
            batch_ordinal=ordinal,
            batch_start=start,
            run_total_records=total,
        )
        for ordinal, start in enumerate(range(0, total, batch_size))
    ]
    run = {
        "schema_version": TAXONOMY_GENERATOR_RUN_SCHEMA_VERSION,
        "generation_plan_sha256": generation_plan_sha256,
        "run_total_records": total,
        "run_controls": controls,
        "batches": batches,
        "generation_run_sha256": sha256_json({
            "generation_plan_sha256": generation_plan_sha256,
            "run_controls": controls,
            "generation_request_sha256s": [batch["generation_request_sha256"] for batch in batches],
        }),
    }
    validate_taxonomy_generator_run(run)
    return run


def validate_taxonomy_generator_run(run: Mapping[str, Any]) -> None:
    """Require complete, non-overlapping batches from one generation plan."""
    required = {
        "schema_version", "generation_plan_sha256", "run_total_records", "run_controls", "batches", "generation_run_sha256",
    }
    if not isinstance(run, Mapping) or set(run) != required:
        raise ValueError("taxonomy generator run has missing or unsupported fields")
    if run.get("schema_version") != TAXONOMY_GENERATOR_RUN_SCHEMA_VERSION:
        raise ValueError("taxonomy generator run schema_version is invalid")
    _require_sha256(run.get("generation_plan_sha256"), label="taxonomy generator run plan")
    if type(run.get("run_total_records")) is not int or run["run_total_records"] <= 0:
        raise ValueError("taxonomy generator run total must be a positive integer")
    _validate_run_controls(run.get("run_controls"))
    batches = run.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("taxonomy generator run batches must be a non-empty array")
    next_start = 0
    seen_ids: set[str] = set()
    previous_id: str | None = None
    for ordinal, batch in enumerate(batches):
        validate_taxonomy_generator_batch(batch)
        if batch["generation_plan_sha256"] != run["generation_plan_sha256"]:
            raise ValueError("taxonomy generator run batch plan does not match run")
        if batch["batch_ordinal"] != ordinal or batch["batch_start"] != next_start:
            raise ValueError("taxonomy generator run batch ordinal or range is not contiguous")
        if batch["run_total_records"] != run["run_total_records"]:
            raise ValueError("taxonomy generator run batch total does not match run")
        next_start = batch["batch_end"]
        for row in batch["mapping_envelope"]:
            corpus_id = row["corpus_id"]
            if corpus_id in seen_ids or (previous_id is not None and corpus_id <= previous_id):
                raise ValueError("taxonomy generator run passage IDs are duplicated or not globally sorted")
            seen_ids.add(corpus_id)
            previous_id = corpus_id
    if next_start != run["run_total_records"]:
        raise ValueError("taxonomy generator run batches are incomplete")
    if len(seen_ids) != run["run_total_records"]:
        raise ValueError("taxonomy generator run passage count does not match declared total")
    _require_sha256(run.get("generation_run_sha256"), label="taxonomy generator run")
    expected_hash = sha256_json({
        "generation_plan_sha256": run["generation_plan_sha256"],
        "run_controls": run["run_controls"],
        "generation_request_sha256s": [batch["generation_request_sha256"] for batch in batches],
    })
    if run["generation_run_sha256"] != expected_hash:
        raise ValueError("taxonomy generator run hash does not match batches")


def validate_taxonomy_generator_run_against_plan(
    run: Mapping[str, Any], generation_plan: Mapping[str, Any],
) -> None:
    """Bind actual batch sizing/grouping to the approved generation plan."""
    validate_taxonomy_generator_run(run)
    validate_taxonomy_generation_plan(generation_plan)
    if run["generation_plan_sha256"] != generation_plan_sha256(generation_plan):
        raise ValueError("taxonomy generator run generation plan does not match approved plan")
    if run["run_controls"] != generation_plan["run_controls"]:
        raise ValueError("taxonomy generator run controls do not match approved plan")


def rejoin_taxonomy_generator_outputs(
    batch: Mapping[str, Any], generator_response: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rejoin request-indexed generator output using only the verified envelope."""
    validate_taxonomy_generator_batch(batch)
    if not isinstance(generator_response, Mapping) or set(generator_response) != {
        "generation_request_sha256", "outputs",
    }:
        raise ValueError("taxonomy generator response has missing or unsupported fields")
    _require_sha256(generator_response.get("generation_request_sha256"), label="taxonomy generator response request")
    if generator_response["generation_request_sha256"] != batch["generation_request_sha256"]:
        raise ValueError("taxonomy generator response generation request does not match batch")
    outputs = generator_response.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("taxonomy generator response outputs must be an array")
    envelope = batch["mapping_envelope"]
    if len(outputs) != len(envelope):
        raise ValueError("taxonomy generator output count does not match input count")
    assignments: list[dict[str, Any]] = []
    outputs_by_index: dict[int, Mapping[str, Any]] = {}
    for output in outputs:
        if not isinstance(output, Mapping) or set(output) != {"request_index", "label_id", "score", "status"}:
            raise ValueError("taxonomy generator output has unsupported fields")
        request_index = output.get("request_index")
        if type(request_index) is not int or request_index < 0 or request_index >= len(envelope):
            raise ValueError("taxonomy generator output request_index is invalid")
        if request_index in outputs_by_index:
            raise ValueError("taxonomy generator output request_index is duplicated")
        outputs_by_index[request_index] = output
    if set(outputs_by_index) != set(range(len(envelope))):
        raise ValueError("taxonomy generator output request_index is missing")
    for envelope_row in envelope:
        output = outputs_by_index[envelope_row["request_index"]]
        label_id = _require_nonempty_string(output.get("label_id"), label="taxonomy generator output label_id")
        score = output.get("score")
        if type(score) not in {int, float} or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("taxonomy generator output score must be finite and within [0, 1]")
        status = output.get("status")
        if status not in {"assigned", "unknown"}:
            raise ValueError("taxonomy generator output status is invalid")
        if status == "unknown" and (label_id != "unknown" or float(score) != 0.0):
            raise ValueError("taxonomy generator unknown output must use unknown label_id and score 0")
        if status == "assigned" and label_id == "unknown":
            raise ValueError("taxonomy generator assigned output must not use unknown label_id")
        assignments.append({
            "corpus_id": envelope_row["corpus_id"],
            "label_id": label_id,
            "score": float(score),
            "status": status,
        })
    return assignments


def rejoin_taxonomy_generator_run_outputs(
    run: Mapping[str, Any], generator_responses: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rejoin all and only the response envelopes for one complete generation run."""
    validate_taxonomy_generator_run(run)
    responses = list(generator_responses)
    batches = run["batches"]
    if len(responses) != len(batches):
        raise ValueError("taxonomy generator run response count does not match batch count")
    responses_by_request: dict[str, Mapping[str, Any]] = {}
    for response in responses:
        if not isinstance(response, Mapping) or set(response) != {"generation_request_sha256", "outputs"}:
            raise ValueError("taxonomy generator run response has missing or unsupported fields")
        request_hash = response.get("generation_request_sha256")
        _require_sha256(request_hash, label="taxonomy generator run response request")
        if request_hash in responses_by_request:
            raise ValueError("taxonomy generator run response request is duplicated")
        responses_by_request[request_hash] = response
    expected_requests = {batch["generation_request_sha256"] for batch in batches}
    if set(responses_by_request) != expected_requests:
        raise ValueError("taxonomy generator run response request is missing or from another run")
    assignments = [
        assignment
        for batch in batches
        for assignment in rejoin_taxonomy_generator_outputs(
            batch, responses_by_request[batch["generation_request_sha256"]]
        )
    ]
    if len(assignments) != run["run_total_records"]:
        raise ValueError("taxonomy generator run assignments are incomplete")
    return assignments


def _validate_generator(generator: Any) -> None:
    if not isinstance(generator, dict):
        raise ValueError("taxonomy generator provenance must be an object")
    for key in (
        "generator_type",
        "generator_version",
        "model_or_algorithm",
        "model_or_tokenizer_version",
        "determinism_mode",
    ):
        _require_nonempty_string(generator.get(key), label=f"taxonomy generator {key}")
    _require_sha256(generator.get("generator_code_sha256"), label="taxonomy generator code")
    template_sha256 = generator.get("prompt_template_sha256")
    if template_sha256 is not None:
        _require_sha256(template_sha256, label="taxonomy generator prompt template")
    if type(generator.get("seed")) is not int:
        raise ValueError("taxonomy generator seed must be an integer")
    if not isinstance(generator.get("parameters"), dict) or not generator["parameters"]:
        raise ValueError("taxonomy generator parameters must be a non-empty object")
    _validate_json_value(generator["parameters"], label="taxonomy generator parameters")
    if generator.get("determinism_mode") not in {"deterministic", "replay_required"}:
        raise ValueError("taxonomy generator determinism_mode is invalid")


def _validate_generation_generator_spec(generator: Any) -> None:
    """Validate the complete execution specification required in a plan.

    Artifact provenance retains a compact, backwards-compatible generator
    description.  A pre-generation plan is stricter: it must be sufficient to
    reproduce an approved model call without guessing a model revision, prompt,
    runtime, batch policy, or label policy.
    """
    _validate_generator(generator)
    required = {
        "generator_type", "generator_version", "generator_code_sha256", "model_or_algorithm",
        "model_or_tokenizer_version", "prompt_template_sha256", "seed", "parameters", "determinism_mode",
        "model_repository", "model_revision", "tokenizer_repository", "tokenizer_revision",
        "pooling", "normalization", "clustering_or_classification_algorithm",
        "clustering_library", "clustering_library_version", "cluster_selection_rule",
        "prompt_template", "generation_controls", "runtime", "label_id_rule",
        "unknown_outlier_handling", "display_label_rule", "vllm_execution",
    }
    if not isinstance(generator, Mapping) or set(generator) != required:
        raise ValueError("taxonomy generation generator specification has missing or unsupported fields")
    for key in (
        "model_repository", "model_revision", "tokenizer_repository", "tokenizer_revision", "pooling",
        "normalization", "clustering_or_classification_algorithm", "clustering_library",
        "clustering_library_version", "cluster_selection_rule", "prompt_template", "label_id_rule",
        "unknown_outlier_handling", "display_label_rule",
    ):
        _require_nonempty_string(generator.get(key), label=f"taxonomy generator specification {key}")
    if generator["prompt_template_sha256"] != _sha256_text(generator["prompt_template"]):
        raise ValueError("taxonomy generator prompt template sha256 does not match template")
    controls = generator.get("generation_controls")
    if not isinstance(controls, Mapping) or set(controls) != {"temperature", "max_tokens"}:
        raise ValueError("taxonomy generator generation_controls are invalid")
    temperature = controls.get("temperature")
    if type(temperature) not in {int, float} or not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError("taxonomy generator temperature must be a finite non-negative number")
    if type(controls.get("max_tokens")) is not int or controls["max_tokens"] <= 0:
        raise ValueError("taxonomy generator max_tokens must be a positive integer")
    _validate_vllm_execution(generator.get("vllm_execution"), generator=generator)
    runtime = generator.get("runtime")
    runtime_required = {
        "model_repository", "model_revision", "tokenizer_repository", "tokenizer_revision",
        "library_versions", "dependency_lock_sha256", "container_digest",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_required:
        raise ValueError("taxonomy generator runtime has missing or unsupported fields")
    for key in ("model_repository", "model_revision", "tokenizer_repository", "tokenizer_revision", "container_digest"):
        _require_nonempty_string(runtime.get(key), label=f"taxonomy generator runtime {key}")
        if key in {"model_repository", "model_revision", "tokenizer_repository", "tokenizer_revision"} and runtime[key] != generator[key]:
            raise ValueError(f"taxonomy generator runtime {key} does not match generator specification")
    _require_sha256(runtime.get("dependency_lock_sha256"), label="taxonomy generator runtime dependency lock")
    versions = runtime.get("library_versions")
    if not isinstance(versions, Mapping) or not versions:
        raise ValueError("taxonomy generator runtime library_versions must be a non-empty object")
    for library, version in versions.items():
        _require_nonempty_string(library, label="taxonomy generator runtime library name")
        _require_nonempty_string(version, label="taxonomy generator runtime library version")


def _validate_vllm_control(value: Any, *, label: str, validator: Any) -> Any:
    if not isinstance(value, Mapping) or set(value) != {"mode", "value"}:
        raise ValueError(f"taxonomy vLLM {label} control is invalid")
    mode = value.get("mode")
    if mode == "not_applicable":
        if value.get("value") is not None:
            raise ValueError(f"taxonomy vLLM {label} not_applicable control must use null value")
        return None
    if mode != "value":
        raise ValueError(f"taxonomy vLLM {label} control mode is invalid")
    validator(value.get("value"), label=label)
    return value["value"]


def _vllm_launch_option_values(arguments: list[str], option: str) -> list[str]:
    """Read one vLLM option without treating prose metadata as launch state."""
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 == len(arguments):
                raise ValueError(f"taxonomy vLLM launch option {option} has no value")
            value = arguments[index + 1]
            if value.startswith("--"):
                raise ValueError(f"taxonomy vLLM launch option {option} has no value")
            values.append(value)
        elif argument.startswith(f"{option}="):
            value = argument.removeprefix(f"{option}=")
            if not value:
                raise ValueError(f"taxonomy vLLM launch option {option} has no value")
            values.append(value)
    return values


def _require_vllm_launch_option(arguments: list[str], option: str, value: str, *, label: str) -> None:
    if _vllm_launch_option_values(arguments, option) != [value]:
        raise ValueError(f"taxonomy vLLM {label} does not match launch arguments")


def _forbid_vllm_launch_option(arguments: list[str], option: str, *, label: str) -> None:
    if _vllm_launch_option_values(arguments, option):
        raise ValueError(f"taxonomy vLLM {label} must not appear in launch arguments")


def _require_single_vllm_launch_option(
    arguments: list[str], option: str, *, label: str, required: bool = True,
) -> str | None:
    """Read one unambiguous launch option from the approved transport command."""
    values = _vllm_launch_option_values(arguments, option)
    if not values:
        if required:
            raise ValueError(f"taxonomy vLLM launch {label} is missing")
        return None
    if len(values) != 1:
        raise ValueError(f"taxonomy vLLM launch {label} is duplicated")
    return values[0]


def _vllm_positional_model(arguments: list[str]) -> str | None:
    """Extract the one supported positional model form without guessing defaults.

    The transport command is recorded separately, so the argument vector may
    begin either with ``MODEL`` or with vLLM's ``serve MODEL`` subcommand form.
    Other bare tokens are deliberately not treated as model identifiers.
    """
    if not arguments or arguments[0].startswith("--"):
        return None
    model_index = 1 if arguments[0] == "serve" else 0
    if model_index == len(arguments) or arguments[model_index].startswith("--"):
        raise ValueError("taxonomy vLLM launch positional model has no value")
    return arguments[model_index]


def _extract_vllm_launch_identity(
    arguments: list[str], *, generator: Mapping[str, Any], binding_kind: str,
) -> dict[str, str | None]:
    """Bind actual launch identity to the approved generator specification.

    This parser intentionally accepts only one model identity: one positional
    model *or* one ``--model``.  Requiring explicit immutable revisions avoids
    silently inheriting mutable server or Hub defaults.
    """
    if binding_kind not in {"actual_execution", "synthetic_test"}:
        raise ValueError("taxonomy vLLM model/tokenizer binding kind is invalid")
    positional_model = _vllm_positional_model(arguments)
    explicit_model = _require_single_vllm_launch_option(
        arguments, "--model", label="model", required=False,
    )
    if positional_model is not None and explicit_model is not None:
        raise ValueError("taxonomy vLLM launch cannot use both positional and --model")
    model = positional_model if positional_model is not None else explicit_model
    if model is None:
        raise ValueError("taxonomy vLLM launch model is missing")
    revision = _require_single_vllm_launch_option(arguments, "--revision", label="revision")
    tokenizer = _require_single_vllm_launch_option(arguments, "--tokenizer", label="tokenizer")
    tokenizer_revision = _require_single_vllm_launch_option(
        arguments, "--tokenizer-revision", label="tokenizer revision",
    )
    served_model_name = _require_single_vllm_launch_option(
        arguments, "--served-model-name", label="served-model-name", required=False,
    )
    expected = {
        "model": generator["model_repository"],
        "revision": generator["model_revision"],
        "tokenizer": generator["tokenizer_repository"],
        "tokenizer revision": generator["tokenizer_revision"],
    }
    actual = {
        "model": model,
        "revision": revision,
        "tokenizer": tokenizer,
        "tokenizer revision": tokenizer_revision,
    }
    for label, value in actual.items():
        if value != expected[label]:
            raise ValueError(f"taxonomy vLLM launch {label} does not match generator specification")
    return {
        "model": model,
        "revision": revision,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "served_model_name": served_model_name,
        "binding_kind": binding_kind,
    }


def _vllm_model_tokenizer_binding_kind(generator: Mapping[str, Any]) -> str:
    """Return the explicit test/actual identity binding classification."""
    execution = generator.get("vllm_execution")
    if not isinstance(execution, Mapping) or not isinstance(execution.get("server_launch"), Mapping):
        raise ValueError("taxonomy vLLM server_launch is invalid")
    binding_kind = execution["server_launch"].get("model_tokenizer_binding_kind")
    if binding_kind not in {"actual_execution", "synthetic_test"}:
        raise ValueError("taxonomy vLLM model/tokenizer binding kind is invalid")
    return binding_kind


def _expected_vllm_request_body_template(
    *, generator: Mapping[str, Any], execution: Mapping[str, Any], sampling_values: Mapping[str, Any],
    request_model: str,
) -> dict[str, Any]:
    """Derive the exact OpenAI-compatible request template from approved controls."""
    structured = execution["structured_output"]
    thinking = execution["thinking"]
    body: dict[str, Any] = {
        "model": request_model,
        "messages": [
            {"role": "system", "content": generator["prompt_template"]},
            {"role": "user", "content": "{{taxonomy_semantic_payload_json}}"},
        ],
    }
    for name in (
        "temperature", "max_tokens", "top_p", "stop", "presence_penalty", "frequency_penalty",
        "repetition_penalty", "seed", "n", "logprobs",
    ):
        if sampling_values[name] is not None:
            body[name] = sampling_values[name]
    extra_body: dict[str, Any] = {}
    for name in ("top_k", "min_p", "stop_token_ids"):
        if sampling_values[name] is not None:
            extra_body[name] = sampling_values[name]
    if thinking["enable_thinking"] in {"enabled", "disabled"}:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": thinking["enable_thinking"] == "enabled",
        }
    if extra_body:
        body["extra_body"] = extra_body
    if structured["mode"] == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": structured["json_schema_name"],
                "schema": structured["json_schema"],
            },
        }
    return {
        "schema_version": TAXONOMY_VLLM_REQUEST_BODY_TEMPLATE_SCHEMA_VERSION,
        "endpoint": "/v1/chat/completions",
        "body": body,
    }


def _validate_vllm_execution(execution: Any, *, generator: Mapping[str, Any]) -> None:
    """Require every serving and request behavior to be explicit in the plan."""
    required = {
        "server_launch", "chat_template", "thinking", "structured_output", "sampling_request_controls",
        "request_body_template", "request_body_template_sha256",
    }
    if not isinstance(execution, Mapping) or set(execution) != required:
        raise ValueError("taxonomy vLLM execution has missing or unsupported fields")
    launch = execution.get("server_launch")
    launch_required = {
        "command", "arguments", "arguments_sha256", "generation_config_mode",
        "server_generation_config", "server_generation_config_sha256",
        "server_generation_config_launch_value", "model_tokenizer_binding_kind",
    }
    if not isinstance(launch, Mapping) or set(launch) != launch_required:
        raise ValueError("taxonomy vLLM server_launch is invalid")
    _require_nonempty_string(launch.get("command"), label="taxonomy vLLM launch command")
    arguments = launch.get("arguments")
    if not isinstance(arguments, list):
        raise ValueError("taxonomy vLLM launch arguments must be an array")
    for argument in arguments:
        _require_nonempty_string(argument, label="taxonomy vLLM launch argument")
    _require_sha256(launch.get("arguments_sha256"), label="taxonomy vLLM launch arguments")
    if launch["arguments_sha256"] != sha256_json(arguments):
        raise ValueError("taxonomy vLLM launch arguments sha256 does not match arguments")
    launch_identity = _extract_vllm_launch_identity(
        arguments,
        generator=generator,
        binding_kind=_vllm_model_tokenizer_binding_kind(generator),
    )
    if launch.get("generation_config_mode") not in {"request_controls_only", "server_generation_config"}:
        raise ValueError("taxonomy vLLM generation_config_mode is invalid")
    _forbid_vllm_launch_option(
        arguments, "--override-generation-config", label="unapproved server generation override",
    )
    server_config_hash = launch.get("server_generation_config_sha256")
    if launch["generation_config_mode"] == "request_controls_only":
        if (
            launch.get("server_generation_config") is not None
            or server_config_hash != "not_applicable"
            or launch.get("server_generation_config_launch_value") != "not_applicable"
        ):
            raise ValueError("taxonomy vLLM unused server generation config must be not_applicable")
        _require_vllm_launch_option(
            arguments, "--generation-config", "vllm", label="request_controls_only generation-config",
        )
    else:
        if not isinstance(launch.get("server_generation_config"), Mapping) or not launch["server_generation_config"]:
            raise ValueError("taxonomy vLLM server generation config is invalid")
        _validate_json_value(launch["server_generation_config"], label="taxonomy vLLM server generation config")
        _require_sha256(server_config_hash, label="taxonomy vLLM server generation config")
        if server_config_hash != sha256_json(launch["server_generation_config"]):
            raise ValueError("taxonomy vLLM server generation config sha256 does not match config")
        config_launch_value = _require_nonempty_string(
            launch.get("server_generation_config_launch_value"),
            label="taxonomy vLLM server generation config launch value",
        )
        _require_vllm_launch_option(
            arguments, "--generation-config", config_launch_value, label="server generation-config",
        )

    template = execution.get("chat_template")
    if not isinstance(template, Mapping) or set(template) != {"mode", "sha256", "content_format", "launch_argument"}:
        raise ValueError("taxonomy vLLM chat_template is invalid")
    if template.get("mode") not in {"resolved_model_template", "explicit_template", "not_applicable"}:
        raise ValueError("taxonomy vLLM chat_template mode is invalid")
    if template["mode"] == "not_applicable":
        if (
            template.get("sha256") != "not_applicable"
            or template.get("content_format") != "not_applicable"
            or template.get("launch_argument") != "not_applicable"
        ):
            raise ValueError("taxonomy vLLM unused chat_template must be explicitly not_applicable")
        _forbid_vllm_launch_option(arguments, "--chat-template", label="unused chat_template")
    else:
        _require_sha256(template.get("sha256"), label="taxonomy vLLM chat_template")
        _require_nonempty_string(template.get("content_format"), label="taxonomy vLLM chat_template content_format")
        if template["mode"] == "explicit_template":
            template_launch_value = _require_nonempty_string(
                template.get("launch_argument"), label="taxonomy vLLM chat_template launch argument",
            )
            _require_vllm_launch_option(
                arguments, "--chat-template", template_launch_value, label="chat_template",
            )
        else:
            if template.get("launch_argument") != "not_applicable":
                raise ValueError("resolved taxonomy vLLM chat_template launch argument must be not_applicable")
            _forbid_vllm_launch_option(arguments, "--chat-template", label="resolved chat_template")

    thinking = execution.get("thinking")
    if not isinstance(thinking, Mapping) or set(thinking) != {
        "enable_thinking", "reasoning_parser", "response_reasoning_content",
    }:
        raise ValueError("taxonomy vLLM thinking policy is invalid")
    if thinking.get("enable_thinking") not in {"enabled", "disabled", "not_applicable"}:
        raise ValueError("taxonomy vLLM thinking enable policy is invalid")
    if thinking["enable_thinking"] == "enabled":
        _require_nonempty_string(thinking.get("reasoning_parser"), label="taxonomy vLLM reasoning_parser")
        if thinking["reasoning_parser"] == "not_applicable":
            raise ValueError("enabled taxonomy vLLM thinking requires a reasoning_parser")
        if thinking.get("response_reasoning_content") not in {"included", "excluded"}:
            raise ValueError("enabled taxonomy vLLM thinking requires response reasoning-content policy")
        _require_vllm_launch_option(
            arguments, "--reasoning-parser", thinking["reasoning_parser"], label="reasoning_parser",
        )
    elif thinking.get("reasoning_parser") != "not_applicable" or thinking.get("response_reasoning_content") != "not_applicable":
        raise ValueError("disabled or unused taxonomy vLLM thinking must be explicitly not_applicable")
    else:
        _forbid_vllm_launch_option(arguments, "--reasoning-parser", label="disabled reasoning_parser")
        _forbid_vllm_launch_option(arguments, "--enable-reasoning", label="disabled reasoning")

    structured = execution.get("structured_output")
    if not isinstance(structured, Mapping) or set(structured) != {
        "mode", "content_format", "json_schema", "json_schema_sha256", "json_schema_name",
    }:
        raise ValueError("taxonomy vLLM structured_output is invalid")
    if structured.get("mode") not in {"json_schema", "not_applicable"}:
        raise ValueError("taxonomy vLLM structured_output mode is invalid")
    if structured["mode"] == "json_schema":
        if not isinstance(structured.get("json_schema"), Mapping) or not structured["json_schema"]:
            raise ValueError("taxonomy vLLM structured output json_schema is invalid")
        _require_nonempty_string(structured.get("json_schema_name"), label="taxonomy vLLM structured output schema name")
        _require_nonempty_string(structured.get("content_format"), label="taxonomy vLLM structured output content_format")
        _require_sha256(structured.get("json_schema_sha256"), label="taxonomy vLLM structured output schema")
        if structured["json_schema_sha256"] != sha256_json(structured["json_schema"]):
            raise ValueError("taxonomy vLLM structured output schema sha256 does not match schema")
    elif (
        structured.get("content_format") != "not_applicable"
        or structured.get("json_schema") is not None
        or structured.get("json_schema_sha256") != "not_applicable"
        or structured.get("json_schema_name") != "not_applicable"
    ):
        raise ValueError("unused taxonomy vLLM structured output must be explicitly not_applicable")

    sampling = execution.get("sampling_request_controls")
    control_specs = {
        "temperature": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be finite and non-negative"))
            if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) < 0 else None
        ),
        "max_tokens": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be a positive integer"))
            if type(value) is not int or value <= 0 else None
        ),
        "top_p": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be in (0, 1]"))
            if type(value) not in {int, float} or not math.isfinite(float(value)) or not 0 < float(value) <= 1 else None
        ),
        "top_k": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be an integer >= -1"))
            if type(value) is not int or value < -1 else None
        ),
        "min_p": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be in [0, 1]"))
            if type(value) not in {int, float} or not math.isfinite(float(value)) or not 0 <= float(value) <= 1 else None
        ),
        "stop": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be an array of strings"))
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value) else None
        ),
        "stop_token_ids": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be an array of non-negative integers"))
            if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value) else None
        ),
        "presence_penalty": lambda value, *, label: _validate_json_value(value, label=f"taxonomy vLLM {label}"),
        "frequency_penalty": lambda value, *, label: _validate_json_value(value, label=f"taxonomy vLLM {label}"),
        "repetition_penalty": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be finite and positive"))
            if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0 else None
        ),
        "seed": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be an integer"))
            if type(value) is not int else None
        ),
        "n": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be a positive integer"))
            if type(value) is not int or value <= 0 else None
        ),
        "logprobs": lambda value, *, label: (
            (_ for _ in ()).throw(ValueError(f"taxonomy vLLM {label} must be boolean"))
            if type(value) is not bool else None
        ),
    }
    if not isinstance(sampling, Mapping) or set(sampling) != set(control_specs):
        raise ValueError("taxonomy vLLM sampling_request_controls are incomplete")
    values = {
        name: _validate_vllm_control(sampling[name], label=name, validator=validator)
        for name, validator in control_specs.items()
    }
    if values["temperature"] != generator["generation_controls"]["temperature"]:
        raise ValueError("taxonomy vLLM temperature does not match generator controls")
    if values["max_tokens"] != generator["generation_controls"]["max_tokens"]:
        raise ValueError("taxonomy vLLM max_tokens does not match generator controls")
    if values["seed"] != generator["seed"]:
        raise ValueError("taxonomy vLLM seed does not match generator seed")
    request_body_template = _expected_vllm_request_body_template(
        generator=generator,
        execution=execution,
        sampling_values=values,
        request_model=launch_identity["served_model_name"] or launch_identity["model"],
    )
    if execution.get("request_body_template") != request_body_template:
        raise ValueError("taxonomy vLLM request body template does not match declared launch/request controls")
    _require_sha256(execution.get("request_body_template_sha256"), label="taxonomy vLLM request body template")
    if execution["request_body_template_sha256"] != sha256_json(request_body_template):
        raise ValueError("taxonomy vLLM request body template sha256 does not match template")


def generator_spec_sha256(generator: Mapping[str, Any]) -> str:
    """Hash the complete, execution-relevant generator specification."""
    _validate_generation_generator_spec(generator)
    return sha256_json(generator)


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("taxonomy artifact provenance must be an object")
    for key in (
        "revision_lock_sha256",
        "preparation_contract_sha256",
        "input_110k_corpus_sha256",
        "input_subset_manifest_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"taxonomy provenance {key}")
    _validate_generator(provenance.get("generator"))


def _validate_source_revisions(source_revisions: Any) -> None:
    if not isinstance(source_revisions, dict) or set(source_revisions) != {"topics_qrels", "corpus"}:
        raise ValueError("taxonomy source_revisions must contain topics_qrels and corpus")
    for key in ("topics_qrels", "corpus"):
        value = source_revisions[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"taxonomy source revision is invalid for {key}")


def _label_catalog_by_id(catalog: Any) -> dict[str, dict[str, str]]:
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("taxonomy label_catalog must be a non-empty array")
    by_id: dict[str, dict[str, str]] = {}
    seen_labels: set[str] = set()
    previous_label_id: str | None = None
    for item in catalog:
        if not isinstance(item, dict):
            raise ValueError("taxonomy label_catalog entries must be objects")
        if set(item) != {"label_id", "label"}:
            raise ValueError("taxonomy label_catalog entries must contain label_id and label only")
        label_id = _require_nonempty_string(item.get("label_id"), label="taxonomy label_id")
        label = _require_nonempty_string(item.get("label"), label="taxonomy label")
        if label != unicodedata.normalize("NFC", label):
            raise ValueError("taxonomy label must use Unicode NFC")
        normalized_label = " ".join(label.split())
        if not normalized_label or label != normalized_label:
            raise ValueError("taxonomy label must use normalized whitespace")
        if label_id in by_id:
            raise ValueError(f"taxonomy label_catalog has duplicate label_id: {label_id}")
        if previous_label_id is not None and label_id <= previous_label_id:
            raise ValueError("taxonomy label_catalog must be sorted by label_id")
        if normalized_label in seen_labels:
            raise ValueError(f"taxonomy label_catalog has duplicate normalized label: {label}")
        by_id[label_id] = {"label_id": label_id, "label": label}
        seen_labels.add(normalized_label)
        previous_label_id = label_id
    if "unknown" not in by_id:
        raise ValueError("taxonomy label_catalog must define the unknown label")
    return by_id


def _assignments_by_id(assignments: Any, *, catalog: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("taxonomy assignments must be a non-empty array")
    by_id: dict[str, dict[str, Any]] = {}
    previous_id: str | None = None
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("taxonomy assignments must contain objects")
        if set(assignment) != {"corpus_id", "label_id", "score", "status"}:
            raise ValueError("taxonomy assignment has unsupported fields")
        corpus_id = _require_nonempty_string(assignment.get("corpus_id"), label="taxonomy assignment corpus_id")
        if corpus_id in by_id:
            raise ValueError(f"taxonomy assignments have duplicate corpus_id: {corpus_id}")
        if previous_id is not None and corpus_id < previous_id:
            raise ValueError("taxonomy assignments must use deterministic corpus_id ordering")
        previous_id = corpus_id
        label_id = _require_nonempty_string(assignment.get("label_id"), label="taxonomy assignment label_id")
        if label_id not in catalog:
            raise ValueError(f"taxonomy assignment references unknown label_id: {label_id}")
        score = assignment.get("score")
        if type(score) not in {int, float} or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("taxonomy assignment score must be finite and within [0, 1]")
        status = assignment.get("status")
        if status not in {"assigned", "unknown"}:
            raise ValueError("taxonomy assignment status must be assigned or unknown")
        if status == "unknown" and (label_id != "unknown" or float(score) != 0.0):
            raise ValueError("taxonomy unknown assignment must use unknown label_id and score 0")
        if status == "assigned" and label_id == "unknown":
            raise ValueError("taxonomy assigned passage cannot use unknown label_id")
        by_id[corpus_id] = {
            "corpus_id": corpus_id,
            "label_id": label_id,
            "score": float(score),
            "status": status,
        }
    return by_id


def taxonomy_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Return a canonical content hash; timestamps live in the outer manifest."""
    return sha256_json(artifact)


def validate_taxonomy_artifact(
    artifact: Mapping[str, Any], *, expected_corpus_ids: set[str] | None = None
) -> dict[str, Any]:
    """Validate one source or projected artifact and return non-performance QA."""
    if not isinstance(artifact, Mapping):
        raise ValueError("taxonomy artifact must be an object")
    required = {
        "schema_version",
        "taxonomy_artifact_id",
        "dataset",
        "language",
        "source_revisions",
        "retrieval_unit",
        "source_scale",
        "projection_scale",
        "input_contract",
        "provenance",
        "label_catalog",
        "assignments",
        "projection",
    }
    if set(artifact) != required:
        raise ValueError("taxonomy artifact has missing or unsupported top-level fields")
    if artifact.get("schema_version") != TAXONOMY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("taxonomy artifact schema_version is invalid")
    if artifact.get("dataset") != "MIRACL" or artifact.get("language") != "ko":
        raise ValueError("taxonomy artifact must identify MIRACL Korean")
    _validate_source_revisions(artifact.get("source_revisions"))
    if artifact.get("retrieval_unit") != RETRIEVAL_UNIT:
        raise ValueError("taxonomy artifact retrieval_unit must be passage")
    _require_nonempty_string(artifact.get("taxonomy_artifact_id"), label="taxonomy artifact id")
    if artifact.get("source_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy artifact source_scale must be 110K")
    if artifact.get("projection_scale") not in SCALE_SIZES:
        raise ValueError("taxonomy artifact projection_scale must be 20K, 50K, or 110K")
    _validate_semantic_input_contract(artifact.get("input_contract"))
    _validate_provenance(artifact.get("provenance"))
    catalog = _label_catalog_by_id(artifact.get("label_catalog"))
    assignments = _assignments_by_id(artifact.get("assignments"), catalog=catalog)
    projection = artifact.get("projection")
    if not isinstance(projection, dict) or set(projection) != {"method", "source_artifact_sha256"}:
        raise ValueError("taxonomy artifact projection metadata is invalid")
    if artifact["projection_scale"] == TAXONOMY_SOURCE_SCALE:
        if projection.get("method") != SOURCE_IDENTITY_METHOD or projection.get("source_artifact_sha256") is not None:
            raise ValueError("110K taxonomy artifact must use the identity projection declaration")
    else:
        if projection.get("method") != FILTER_PROJECTION_METHOD:
            raise ValueError("projected taxonomy artifact must use fixed 110K filter projection")
        _require_sha256(projection.get("source_artifact_sha256"), label="taxonomy projection source artifact")
    if expected_corpus_ids is not None:
        actual_ids = set(assignments)
        missing = expected_corpus_ids - actual_ids
        orphan = actual_ids - expected_corpus_ids
        if orphan:
            raise ValueError(f"taxonomy artifact has orphan passage mapping: {sorted(orphan)[0]}")
        if missing:
            raise ValueError(f"taxonomy artifact coverage is missing passage: {sorted(missing)[0]}")

    assigned_labels = Counter(
        assignment["label_id"] for assignment in assignments.values()
        if assignment["status"] == "assigned"
    )
    unknown_count = sum(assignment["status"] == "unknown" for assignment in assignments.values())
    assigned_count = len(assignments) - unknown_count
    largest_label_count = max(assigned_labels.values(), default=0)
    return {
        "taxonomy_artifact_id": artifact["taxonomy_artifact_id"],
        "artifact_sha256": taxonomy_artifact_sha256(artifact),
        "passage_count": len(assignments),
        "coverage_rate": 1.0 if expected_corpus_ids is not None else None,
        "unknown_assignment_count": unknown_count,
        "unknown_assignment_rate": unknown_count / len(assignments),
        "assigned_label_count": len(assigned_labels),
        "label_distribution": dict(sorted(assigned_labels.items())),
        "minimum_assigned_label_size": min(assigned_labels.values(), default=0),
        "maximum_assigned_label_size": largest_label_count,
        "maximum_assigned_label_share": (
            largest_label_count / assigned_count if assigned_count else None
        ),
    }


def build_flat_l1_consumer_adapter(
    artifact: Mapping[str, Any], *, expected_corpus_ids: set[str]
) -> dict[str, Any]:
    """Derive the one approved future-consumer shape without wiring it in.

    The existing retriever performs an exact dictionary match against L1/L2.
    This adapter therefore uses the artifact display label as L1, exposes the
    same labels to the future agent prompt, keeps L2 empty, and never forwards
    assignment confidence as a retrieval weight.
    """
    validate_taxonomy_artifact(artifact, expected_corpus_ids=expected_corpus_ids)
    catalog = _label_catalog_by_id(artifact["label_catalog"])
    document_taxonomy = {
        assignment["corpus_id"]: {"L1": catalog[assignment["label_id"]]["label"]}
        for assignment in artifact["assignments"]
        if assignment["status"] == "assigned"
    }
    adapter = {
        "schema_version": TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION,
        "taxonomy_artifact_id": artifact["taxonomy_artifact_id"],
        "taxonomy_artifact_sha256": taxonomy_artifact_sha256(artifact),
        "dataset": artifact["dataset"],
        "language": artifact["language"],
        "retrieval_unit": artifact["retrieval_unit"],
        "projection_scale": artifact["projection_scale"],
        "agent_taxonomy_schema": {
            "L1": [item["label"] for item in artifact["label_catalog"] if item["label_id"] != "unknown"],
            "L2": {},
        },
        "document_taxonomy": document_taxonomy,
        "score_handling": FLAT_L1_SCORE_HANDLING,
        "unknown_handling": FLAT_L1_UNKNOWN_HANDLING,
    }
    validate_flat_l1_consumer_adapter(adapter, artifact, expected_corpus_ids=expected_corpus_ids)
    return adapter


def validate_flat_l1_consumer_adapter(
    adapter: Mapping[str, Any], artifact: Mapping[str, Any], *, expected_corpus_ids: set[str]
) -> None:
    """Require an exact, label-preserving flat-L1 consumer projection."""
    validate_taxonomy_artifact(artifact, expected_corpus_ids=expected_corpus_ids)
    required = {
        "schema_version", "taxonomy_artifact_id", "taxonomy_artifact_sha256", "dataset", "language",
        "retrieval_unit", "projection_scale", "agent_taxonomy_schema", "document_taxonomy",
        "score_handling", "unknown_handling",
    }
    if not isinstance(adapter, Mapping) or set(adapter) != required:
        raise ValueError("flat L1 consumer adapter has missing or unsupported fields")
    if adapter.get("schema_version") != TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION:
        raise ValueError("flat L1 consumer adapter schema_version is invalid")
    for key in ("taxonomy_artifact_id", "dataset", "language", "retrieval_unit", "projection_scale"):
        if adapter.get(key) != artifact.get(key):
            raise ValueError(f"flat L1 consumer adapter {key} does not match artifact")
    if adapter.get("taxonomy_artifact_sha256") != taxonomy_artifact_sha256(artifact):
        raise ValueError("flat L1 consumer adapter artifact hash does not match")
    if adapter.get("score_handling") != FLAT_L1_SCORE_HANDLING:
        raise ValueError("flat L1 consumer adapter must not use score as a boost weight")
    if adapter.get("unknown_handling") != FLAT_L1_UNKNOWN_HANDLING:
        raise ValueError("flat L1 consumer adapter must exclude unknown assignments from boost eligibility")
    catalog = _label_catalog_by_id(artifact["label_catalog"])
    expected_labels = [item["label"] for item in artifact["label_catalog"] if item["label_id"] != "unknown"]
    if adapter.get("agent_taxonomy_schema") != {"L1": expected_labels, "L2": {}}:
        raise ValueError("flat L1 agent taxonomy schema must expose artifact display labels exactly")
    document_taxonomy = adapter.get("document_taxonomy")
    if not isinstance(document_taxonomy, dict):
        raise ValueError("flat L1 consumer document_taxonomy must be an object")
    expected_mapping = {
        assignment["corpus_id"]: {"L1": catalog[assignment["label_id"]]["label"]}
        for assignment in artifact["assignments"]
        if assignment["status"] == "assigned"
    }
    unknown_ids = {
        assignment["corpus_id"] for assignment in artifact["assignments"]
        if assignment["status"] == "unknown"
    }
    if set(document_taxonomy) & unknown_ids:
        raise ValueError("flat L1 consumer adapter includes unknown assignment in boost eligibility")
    if set(document_taxonomy) != set(expected_mapping):
        raise ValueError("flat L1 consumer adapter passage coverage does not match assigned artifact passages")
    if document_taxonomy != expected_mapping:
        raise ValueError("flat L1 consumer adapter must use artifact display label for exact match")


def project_taxonomy_artifact(
    source_artifact: Mapping[str, Any], target_corpus_ids: set[str], *, target_scale: int
) -> dict[str, Any]:
    """Filter the one 110K mapping without relabeling or reassigning passages."""
    validate_taxonomy_artifact(source_artifact)
    if source_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy projection source must be the 110K identity artifact")
    if target_scale not in SCALE_SIZES or target_scale == TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy filter projection target must be 20K or 50K")
    source_assignments = {row["corpus_id"]: row for row in source_artifact["assignments"]}
    unknown = target_corpus_ids - set(source_assignments)
    if unknown:
        raise ValueError(f"taxonomy projection target has passage outside 110K source: {sorted(unknown)[0]}")
    projected = {
        **dict(source_artifact),
        "projection_scale": target_scale,
        "assignments": [
            dict(row) for row in source_artifact["assignments"]
            if row["corpus_id"] in target_corpus_ids
        ],
        "projection": {
            "method": FILTER_PROJECTION_METHOD,
            "source_artifact_sha256": taxonomy_artifact_sha256(source_artifact),
        },
    }
    validate_taxonomy_projection(
        source_artifact,
        projected,
        expected_corpus_ids=target_corpus_ids,
        target_scale=target_scale,
    )
    return projected


def validate_taxonomy_projection(
    source_artifact: Mapping[str, Any], projected_artifact: Mapping[str, Any], *,
    expected_corpus_ids: set[str], target_scale: int,
) -> None:
    """Require a byte-independent, assignment-exact filter projection."""
    validate_taxonomy_artifact(source_artifact)
    validate_taxonomy_artifact(projected_artifact, expected_corpus_ids=expected_corpus_ids)
    if source_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy projection source must remain the 110K artifact")
    if target_scale not in {20_000, 50_000} or projected_artifact.get("projection_scale") != target_scale:
        raise ValueError("taxonomy projection target scale is invalid")
    for field in (
        "schema_version",
        "taxonomy_artifact_id",
        "dataset",
        "language",
        "source_revisions",
        "retrieval_unit",
        "source_scale",
        "input_contract",
        "provenance",
        "label_catalog",
    ):
        if projected_artifact.get(field) != source_artifact.get(field):
            raise ValueError(f"taxonomy projection must preserve {field}")
    projection = projected_artifact.get("projection")
    if projection != {
        "method": FILTER_PROJECTION_METHOD,
        "source_artifact_sha256": taxonomy_artifact_sha256(source_artifact),
    }:
        raise ValueError("taxonomy projection source hash or method is invalid")
    expected_assignments = [
        row for row in source_artifact["assignments"]
        if row["corpus_id"] in expected_corpus_ids
    ]
    if projected_artifact.get("assignments") != expected_assignments:
        raise ValueError("taxonomy projection changes a shared passage assignment")


def _validate_file_record(record: Any, *, data_dir: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} file record must be an object")
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} file record has invalid relative_path")
    root = data_dir.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"{label} artifact file is missing: {relative_path}")
    if record.get("byte_size") != path.stat().st_size:
        raise ValueError(f"{label} artifact byte_size does not match manifest")
    _require_sha256(record.get("sha256"), label=f"{label} artifact")
    if record["sha256"] != sha256_file(path):
        raise ValueError(f"{label} artifact sha256 does not match manifest")
    return path


def _load_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} taxonomy artifact JSON is invalid") from error
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} taxonomy artifact must be an object")
    return artifact


def build_generator_code_contract(repo_root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    """Hash the exact generator files independently of the result artifact."""
    root = repo_root.resolve()
    paths = sorted(set(relative_paths))
    if not paths:
        raise ValueError("taxonomy generator code contract requires at least one generator file")
    records: list[dict[str, str]] = []
    for relative_path in paths:
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError("taxonomy generator code contract has invalid relative path")
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"taxonomy generator code file is missing: {relative_path}")
        records.append({"relative_path": relative_path, "sha256": sha256_file(path)})
    aggregate_payload = {
        "aggregate_hash_method": GENERATOR_CODE_AGGREGATE_METHOD,
        "generator_files": records,
    }
    contract = {
        "schema_version": TAXONOMY_GENERATOR_CODE_CONTRACT_SCHEMA_VERSION,
        **aggregate_payload,
        "generator_code_sha256": sha256_json(aggregate_payload),
    }
    validate_generator_code_contract(contract, repo_root=root)
    return contract


def validate_generator_code_contract(
    contract: Mapping[str, Any], *, repo_root: Path | None = None
) -> None:
    """Validate the file-list aggregate and, when available, each file byte."""
    required = {"schema_version", "aggregate_hash_method", "generator_files", "generator_code_sha256"}
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError("taxonomy generator code contract has missing or unsupported fields")
    if contract.get("schema_version") != TAXONOMY_GENERATOR_CODE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("taxonomy generator code contract schema_version is invalid")
    if contract.get("aggregate_hash_method") != GENERATOR_CODE_AGGREGATE_METHOD:
        raise ValueError("taxonomy generator code aggregate hash method is invalid")
    files = contract.get("generator_files")
    if not isinstance(files, list) or not files:
        raise ValueError("taxonomy generator code contract requires generator_files")
    previous_path: str | None = None
    for record in files:
        if not isinstance(record, dict) or set(record) != {"relative_path", "sha256"}:
            raise ValueError("taxonomy generator code file record is invalid")
        relative_path = record.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("taxonomy generator code file path is invalid")
        if previous_path is not None and relative_path <= previous_path:
            raise ValueError("taxonomy generator code file records must be uniquely sorted")
        previous_path = relative_path
        _require_sha256(record.get("sha256"), label="taxonomy generator code file")
    aggregate_payload = {
        "aggregate_hash_method": contract["aggregate_hash_method"],
        "generator_files": files,
    }
    if contract.get("generator_code_sha256") != sha256_json(aggregate_payload):
        raise ValueError("taxonomy generator code aggregate hash does not match file list")
    if repo_root is not None:
        root = repo_root.resolve()
        for record in files:
            path = (root / record["relative_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"taxonomy generator code file is missing: {record['relative_path']}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"taxonomy generator code file hash does not match: {record['relative_path']}")


def generator_contract_sha256(contract: Mapping[str, Any]) -> str:
    validate_generator_code_contract(contract)
    return sha256_json(contract)


def _validate_input_provenance(input_provenance: Any) -> None:
    if not isinstance(input_provenance, dict):
        raise ValueError("taxonomy generation input provenance must be an object")
    required = {
        "revision_lock_sha256",
        "preparation_contract_sha256",
        "input_110k_corpus_sha256",
        "input_subset_manifest_sha256",
    }
    if set(input_provenance) != required:
        raise ValueError("taxonomy generation input provenance has missing or unsupported fields")
    for key in sorted(required):
        _require_sha256(input_provenance.get(key), label=f"taxonomy generation input {key}")


def build_taxonomy_generation_plan(
    *, input_provenance: Mapping[str, str], generator_source_commit: str,
    generator_code_contract_sha256: str, generator_code_sha256: str,
    generator: Mapping[str, Any], input_contract: Mapping[str, Any],
    run_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the artifact-free object that must be approved before generation."""
    plan = {
        "schema_version": TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION,
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "source_scale": TAXONOMY_SOURCE_SCALE,
        "input_provenance": dict(input_provenance),
        "generator_source_commit": generator_source_commit,
        "generator_code_contract_sha256": generator_code_contract_sha256,
        "generator_code_sha256": generator_code_sha256,
        "generator": dict(generator),
        "generator_spec_sha256": generator_spec_sha256(generator),
        "input_contract": dict(input_contract),
        "batch_contract_schema_version": TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION,
        "run_controls": dict(run_controls),
        "output_artifact_schema_version": TAXONOMY_ARTIFACT_SCHEMA_VERSION,
        "projection_method": FILTER_PROJECTION_METHOD,
        "determinism_mode": generator.get("determinism_mode"),
    }
    validate_taxonomy_generation_plan(plan)
    return plan


def validate_taxonomy_generation_plan(plan: Mapping[str, Any]) -> None:
    """Validate an artifact-free pre-generation plan."""
    required = {
        "schema_version", "dataset", "language", "retrieval_unit", "source_scale",
        "input_provenance", "generator_source_commit", "generator_code_contract_sha256",
        "generator_code_sha256", "generator", "generator_spec_sha256", "input_contract",
        "batch_contract_schema_version", "run_controls", "output_artifact_schema_version",
        "projection_method", "determinism_mode",
    }
    if not isinstance(plan, Mapping) or set(plan) != required:
        raise ValueError("taxonomy generation plan has missing, artifact-derived, or unsupported fields")
    if plan.get("schema_version") != TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION:
        raise ValueError("taxonomy generation plan schema_version is invalid")
    if plan.get("dataset") != "MIRACL" or plan.get("language") != "ko":
        raise ValueError("taxonomy generation plan must identify MIRACL Korean")
    if plan.get("retrieval_unit") != RETRIEVAL_UNIT or plan.get("source_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy generation plan retrieval unit or source scale is invalid")
    _validate_input_provenance(plan.get("input_provenance"))
    if not isinstance(plan.get("generator_source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", plan["generator_source_commit"]
    ):
        raise ValueError("taxonomy generation plan generator source commit is invalid")
    _require_sha256(plan.get("generator_code_contract_sha256"), label="taxonomy generation plan code contract")
    _require_sha256(plan.get("generator_code_sha256"), label="taxonomy generation plan code aggregate")
    _validate_generation_generator_spec(plan.get("generator"))
    if plan["generator"]["generator_code_sha256"] != plan["generator_code_sha256"]:
        raise ValueError("taxonomy generation plan generator code hash does not match generator metadata")
    _require_sha256(plan.get("generator_spec_sha256"), label="taxonomy generation plan generator specification")
    if plan["generator_spec_sha256"] != generator_spec_sha256(plan["generator"]):
        raise ValueError("taxonomy generation plan generator specification hash does not match generator metadata")
    _validate_semantic_input_contract(plan.get("input_contract"))
    if plan.get("batch_contract_schema_version") != TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION:
        raise ValueError("taxonomy generation plan batch contract version is invalid")
    _validate_run_controls(plan.get("run_controls"))
    if plan.get("output_artifact_schema_version") != TAXONOMY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("taxonomy generation plan output artifact schema version is invalid")
    if plan.get("projection_method") != FILTER_PROJECTION_METHOD:
        raise ValueError("taxonomy generation plan projection method is invalid")
    if plan.get("determinism_mode") != plan["generator"].get("determinism_mode"):
        raise ValueError("taxonomy generation plan determinism mode does not match generator metadata")


def generation_plan_sha256(plan: Mapping[str, Any]) -> str:
    validate_taxonomy_generation_plan(plan)
    return sha256_json(plan)


def validate_taxonomy_approval_record(record: Mapping[str, Any], *, allow_synthetic: bool = False) -> None:
    """Validate an approval object; synthetic records cannot authorize execution."""
    required = {
        "schema_version", "approval_kind", "status", "approved_generator_source_commit",
        "approved_generation_plan_sha256", "approved_generator_contract_sha256",
        "approved_generator_code_sha256",
        "approved_by", "approved_at", "approval_basis",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("taxonomy approval record has missing or unsupported fields")
    if record.get("schema_version") != TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION:
        raise ValueError("taxonomy approval record schema_version is invalid")
    if not isinstance(record.get("approved_generator_source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", record["approved_generator_source_commit"]
    ):
        raise ValueError("taxonomy approval record generator source commit is invalid")
    _require_sha256(record.get("approved_generation_plan_sha256"), label="taxonomy approval generation plan")
    _require_sha256(record.get("approved_generator_contract_sha256"), label="taxonomy approval generator contract")
    _require_sha256(record.get("approved_generator_code_sha256"), label="taxonomy approval generator code")
    _require_nonempty_string(record.get("approved_by"), label="taxonomy approval approver")
    _require_rfc3339_timestamp(record.get("approved_at"), label="taxonomy approval timestamp")
    _require_nonempty_string(record.get("approval_basis"), label="taxonomy approval basis")
    kind = record.get("approval_kind")
    status = record.get("status")
    if kind == "actual_execution" and status == "approved_for_generation":
        return
    if kind == "synthetic_test" and status == "synthetic_only":
        if allow_synthetic:
            return
        raise ValueError("synthetic taxonomy approval record cannot authorize actual generation")
    raise ValueError("taxonomy approval record kind/status is invalid")


def approval_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash an approval record for post-generation manifest binding."""
    validate_taxonomy_approval_record(record, allow_synthetic=True)
    return sha256_json(record)


def _generator_source_git_state(generator_source_root: Path) -> tuple[str, str]:
    """Read the actual generator checkout state; control files live elsewhere."""
    root = generator_source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"taxonomy generator source root is missing: {root}")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("taxonomy generator source root must be a Git checkout") from error
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("taxonomy generator source Git HEAD is invalid")
    return head, porcelain


def validate_taxonomy_generation_preflight(
    generation_plan: Mapping[str, Any], *, approval_record: Mapping[str, Any] | None,
    generator_code_contract: Mapping[str, Any] | None,
    verified_input_provenance: Mapping[str, str], generator_source_root: Path,
) -> None:
    """Require independently approved code/source and a clean checkout.

    A real generator must call this before generation.  The generator source
    checkout is inspected directly. Plan/approval/control artifacts are read
    separately and therefore cannot make that source checkout dirty.
    """
    if approval_record is None:
        raise FileNotFoundError("taxonomy generation approval record is missing")
    if generator_code_contract is None:
        raise FileNotFoundError("taxonomy generator code contract is missing")
    validate_taxonomy_generation_plan(generation_plan)
    if _vllm_model_tokenizer_binding_kind(generation_plan["generator"]) == "synthetic_test":
        raise ValueError("synthetic taxonomy vLLM model/tokenizer binding cannot authorize actual generation")
    validate_taxonomy_approval_record(approval_record)
    actual_generator_source_commit, git_status_porcelain = _generator_source_git_state(generator_source_root)
    if git_status_porcelain.strip():
        raise ValueError("taxonomy generation dirty Git worktree is prohibited")
    if generation_plan["generator_source_commit"] != actual_generator_source_commit:
        raise ValueError("taxonomy generation plan generator source commit does not match checkout")
    validate_generator_code_contract(generator_code_contract, repo_root=generator_source_root)
    contract_hash = generator_contract_sha256(generator_code_contract)
    if generation_plan["generator_code_contract_sha256"] != contract_hash:
        raise ValueError("taxonomy generation plan code contract hash does not match verified contract")
    code_hash = generator_code_contract["generator_code_sha256"]
    if generation_plan["generator_code_sha256"] != code_hash:
        raise ValueError("taxonomy generation plan code aggregate does not match verified code contract")
    _validate_input_provenance(dict(verified_input_provenance))
    if generation_plan["input_provenance"] != dict(verified_input_provenance):
        raise ValueError("taxonomy generation plan input provenance does not match verified inputs")
    plan_hash = generation_plan_sha256(generation_plan)
    if approval_record["approved_generation_plan_sha256"] != plan_hash:
        raise ValueError("taxonomy approval generation plan does not match current plan")
    if approval_record["approved_generator_source_commit"] != generation_plan["generator_source_commit"]:
        raise ValueError("taxonomy approval generator source commit does not match generation plan")
    if approval_record["approved_generator_contract_sha256"] != contract_hash:
        raise ValueError("taxonomy approval generator contract does not match generation plan")
    if approval_record["approved_generator_code_sha256"] != code_hash:
        raise ValueError("taxonomy approval generator code does not match generation plan")


def _validate_receipt_raw_response_record(record: Any, *, raw_response_dir: Path | None = None) -> Mapping[str, Any]:
    required = {
        "batch_ordinal", "generation_request_sha256", "relative_path", "byte_size", "sha256",
        "attempt_count", "retry_count",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("taxonomy generation receipt raw response record has missing or unsupported fields")
    if type(record.get("batch_ordinal")) is not int or record["batch_ordinal"] < 0:
        raise ValueError("taxonomy generation receipt raw response batch_ordinal is invalid")
    _require_sha256(record.get("generation_request_sha256"), label="taxonomy generation receipt raw response request")
    if not isinstance(record.get("relative_path"), str) or not record["relative_path"]:
        raise ValueError("taxonomy generation receipt raw response relative_path is invalid")
    if Path(record["relative_path"]).is_absolute() or ".." in Path(record["relative_path"]).parts:
        raise ValueError("taxonomy generation receipt raw response relative_path escapes response root")
    if type(record.get("byte_size")) is not int or record["byte_size"] < 0:
        raise ValueError("taxonomy generation receipt raw response byte_size is invalid")
    _require_sha256(record.get("sha256"), label="taxonomy generation receipt raw response")
    if type(record.get("attempt_count")) is not int or record["attempt_count"] <= 0:
        raise ValueError("taxonomy generation receipt raw response attempt_count is invalid")
    if type(record.get("retry_count")) is not int or record["retry_count"] < 0:
        raise ValueError("taxonomy generation receipt raw response retry_count is invalid")
    if record["attempt_count"] != record["retry_count"] + 1:
        raise ValueError("taxonomy generation receipt raw response attempt/retry counts do not match")
    if raw_response_dir is not None:
        root = raw_response_dir.resolve()
        path = (root / record["relative_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"taxonomy generation raw response is missing: {record['relative_path']}")
        if path.stat().st_size != record["byte_size"]:
            raise ValueError("taxonomy generation raw response byte_size does not match receipt")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("taxonomy generation raw response sha256 does not match receipt")
    return record


def _validate_taxonomy_generation_receipt_shape(receipt: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "status", "generation_plan_sha256", "approval_record_sha256",
        "generator_contract_sha256", "generator_source_commit", "generation_run_sha256",
        "ordered_generation_request_sha256s", "raw_response_files", "assignment_canonical_sha256",
        "batch_summary", "started_at", "ended_at", "runtime", "determinism_status", "replay_required",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("taxonomy generation receipt has missing or unsupported fields")
    if receipt.get("schema_version") != TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("taxonomy generation receipt schema_version is invalid")
    if receipt.get("status") not in {"complete", "failed", "partial"}:
        raise ValueError("taxonomy generation receipt status is invalid")
    for key in (
        "generation_plan_sha256", "approval_record_sha256", "generator_contract_sha256", "generation_run_sha256",
    ):
        _require_sha256(receipt.get(key), label=f"taxonomy generation receipt {key}")
    if not isinstance(receipt.get("generator_source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", receipt["generator_source_commit"]
    ):
        raise ValueError("taxonomy generation receipt generator_source_commit is invalid")
    requests = receipt.get("ordered_generation_request_sha256s")
    if not isinstance(requests, list) or not requests:
        raise ValueError("taxonomy generation receipt ordered request hashes must be a non-empty array")
    for request in requests:
        _require_sha256(request, label="taxonomy generation receipt ordered request")
    if len(set(requests)) != len(requests):
        raise ValueError("taxonomy generation receipt ordered request hashes are duplicated")
    raw_responses = receipt.get("raw_response_files")
    if not isinstance(raw_responses, list):
        raise ValueError("taxonomy generation receipt raw_response_files must be an array")
    summary = receipt.get("batch_summary")
    if not isinstance(summary, Mapping) or set(summary) != {"total", "succeeded", "failed", "retries"}:
        raise ValueError("taxonomy generation receipt batch_summary is invalid")
    for key in ("total", "succeeded", "failed", "retries"):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise ValueError(f"taxonomy generation receipt batch_summary {key} is invalid")
    if summary["total"] != len(requests) or summary["succeeded"] + summary["failed"] != summary["total"]:
        raise ValueError("taxonomy generation receipt batch_summary totals do not match requests")
    if len(raw_responses) != summary["succeeded"]:
        raise ValueError("taxonomy generation receipt successful batch count does not match raw response records")
    seen_ordinals: set[int] = set()
    for record in raw_responses:
        _validate_receipt_raw_response_record(record)
        if record["batch_ordinal"] in seen_ordinals:
            raise ValueError("taxonomy generation receipt raw response batch_ordinal is duplicated")
        seen_ordinals.add(record["batch_ordinal"])
    if receipt["status"] == "complete":
        if summary["succeeded"] != summary["total"] or summary["failed"] != 0:
            raise ValueError("complete taxonomy generation receipt must succeed for every batch")
        _require_sha256(receipt.get("assignment_canonical_sha256"), label="taxonomy generation receipt assignment")
    elif receipt.get("assignment_canonical_sha256") is not None:
        raise ValueError("partial or failed taxonomy generation receipt cannot contain assignment hash")
    _require_rfc3339_timestamp(receipt.get("started_at"), label="taxonomy generation receipt started_at")
    _require_rfc3339_timestamp(receipt.get("ended_at"), label="taxonomy generation receipt ended_at")
    started = datetime.fromisoformat(receipt["started_at"].replace("Z", "+00:00"))
    ended = datetime.fromisoformat(receipt["ended_at"].replace("Z", "+00:00"))
    if ended < started:
        raise ValueError("taxonomy generation receipt ended_at precedes started_at")
    if receipt.get("determinism_status") not in {"deterministic", "replay_required"}:
        raise ValueError("taxonomy generation receipt determinism_status is invalid")
    if type(receipt.get("replay_required")) is not bool:
        raise ValueError("taxonomy generation receipt replay_required must be a boolean")
    if receipt["replay_required"] != (receipt["determinism_status"] == "replay_required"):
        raise ValueError("taxonomy generation receipt replay_required does not match determinism_status")


def taxonomy_generation_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash a structurally valid execution receipt before manifest binding."""
    _validate_taxonomy_generation_receipt_shape(receipt)
    return sha256_json(receipt)


def _require_verified_taxonomy_generation_receipt(
    receipt: Any,
) -> VerifiedTaxonomyGenerationReceipt:
    if (
        not isinstance(receipt, VerifiedTaxonomyGenerationReceipt)
        or receipt._verification_token is not _VERIFIED_RECEIPT_TOKEN
    ):
        raise ValueError(
            "taxonomy artifact manifest requires a verified generation receipt from raw-response validation"
        )
    return receipt


def validate_taxonomy_generation_receipt(
    receipt: Mapping[str, Any], *, generation_plan: Mapping[str, Any],
    approval_record: Mapping[str, Any], generator_code_contract: Mapping[str, Any],
    generation_run: Mapping[str, Any], raw_response_dir: Path,
) -> VerifiedTaxonomyGenerationReceipt | None:
    """Validate raw response bytes and the complete-run assignment receipt.

    A receipt is the only bridge from transport responses to a future artifact.
    It may describe a failed or partial run for diagnosis, but only a complete
    receipt returns verified assignments and can be bound to a manifest.
    """
    _validate_taxonomy_generation_receipt_shape(receipt)
    validate_taxonomy_generation_plan(generation_plan)
    if _vllm_model_tokenizer_binding_kind(generation_plan["generator"]) == "synthetic_test":
        raise ValueError("synthetic taxonomy vLLM model/tokenizer binding cannot produce a verified receipt")
    validate_taxonomy_approval_record(approval_record)
    validate_generator_code_contract(generator_code_contract)
    validate_taxonomy_generator_run_against_plan(generation_run, generation_plan)
    plan_hash = generation_plan_sha256(generation_plan)
    contract_hash = generator_contract_sha256(generator_code_contract)
    code_hash = generator_code_contract["generator_code_sha256"]
    approval_hash = approval_record_sha256(approval_record)
    if receipt["generation_plan_sha256"] != plan_hash:
        raise ValueError("taxonomy generation receipt plan hash does not match plan")
    if receipt["approval_record_sha256"] != approval_hash:
        raise ValueError("taxonomy generation receipt approval hash does not match approval record")
    if receipt["generator_contract_sha256"] != contract_hash:
        raise ValueError("taxonomy generation receipt generator contract hash does not match contract")
    if generation_plan["generator_code_contract_sha256"] != contract_hash:
        raise ValueError("taxonomy generation receipt plan code contract does not match actual contract")
    if generation_plan["generator_code_sha256"] != code_hash:
        raise ValueError("taxonomy generation receipt plan generator code does not match actual contract")
    if generation_plan["generator"]["generator_code_sha256"] != code_hash:
        raise ValueError("taxonomy generation receipt plan generator metadata code does not match actual contract")
    if approval_record["approved_generation_plan_sha256"] != plan_hash:
        raise ValueError("taxonomy approval generation plan does not match receipt plan")
    if approval_record["approved_generator_source_commit"] != generation_plan["generator_source_commit"]:
        raise ValueError("taxonomy approval generator source commit does not match receipt plan")
    if approval_record["approved_generator_contract_sha256"] != contract_hash:
        raise ValueError("taxonomy approval generator contract does not match receipt contract")
    if approval_record["approved_generator_code_sha256"] != code_hash:
        raise ValueError("taxonomy approval generator code does not match actual contract")
    if receipt["generator_source_commit"] != generation_plan["generator_source_commit"]:
        raise ValueError("taxonomy generation receipt generator source commit does not match plan")
    if receipt["generation_run_sha256"] != generation_run["generation_run_sha256"]:
        raise ValueError("taxonomy generation receipt generation run hash does not match run")
    if receipt["runtime"] != generation_plan["generator"]["runtime"]:
        raise ValueError("taxonomy generation receipt runtime does not match approved generator specification")
    if receipt["determinism_status"] != generation_plan["determinism_mode"]:
        raise ValueError("taxonomy generation receipt determinism_status does not match plan")
    expected_requests = [batch["generation_request_sha256"] for batch in generation_run["batches"]]
    if receipt["ordered_generation_request_sha256s"] != expected_requests:
        raise ValueError("taxonomy generation receipt ordered request hashes do not match run")
    records: dict[int, Mapping[str, Any]] = {}
    responses: list[Mapping[str, Any]] = []
    retries = 0
    max_retries_per_batch = generation_plan["run_controls"]["max_retries"]
    for record in receipt["raw_response_files"]:
        _validate_receipt_raw_response_record(record, raw_response_dir=raw_response_dir)
        if record["retry_count"] > max_retries_per_batch:
            raise ValueError("taxonomy generation receipt batch retry_count exceeds approved per-batch max_retries")
        ordinal = record["batch_ordinal"]
        if ordinal in records or ordinal >= len(generation_run["batches"]):
            raise ValueError("taxonomy generation receipt has duplicate or out-of-range successful batch")
        expected_batch = generation_run["batches"][ordinal]
        if record["generation_request_sha256"] != expected_batch["generation_request_sha256"]:
            raise ValueError("taxonomy generation receipt raw response belongs to another batch or run")
        path = (raw_response_dir.resolve() / record["relative_path"]).resolve()
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("taxonomy generation receipt raw response JSON is invalid") from error
        if not isinstance(response, Mapping):
            raise ValueError("taxonomy generation receipt raw response must be an object")
        if response.get("generation_request_sha256") != record["generation_request_sha256"]:
            raise ValueError("taxonomy generation receipt raw response request does not match file record")
        records[ordinal] = record
        responses.append(response)
        retries += record["retry_count"]
    if list(records) != sorted(records):
        raise ValueError("taxonomy generation receipt raw response records must be batch-ordinal sorted")
    summary = receipt["batch_summary"]
    if len(records) != summary["succeeded"] or retries != summary["retries"]:
        raise ValueError("taxonomy generation receipt successful batch or retry count does not match raw records")
    if receipt["status"] != "complete":
        return None
    assignments = rejoin_taxonomy_generator_run_outputs(generation_run, responses)
    if receipt["assignment_canonical_sha256"] != sha256_json(assignments):
        raise ValueError("taxonomy generation receipt assignment hash does not match raw responses")
    return VerifiedTaxonomyGenerationReceipt(
        receipt_sha256=taxonomy_generation_receipt_sha256(receipt),
        generation_plan_sha256=receipt["generation_plan_sha256"],
        approval_record_sha256=receipt["approval_record_sha256"],
        generator_contract_sha256=receipt["generator_contract_sha256"],
        generator_code_sha256=code_hash,
        generator_spec_sha256=generation_plan["generator_spec_sha256"],
        generator_source_commit=receipt["generator_source_commit"],
        generation_run_sha256=receipt["generation_run_sha256"],
        assignment_canonical_sha256=receipt["assignment_canonical_sha256"],
        assignments=tuple(dict(row) for row in assignments),
        _verification_token=_VERIFIED_RECEIPT_TOKEN,
    )


def build_taxonomy_artifact_manifest(
    *, full_artifact: Mapping[str, Any], artifact_records: Mapping[int, Mapping[str, Any]],
    generator_source_commit: str, generator_contract_sha256: str, generation_plan_sha256: str,
    approval_record_sha256: str, verified_receipt: VerifiedTaxonomyGenerationReceipt,
) -> dict[str, Any]:
    """Build the tracked manifest; large artifact bodies remain outside Git."""
    full_audit = validate_taxonomy_artifact(full_artifact)
    if full_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy manifest requires the 110K identity artifact")
    if set(artifact_records) != set(SCALE_SIZES):
        raise ValueError("taxonomy manifest must contain 20K/50K/110K artifact records")
    if not isinstance(generator_source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", generator_source_commit):
        raise ValueError("taxonomy manifest generator_source_commit is invalid")
    _require_sha256(generator_contract_sha256, label="taxonomy manifest generator contract")
    _require_sha256(generation_plan_sha256, label="taxonomy manifest generation plan")
    _require_sha256(approval_record_sha256, label="taxonomy manifest approval record")
    verified_receipt = _require_verified_taxonomy_generation_receipt(verified_receipt)
    if verified_receipt.generation_plan_sha256 != generation_plan_sha256:
        raise ValueError("taxonomy artifact manifest receipt plan hash does not match manifest")
    if verified_receipt.approval_record_sha256 != approval_record_sha256:
        raise ValueError("taxonomy artifact manifest receipt approval hash does not match manifest")
    if verified_receipt.generator_contract_sha256 != generator_contract_sha256:
        raise ValueError("taxonomy artifact manifest receipt generator contract hash does not match manifest")
    if verified_receipt.generator_source_commit != generator_source_commit:
        raise ValueError("taxonomy artifact manifest receipt generator source commit does not match manifest")
    artifact_generator = full_artifact["provenance"]["generator"]
    if artifact_generator["generator_code_sha256"] != verified_receipt.generator_code_sha256:
        raise ValueError("taxonomy artifact manifest generator code does not match verified receipt")
    if generator_spec_sha256(artifact_generator) != verified_receipt.generator_spec_sha256:
        raise ValueError("taxonomy artifact manifest generator specification does not match verified receipt")
    if verified_receipt.assignment_canonical_sha256 != sha256_json(full_artifact["assignments"]):
        raise ValueError("taxonomy artifact manifest receipt assignment hash does not match artifact")
    normalized_records: dict[str, dict[str, Any]] = {}
    for scale in SCALE_SIZES:
        record = dict(artifact_records[scale])
        _require_sha256(record.get("sha256"), label=f"taxonomy manifest {scale} artifact")
        if not isinstance(record.get("relative_path"), str) or not record["relative_path"]:
            raise ValueError("taxonomy manifest artifact relative_path is invalid")
        if not isinstance(record.get("byte_size"), int) or record["byte_size"] < 0:
            raise ValueError("taxonomy manifest artifact byte_size is invalid")
        normalized_records[str(scale)] = record
    provenance = dict(full_artifact["provenance"])
    return {
        "schema_version": TAXONOMY_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "taxonomy_artifact_id": full_artifact["taxonomy_artifact_id"],
        "dataset": "MIRACL",
        "language": "ko",
        "source_revisions": dict(full_artifact["source_revisions"]),
        "retrieval_unit": RETRIEVAL_UNIT,
        "generator_source_commit": generator_source_commit,
        "generator_contract_sha256": generator_contract_sha256,
        "generation_plan_sha256": generation_plan_sha256,
        "approval_record_sha256": approval_record_sha256,
        "generation_receipt_sha256": verified_receipt.receipt_sha256,
        "source_scale": TAXONOMY_SOURCE_SCALE,
        "input_contract": dict(full_artifact["input_contract"]),
        "provenance": provenance,
        "source_artifact_content_sha256": full_audit["artifact_sha256"],
        "artifacts": {
            str(scale): {
                "role": "source" if scale == TAXONOMY_SOURCE_SCALE else "filter_projection",
                "artifact_file": normalized_records[str(scale)],
                "source_artifact_content_sha256": (
                    None if scale == TAXONOMY_SOURCE_SCALE else full_audit["artifact_sha256"]
                ),
                "projection_method": (
                    SOURCE_IDENTITY_METHOD if scale == TAXONOMY_SOURCE_SCALE else FILTER_PROJECTION_METHOD
                ),
            }
            for scale in SCALE_SIZES
        },
    }


def validate_taxonomy_artifact_manifest(
    manifest: Mapping[str, Any], *, data_dir: Path,
    expected_ids_by_scale: Mapping[int, set[str]],
    expected_provenance: Mapping[str, str] | None = None,
    expected_source_revisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recheck ignored bodies against the tracked manifest and nested fixtures."""
    if not isinstance(manifest, Mapping):
        raise ValueError("taxonomy artifact manifest must be an object")
    required = {
        "schema_version", "generated_at", "taxonomy_artifact_id", "dataset", "language", "source_revisions", "retrieval_unit",
        "generator_source_commit", "source_scale", "input_contract", "provenance",
        "generator_contract_sha256", "generation_plan_sha256", "approval_record_sha256",
        "generation_receipt_sha256",
        "source_artifact_content_sha256", "artifacts",
    }
    if set(manifest) != required:
        raise ValueError("taxonomy artifact manifest has missing or unsupported fields")
    if manifest.get("schema_version") != TAXONOMY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("taxonomy artifact manifest schema_version is invalid")
    _require_rfc3339_timestamp(manifest.get("generated_at"), label="taxonomy artifact manifest generated_at")
    if manifest.get("dataset") != "MIRACL" or manifest.get("language") != "ko":
        raise ValueError("taxonomy artifact manifest must identify MIRACL Korean")
    _validate_source_revisions(manifest.get("source_revisions"))
    if expected_source_revisions is not None:
        _validate_source_revisions(dict(expected_source_revisions))
        if manifest["source_revisions"] != dict(expected_source_revisions):
            raise ValueError("taxonomy manifest source_revisions do not match verified fixture")
    if manifest.get("retrieval_unit") != RETRIEVAL_UNIT or manifest.get("source_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy artifact manifest has invalid retrieval unit or source scale")
    if not isinstance(manifest.get("generator_source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["generator_source_commit"]
    ):
        raise ValueError("taxonomy artifact manifest generator_source_commit is invalid")
    _require_sha256(manifest.get("generator_contract_sha256"), label="taxonomy artifact manifest generator contract")
    _require_sha256(manifest.get("generation_plan_sha256"), label="taxonomy artifact manifest generation plan")
    _require_sha256(manifest.get("approval_record_sha256"), label="taxonomy artifact manifest approval record")
    _require_sha256(manifest.get("generation_receipt_sha256"), label="taxonomy artifact manifest generation receipt")
    _validate_semantic_input_contract(manifest.get("input_contract"))
    _validate_provenance(manifest.get("provenance"))
    if expected_provenance is not None:
        for key in (
            "revision_lock_sha256",
            "preparation_contract_sha256",
            "input_110k_corpus_sha256",
            "input_subset_manifest_sha256",
        ):
            expected = expected_provenance.get(key)
            _require_sha256(expected, label=f"expected taxonomy provenance {key}")
            if manifest["provenance"].get(key) != expected:
                raise ValueError(f"taxonomy manifest provenance {key} does not match verified input")
    _require_sha256(manifest.get("source_artifact_content_sha256"), label="taxonomy manifest source artifact")
    if set(expected_ids_by_scale) != set(SCALE_SIZES):
        raise ValueError("taxonomy preflight requires expected passage IDs for 20K/50K/110K")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {str(scale) for scale in SCALE_SIZES}:
        raise ValueError("taxonomy artifact manifest must contain all scale artifacts")

    loaded: dict[int, dict[str, Any]] = {}
    for scale in SCALE_SIZES:
        record = artifacts[str(scale)]
        if not isinstance(record, Mapping):
            raise ValueError(f"taxonomy manifest artifact record is invalid for {scale}")
        expected_role = "source" if scale == TAXONOMY_SOURCE_SCALE else "filter_projection"
        expected_method = SOURCE_IDENTITY_METHOD if scale == TAXONOMY_SOURCE_SCALE else FILTER_PROJECTION_METHOD
        if record.get("role") != expected_role or record.get("projection_method") != expected_method:
            raise ValueError(f"taxonomy manifest projection declaration is invalid for {scale}")
        if scale == TAXONOMY_SOURCE_SCALE:
            if record.get("source_artifact_content_sha256") is not None:
                raise ValueError("110K taxonomy manifest source artifact must not point to another source")
        elif record.get("source_artifact_content_sha256") != manifest["source_artifact_content_sha256"]:
            raise ValueError(f"taxonomy manifest projection source hash is invalid for {scale}")
        path = _validate_file_record(record.get("artifact_file"), data_dir=data_dir, label=f"taxonomy {scale}")
        artifact = _load_artifact(path, label=f"taxonomy {scale}")
        audit = validate_taxonomy_artifact(artifact, expected_corpus_ids=set(expected_ids_by_scale[scale]))
        if artifact.get("taxonomy_artifact_id") != manifest["taxonomy_artifact_id"]:
            raise ValueError(f"taxonomy artifact id does not match manifest for {scale}")
        if artifact.get("source_revisions") != manifest["source_revisions"]:
            raise ValueError(f"taxonomy source_revisions do not match manifest for {scale}")
        if artifact.get("input_contract") != manifest["input_contract"]:
            raise ValueError(f"taxonomy input contract does not match manifest for {scale}")
        if artifact.get("provenance") != manifest["provenance"]:
            raise ValueError(f"taxonomy provenance does not match manifest for {scale}")
        loaded[scale] = {"artifact": artifact, "audit": audit}

    source = loaded[TAXONOMY_SOURCE_SCALE]["artifact"]
    if taxonomy_artifact_sha256(source) != manifest["source_artifact_content_sha256"]:
        raise ValueError("taxonomy 110K artifact content sha256 does not match manifest")
    for scale in (20_000, 50_000):
        validate_taxonomy_projection(
            source,
            loaded[scale]["artifact"],
            expected_corpus_ids=set(expected_ids_by_scale[scale]),
            target_scale=scale,
        )
    return {
        "status": "ready",
        "taxonomy_artifact_id": manifest["taxonomy_artifact_id"],
        "artifact_sha256": manifest["source_artifact_content_sha256"],
        "scales": {str(scale): loaded[scale]["audit"] for scale in SCALE_SIZES},
    }


def validate_taxonomy_artifact_authorization(
    manifest: Mapping[str, Any], *, generation_plan: Mapping[str, Any],
    approval_record: Mapping[str, Any] | None, generator_code_contract: Mapping[str, Any] | None,
    verified_input_provenance: Mapping[str, str], generator_source_root: Path,
    generation_run: Mapping[str, Any] | None, generation_receipt: Mapping[str, Any] | None,
    raw_response_dir: Path | None, source_artifact: Mapping[str, Any],
) -> None:
    """Bind an integrity-checked artifact to an approved pre-generation plan."""
    validate_taxonomy_generation_preflight(
        generation_plan,
        approval_record=approval_record,
        generator_code_contract=generator_code_contract,
        verified_input_provenance=verified_input_provenance,
        generator_source_root=generator_source_root,
    )
    if generation_run is None:
        raise FileNotFoundError("taxonomy generation run is missing")
    if generation_receipt is None:
        raise FileNotFoundError("taxonomy generation receipt is missing")
    if raw_response_dir is None:
        raise FileNotFoundError("taxonomy generation raw response directory is missing")
    if approval_record is None or generator_code_contract is None:
        raise FileNotFoundError("taxonomy generation authorization control is missing")
    verified_receipt = validate_taxonomy_generation_receipt(
        generation_receipt,
        generation_plan=generation_plan,
        approval_record=approval_record,
        generator_code_contract=generator_code_contract,
        generation_run=generation_run,
        raw_response_dir=raw_response_dir,
    )
    if verified_receipt is None:
        raise ValueError("partial or failed taxonomy generation receipt cannot authorize an artifact")
    plan_hash = generation_plan_sha256(generation_plan)
    if manifest.get("generation_plan_sha256") != plan_hash:
        raise ValueError("taxonomy artifact manifest generation plan hash does not match authorization")
    if manifest.get("generator_source_commit") != generation_plan["generator_source_commit"]:
        raise ValueError("taxonomy artifact manifest generator source commit does not match generation plan")
    if manifest.get("generator_contract_sha256") != generation_plan["generator_code_contract_sha256"]:
        raise ValueError("taxonomy artifact manifest generator contract does not match generation plan")
    if manifest.get("input_contract") != generation_plan["input_contract"]:
        raise ValueError("taxonomy artifact manifest input contract does not match generation plan")
    if manifest.get("provenance", {}).get("generator") != generation_plan["generator"]:
        raise ValueError("taxonomy artifact manifest generator metadata does not match generation plan")
    if approval_record is None or manifest.get("approval_record_sha256") != approval_record_sha256(approval_record):
        raise ValueError("taxonomy artifact manifest approval record does not match authorization")
    if manifest.get("generation_receipt_sha256") != verified_receipt.receipt_sha256:
        raise ValueError("taxonomy artifact manifest generation receipt does not match authorization")
    manifest_input_provenance = {
        key: manifest.get("provenance", {}).get(key)
        for key in generation_plan["input_provenance"]
    }
    if manifest_input_provenance != generation_plan["input_provenance"]:
        raise ValueError("taxonomy artifact manifest input provenance does not match generation plan")
    # The source identity artifact assignment order is canonical by corpus_id;
    # this exact comparison prevents a valid receipt from being bound to a
    # differently assembled taxonomy body.
    validate_taxonomy_artifact(source_artifact)
    if taxonomy_artifact_sha256(source_artifact) != manifest.get("source_artifact_content_sha256"):
        raise ValueError("taxonomy artifact manifest source body does not match authorization")
    if sha256_json(source_artifact["assignments"]) != verified_receipt.assignment_canonical_sha256:
        raise ValueError("taxonomy artifact assignments do not match generation receipt")


def load_integrity_only_taxonomy_projection(
    manifest_path: Path, *, data_dir: Path, expected_ids_by_scale: Mapping[int, set[str]], scale: int,
    expected_source_revisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a projection after integrity checks only; never authorize experiment use."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"taxonomy artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("taxonomy artifact manifest JSON is invalid") from error
    if scale not in SCALE_SIZES:
        raise ValueError("taxonomy projection scale is invalid")
    validate_taxonomy_artifact_manifest(
        manifest,
        data_dir=data_dir,
        expected_ids_by_scale=expected_ids_by_scale,
        expected_source_revisions=expected_source_revisions,
    )
    record = manifest["artifacts"][str(scale)]["artifact_file"]
    return _load_artifact(_validate_file_record(record, data_dir=data_dir, label=f"taxonomy {scale}"), label=f"taxonomy {scale}")


def load_authorized_taxonomy_projection(
    manifest_path: Path, *, data_dir: Path, expected_ids_by_scale: Mapping[int, set[str]], scale: int,
    generation_plan: Mapping[str, Any], approval_record: Mapping[str, Any] | None,
    generator_code_contract: Mapping[str, Any] | None,
    verified_input_provenance: Mapping[str, str], generator_source_root: Path,
    generation_run: Mapping[str, Any] | None, generation_receipt: Mapping[str, Any] | None,
    raw_response_dir: Path | None,
    expected_source_revisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a projection only after integrity and generation authorization pass."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"taxonomy artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("taxonomy artifact manifest JSON is invalid") from error
    validate_taxonomy_artifact_manifest(
        manifest,
        data_dir=data_dir,
        expected_ids_by_scale=expected_ids_by_scale,
        expected_source_revisions=expected_source_revisions,
    )
    validate_taxonomy_artifact_authorization(
        manifest,
        generation_plan=generation_plan,
        approval_record=approval_record,
        generator_code_contract=generator_code_contract,
        verified_input_provenance=verified_input_provenance,
        generator_source_root=generator_source_root,
        generation_run=generation_run,
        generation_receipt=generation_receipt,
        raw_response_dir=raw_response_dir,
        source_artifact=_load_artifact(
            _validate_file_record(
                manifest["artifacts"][str(TAXONOMY_SOURCE_SCALE)]["artifact_file"],
                data_dir=data_dir,
                label="taxonomy 110K",
            ),
            label="taxonomy 110K",
        ),
    )
    record = manifest["artifacts"][str(scale)]["artifact_file"]
    return _load_artifact(_validate_file_record(record, data_dir=data_dir, label=f"taxonomy {scale}"), label=f"taxonomy {scale}")


def validate_deterministic_regeneration(
    reference_artifact: Mapping[str, Any], regenerated_artifact: Mapping[str, Any]
) -> None:
    """Require byte-independent identical content for a deterministic generator."""
    validate_taxonomy_artifact(reference_artifact)
    validate_taxonomy_artifact(regenerated_artifact)
    for artifact in (reference_artifact, regenerated_artifact):
        if artifact["provenance"]["generator"]["determinism_mode"] != "deterministic":
            raise ValueError("taxonomy generator does not claim deterministic regeneration")
    if taxonomy_artifact_sha256(reference_artifact) != taxonomy_artifact_sha256(regenerated_artifact):
        raise ValueError("taxonomy deterministic regeneration changed artifact content")


def validate_focused_taxonomy_treatment_pair(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> None:
    """Freeze the later A/B to one treatment difference: taxonomy score boost."""
    if control.get("taxonomy") is not False or treatment.get("taxonomy") is not True:
        raise ValueError("focused taxonomy pair must toggle taxonomy score boost only")
    for key in ("taxonomy_prompt_schema", "workspace_taxonomy"):
        if control.get(key) != treatment.get(key):
            raise ValueError("focused taxonomy pair changes more than taxonomy score boost")
    if control.get("taxonomy_prompt_schema") is not True or control.get("workspace_taxonomy") is not False:
        raise ValueError("focused taxonomy pair requires shared prompt and disabled workspace taxonomy")
    ignored = {"name", "description", "taxonomy"}
    changed = {
        key for key in set(control) | set(treatment)
        if key not in ignored and control.get(key) != treatment.get(key)
    }
    if changed:
        raise ValueError("focused taxonomy pair changes more than taxonomy score boost")
