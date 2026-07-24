#!/usr/bin/env python3
"""Emit and verify the exact local vLLM launch contract from an approved plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = BUNDLE_ROOT / "generator"
if str(GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATOR_ROOT))

from src.miracl_ko.kt_bundle import (  # noqa: E402
    atomic_write_json,
    load_bundle_controls,
    parse_env_file,
    path_from_config,
    require_config,
    utc_now,
    validate_execution_lock,
)
from src.miracl_ko.preparation import sha256_json  # noqa: E402


def _load_lock(config_path: Path) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    config = require_config(parse_env_file(config_path))
    control_dir = path_from_config(BUNDLE_ROOT, config["KT_CONTROL_DIR"], label="KT_CONTROL_DIR")
    plan, approval, contract = load_bundle_controls(control_dir)
    lock = validate_execution_lock(
        bundle_root=BUNDLE_ROOT, config=config, plan=plan, approval=approval, supplied_contract=contract,
    )
    return config, plan, lock


def launch_spec(config_path: Path) -> dict[str, Any]:
    config, plan, lock = _load_lock(config_path)
    launch = plan["generator"]["vllm_execution"]["server_launch"]
    return {
        "generation_plan_sha256": lock["generation_plan_sha256"],
        "generator_source_commit": lock["generator_source_commit"],
        "container_digest": lock["container_digest"],
        "launch_arguments": launch["arguments"],
        "launch_arguments_sha256": launch["arguments_sha256"],
        "model_repository": lock["model_repository"],
        "model_revision": lock["model_revision"],
        "tokenizer_repository": lock["tokenizer_repository"],
        "tokenizer_revision": lock["tokenizer_revision"],
        "request_model": plan["generator"]["vllm_execution"]["request_body_template"]["body"]["model"],
        "vllm_image": config["KT_VLLM_IMAGE"],
        "model_cache_dir": config["KT_MODEL_CACHE_DIR"],
        "host": config["KT_VLLM_HOST"],
        "port": config["KT_VLLM_PORT"],
    }


def _local_models(host: str, port: str) -> list[str]:
    url = f"http://{host}:{port}/v1/models"
    try:
        with urlopen(url, timeout=10) as response:  # nosec B310: local address comes from strict config
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"local vLLM model identity endpoint is unavailable: {error}") from error
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("local vLLM /v1/models response has no data array")
    identifiers = []
    for row in rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        if isinstance(identifier, str) and identifier:
            identifiers.append(identifier)
    if not identifiers:
        raise ValueError("local vLLM /v1/models response has no model identity")
    return sorted(set(identifiers))


def verify_identity(config_path: Path) -> int:
    config, plan, lock = _load_lock(config_path)
    output_dir = path_from_config(BUNDLE_ROOT, config["KT_OUTPUT_DIR"], label="KT_OUTPUT_DIR")
    expected = plan["generator"]["vllm_execution"]["request_body_template"]["body"]["model"]
    report: dict[str, Any] = {
        "schema_version": "dr-dci.miracl-ko-kt-model-identity.v1",
        "generated_at": utc_now(),
        "generation_plan_sha256": lock["generation_plan_sha256"],
        "expected_request_model": expected,
        "model_revision": lock["model_revision"],
        "tokenizer_revision": lock["tokenizer_revision"],
        "launch_arguments_sha256": plan["generator"]["vllm_execution"]["server_launch"]["arguments_sha256"],
    }
    try:
        launch = json.loads((output_dir / "environment" / "vllm_launch.json").read_text(encoding="utf-8"))
        if (
            launch.get("generation_plan_sha256") != lock["generation_plan_sha256"]
            or launch.get("container_digest") != lock["container_digest"]
            or launch.get("launch_arguments_sha256") != report["launch_arguments_sha256"]
        ):
            raise ValueError("recorded local vLLM launch does not match the approved exact plan")
        returned = _local_models(config["KT_VLLM_HOST"], config["KT_VLLM_PORT"])
        if expected not in returned:
            raise ValueError("local vLLM served model does not match the approved request model")
        report.update({"status": "passed", "served_model_ids": returned})
    except Exception as error:
        report.update({
            "status": "failed",
            "error_class": type(error).__name__,
            "error": "local model identity validation failed",
            "error_fingerprint_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        })
    atomic_write_json(output_dir / "environment" / "model_identity.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch-spec", "verify-identity"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "launch-spec":
        print(json.dumps(launch_spec(args.config), ensure_ascii=False, sort_keys=True))
        return 0
    return verify_identity(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
