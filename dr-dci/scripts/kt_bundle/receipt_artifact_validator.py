#!/usr/bin/env python3
"""Standalone Git-free audit for KT taxonomy generation and projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = BUNDLE_ROOT / "generator"
if str(GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATOR_ROOT))

from src.miracl_ko.kt_bundle import (  # noqa: E402
    atomic_write_json,
    input_provenance,
    load_bundle_controls,
    load_json_object,
    parse_env_file,
    path_from_config,
    require_config,
    utc_now,
    validate_execution_lock,
    validate_subset_fixture_files,
)
from src.miracl_ko.preparation import SCALE_SIZES, iter_jsonl, sha256_json  # noqa: E402
from src.miracl_ko.taxonomy_artifact import (  # noqa: E402
    taxonomy_artifact_sha256,
    validate_taxonomy_artifact,
    validate_taxonomy_artifact_manifest,
    validate_taxonomy_generation_receipt,
)


def ids_by_scale(data_dir: Path) -> dict[int, set[str]]:
    subset = validate_subset_fixture_files(data_dir)
    result: dict[int, set[str]] = {}
    for scale in SCALE_SIZES:
        relative_path = subset["subsets"][str(scale)]["corpus"]["relative_path"]
        path = (data_dir / relative_path).resolve()
        if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
            raise FileNotFoundError(f"MIRACL {scale} corpus is missing")
        result[scale] = {str(row["corpus_id"]) for row in iter_jsonl(path)}
        if len(result[scale]) != scale:
            raise ValueError(f"MIRACL {scale} corpus IDs are not unique")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("source", "full"), required=True)
    args = parser.parse_args()
    config = require_config(parse_env_file(args.config))
    output_dir = path_from_config(BUNDLE_ROOT, config["KT_OUTPUT_DIR"], label="KT_OUTPUT_DIR")
    report: dict[str, Any] = {"schema_version": "dr-dci.miracl-ko-kt-audit.v1", "stage": args.stage, "generated_at": utc_now()}
    try:
        control_dir = path_from_config(BUNDLE_ROOT, config["KT_CONTROL_DIR"], label="KT_CONTROL_DIR")
        plan, approval, contract = load_bundle_controls(control_dir)
        lock = validate_execution_lock(
            bundle_root=BUNDLE_ROOT, config=config, plan=plan, approval=approval, supplied_contract=contract,
        )
        generation_dir = output_dir / "generation"
        run = load_json_object(generation_dir / "generation_run.json", label="generation run")
        receipt = load_json_object(generation_dir / "generation_receipt.json", label="generation receipt")
        verified = validate_taxonomy_generation_receipt(
            receipt,
            generation_plan=plan,
            approval_record=approval,
            generator_code_contract=contract,
            generation_run=run,
            raw_response_dir=generation_dir / "raw-responses",
        )
        if verified is None:
            raise ValueError("partial or failed receipt cannot pass standalone artifact audit")
        source = load_json_object(output_dir / "artifacts" / "110000.json", label="110K taxonomy artifact")
        data_dir = Path(lock["data_dir"])
        expected_ids = ids_by_scale(data_dir)
        source_audit = validate_taxonomy_artifact(source, expected_corpus_ids=expected_ids[110_000])
        if source["provenance"] != {**plan["input_provenance"], "generator": plan["generator"]}:
            raise ValueError("110K taxonomy artifact provenance does not match approved generation plan")
        if sha256_json(source["assignments"]) != verified.assignment_canonical_sha256:
            raise ValueError("110K taxonomy assignments do not match verified receipt")
        report.update({
            "status": "passed",
            "generation_plan_sha256": lock["generation_plan_sha256"],
            "generation_receipt_sha256": verified.receipt_sha256,
            "source_artifact_sha256": taxonomy_artifact_sha256(source),
            "source_audit": source_audit,
        })
        if args.stage == "full":
            manifest = load_json_object(output_dir / "artifacts" / "taxonomy_artifact_manifest.json", label="taxonomy artifact manifest")
            subset = load_json_object(data_dir / "subsets" / "manifest.json", label="MIRACL subset manifest")
            report["full_audit"] = validate_taxonomy_artifact_manifest(
                manifest,
                data_dir=output_dir / "artifacts",
                expected_ids_by_scale=expected_ids,
                expected_provenance=input_provenance(
                    data_dir=data_dir, revision_lock_path=BUNDLE_ROOT / "source" / "miracl_ko_revision_lock.json",
                ),
                expected_source_revisions=subset["source_revisions"],
            )
    except Exception as error:
        report.update({
            "status": "failed",
            "error_class": type(error).__name__,
            "error": "standalone audit failed",
            "error_fingerprint_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        })
    target = output_dir / "audit" / f"{args.stage}_audit.json"
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
