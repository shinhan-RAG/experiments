#!/usr/bin/env python3
"""Create only the approved 20K/50K filter projections after a passed source audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from src.miracl_ko.preparation import SCALE_SIZES, file_record, iter_jsonl  # noqa: E402
from src.miracl_ko.taxonomy_artifact import (  # noqa: E402
    build_taxonomy_artifact_manifest,
    project_taxonomy_artifact,
    validate_taxonomy_artifact,
    validate_taxonomy_generation_receipt,
)


def _ids_by_scale(data_dir: Path) -> dict[int, set[str]]:
    subset = validate_subset_fixture_files(data_dir)
    result: dict[int, set[str]] = {}
    for scale in SCALE_SIZES:
        try:
            relative_path = subset["subsets"][str(scale)]["corpus"]["relative_path"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"MIRACL subset manifest lacks {scale} corpus") from error
        path = (data_dir / relative_path).resolve()
        if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
            raise FileNotFoundError(f"MIRACL {scale} corpus is missing")
        ids = {str(row["corpus_id"]) for row in iter_jsonl(path)}
        if len(ids) != scale:
            raise ValueError(f"MIRACL {scale} corpus IDs are not unique or complete")
        result[scale] = ids
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = require_config(parse_env_file(args.config))
    output_dir = path_from_config(BUNDLE_ROOT, config["KT_OUTPUT_DIR"], label="KT_OUTPUT_DIR")
    source_audit = load_json_object(output_dir / "audit" / "source_audit.json", label="source artifact audit")
    if source_audit.get("status") != "passed":
        raise ValueError("20K/50K projection is blocked until the source standalone audit passes")
    control_dir = path_from_config(BUNDLE_ROOT, config["KT_CONTROL_DIR"], label="KT_CONTROL_DIR")
    plan, approval, contract = load_bundle_controls(control_dir)
    lock = validate_execution_lock(
        bundle_root=BUNDLE_ROOT, config=config, plan=plan, approval=approval, supplied_contract=contract,
    )
    if source_audit.get("generation_plan_sha256") != lock["generation_plan_sha256"]:
        raise ValueError("source audit does not bind the exact approved generation plan")
    generation_dir = output_dir / "generation"
    receipt = load_json_object(generation_dir / "generation_receipt.json", label="generation receipt")
    run = load_json_object(generation_dir / "generation_run.json", label="generation run")
    verified = validate_taxonomy_generation_receipt(
        receipt,
        generation_plan=plan,
        approval_record=approval,
        generator_code_contract=contract,
        generation_run=run,
        raw_response_dir=generation_dir / "raw-responses",
    )
    if verified is None:
        raise ValueError("partial or failed generation receipt cannot be projected")
    artifact_dir = output_dir / "artifacts"
    source = load_json_object(artifact_dir / "110000.json", label="110K taxonomy artifact")
    data_dir = Path(lock["data_dir"])
    ids_by_scale = _ids_by_scale(data_dir)
    validate_taxonomy_artifact(source, expected_corpus_ids=ids_by_scale[110_000])
    projected_records: dict[int, dict[str, object]] = {}
    for scale in (20_000, 50_000):
        projected = project_taxonomy_artifact(source, ids_by_scale[scale], target_scale=scale)
        target = artifact_dir / f"{scale}.json"
        atomic_write_json(target, projected)
        projected_records[scale] = file_record(target, relative_to=artifact_dir)
    projected_records[110_000] = file_record(artifact_dir / "110000.json", relative_to=artifact_dir)
    manifest = build_taxonomy_artifact_manifest(
        full_artifact=source,
        artifact_records=projected_records,
        generator_source_commit=lock["generator_source_commit"],
        generator_contract_sha256=lock["generator_contract_sha256"],
        generation_plan_sha256=lock["generation_plan_sha256"],
        approval_record_sha256=lock["approval_record_sha256"],
        verified_receipt=verified,
    )
    atomic_write_json(artifact_dir / "taxonomy_artifact_manifest.json", manifest)
    result = {
        "schema_version": "dr-dci.miracl-ko-kt-projection.v1",
        "status": "passed",
        "generated_at": utc_now(),
        "generation_plan_sha256": lock["generation_plan_sha256"],
        "projection_scales": [20_000, 50_000],
        "artifact_manifest": file_record(artifact_dir / "taxonomy_artifact_manifest.json", relative_to=output_dir),
    }
    atomic_write_json(output_dir / "projection" / "projection_result.json", result)
    print("projection passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"KT taxonomy projection failed loudly: {error}", file=sys.stderr)
        raise
