#!/usr/bin/env python3
"""Validate a pre-generated MIRACL Korean taxonomy artifact without models.

This is a fail-loud preflight only.  It reads the locked fixture and an
already-created taxonomy artifact; it never creates labels, embeddings, query
tags, retrieval rankings, or Agent/LLM calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.preparation import (
    SCALE_SIZES,
    iter_jsonl,
    sha256_file,
    validate_miracl_ko_subset_files,
)
from src.miracl_ko.taxonomy_artifact import (
    validate_taxonomy_artifact_manifest,
    validate_taxonomy_artifact_authorization,
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def subset_ids_by_scale(data_dir: Path, subset_manifest: dict[str, Any]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for scale in SCALE_SIZES:
        record = subset_manifest["subsets"][str(scale)]["corpus"]
        relative_path = record["relative_path"]
        corpus_path = (data_dir / relative_path).resolve()
        if not corpus_path.is_relative_to(data_dir.resolve()) or not corpus_path.is_file():
            raise FileNotFoundError(f"MIRACL subset corpus is missing: {relative_path}")
        result[scale] = {str(row["corpus_id"]) for row in iter_jsonl(corpus_path)}
        if len(result[scale]) != scale:
            raise ValueError(f"MIRACL subset {scale} has duplicate or missing passage IDs")
    return result


def verified_taxonomy_provenance(
    *, data_dir: Path, subset_manifest: dict[str, Any], revision_lock_path: Path
) -> dict[str, str]:
    corpus_record = subset_manifest["subsets"][str(110_000)]["corpus"]
    return {
        "revision_lock_sha256": sha256_file(revision_lock_path),
        "preparation_contract_sha256": subset_manifest["preparation_contract_sha256"],
        "input_110k_corpus_sha256": corpus_record["sha256"],
        "input_subset_manifest_sha256": sha256_file(data_dir / "subsets" / "manifest.json"),
    }


def require_external_control_path(path: Path, *, generator_source_root: Path, label: str) -> None:
    """Keep mutable plan/approval/result controls out of the generator checkout."""
    if path.resolve().is_relative_to(generator_source_root.resolve()):
        raise ValueError(
            f"{label} must be outside --generator-source-root; use a separate read-only control artifact path"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_artifact_manifest.json"),
    )
    parser.add_argument("--revision-lock", type=Path, default=Path("config/miracl_ko_revision_lock.json"))
    parser.add_argument(
        "--generation-plan",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_generation_plan.json"),
    )
    parser.add_argument(
        "--approval-record",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_approval.json"),
    )
    parser.add_argument(
        "--generator-code-contract",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_generator_code_contract.json"),
    )
    parser.add_argument(
        "--generation-run",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_generation_run.json"),
    )
    parser.add_argument(
        "--generation-receipt",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_generation_receipt.json"),
    )
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        default=Path("data/miracl-ko/taxonomy/raw-responses"),
    )
    parser.add_argument(
        "--generator-source-root",
        type=Path,
        default=REPO_ROOT,
        help="clean Git checkout containing the generator code; control artifacts may live elsewhere",
    )
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="validate body/hash/projection only; never marks an artifact experiment-ready",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    manifest_path = (REPO_ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    revision_lock_path = (
        (REPO_ROOT / args.revision_lock).resolve()
        if not args.revision_lock.is_absolute()
        else args.revision_lock
    )
    plan_path = (REPO_ROOT / args.generation_plan).resolve() if not args.generation_plan.is_absolute() else args.generation_plan
    approval_path = (REPO_ROOT / args.approval_record).resolve() if not args.approval_record.is_absolute() else args.approval_record
    generator_contract_path = (
        (REPO_ROOT / args.generator_code_contract).resolve()
        if not args.generator_code_contract.is_absolute()
        else args.generator_code_contract
    )
    generation_run_path = (
        (REPO_ROOT / args.generation_run).resolve()
        if not args.generation_run.is_absolute()
        else args.generation_run
    )
    generation_receipt_path = (
        (REPO_ROOT / args.generation_receipt).resolve()
        if not args.generation_receipt.is_absolute()
        else args.generation_receipt
    )
    raw_response_dir = (
        (REPO_ROOT / args.raw_response_dir).resolve()
        if not args.raw_response_dir.is_absolute()
        else args.raw_response_dir.resolve()
    )
    generator_source_root = (
        (REPO_ROOT / args.generator_source_root).resolve()
        if not args.generator_source_root.is_absolute()
        else args.generator_source_root.resolve()
    )
    if not revision_lock_path.is_file():
        raise FileNotFoundError(f"MIRACL revision lock is missing: {revision_lock_path}")
    subset_manifest_path = data_dir / "subsets" / "manifest.json"
    if not subset_manifest_path.is_file():
        raise FileNotFoundError(f"MIRACL subset manifest is missing: {subset_manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MIRACL taxonomy artifact manifest is missing: {manifest_path}")
    subset_manifest = load_json(subset_manifest_path)
    validate_miracl_ko_subset_files(data_dir, subset_manifest)
    taxonomy_manifest = load_json(manifest_path)
    integrity = validate_taxonomy_artifact_manifest(
        taxonomy_manifest,
        data_dir=data_dir,
        expected_ids_by_scale=subset_ids_by_scale(data_dir, subset_manifest),
        expected_source_revisions=subset_manifest["source_revisions"],
        expected_provenance=verified_taxonomy_provenance(
            data_dir=data_dir,
            subset_manifest=subset_manifest,
            revision_lock_path=revision_lock_path,
        ),
    )
    if args.integrity_only:
        report = {
            "artifact_integrity": integrity,
            "generation_authorization": "integrity_only_not_authorized",
            "experiment_ready": False,
        }
    else:
        for path, label in (
            (plan_path, "taxonomy generation plan"),
            (approval_path, "taxonomy approval record"),
            (generator_contract_path, "taxonomy generator code contract"),
            (generation_run_path, "taxonomy generation run"),
            (generation_receipt_path, "taxonomy generation receipt"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")
        for path, label in (
            (manifest_path, "taxonomy artifact manifest"),
            (plan_path, "taxonomy generation plan"),
            (approval_path, "taxonomy approval record"),
            (generator_contract_path, "taxonomy generator code contract"),
            (generation_run_path, "taxonomy generation run"),
            (generation_receipt_path, "taxonomy generation receipt"),
            (raw_response_dir, "taxonomy generation raw response directory"),
        ):
            require_external_control_path(path, generator_source_root=generator_source_root, label=label)
        input_provenance = verified_taxonomy_provenance(
            data_dir=data_dir,
            subset_manifest=subset_manifest,
            revision_lock_path=revision_lock_path,
        )
        validate_taxonomy_artifact_authorization(
            taxonomy_manifest,
            generation_plan=load_json(plan_path),
            approval_record=load_json(approval_path),
            generator_code_contract=load_json(generator_contract_path),
            verified_input_provenance=input_provenance,
            generator_source_root=generator_source_root,
            generation_run=load_json(generation_run_path),
            generation_receipt=load_json(generation_receipt_path),
            raw_response_dir=raw_response_dir,
            source_artifact=load_json(
                data_dir / taxonomy_manifest["artifacts"][str(110_000)]["artifact_file"]["relative_path"]
            ),
        )
        report = {
            "artifact_integrity": integrity,
            "generation_authorization": "ready",
            "experiment_ready": True,
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MIRACL taxonomy artifact audit failed loudly: {error}", file=sys.stderr)
        raise
