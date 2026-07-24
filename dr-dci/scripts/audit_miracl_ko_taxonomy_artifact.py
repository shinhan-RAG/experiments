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
import subprocess
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
    validate_taxonomy_generation_preflight,
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


def git_source_state(repo_root: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return head, porcelain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/miracl_ko_taxonomy_artifact_manifest.json"),
    )
    parser.add_argument("--revision-lock", type=Path, default=Path("config/miracl_ko_revision_lock.json"))
    parser.add_argument("--approval-record", type=Path)
    parser.add_argument("--generator-code-contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    manifest_path = (REPO_ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    revision_lock_path = (
        (REPO_ROOT / args.revision_lock).resolve()
        if not args.revision_lock.is_absolute()
        else args.revision_lock
    )
    approval_path = (
        (REPO_ROOT / args.approval_record).resolve()
        if args.approval_record is not None and not args.approval_record.is_absolute()
        else args.approval_record
    )
    generator_contract_path = (
        (REPO_ROOT / args.generator_code_contract).resolve()
        if args.generator_code_contract is not None and not args.generator_code_contract.is_absolute()
        else args.generator_code_contract
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
    report = validate_taxonomy_artifact_manifest(
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
    if (approval_path is None) != (generator_contract_path is None):
        raise ValueError("taxonomy approval record and generator code contract must be provided together")
    if approval_path is not None and generator_contract_path is not None:
        if not approval_path.is_file():
            raise FileNotFoundError(f"taxonomy approval record is missing: {approval_path}")
        if not generator_contract_path.is_file():
            raise FileNotFoundError(f"taxonomy generator code contract is missing: {generator_contract_path}")
        actual_source_git_commit, git_status_porcelain = git_source_state(REPO_ROOT)
        validate_taxonomy_generation_preflight(
            taxonomy_manifest,
            approval_record=load_json(approval_path),
            generator_code_contract=load_json(generator_contract_path),
            actual_source_git_commit=actual_source_git_commit,
            git_status_porcelain=git_status_porcelain,
            generator_repo_root=REPO_ROOT,
        )
        report["generation_approval_preflight"] = "ready"
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
