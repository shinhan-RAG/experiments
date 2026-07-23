#!/usr/bin/env python3
"""Prepare and validate official MIRACL Korean artifacts outside Part 1/2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.preparation import (
    acquire_miracl_ko,
    build_miracl_ko_subsets,
    load_current_miracl_ko_eda_report,
    load_revision_lock,
    load_verified_acquisition_manifest,
    load_verified_normalization_manifest,
    normalize_miracl_ko,
    raw_data_paths_are_ignored,
    validate_miracl_ko,
    validate_miracl_ko_subset_files,
    validate_nested_subset_manifest,
    write_miracl_ko_eda_report,
    write_miracl_pretest_artifacts,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("acquire", "normalize", "validate", "subsets", "report", "all"),
        help="data-preparation stage; no taxonomy, embeddings, agent, or Part 1/2 execution",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--revision-lock", type=Path, default=Path("config/miracl_ko_revision_lock.json"),
        help="committed immutable source revisions; remote dataset HEAD is never adopted automatically",
    )
    parser.add_argument("--stamp", default="20260723")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    docs_dir = (repo_root / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir
    revision_lock_path = (
        (repo_root / args.revision_lock).resolve()
        if not args.revision_lock.is_absolute() else args.revision_lock
    )
    revision_lock = load_revision_lock(revision_lock_path)
    if not raw_data_paths_are_ignored(repo_root, data_dir):
        raise RuntimeError("MIRACL raw data directory must remain below ignored data/")

    acquisition_path = data_dir / "acquisition_manifest.json"
    normalization_path = data_dir / "normalization_manifest.json"
    subset_path = data_dir / "subsets" / "manifest.json"
    script_path = Path(__file__).resolve()

    if args.command in {"acquire", "all"}:
        acquisition = acquire_miracl_ko(
            data_dir, acquisition_script=script_path, revision_lock=revision_lock
        )
        print(f"acquired {len(acquisition['files'])} official MIRACL-ko files")
    elif not acquisition_path.exists():
        raise FileNotFoundError("run acquire first: missing acquisition_manifest.json")
    elif args.command == "report":
        acquisition = load_verified_acquisition_manifest(data_dir, revision_lock)

    if args.command in {"normalize", "all"}:
        normalization = normalize_miracl_ko(data_dir, revision_lock=revision_lock)
        print("normalized MIRACL-ko raw artifacts")
    elif args.command in {"validate", "subsets", "report"} and not normalization_path.exists():
        raise FileNotFoundError("run normalize first: missing normalization_manifest.json")
    elif args.command == "report":
        normalization = load_verified_normalization_manifest(data_dir)

    if args.command in {"validate", "all"}:
        eda = validate_miracl_ko(data_dir, revision_lock=revision_lock)
        if eda["integrity_status"] != "passed":
            raise RuntimeError(f"MIRACL-ko integrity blocked: {eda['violations']}")
        write_miracl_ko_eda_report(data_dir, eda)
        print("validated MIRACL-ko passage/query/qrel integrity")
    elif args.command in {"subsets", "report"}:
        eda = load_current_miracl_ko_eda_report(data_dir)
    else:
        eda = None

    if args.command == "subsets":
        subsets = build_miracl_ko_subsets(data_dir, eda, revision_lock=revision_lock)
        print("built deterministic 20K/50K/110K passage subsets")
    elif args.command == "all":
        subprocess.run(
            [
                sys.executable, str(script_path), "subsets",
                "--data-dir", str(data_dir), "--docs-dir", str(docs_dir),
                "--revision-lock", str(revision_lock_path), "--stamp", args.stamp,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, str(script_path), "report",
                "--data-dir", str(data_dir), "--docs-dir", str(docs_dir),
                "--revision-lock", str(revision_lock_path), "--stamp", args.stamp,
            ],
            check=True,
        )
        return
    elif args.command == "report":
        if not subset_path.is_file():
            raise FileNotFoundError("run subsets first: missing MIRACL subset manifest")
        subsets = load_json(subset_path)
        validate_nested_subset_manifest(subsets, data_dir=data_dir)
        validate_miracl_ko_subset_files(data_dir, subsets)
    elif subset_path.exists():
        subsets = load_json(subset_path)
    else:
        subsets = None

    if args.command == "report":
        smoke = {
            "status": "blocked",
            "reason": (
                "Pyserini is not installed and no existing Korean lexical retrieval dependency is approved; "
                "the repository's small regex BM25 helper is not promoted to a Korean tokenizer"
            ),
        }
        paths = write_miracl_pretest_artifacts(
            docs_dir,
            acquisition=acquisition,
            normalization=normalization,
            eda=eda,
            subsets=subsets,
            smoke=smoke,
            stamp=args.stamp,
        )
        print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MIRACL-ko preparation failed loudly: {error}", file=sys.stderr)
        raise
