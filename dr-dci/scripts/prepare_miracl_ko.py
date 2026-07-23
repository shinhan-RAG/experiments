#!/usr/bin/env python3
"""Prepare and validate official MIRACL Korean artifacts outside Part 1/2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.preparation import (
    acquire_miracl_ko,
    build_miracl_ko_subsets,
    normalize_miracl_ko,
    raw_data_paths_are_ignored,
    validate_miracl_ko,
    write_miracl_pretest_artifacts,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("acquire", "normalize", "validate", "subsets", "all"),
        help="data-preparation stage; no taxonomy, embeddings, agent, or Part 1/2 execution",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stamp", default="20260723")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    docs_dir = (repo_root / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir
    if not raw_data_paths_are_ignored(repo_root, data_dir):
        raise RuntimeError("MIRACL raw data directory must remain below ignored data/")

    acquisition_path = data_dir / "acquisition_manifest.json"
    normalization_path = data_dir / "normalization_manifest.json"
    subset_path = data_dir / "subsets" / "manifest.json"
    script_path = Path(__file__).resolve()

    if args.command in {"acquire", "all"}:
        acquisition = acquire_miracl_ko(data_dir, acquisition_script=script_path)
        print(f"acquired {len(acquisition['files'])} official MIRACL-ko files")
    elif not acquisition_path.exists():
        raise FileNotFoundError("run acquire first: missing acquisition_manifest.json")

    if args.command in {"normalize", "all"}:
        normalization = normalize_miracl_ko(data_dir)
        print("normalized MIRACL-ko raw artifacts")
    elif args.command in {"validate", "subsets"} and not normalization_path.exists():
        raise FileNotFoundError("run normalize first: missing normalization_manifest.json")

    if args.command in {"validate", "subsets", "all"}:
        eda = validate_miracl_ko(data_dir)
        if eda["integrity_status"] != "passed":
            raise RuntimeError(f"MIRACL-ko integrity blocked: {eda['violations']}")
        print("validated MIRACL-ko passage/query/qrel integrity")
    else:
        eda = None

    if args.command in {"subsets", "all"}:
        subsets = build_miracl_ko_subsets(data_dir, eda)
        print("built deterministic 20K/50K/110K passage subsets")
    elif subset_path.exists():
        subsets = load_json(subset_path)
    else:
        subsets = None

    if args.command == "all":
        smoke = {
            "status": "blocked",
            "reason": (
                "no existing Korean lexical retrieval dependency is approved for MIRACL; "
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
