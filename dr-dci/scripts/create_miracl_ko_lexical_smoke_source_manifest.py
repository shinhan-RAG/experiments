#!/usr/bin/env python3
"""Record clean Git source evidence before a Git-free MIRACL smoke bundle run.

The archive itself is created by the caller's approved transfer workflow.  This
script only binds that archive's byte hash to a clean checkout and the exact
lexical-smoke code/config/data-manifest contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.lexical_smoke import (
    EXPERIMENT_CONTRACT_PATHS,
    SOURCE_MANIFEST_SCHEMA,
    build_experiment_contract,
)
from src.miracl_ko.preparation import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_clean_head() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("source manifest creation requires a valid Git checkout") from error
    if dirty:
        raise RuntimeError("source manifest creation rejects a dirty Git checkout")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("Git HEAD is not a full lowercase SHA-1")
    return head


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    args = parser.parse_args()

    source_archive = args.source_archive.resolve()
    output = args.output.resolve()
    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    if not source_archive.is_file():
        raise RuntimeError("source archive is missing")
    subset_manifest = data_dir / "subsets" / "manifest.json"
    if not subset_manifest.is_file():
        raise RuntimeError("MIRACL subset manifest is missing")

    source_git_commit = require_clean_head()
    file_sha256 = {
        relative_path: sha256_file(REPO_ROOT / relative_path)
        for relative_path in EXPERIMENT_CONTRACT_PATHS
    }
    contract = build_experiment_contract(
        file_sha256,
        subset_manifest_sha256=sha256_file(subset_manifest),
    )
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "generated_at": utc_now(),
        "source_git_commit": source_git_commit,
        "source_archive_sha256": sha256_file(source_archive),
        "experiment_contract": contract,
        "experiment_contract_sha256": contract["experiment_contract_sha256"],
    }
    atomic_write_json(output, manifest)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MIRACL lexical smoke source manifest failed loudly: {error}", file=sys.stderr)
        raise
