#!/usr/bin/env python3
"""Convert academic-paper collections into frozen collection_eval artifacts.

Private-data smoke/full-run CLI for the element_alignment stage. Reads the
source archives strictly read-only, writes document/chunk/element/alignment
JSONL plus a conversion manifest, then re-verifies every output. Prints only
aggregate counts and hashes; raw document text never reaches stdout or Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.data import collection_academic as academic  # noqa: E402
from src.eval.collection_contract import sha256_file  # noqa: E402


DEFAULT_SOURCE_CONFIG = BASE / "config" / "collection_academic" / "academic_paper.yaml"
DEFAULT_CHUNKING_CONFIG = (
    BASE / "config" / "collection_academic" / "chunking.academic_element_packed.v1.yaml"
)
ALIGNMENT_SCHEMA = BASE / "config" / "collection_academic" / "element_alignment.schema.json"


def code_fingerprint() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "adapter_sha256": sha256_file(BASE / "src" / "data" / "collection_academic.py"),
        "cli_sha256": sha256_file(Path(__file__).resolve()),
        "alignment_schema_sha256": sha256_file(ALIGNMENT_SCHEMA),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="read-only root that contains the archive relative paths",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--chunking-config", type=Path, default=DEFAULT_CHUNKING_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="output directory outside Git (or under the ignored data/ path)",
    )
    parser.add_argument("--collections", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=0,
        help="smoke mode: convert only the first N sorted stems per cell (0 = all)",
    )
    parser.add_argument(
        "--skip-archive-sha256",
        action="store_true",
        help="skip full-archive SHA-256 verification (bytes are always verified); "
        "recorded in the manifest",
    )
    parser.add_argument(
        "--skip-source-member-hash",
        action="store_true",
        help="record source pdf/pptx member bytes without SHA-256 digests; "
        "recorded in the manifest",
    )
    args = parser.parse_args()

    source_config = academic.load_source_config(args.config)
    chunking_config = academic.load_chunking_config(args.chunking_config)
    academic.ensure_output_dir_outside_git(args.output_dir, code_base=BASE)

    fingerprint = code_fingerprint()
    fingerprint["source_config_sha256"] = sha256_file(args.config)
    fingerprint["chunking_config_sha256"] = sha256_file(args.chunking_config)

    manifest = academic.convert_collections(
        data_root=args.data_root,
        source_config=source_config,
        chunking_config=chunking_config,
        output_dir=args.output_dir,
        collections=args.collections,
        splits=args.splits,
        limit_documents=args.limit_documents,
        verify_archive_sha256=not args.skip_archive_sha256,
        hash_source_members=not args.skip_source_member_hash,
        code_fingerprint=fingerprint,
        progress=lambda message: print(message, flush=True),
    )

    verification = {}
    for collection_id in manifest["selection"]["collections"]:
        verification[collection_id] = academic.verify_collection_outputs(
            args.output_dir,
            collection_id,
            separator=chunking_config["separator"],
            element_atomic=chunking_config["element_atomic"],
        )
    manifest["verification"] = verification
    manifest_path = academic.write_manifest(manifest, args.output_dir)

    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "totals": manifest["totals"],
        "cells": manifest["cells"],
        "verification": {
            collection_id: value["status"] for collection_id, value in verification.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
