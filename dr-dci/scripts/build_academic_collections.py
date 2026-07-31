#!/usr/bin/env python3
"""Convert academic-paper collections into frozen collection_eval artifacts.

Private-data smoke/full-run CLI for the element_alignment stage. Reads the
source archives strictly read-only, then converts through the lock-protected
staging/atomic-publication path: a failure never leaves a partial target, an
identical rerun re-verifies and reuses the published target, and a
different-identity target fails loudly. Prints only aggregate counts and
hashes; raw document text never reaches stdout or Git.

Exit codes: 0 success (published or reused), 2 conversion/contract failure,
3 stable lock conflict with a concurrent same-target process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.data import collection_academic as academic  # noqa: E402
from src.data.collection_publication import LockConflictError  # noqa: E402
from src.eval.collection_contract import ContractError, sha256_file  # noqa: E402


DEFAULT_SOURCE_CONFIG = BASE / "config" / "collection_academic" / "academic_paper.yaml"
DEFAULT_CHUNKING_CONFIG = (
    BASE / "config" / "collection_academic" / "chunking.academic_element_packed.v1.yaml"
)
DEFAULT_ACCEPTANCE_CONTRACT = (
    BASE / "config" / "collection_academic" / "accepted_full_run.v1.yaml"
)


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
        "--acceptance-contract", type=Path, default=DEFAULT_ACCEPTANCE_CONTRACT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="publication target outside Git (or under the ignored data/ path)",
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
        "recorded in the manifest and makes the run acceptance-ineligible",
    )
    parser.add_argument(
        "--skip-source-member-hash",
        action="store_true",
        help="record source pdf/pptx member bytes without SHA-256 digests; "
        "recorded in the manifest and makes the run acceptance-ineligible",
    )
    args = parser.parse_args()

    source_config = academic.load_source_config(args.config)
    chunking_config = academic.load_chunking_config(args.chunking_config)
    acceptance_contract = academic.load_acceptance_contract(args.acceptance_contract)
    academic.ensure_output_dir_outside_git(args.output_dir, code_base=BASE)

    try:
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
            runtime_identity=academic.collect_runtime_identity(BASE),
            acceptance_contract=acceptance_contract,
            progress=lambda message: print(message, flush=True),
        )
    except LockConflictError as error:
        print(f"lock conflict: {error}", file=sys.stderr)
        raise SystemExit(3) from error
    except (academic.ConversionError, ContractError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    manifest_path = Path(args.output_dir) / academic.MANIFEST_FILE_NAME
    summary = {
        "publication_status": manifest.get("publication_status"),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "run_identity_sha256": manifest["run_identity"]["identity_sha256"],
        "acceptance": manifest["acceptance"],
        "retrieval_compatibility": manifest["retrieval_compatibility"]["status"],
        "totals": manifest["totals"],
        "cells": manifest["cells"],
        "verification": {
            collection_id: value["status"]
            for collection_id, value in manifest["verification"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
