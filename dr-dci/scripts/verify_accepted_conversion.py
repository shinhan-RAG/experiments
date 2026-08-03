#!/usr/bin/env python3
"""Independently replay the acceptance gate against a preserved target.

Controlled internal verification procedure for the private full-run
evidence: recomputes the manifest SHA-256 against the caller's pin,
re-hashes all declared artifacts inside the confined layout, re-runs the
streaming semantic verifier, and recomputes the acceptance decision. Prints
aggregate counts and hashes only; raw text never reaches stdout.

Exit codes: 0 accepted, 2 any failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.data import collection_academic as academic  # noqa: E402
from src.eval.collection_contract import ContractError  # noqa: E402


DEFAULT_ACCEPTANCE_CONTRACT = (
    BASE / "config" / "collection_academic" / "accepted_full_run.v1.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="preserved conversion target directory (outside Git)",
    )
    parser.add_argument(
        "--acceptance-contract", type=Path, default=DEFAULT_ACCEPTANCE_CONTRACT
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="externally pinned conversion manifest SHA-256",
    )
    args = parser.parse_args()

    contract = academic.load_acceptance_contract(args.acceptance_contract)
    try:
        manifest = academic.require_accepted_conversion(
            args.target / academic.MANIFEST_FILE_NAME,
            acceptance_contract=contract,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except (academic.ConversionError, ContractError) as error:
        print(f"accepted-conversion replay: failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    summary = {
        "status": "accepted",
        "manifest_sha256": args.expected_manifest_sha256,
        "run_identity_sha256": manifest["run_identity"]["identity_sha256"],
        "git_commit": manifest["run_identity"]["runtime"]["git_commit"],
        "totals": manifest["totals"],
        "acceptance": manifest["acceptance"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
