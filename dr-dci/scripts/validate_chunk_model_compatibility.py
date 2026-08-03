#!/usr/bin/env python3
"""Prove a chunk corpus fits a frozen model input contract, or fail loud.

The academic chunk corpus is retrieval-unapproved until the embedding
tokenizer/revision and input budget are frozen. This validator takes that
frozen contract (see
``config/collection_academic/chunk_model_compat.template.yaml``) and proves
every chunk fits ``max_input_chars``. An owner-approved deterministic
long-element split policy (file bytes/SHA-256 verified) does not make an
oversized corpus usable: the status becomes ``requires_rebuild`` and the
exit code stays non-zero until the split corpus is rebuilt, alignment is
regenerated, and this validation passes with zero violations. It never
calls a model.

Exit codes: 0 compatible (zero violations), 4 requires_rebuild (approved
split policy recorded but oversized chunks remain), 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.data import collection_academic as academic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path, help="chunks.jsonl to validate")
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="frozen chunk/model compatibility contract YAML",
    )
    args = parser.parse_args()

    contract = academic.load_compat_contract(args.contract)
    try:
        result = academic.validate_chunk_model_compatibility(
            args.chunks, contract, contract_dir=args.contract.resolve().parent
        )
    except academic.ConversionError as error:
        print(f"chunk/model compatibility: failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "compatible":
        print(
            "chunk/model compatibility: corpus is not usable yet "
            f"(status={result['status']})",
            file=sys.stderr,
        )
        raise SystemExit(4)


if __name__ == "__main__":
    main()
