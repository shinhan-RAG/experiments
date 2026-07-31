#!/usr/bin/env python3
"""Prove a chunk corpus fits a frozen model input contract, or fail loud.

The academic chunk corpus is retrieval-unapproved until the embedding
tokenizer/revision and input budget are frozen. This validator takes that
frozen contract (see
``config/collection_academic/chunk_model_compat.template.yaml``) and either
proves every chunk fits ``max_input_chars``, or reports an explicitly
approved deterministic long-element split policy. It never calls a model.

Exit codes: 0 compatible or approved-split-policy recorded, 2 otherwise.
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
        result = academic.validate_chunk_model_compatibility(args.chunks, contract)
    except academic.ConversionError as error:
        print(f"chunk/model compatibility: failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
