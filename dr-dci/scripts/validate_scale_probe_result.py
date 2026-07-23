#!/usr/bin/env python3
"""Validate a saved Part 2 dense scale-probe result without model calls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.scale_probe_contract import validate_scale_probe_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="results/part2_scale_probe/<timestamp>.json")
    args = parser.parse_args()
    with args.result.open(encoding="utf-8") as stream:
        errors = validate_scale_probe_result(json.load(stream))
    if errors:
        print("INVALID")
        print("\n".join(f"- {error}" for error in errors))
        return 2
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
