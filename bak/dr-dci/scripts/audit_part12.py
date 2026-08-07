#!/usr/bin/env python3
"""Run the model-free Part 1/2 experiment preflight."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.part12_contracts import audit_part12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step", action="append", dest="steps",
                        help="Part 1 arm to audit; repeatable. Defaults to baseline + taxonomy_only")
    parser.add_argument("--size", action="append", type=int, dest="sizes",
                        help="Corpus size to audit; repeatable. Defaults to Part 2 sizes")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    selected = set(args.steps or ["baseline", "taxonomy_only"])
    report = audit_part12(config, args.data_dir, step_names=selected, sizes=args.sizes)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
