#!/usr/bin/env python3
"""Build an artifact-free MIRACL-ko Flat-L1 generation-plan candidate.

The output belongs in a separately controlled read-only directory. It is not
an approval record and it does not start a model or access a network service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.flat_l1 import (  # noqa: E402
    FLAT_L1_GENERATOR_SOURCE_FILES,
    build_flat_l1_generation_plan,
)
from src.miracl_ko.taxonomy_artifact import (  # noqa: E402
    build_generator_code_contract,
    generation_plan_sha256,
    generator_contract_sha256,
)


def clean_head() -> str:
    for arguments in (("diff", "--quiet", "HEAD"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["git", "-C", str(REPO_ROOT), *arguments], check=False).returncode != 0:
            raise RuntimeError("build the plan only from a clean tracked generator source checkout")
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_commit = clean_head()
    contract = build_generator_code_contract(REPO_ROOT, FLAT_L1_GENERATOR_SOURCE_FILES)
    plan = build_flat_l1_generation_plan(
        repo_root=REPO_ROOT,
        generator_source_commit=source_commit,
        generator_code_contract_sha256=generator_contract_sha256(contract),
        generator_code_sha256=contract["generator_code_sha256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "generation_plan_sha256": generation_plan_sha256(plan),
        "generator_source_commit": source_commit,
        "generator_contract_sha256": generator_contract_sha256(contract),
        "generator_code_sha256": contract["generator_code_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
