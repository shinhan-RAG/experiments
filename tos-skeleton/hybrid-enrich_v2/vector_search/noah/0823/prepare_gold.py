#!/usr/bin/env python3
"""Report the default test90 denominator or build strict raw-v3 test138 from an external spanmap."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_90 = ROOT / "out" / "noah" / "gold_v4_test_full.jsonl"
RAW_149 = ROOT / "out" / "gold_mapped_noah_v3_149_test.jsonl"
BUILDER = HERE.parent / "0820" / "build_gold_spans_test.py"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_gold(path: Path) -> dict:
    rows = list(map(json.loads, path.open(encoding="utf-8")))
    eligible = [x for x in rows if x.get("groups") and x.get("status", "ok") == "ok"]
    return {
        "path": str(path.resolve()), "sha256": digest(path), "n_source": len(rows), "n_eligible": len(eligible),
        "n_c3_partial": sum(bool(x.get("c3_partial")) for x in eligible),
        "n_strict_nonpartial": sum(not bool(x.get("c3_partial")) for x in eligible),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spanmap", type=Path, help="cowork 원문 span 매핑 파일; 없으면 test90 manifest만 생성")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "gold_spans_lsh_test_v3_138.jsonl")
    args = ap.parse_args()
    manifest = {"default_test90": count_gold(DEFAULT_90), "raw_test149": {"path": str(RAW_149.resolve()), "sha256": digest(RAW_149)}}
    if args.spanmap:
        if not args.spanmap.exists():
            raise SystemExit(f"spanmap 없음: {args.spanmap}")
        subprocess.run([sys.executable, str(BUILDER), "--spanmap", str(args.spanmap), "--validate"], check=True, cwd=HERE)
        subprocess.run([
            sys.executable, str(BUILDER), "--spanmap", str(args.spanmap), "--test-gold", str(RAW_149),
            "--out", str(args.out), "--expected-empty", "11", "--submit-cap", "10",
        ], check=True, cwd=HERE)
        built = count_gold(args.out)
        if built["n_eligible"] != 138:
            raise SystemExit(f"고정 분모 위반: expected=138 actual={built['n_eligible']}")
        manifest["optional_test138"] = built
        manifest["official_if_present"] = "optional_test138"
    else:
        manifest["spanmap_status"] = "not_available; external transfer required"
        manifest["official_if_present"] = "default_test90"
    out = HERE / "out" / "gold_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
