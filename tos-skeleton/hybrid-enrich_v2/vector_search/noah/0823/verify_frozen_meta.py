#!/usr/bin/env python3
"""Compare 0820 and 0821 msearch top-k outputs and record frozen V9 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "0820" / "agent_tools.py"
TRAIN = ROOT / "out" / "noah" / "gold_v4_train_full.jsonl"
VS = ROOT / "vector_search"
FILES = [
    VS / "hybrid_search.py", VS / "out" / "view_V9.jsonl", VS / "out" / "chunks.jsonl",
    VS / "out" / "emb" / "chunk_V9.npy", VS / "out" / "emb" / "chunk_V9_ids.json",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def call(script: Path, query: str, arm: dict, session: Path) -> dict:
    env = dict(os.environ, SEMTAG_SESSION=str(session), SEMTAG_QID="parity", SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    result = subprocess.run(
        [sys.executable, str(script), "msearch", "--q", query], cwd=script.parent, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        raise SystemExit(f"V9 필수 파일 없음: {missing}")
    report = {"files": {str(p.resolve()): digest(p) for p in FILES}, "n_queries": 0, "parity": None}
    if not args.manifest_only:
        queries = [x["q"] for x in list(map(json.loads, TRAIN.open(encoding="utf-8")))[: args.n]]
        old_arm = {"tags": "tags_u2_rules.jsonl", "mode": "clm", "lex": "count", "meta": 1, "meta_view": "V9"}
        new_arm = {"tag_weights": {"clm": 1.0, "sparse": 0.0}, "meta": 1, "meta_view": "V9"}
        mismatches = []
        with tempfile.TemporaryDirectory(prefix="meta_parity_") as temp:
            base = Path(temp)
            for i, query in enumerate(queries):
                old = call(OLD, query, old_arm, base / f"old_{i}")
                new = call(HERE / "agent_tools.py", query, new_arm, base / f"new_{i}")
                old_ids = [(x["id"], x.get("jo", "")) for x in old.get("results", [])]
                new_ids = [(x["id"], x.get("jo", "")) for x in new.get("results", [])]
                if old_ids != new_ids:
                    mismatches.append({"query": query, "old": old_ids, "new": new_ids})
        report.update({"n_queries": len(queries), "parity": not mismatches, "mismatches": mismatches})
        if mismatches:
            raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    out = HERE / "out" / "frozen_meta_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
