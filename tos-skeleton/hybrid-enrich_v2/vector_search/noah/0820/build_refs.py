#!/usr/bin/env python3
"""참조 그래프 사전계산 — filesearch/out/elements_u2jo.jsonl → out/refs_jo.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(HERE))
from enhance import build_refs

J = [json.loads(l) for l in open(FS / "out" / "elements_u2jo.jsonl", encoding="utf-8")]
refs = build_refs(J)
(HERE / "out").mkdir(exist_ok=True)
json.dump(refs, open(HERE / "out" / "refs_jo.json", "w", encoding="utf-8"), ensure_ascii=False)
n_edges = sum(len(v) for v in refs.values())
print(f"조 {len(J):,} / 참조 보유 조 {len(refs):,} / edge {n_edges:,} -> out/refs_jo.json")
