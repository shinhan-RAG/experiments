#!/usr/bin/env python3
"""이미 실행된 run(results.jsonl 의 submitted)을 다른 gold 로 재채점. LLM 0회."""
import argparse, collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import score
from units import Units

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="쉼표 구분 run 이름(out/agent/<run>)")
ap.add_argument("--gold", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
U = Units()
G = {g["qid"]: g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"}
print(f"gold={Path(a.gold).name} n_q_available={len(G)}")
print("| run | n | R@1 | R@5 | R@10 | S@5 | suff@10 | MRR@10 | core R@5 | 비core R@5 |")
print("|---" * 10 + "|")
for run in a.runs.split(","):
    rows = [json.loads(l) for l in open(HERE / "out/agent" / run / "results.jsonl")]
    byq = collections.defaultdict(list)
    for r in rows:
        g = G.get(r["qid"])
        if not g:
            continue
        sc = score(U.resolve(r["submitted"]), g["groups"], ks=(1, 5, 10, 20))
        sc["_core"] = g["core_retrieval"] == "True"
        byq[r["qid"]].append(sc)
    agg = {k: sum(sum(x[k] for x in v) / len(v) for v in byq.values()) / len(byq) for k in ("R@1", "R@5", "R@10", "S@5", "suff@10", "RR@10")}
    core = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if v[0]["_core"]]
    nc = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if not v[0]["_core"]]
    print(f"| {run} | {len(byq)} | " + " | ".join(f"{agg[k]:.3f}" for k in ("R@1", "R@5", "R@10", "S@5", "suff@10", "RR@10")) + f" | {sum(core)/max(1,len(core)):.3f} | {sum(nc)/max(1,len(nc)):.3f} |")
