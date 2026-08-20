#!/usr/bin/env python3
"""alias 확장 후 ceiling 재계산 — 메타 채널은 ceiling_train.json 캐시(in_meta 플래그) 재사용.

union' = tag(alias 확장) top-40 ∪ meta top-40(캐시). 임베딩 재계산 없음.
suff 상한' = 모든 그룹이 union' 에 존재하는 문항 비율.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import overlaps
from units import Units

ceil = json.load(open(HERE / "out/ceiling_train.json", encoding="utf-8"))
gold = {g["qid"]: g for g in (json.loads(l) for l in open(FS / "out/gold_spans_lsh_train.jsonl", encoding="utf-8"))}
ranks = {json.loads(l)["qid"]: json.loads(l) for l in open(HERE / "out/det_ranks_clm_lex-count_w-contract-2_alias-1.jsonl", encoding="utf-8")}
U = Units()
jix = {u["element_id"]: u for u in U.J}

tot = cov_old = cov_new = 0
suff_old = suff_new = 0
recovered = []
for r in ceil["rows"]:
    g = gold[r["qid"]]
    tag_jo = [jix[i] for i in ranks[r["qid"]]["jo_top"][:40] if i in jix]
    ok_old = ok_new = True
    for gr, st in zip(g["groups"], r["groups"]):
        tot += 1
        old = st["union_rank"] is not None
        new = old or st["in_meta"] or any(overlaps(u, gr) for u in tag_jo)
        cov_old += old; cov_new += new
        ok_old &= old; ok_new &= new
        if new and not old:
            recovered.append((r["qid"], g["q"][:50]))
    suff_old += ok_old; suff_new += ok_new

n = len(ceil["rows"])
print(f"그룹 커버리지 상한: {100*cov_old/tot:.1f}% -> {100*cov_new/tot:.1f}% (+{100*(cov_new-cov_old)/tot:.1f}%p, 회수 {cov_new-cov_old}그룹)")
print(f"suff 상한(문항):     {100*suff_old/n:.1f}% -> {100*suff_new/n:.1f}% (+{100*(suff_new-suff_old)/n:.1f}%p)")
print("\n회수된 그룹:")
for q, t in recovered:
    print(f"  {q} | {t}")
