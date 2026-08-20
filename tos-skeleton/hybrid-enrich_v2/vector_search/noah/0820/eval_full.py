#!/usr/bin/env python3
"""교정 train 337 전체 실측 병합 채점 — base_c3 vs f3_both 페어드 비교.

분모 병기: n=337(전체) / n=298(c3_partial 39 제외).
런은 반쪽 분할(full_*_a + full_*_b)을 병합한다. 러너가 이미 gold_v4 기준으로 채점했으므로
results.jsonl 의 지표를 그대로 쓰되, 페어드 Δ와 부호검정 p 를 추가한다.
"""
import json, glob, math, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
gold = {g["qid"]: g for g in (json.loads(l) for l in open(HERE.parents[2] / "out/noah/gold_v4_train_full.jsonl", encoding="utf-8"))}
PART = {q for q, g in gold.items() if g.get("c3_partial")}

def load(runs):
    rows = {}
    for r in runs:
        p = HERE / "out/agent" / r / "results.jsonl"
        if not p.exists():
            print("경고: 없음", p)
            continue
        for x in (json.loads(l) for l in open(p, encoding="utf-8")):
            rows[x["qid"]] = x
    return rows

base = load(["full_base_a", "full_base_b"])
f3 = load(["full_f3_a", "full_f3_b"])
common = sorted(set(base) & set(f3))
print("base %d문항, f3 %d문항, 공통 %d" % (len(base), len(f3), len(common)))

def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0); neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    # 이항 양측 p (정규근사 없이 정확계산)
    from math import comb
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / 2 ** n
    return min(1.0, p)

for label, qs in (("n=337(전체)", common), ("n=298(c3_partial 제외)", [q for q in common if q not in PART])):
    print("\n==", label, f"실제 {len(qs)}문항")
    print(f"{'지표':10s} {'base':>8s} {'f3_both':>8s} {'Δ':>8s} {'개선/악화':>10s} {'p(부호)':>8s}")
    for k in ("R@5", "R@10", "suff@5", "suff@10"):
        b = [base[q][k] for q in qs]; f = [f3[q][k] for q in qs]
        d = [fi - bi for bi, fi in zip(b, f)]
        p = sign_test(d)
        print(f"{k:10s} {sum(b)/len(b):8.4f} {sum(f)/len(f):8.4f} {sum(d)/len(d):+8.4f} "
              f"{sum(1 for x in d if x>0):4d}/{sum(1 for x in d if x<0):<4d} {p:8.3f}")
    bytt = collections.defaultdict(lambda: [[], []])
    for q in qs:
        tt = gold[q]["task_type"]
        bytt[tt][0].append(base[q]["R@5"]); bytt[tt][1].append(f3[q]["R@5"])
    print("  유형별 R@5:")
    for tt, (b, f) in sorted(bytt.items()):
        print(f"    {tt:18s} n={len(b):3d} base={sum(b)/len(b):.3f} f3={sum(f)/len(f):.3f} Δ={sum(f)/len(f)-sum(b)/len(b):+.3f}")
