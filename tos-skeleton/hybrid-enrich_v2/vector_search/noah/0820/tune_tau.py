#!/usr/bin/env python3
"""폴백 τ 튜닝 — eval_det 의 det_ranks 파일에서 τ 민감도 곡선을 뽑는다.

폴백이 이상적으로 발화해야 하는 문항 = gold 가 태그 후보(limit 안)에 없는 문항.
트리거 규칙: n_cand < min_n 또는 top_score < τ.
각 τ 에 대해 (발화율, 이상발화 재현율/정밀도, F1) 을 출력 — 과적합 방지용 곡선 기록.

Usage: python tune_tau.py --ranks out/det_ranks_clm_lex-count_w-contract-2.jsonl
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", required=True)
ap.add_argument("--min-n", type=int, default=5)
a = ap.parse_args()

R = [json.loads(l) for l in open(a.ranks, encoding="utf-8")]
need = [not r["gold_in_cand"] for r in R]  # 폴백이 필요한 문항(태그 후보에 gold 없음)
print(f"n={len(R)} / 폴백 필요(태그 후보에 gold 없음) {sum(need)} ({100*sum(need)/len(R):.1f}%)")
taus = sorted({round(r["top_score"], 1) for r in R} | {0.0})
print("| tau | 발화율 | 재현율 | 정밀도 | F1 |\n|---|---|---|---|---|")
best = (0, None)
for tau in taus:
    fire = [(r["n_cand"] < a.min_n) or (r["top_score"] < tau) for r in R]
    tp = sum(1 for f, n in zip(fire, need) if f and n)
    prec = tp / max(sum(fire), 1)
    rec = tp / max(sum(need), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"| {tau} | {sum(fire)/len(R):.3f} | {rec:.3f} | {prec:.3f} | {f1:.3f} |")
    if f1 > best[0]:
        best = (f1, tau)
print(f"\n권장 tau = {best[1]} (F1 {best[0]:.3f}, min_n={a.min_n}) — arms.json 의 c4_fb/c4_all 에 반영")
