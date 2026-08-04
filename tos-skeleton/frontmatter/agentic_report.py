#!/usr/bin/env python3
"""본실험 채점 — 검색층: repo(span_metrics) 지표 + McNemar. 답변층: judge 결과 병합.

사용: python3 agentic_report.py            # 검색층만
      python3 agentic_report.py --judge    # judge 캐시 병합해 답변층까지
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/seyoung/workspace/braincrew/experiments/dr-dci")
from src.eval.span_metrics import recall_at_k, ndcg_at_k  # repo 표준 지표
from gate2 import BASE, QA

OUT = BASE / "frontmatter" / "out"


def mcnemar_exact(b, c):
    """양측 exact McNemar p값 (이항)."""
    n = b + c
    if n == 0:
        return 1.0
    def binom(k):
        return math.comb(n, k) * 0.5 ** n
    p = sum(binom(k) for k in range(0, min(b, c) + 1)) * 2
    return min(1.0, p)


def main(with_judge=False):
    rows = [json.loads(l) for l in open(OUT / "agentic3_full.jsonl")]
    qa = {j["질문"]: j for j in (json.loads(l) for l in open(QA))}
    judge = {}
    if with_judge and (OUT / "judge_cache.jsonl").exists():
        for l in open(OUT / "judge_cache.jsonl"):
            d = json.loads(l)
            judge[(d["arm"], d["q"])] = d["correct"]

    by_q = defaultdict(dict)
    for r in rows:
        if "error" not in r:
            by_q[r["q"]][r["arm"]] = r
    qs = [q for q, d in by_q.items() if len(d) == 3]
    print(f"3안 모두 완료된 문항 {len(qs)} (전체 행 {len(rows)})\n")

    print(f"{'':4s}{'r@1':>7s}{'r@3':>7s}{'r@5':>7s}{'ndcg@5':>8s}{'홉':>6s}{'초':>6s}"
          + ("{:>8s}".format("정답률") if with_judge else ""))
    stats = {}
    for arm in ("A", "B1", "B3"):
        r1 = r3 = r5 = nd = 0.0
        turns, durs, correct, njudge = [], [], 0, 0
        for q in qs:
            r = by_q[q][arm]
            ranked, gold = r["elements"], r["gold"]
            r1 += recall_at_k(ranked, gold, 1) > 0
            r3 += recall_at_k(ranked, gold, 3) > 0
            r5 += recall_at_k(ranked, gold, 5) > 0
            nd += ndcg_at_k(ranked, gold, 5)
            if r.get("turns"):
                turns.append(r["turns"])
            if r.get("dur_ms"):
                durs.append(r["dur_ms"])
            if (arm, q) in judge:
                njudge += 1
                correct += judge[(arm, q)]
        n = len(qs)
        stats[arm] = {"hit5": {q: recall_at_k(by_q[q][arm]["elements"], by_q[q][arm]["gold"], 5) > 0 for q in qs}}
        line = (f"{arm:4s}{r1/n:7.3f}{r3/n:7.3f}{r5/n:7.3f}{nd/n:8.3f}"
                f"{sum(turns)/len(turns):6.1f}{sum(durs)/len(durs)/1000:6.0f}")
        if with_judge and njudge:
            line += f"{correct/njudge:8.3f}"
        print(line)

    print("\nMcNemar (hit@5):")
    for x, y in (("A", "B1"), ("A", "B3"), ("B1", "B3")):
        b = sum(1 for q in qs if stats[x]["hit5"][q] and not stats[y]["hit5"][q])
        c = sum(1 for q in qs if not stats[x]["hit5"][q] and stats[y]["hit5"][q])
        print(f"  {x} vs {y}: {x}만 {b} | {y}만 {c} | p={mcnemar_exact(b,c):.3f}")

    if with_judge:
        print("\n근거×답변 교차 (B1):")
        cells = defaultdict(int)
        for q in qs:
            if ("B1", q) not in judge:
                continue
            cells[(stats["B1"]["hit5"][q], judge[("B1", q)])] += 1
        print(f"  근거O답O {cells[(True,1)]} | 근거O답X {cells[(True,0)]} | "
              f"근거X답O {cells[(False,1)]} | 근거X답X {cells[(False,0)]}")


if __name__ == "__main__":
    main("--judge" in sys.argv)
