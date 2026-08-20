#!/usr/bin/env python3
"""에이전트 run 두 개를 qid별 rep 평균 후 paired 비교한다."""
import argparse
import collections
import json

from stats import bootstrap_ci, sign_test


def load(path, key):
    byq = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        byq[row["qid"]].append(float(row[key]))
    return {qid: sum(values) / len(values) for qid, values in byq.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="challenger results.jsonl")
    ap.add_argument("--b", required=True, help="baseline results.jsonl")
    ap.add_argument("--key", default="R@5")
    a = ap.parse_args()
    A, B = load(a.a, a.key), load(a.b, a.key)
    qids = sorted(set(A) & set(B))
    av, bv = [A[q] for q in qids], [B[q] for q in qids]
    print(json.dumps({
        "key": a.key, "n_q": len(qids),
        "challenger": sum(av) / len(av), "baseline": sum(bv) / len(bv),
        "sign": sign_test(av, bv), "bca": bootstrap_ci(av, bv),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
