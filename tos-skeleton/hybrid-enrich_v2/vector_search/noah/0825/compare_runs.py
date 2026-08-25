#!/usr/bin/env python3
"""Paired report for baseline vs candidate deterministic results."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ("R@5", "R@10", "suff@5", "suff@10")


def load(path: Path) -> dict:
    rows = list(map(json.loads, path.open(encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    return {
        qid: {**items[0], **{key: sum(x[key] for x in items) / len(items) for key in KEYS}}
        for qid, items in grouped.items()
    }


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def report(base: dict, cand: dict, qids: list[str]) -> dict:
    result = {"n": len(qids), "metrics": {}}
    for key in KEYS:
        bv = sum(base[q][key] for q in qids) / len(qids)
        cv = sum(cand[q][key] for q in qids) / len(qids)
        diffs = [cand[q][key] - base[q][key] for q in qids]
        wins = sum(x > 0 for x in diffs)
        losses = sum(x < 0 for x in diffs)
        result["metrics"][key] = {
            "baseline": round(bv, 6), "candidate": round(cv, 6),
            "delta": round(cv - bv, 6),
            "wins": wins, "losses": losses,
            "ties": len(qids) - wins - losses,
            "sign_test_p": round(sign_p(wins, losses), 6),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "comparison.json")
    args = ap.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    common = sorted(set(base) & set(cand))
    if not common:
        raise SystemExit("no common QIDs")
    if set(base) != set(cand):
        print(f"warning: QID mismatch baseline={len(base)} candidate={len(cand)} common={len(common)}")

    output = {"all": report(base, cand, common)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
