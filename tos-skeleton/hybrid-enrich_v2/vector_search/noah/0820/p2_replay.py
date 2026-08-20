#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2(submit_cover) 오프라인 재적용 — LLM 0회.

기존 런(기본 out/agent/v4_s_all60/results.jsonl)의 submitted 리스트에 agent_tools.cover_reorder
(top-5 근사중복 조 압축, 같은 군 최대 2개)를 그대로 적용해 재채점(R@5/R@10 before/after).
주의: 원 런은 submit_pad/ref_merge 가 이미 적용된 최종 리스트이므로, 실서빙과 동일하게
pad/merge 이후 시점에 cover 정렬을 얹는 재현이 된다.

사용: PYTHONUTF8=1 python p2_replay.py [--results out/agent/v4_s_all60/results.jsonl]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))

from units import Units
from scoring import score
from agent_tools import cover_reorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out/agent/v4_s_all60/results.jsonl")
    a = ap.parse_args()

    gold = {g["qid"]: g for g in (json.loads(l) for l in open(HERE / "out/gold_v4_train60.jsonl", encoding="utf-8"))}
    recs = [json.loads(l) for l in open(HERE / a.results, encoding="utf-8")]
    U = Units()
    cache = {}

    def jo_of(x):
        if x not in cache:
            r = U.resolve([x])
            cache[x] = r[0] if r else None
        return cache[x]

    rows, n_moved_q, n_changed_score = [], 0, 0
    for rec in recs:
        g = gold.get(rec["qid"])
        ids = rec.get("submitted") or []
        if g is None:
            continue
        before = score(U.resolve(ids), g["groups"], ks=(5, 10))
        new_ids, moved = cover_reorder(ids, jo_of)
        after = score(U.resolve(new_ids), g["groups"], ks=(5, 10))
        if moved:
            n_moved_q += 1
        d5 = after["R@5"] - before["R@5"]
        if d5 != 0 or after["R@10"] != before["R@10"]:
            n_changed_score += 1
            print(f"  {rec['qid']}: R@5 {before['R@5']:.2f}→{after['R@5']:.2f} "
                  f"R@10 {before['R@10']:.2f}→{after['R@10']:.2f} moved={moved}")
        rows.append((before, after, moved))

    n = len(rows)
    print(f"\n== P2 submit_cover 오프라인 재적용 ({a.results}, n={n}) ==")
    print(f"cover 정렬로 top-5 에서 밀린 조가 있는 문항: {n_moved_q}/{n}, 점수 변동 문항: {n_changed_score}/{n}")
    print(f"{'지표':<8}{'before':>10}{'after':>10}{'Δ':>10}")
    for k in ("R@5", "R@10"):
        b = sum(r[0][k] for r in rows) / n
        f = sum(r[1][k] for r in rows) / n
        print(f"{k:<8}{b:>10.4f}{f:>10.4f}{f - b:>+10.4f}")


if __name__ == "__main__":
    main()
