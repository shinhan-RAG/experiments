#!/usr/bin/env python3
"""채점 공통 함수 — gold span(char 구간) 대비 element 순위 채점.
Recall@K = fractional evidence-group(Top-K 가 덮은 group / 전체 group), Success@K, RR@10, 글자예산 recall.
"""
import collections

def bigrams(s):
    s = "".join(s.split())
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]

def overlaps(e, g):
    """element e 가 gold group g 와 겹치는가. g 는 {c0,c1} 단일 구간 또는 {members:[{c0,c1},…]} (OR 멤버 — a/a' 동치·동일문구 출현)."""
    mem = g.get("members") or [g]
    return any(e["char_start"] < m["c1"] and e["char_end"] > m["c0"] for m in mem)

def score(ranked_units, groups, ks=(1, 5, 10, 20)):
    """ranked_units: element dict 리스트(순위순). 반환: recall@k(fractional), success@k, rr@10"""
    out = {}
    hit_rank = {}
    for r, e in enumerate(ranked_units):
        for gi, g in enumerate(groups):
            if gi not in hit_rank and overlaps(e, g):
                hit_rank[gi] = r + 1
    for k in ks:
        n = sum(1 for gi in hit_rank if hit_rank[gi] <= k)
        out[f"R@{k}"] = n / len(groups)
        out[f"S@{k}"] = 1.0 if n else 0.0
    fr = min(hit_rank.values()) if hit_rank else None
    out["RR@10"] = 1.0 / fr if fr and fr <= 10 else 0.0
    # sufficient@k: 모든 group(AND) 이 top-k 안에서 회수(OR 은 멤버 1개면 충족) — 정답기준 명세 v1 주지표
    for k in (5, 10):
        out[f"suff@{k}"] = 1.0 if groups and all(hit_rank.get(gi, 10**9) <= k for gi in range(len(groups))) else 0.0
    return out

def budget_recall(ranked_units, groups, B):
    acc, tot = [], 0
    for e in ranked_units:
        acc.append(e); tot += len(e["text"])
        if tot >= B:
            break
    hit = sum(1 for g in groups if any(overlaps(e, g) for e in acc))
    return hit / len(groups)
