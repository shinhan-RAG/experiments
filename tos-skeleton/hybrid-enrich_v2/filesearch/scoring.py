#!/usr/bin/env python3
"""채점 공통 함수 — gold span(char 구간) 대비 element 순위 채점.
Recall@K = fractional evidence-group(Top-K 가 덮은 group / 전체 group), Success@K, RR@10, 글자예산 recall.
"""
import collections

def bigrams(s):
    s = "".join(s.split())
    return [s[i:i + 2] for i in range(len(s) - 1)] if len(s) > 1 else [s]

def overlaps(e, g):
    return e["char_start"] < g["c1"] and e["char_end"] > g["c0"]

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
    return out

def budget_recall(ranked_units, groups, B):
    acc, tot = [], 0
    for e in ranked_units:
        acc.append(e); tot += len(e["text"])
        if tot >= B:
            break
    hit = sum(1 for g in groups if any(overlaps(e, g) for e in acc))
    return hit / len(groups)
