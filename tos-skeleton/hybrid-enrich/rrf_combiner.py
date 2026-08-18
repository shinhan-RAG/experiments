#!/usr/bin/env python3
"""Reciprocal Rank Fusion — 여러 검색 채널의 순위 리스트를 결합한다.

참조: bak/metajson-v6/meta-search-v4/retrieve.py:86-96
공식: score(d) = Σ_channel 1/(k + rank_in_channel(d))
"""
from __future__ import annotations

import collections


def rrf_scores(*rankings: list[str], k: int = 60) -> dict[str, float]:
    score: dict[str, float] = collections.defaultdict(float)
    for r in rankings:
        for rank, uid in enumerate(r, 1):
            score[uid] += 1.0 / (k + rank)
    return dict(score)


def rrf_merge(*rankings: list[str], k: int = 60, top_n: int = 10) -> list[str]:
    score = rrf_scores(*rankings, k=k)
    return sorted(score, key=lambda uid: -score[uid])[:top_n]


def fill_with_rrf(agent_top: list[str], rrf_ranked: list[str], max_slots: int = 10) -> list[str]:
    """에이전트 결과 뒤에 RRF 순위로 빈 슬롯을 채운다."""
    seen = set(agent_top)
    out = list(agent_top[:max_slots])
    for uid in rrf_ranked:
        if len(out) >= max_slots:
            break
        if uid not in seen:
            out.append(uid)
            seen.add(uid)
    return out
