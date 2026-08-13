#!/usr/bin/env python3
"""가중 RRF — 등가중 융합이 기각된 이유를 직접 겨냥한 변형.

무엇이 문제였나
    `bak/dr-dci/src/retrieval/fusion.py`의 RRF는 가중치 인자가 없다.

        scores[doc_id] += 1.0 / (k + rank)

    모든 채널의 1위가 같은 가산점을 받는다. dr-dci `docs/PULL_BACKEND_EXPERIMENT.md`
    §7.1-7.3이 이 설계로 FiQA에서 기각당했다 — R@20 .6669 -> .6194,
    **신규로 얻은 gold 21건 vs 밀려난 dense gold 117건**. 약한 채널이 강한 채널의
    상위권을 밀어낸 것이다. 그 문서의 자체 진단도 "비등가 융합 가중치"를 지목한다.

    tos-skeleton에서도 같은 일이 재현됐다. J1 dense R@5 .5366 -> RRF .5000.
    다만 깊은 곳은 올랐다(R@100 .8537 -> .8902). 즉 **머리를 잃고 꼬리를 얻는다**.

가중치가 하는 일
    상위권 가산점의 크기를 보면 왜 w가 듣는지 보인다(k=60 기준).

        dense 1위   1.00/61 = .01639
        어휘  1위   w/61
        dense 5위   1.00/65 = .01538

    w=1.0이면 어휘 1위(.01639)가 dense 5위(.01538)보다 커서 상위권을 재배열한다.
    w=0.35면 어휘 1위가 .00574로 dense 100위(.00621)보다도 작아, 상위권은 건드리지
    못하고 **dense가 놓친 꼬리만 끌어올린다**. 이것이 "신규는 얻되 밀어내지는 않는"
    영역이고, 이번 스모크가 확인하려는 것이다.

기본값은 등가중과 비트 동일
    weights=None, k=60이면 dr-dci와 완전히 같은 값을 낸다. 비교 기준선이 흔들리지
    않도록 일부러 그렇게 뒀다.
"""
from __future__ import annotations

import collections

RRF_K = 60


def rrf_scores(*rankings: list[int], weights: list[float] | None = None,
               k: int = RRF_K) -> dict[int, float]:
    """가중 RRF. weights=None이면 전부 1.0(등가중)."""
    ws = weights if weights is not None else [1.0] * len(rankings)
    if len(ws) != len(rankings):
        raise ValueError(f"가중치 {len(ws)}개 != 랭킹 {len(rankings)}개")
    score: dict[int, float] = collections.defaultdict(float)
    for w, r in zip(ws, rankings):
        if w == 0.0:
            continue
        for rank, i in enumerate(r, 1):
            score[i] += w / (k + rank)
    return score


def fuse(*rankings: list[int], weights: list[float] | None = None,
         k: int = RRF_K, topk: int | None = None) -> list[int]:
    """융합 순위. 동점은 인덱스 오름차순으로 깨서 실행 간 재현성을 보장한다."""
    score = rrf_scores(*rankings, weights=weights, k=k)
    order = sorted(score, key=lambda i: (-score[i], i))
    return order[:topk] if topk else order


def displacement(base: list[int], fused: list[int], gold: set[int], k: int = 5) -> dict:
    """융합이 상위 k에서 무엇을 얻고 무엇을 잃었는지.

    dr-dci가 "신규 21 / 밀려난 117"을 셌던 것과 같은 진단이다. 총점만 보면
    이득과 손실이 상쇄돼 보이지만, 실제로는 서로 다른 질의에서 일어난다.
    """
    b, f = set(base[:k]), set(fused[:k])
    return {
        "gained": len((f - b) & gold),
        "displaced": len((b - f) & gold),
        "base_hit": len(b & gold),
        "fused_hit": len(f & gold),
    }
