"""Rank fusion helpers."""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item["doc_id"]] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    if top_k is not None:
        fused = fused[:top_k]
    return [
        {"doc_id": doc_id, "score": float(score)}
        for doc_id, score in fused
    ]
