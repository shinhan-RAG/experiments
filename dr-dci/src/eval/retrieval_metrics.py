"""Rank-based retrieval metrics for the pull-backend probe.

The agent-loop primary endpoint stays workspace gold recall. These metrics
answer the retrieval-quality axis on the ranked candidate list that one pull
produces for the original query text, independent of agent query rewriting.
Callers must exclude queries without positive gold (denominator = judged
queries only); an empty gold set here is a caller bug and fails loudly.
"""


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("recall_at_k requires a non-empty gold set")
    hits = len(set(ranked_ids[:k]) & set(gold_ids))
    return hits / len(gold_ids)


def hit_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("hit_at_k requires a non-empty gold set")
    return 1.0 if set(ranked_ids[:k]) & set(gold_ids) else 0.0


def rank_metrics(ranked_ids: list[str], gold_ids: set[str]) -> dict:
    """Fixed metric set for the backend comparison (same keys in both arms)."""
    return {
        "recall_at_5": recall_at_k(ranked_ids, gold_ids, 5),
        "recall_at_20": recall_at_k(ranked_ids, gold_ids, 20),
        "hit_at_5": hit_at_k(ranked_ids, gold_ids, 5),
        "hit_at_10": hit_at_k(ranked_ids, gold_ids, 10),
    }
