"""Rank-based retrieval metrics for the pull-backend probe.

The agent-loop primary endpoint stays workspace document gold recall. These metrics
answer the retrieval-quality axis on the ranked candidate list that one pull
produces for the original query text, independent of agent query rewriting.
Callers must exclude queries without positive gold (denominator = judged
queries only); an empty gold set here is a caller bug and fails loudly.

On deeply judged collections (TREC-COVID: hundreds of positives per query)
Recall@20 has a low structural ceiling and Hit@k saturates, so nDCG@10 and
Precision@20 are the primary scale-probe metrics. nDCG follows the
pytrec_eval/BEIR convention: linear graded gain with a log2(rank+1) discount
(Järvelin & Kekäläinen 2002; Thakur et al. 2021, arXiv:2104.08663).
"""

import math


def validate_unique_ranked_ids(ranked_ids: list[str]) -> None:
    """A rank metric is defined only over unique retrieval item IDs."""
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked list must contain unique IDs")


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("recall_at_k requires a non-empty gold set")
    hits = len(set(ranked_ids[:k]) & set(gold_ids))
    return hits / len(gold_ids)


def hit_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("hit_at_k requires a non-empty gold set")
    return 1.0 if set(ranked_ids[:k]) & set(gold_ids) else 0.0


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("precision_at_k requires a non-empty gold set")
    return len(set(ranked_ids[:k]) & set(gold_ids)) / k


def ndcg_at_k(ranked_ids: list[str], gains: dict[str, float], k: int) -> float:
    if not gains:
        raise ValueError("ndcg_at_k requires non-empty graded gains")
    validate_unique_ranked_ids(ranked_ids)
    dcg = sum(
        gains.get(doc_id, 0.0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_ids[:k])
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    value = dcg / idcg if idcg > 0 else 0.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"nDCG@{k} outside [0, 1]: {value}")
    return value


def rank_metrics(ranked_ids: list[str], gold_ids: set[str],
                 gains: dict[str, float] = None) -> dict:
    """Fixed metric set for the backend comparison (same keys in both arms).

    Without graded gains, nDCG falls back to binary gains over the positive
    gold set so the metric keys stay identical across callers.
    """
    validate_unique_ranked_ids(ranked_ids)
    if gains is None:
        gains = {doc_id: 1.0 for doc_id in gold_ids}
    return {
        "recall_at_5": recall_at_k(ranked_ids, gold_ids, 5),
        "recall_at_20": recall_at_k(ranked_ids, gold_ids, 20),
        "hit_at_5": hit_at_k(ranked_ids, gold_ids, 5),
        "hit_at_10": hit_at_k(ranked_ids, gold_ids, 10),
        "precision_at_20": precision_at_k(ranked_ids, gold_ids, 20),
        "ndcg_at_10": ndcg_at_k(ranked_ids, gains, 10),
    }
