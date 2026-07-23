"""
동일-K retrieval 지표와 통계 도구 (P1-1, P1-11)

- recall_at_k / ndcg_at_k: ranked list 기반, 시스템 간 동일 K 비교용
- bootstrap_ci: query-level 평균의 95% CI
- paired_bootstrap_test: 동일 query 단위 paired 비교 p-value
"""

import math
import random


def recall_at_k(ranked_ids: list[str], gold_ids: set, k: int) -> float:
    if not gold_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & set(gold_ids)) / len(gold_ids)


def ndcg_at_k(ranked_ids: list[str], qrel_scores: dict, k: int) -> float:
    """qrel_scores: {doc_id: graded relevance score(int)}"""
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        rel = qrel_scores.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 2)

    ideal = sorted(qrel_scores.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(ranked_ids: list[str], gold_ids: set, k: int) -> float:
    for i, did in enumerate(ranked_ids[:k]):
        if did in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """query-level 값들의 평균과 percentile bootstrap 95% CI."""
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def paired_bootstrap_test(values_a: list[float], values_b: list[float],
                          n_boot: int = 2000, seed: int = 42) -> dict:
    """동일 query 순서로 정렬된 두 시스템 값의 paired bootstrap.

    반환: 평균 차이(a-b), 95% CI, two-sided p-value 근사.
    """
    assert len(values_a) == len(values_b), "paired test requires equal-length aligned lists"
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if not diffs:
        return {"mean_diff": None, "ci_low": None, "ci_high": None, "p_value": None}
    rng = random.Random(seed)
    n = len(diffs)
    mean_diff = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
    # two-sided p: 부호가 뒤집히는 bootstrap 비율의 2배 (근사)
    if mean_diff >= 0:
        p = 2 * sum(1 for b in boots if b <= 0) / n_boot
    else:
        p = 2 * sum(1 for b in boots if b >= 0) / n_boot
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(min(p, 1.0), 4),
    }
