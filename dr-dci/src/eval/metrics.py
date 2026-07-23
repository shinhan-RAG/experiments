"""
동일-K retrieval 지표와 통계 도구 (P1-1, P1-11)

- recall_at_k / ndcg_at_k: ranked list 기반, 시스템 간 동일 K 비교용
- bootstrap_ci: query-level 평균의 95% CI
- paired_bootstrap_test: 동일 query 단위 paired 비교 p-value
"""

import random

from .retrieval_metrics import ndcg_at_k, recall_at_k


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
    """동일 query 순서로 정렬된 두 시스템 값의 paired 비교.

    평균 차이의 CI는 paired bootstrap, p-value는 질의 단위 sign-flip
    randomization으로 계산한다. bootstrap 분포를 그대로 귀무가설
    p-value로 해석하지 않는다.
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
    observed = abs(mean_diff)
    extreme = 0
    for _ in range(n_boot):
        randomized = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(randomized) >= observed - 1e-15:
            extreme += 1
    p = (extreme + 1) / (n_boot + 1)
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(min(p, 1.0), 4),
    }
