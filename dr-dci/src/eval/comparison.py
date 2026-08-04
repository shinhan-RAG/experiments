"""Paired comparisons for controlled retrieval experiments."""

import numpy as np


def paired_bootstrap_delta(
    control: list[float],
    treatment: list[float],
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict:
    if len(control) != len(treatment):
        raise ValueError("paired samples must have the same length")
    if not control:
        raise ValueError("paired samples must not be empty")

    control_arr = np.asarray(control, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    deltas = treatment_arr - control_arr
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])

    # p-value: 질의 단위 sign-flip randomization (귀무가설: 처치 효과 없음).
    # bootstrap CI를 p-value로 재해석하지 않는다 — metrics.paired_bootstrap_test와
    # 같은 정의((extreme+1)/(n+1))를 numpy로 벡터화한 것.
    observed = abs(float(deltas.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(deltas)))
    randomized_means = np.abs((signs * deltas).mean(axis=1))
    extreme = int(np.sum(randomized_means >= observed - 1e-15))
    p_value = min((extreme + 1) / (iterations + 1), 1.0)

    control_mean = float(control_arr.mean())
    mean_delta = float(deltas.mean())
    # 상대 %는 baseline 절대값과 항상 병기한다 — 바닥(0.039) 위의 +29% 같은
    # 착시를 절대 델타 없이 단독 보고하지 않기 위한 구조적 장치.
    relative_pct = (
        round(mean_delta / control_mean * 100, 2)
        if abs(control_mean) > 1e-12 else None
    )
    return {
        "n": len(deltas),
        "control_mean": round(control_mean, 6),
        "treatment_mean": round(float(treatment_arr.mean()), 6),
        "mean_delta": round(mean_delta, 6),
        "relative_pct": relative_pct,
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
        "p_value": round(p_value, 6),
        "iterations": iterations,
        "seed": seed,
    }


def compare_paired_results(
    control: list[dict],
    treatment: list[dict],
    *,
    seed: int,
) -> dict:
    control_by_id = {row["query_id"]: row for row in control}
    treatment_by_id = {row["query_id"]: row for row in treatment}
    shared_ids = sorted(set(control_by_id) & set(treatment_by_id))
    if not shared_ids:
        raise ValueError("no shared query IDs for paired comparison")

    recall = paired_bootstrap_delta(
        [control_by_id[qid].get("gold_recall", 0.0) for qid in shared_ids],
        [treatment_by_id[qid].get("gold_recall", 0.0) for qid in shared_ids],
        seed=seed,
    )
    return {
        "primary_metric": "gold_recall",
        "treatment_minus_control": recall,
        "control_query_count": len(control_by_id),
        "treatment_query_count": len(treatment_by_id),
        "paired_query_count": len(shared_ids),
    }


PROBE_METRIC_KEYS = ("recall_at_5", "recall_at_20", "hit_at_5", "hit_at_10",
                     "precision_at_20", "ndcg_at_10", "probe_latency_seconds")

RESULT_METRIC_KEYS = (
    "gold_recall",
    "efficiency",
    "pull_count",
    "taxonomy_filtered_pulls",
    "latency_seconds",
    "llm_prompt_tokens",
    "llm_completion_tokens",
)


def compare_probe_rows(
    control: list[dict],
    treatment: list[dict],
    *,
    seed: int,
    metric_keys: tuple[str, ...] = PROBE_METRIC_KEYS,
) -> dict:
    """retrieval-only probe의 질의별 paired 비교 — rank 지표 각각에 대해
    bootstrap 95% CI를 병기한다(집계 평균만으로 비교 금지 원칙)."""
    control_by_id = {row["query_id"]: row for row in control}
    treatment_by_id = {row["query_id"]: row for row in treatment}
    shared_ids = sorted(set(control_by_id) & set(treatment_by_id))
    if not shared_ids:
        raise ValueError("no shared query IDs for probe comparison")

    out = {"paired_query_count": len(shared_ids)}
    for key in metric_keys:
        out[key] = paired_bootstrap_delta(
            [float(control_by_id[qid].get(key, 0.0)) for qid in shared_ids],
            [float(treatment_by_id[qid].get(key, 0.0)) for qid in shared_ids],
            seed=seed,
        )
    return out


def compare_result_rows(
    control: list[dict],
    treatment: list[dict],
    *,
    seed: int,
) -> dict:
    """Compare agent runs query by query instead of comparing aggregate means.

    Positive deltas always mean ``treatment - control``.  Callers must interpret
    cost metrics (pulls, latency, and tokens) in the opposite direction from
    quality metrics.
    """
    control_by_id = {str(row["query_id"]): row for row in control}
    treatment_by_id = {str(row["query_id"]): row for row in treatment}
    shared_ids = sorted(set(control_by_id) & set(treatment_by_id))
    if not shared_ids:
        raise ValueError("no shared query IDs for paired comparison")

    out = {
        "paired_query_count": len(shared_ids),
        "delta_direction": "treatment_minus_control",
    }
    for key in RESULT_METRIC_KEYS:
        out[key] = paired_bootstrap_delta(
            [float(control_by_id[qid].get(key, 0.0) or 0.0) for qid in shared_ids],
            [float(treatment_by_id[qid].get(key, 0.0) or 0.0) for qid in shared_ids],
            seed=seed,
        )

    judged_ids = [
        qid for qid in shared_ids
        if control_by_id[qid].get("judgment") in {"correct", "incorrect"}
        and treatment_by_id[qid].get("judgment") in {"correct", "incorrect"}
    ]
    out["accuracy"] = (
        paired_bootstrap_delta(
            [1.0 if control_by_id[qid]["judgment"] == "correct" else 0.0
             for qid in judged_ids],
            [1.0 if treatment_by_id[qid]["judgment"] == "correct" else 0.0
             for qid in judged_ids],
            seed=seed,
        )
        if judged_ids else None
    )
    out["paired_judged_query_count"] = len(judged_ids)
    return out
