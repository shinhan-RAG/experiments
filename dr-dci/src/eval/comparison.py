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

    deltas = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n": len(deltas),
        "mean_delta": round(float(deltas.mean()), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
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
