"""Validation contract for stored Part 2 retrieval-only scale-probe results."""

from __future__ import annotations

from typing import Any


ROW_METRICS = (
    "recall_at_5",
    "recall_at_20",
    "hit_at_5",
    "hit_at_10",
    "precision_at_20",
    "ndcg_at_10",
    "probe_latency_seconds",
)

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "single_variable",
    "design",
    "backend",
    "dataset",
    "subset_sizes",
    "primary_metrics",
    "secondary_metrics",
    "seed",
    "git_commit",
    "dataset_provenance",
    "experiment_config",
    "execution_environment",
)


def _comparison_key(control_size: int, treatment_size: int) -> str:
    return (
        f"dense_{treatment_size // 1000}k_minus_"
        f"{control_size // 1000}k"
    )


def validate_scale_probe_result(payload: dict[str, Any]) -> list[str]:
    """Return all reproducibility-contract failures without mutating a result."""
    errors: list[str] = []
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"manifest missing {key}")

    if manifest.get("schema_version") != "dr-dci.part2-scale-probe.v1":
        errors.append("manifest schema_version is not dr-dci.part2-scale-probe.v1")
    if manifest.get("single_variable") != "distractor_count":
        errors.append("manifest single_variable must be distractor_count")
    if manifest.get("backend") != "dense":
        errors.append("manifest backend must be dense")
    if manifest.get("primary_metrics") != ["ndcg_at_10", "precision_at_20"]:
        errors.append("manifest primary_metrics must be [ndcg_at_10, precision_at_20]")
    provenance = manifest.get("dataset_provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("files"), dict):
        errors.append("manifest dataset_provenance.files must be an object")
    elif not {"corpus", "queries", "qrels"} <= set(provenance["files"]):
        errors.append("manifest dataset_provenance.files misses a raw input hash")
    config = manifest.get("experiment_config")
    if not isinstance(config, dict) or not config.get("sha256"):
        errors.append("manifest experiment_config.sha256 is required")

    sizes = manifest.get("subset_sizes")
    if not isinstance(sizes, list) or len(sizes) < 2 or any(not isinstance(size, int) for size in sizes):
        errors.append("manifest subset_sizes must contain at least two integer scales")
        return errors

    full_results = payload.get("full_results")
    if not isinstance(full_results, dict):
        return [*errors, "full_results must be an object"]
    query_ids_by_size: dict[int, set[str]] = {}
    for size in sizes:
        key = f"dense_{size // 1000}k"
        arm = full_results.get(key)
        if not isinstance(arm, dict):
            errors.append(f"full_results missing {key}")
            continue
        rows = arm.get("probe_rows")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{key}.probe_rows must be a non-empty list")
            continue
        query_ids = [str(row.get("query_id", "")) for row in rows if isinstance(row, dict)]
        if len(query_ids) != len(rows) or not all(query_ids):
            errors.append(f"{key}.probe_rows has an invalid query_id")
        if len(query_ids) != len(set(query_ids)):
            errors.append(f"{key}.probe_rows contains duplicate query_id values")
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f"{key}.probe_rows contains a non-object row")
                continue
            missing_metrics = [metric for metric in ROW_METRICS if metric not in row]
            if missing_metrics:
                errors.append(f"{key}.probe_rows misses {', '.join(missing_metrics)}")
                break
        query_ids_by_size[size] = set(query_ids)

    if len(query_ids_by_size) == len(sizes):
        expected_ids = query_ids_by_size[sizes[0]]
        for size in sizes[1:]:
            if query_ids_by_size[size] != expected_ids:
                errors.append("probe rows must contain the same positive-gold query IDs at every scale")
                break

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return [*errors, "analysis must be an object"]
    for i, control_size in enumerate(sizes):
        for treatment_size in sizes[i + 1:]:
            key = _comparison_key(control_size, treatment_size)
            comparison = analysis.get(key)
            if not isinstance(comparison, dict):
                errors.append(f"analysis missing {key}")
                continue
            if comparison.get("paired_query_count") != len(query_ids_by_size.get(control_size, set())):
                errors.append(f"{key} paired_query_count does not match raw rows")
            for metric in ROW_METRICS:
                delta = comparison.get(metric)
                if not isinstance(delta, dict) or not {"n", "mean_delta", "ci95_low", "ci95_high"} <= set(delta):
                    errors.append(f"{key}.{metric} lacks paired delta and confidence interval")
    return errors
