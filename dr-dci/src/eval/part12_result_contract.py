"""Validation contract for focused Part 1 and Part 2 agent results."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any

from src.eval.comparison import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    classify_practical_effect,
    compare_result_rows,
)


REQUIRED_MANIFEST_KEYS = (
    "focused",
    "dataset_provenance",
    "experiment_config",
    "execution_environment",
    "git_commit",
    "experiment_contract_sha256",
    "preflight",
    "arms",
)

REQUIRED_CONTROL_KEYS = (
    "analysis_bootstrap_seed",
    "analysis_bootstrap_iterations",
    "analysis_seed_purpose",
    "minimum_practical_effect_size",
    "minimum_practical_effect_version",
    "embedding_model",
    "agent_model",
    "agent_temperature",
    "agent_max_tokens",
    "agent_generation_seed",
    "judge_model",
    "judge_temperature",
    "judge_max_tokens",
    "judge_generation_seed",
    "pull_top_k",
    "workspace_max_docs",
    "max_turns",
)

REQUIRED_AGENT_ROW_KEYS = (
    "query_id",
    "gold_recall",
    "latency_seconds",
    "latency_without_taxonomy_boost_telemetry_seconds",
    "taxonomy_boost_telemetry_seconds",
    "pull_count",
    "pull_queries",
    "pull_traces",
    "first_pull_document_gold_recall",
    "workspace_expansion_document_gold_recall",
)

NUMERIC_TOLERANCE = 1e-6
REQUIRED_RUNTIME_DEPENDENCIES = ("numpy", "requests", "PyYAML")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_agent_measurements(label: str, row: dict[str, Any],
                                 errors: list[str]) -> None:
    gold_recall = row["gold_recall"]
    if not _is_finite_number(gold_recall) or not 0 <= gold_recall <= 1:
        errors.append(f"{label}.results gold_recall must be within [0, 1]")

    latency = row["latency_seconds"]
    latency_without = row["latency_without_taxonomy_boost_telemetry_seconds"]
    telemetry = row["taxonomy_boost_telemetry_seconds"]
    if not all(_is_finite_number(value) and value >= 0 for value in (
        latency, latency_without, telemetry,
    )):
        errors.append(f"{label}.results latency values must be finite and non-negative")
    elif not math.isclose(
        latency_without + telemetry,
        latency,
        rel_tol=0.0,
        abs_tol=NUMERIC_TOLERANCE,
    ):
        errors.append(
            f"{label}.results latency_without_taxonomy_boost_telemetry_seconds + "
            "taxonomy_boost_telemetry_seconds must equal latency_seconds"
        )

    pull_count = row.get("pull_count")
    first_pull_recall = row["first_pull_document_gold_recall"]
    expansion_recall = row["workspace_expansion_document_gold_recall"]
    if isinstance(pull_count, int) and pull_count > 0:
        if not all(_is_finite_number(value) and 0 <= value <= 1 for value in (
            first_pull_recall, expansion_recall,
        )):
            errors.append(
                f"{label}.results first-pull and workspace expansion recall "
                "must be within [0, 1]"
            )
        elif _is_finite_number(gold_recall) and not math.isclose(
            first_pull_recall + expansion_recall,
            gold_recall,
            rel_tol=0.0,
            abs_tol=NUMERIC_TOLERANCE,
        ):
            errors.append(
                f"{label}.results first-pull recall + workspace expansion recall "
                "must equal gold_recall"
            )


def _validate_manifest_common(manifest: dict[str, Any], errors: list[str]) -> None:
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"manifest missing {key}")
    if manifest.get("focused") is not True:
        errors.append("manifest focused must be true")
    execution_environment = manifest.get("execution_environment")
    dependencies = (
        execution_environment.get("dependencies")
        if isinstance(execution_environment, dict) else None
    )
    if not isinstance(dependencies, dict) or set(dependencies) != set(
        REQUIRED_RUNTIME_DEPENDENCIES
    ) or not all(isinstance(version, str) and version for version in dependencies.values()):
        errors.append(
            "manifest execution_environment must record numpy, requests, and PyYAML versions"
        )
    contract_sha256 = manifest.get("experiment_contract_sha256")
    if (
        not isinstance(contract_sha256, str)
        or len(contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in contract_sha256.lower())
    ):
        errors.append("manifest experiment_contract_sha256 must be a SHA-256 digest")
    arm_names = [arm.get("name") for arm in manifest.get("arms", []) if isinstance(arm, dict)]
    if arm_names != ["baseline", "taxonomy_only"]:
        errors.append("manifest arms must be baseline then taxonomy_only")
    controls = manifest.get("controls")
    if not isinstance(controls, dict):
        errors.append("manifest missing controls")
        return
    missing_controls = [key for key in REQUIRED_CONTROL_KEYS if key not in controls]
    if missing_controls:
        errors.append(f"manifest controls missing {', '.join(missing_controls)}")
    if controls.get("analysis_seed_purpose") != "paired_bootstrap":
        errors.append("manifest controls must distinguish the analysis seed purpose")
    if not isinstance(controls.get("analysis_bootstrap_seed"), int):
        errors.append("manifest controls analysis_bootstrap_seed must be an integer")
    iterations = controls.get("analysis_bootstrap_iterations")
    if not isinstance(iterations, int) or iterations <= 0:
        errors.append("manifest controls analysis_bootstrap_iterations must be a positive integer")
    elif iterations != DEFAULT_BOOTSTRAP_ITERATIONS:
        errors.append(
            "manifest controls analysis_bootstrap_iterations must equal "
            f"{DEFAULT_BOOTSTRAP_ITERATIONS}"
        )
    minimum_effect = controls.get("minimum_practical_effect_size")
    if not _is_finite_number(minimum_effect) or minimum_effect < 0:
        errors.append("manifest controls minimum_practical_effect_size must be non-negative")
    if not isinstance(controls.get("minimum_practical_effect_version"), str) or not controls[
        "minimum_practical_effect_version"
    ]:
        errors.append("manifest controls minimum_practical_effect_version is required")


def _validate_rows(label: str, arm: Any, errors: list[str]) -> set[str]:
    if not isinstance(arm, dict):
        errors.append(f"full_results missing {label}")
        return set()
    rows = arm.get("results")
    if not isinstance(rows, list) or not rows:
        errors.append(f"{label}.results must be a non-empty list")
        return set()
    query_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append(f"{label}.results contains a non-object row")
            continue
        missing = [key for key in REQUIRED_AGENT_ROW_KEYS if key not in row]
        if missing:
            errors.append(f"{label}.results missing {', '.join(missing)}")
            continue
        if not isinstance(row["pull_queries"], list) or not isinstance(row["pull_traces"], list):
            errors.append(f"{label}.results pull traces must be lists")
        _validate_agent_measurements(label, row, errors)
        pull_count = row.get("pull_count")
        if not isinstance(pull_count, int) or pull_count < 0:
            errors.append(f"{label}.results pull_count must be a non-negative integer")
        elif pull_count > 0:
            if not row["pull_traces"]:
                errors.append(f"{label}.results has pulls but no pull traces")
            if row["first_pull_document_gold_recall"] is None:
                errors.append(f"{label}.results has pulls but no first-pull recall")
            if row["workspace_expansion_document_gold_recall"] is None:
                errors.append(f"{label}.results has pulls but no workspace expansion recall")
        query_id = str(row["query_id"])
        if not query_id:
            errors.append(f"{label}.results has an empty query_id")
        else:
            query_ids.append(query_id)
    if len(query_ids) != len(set(query_ids)):
        errors.append(f"{label}.results has duplicate query_id values")
    return set(query_ids)


def _validate_comparison(label: str, analysis: dict[str, Any], expected_n: int,
                         errors: list[str]) -> None:
    comparison = analysis.get(label)
    if not isinstance(comparison, dict):
        errors.append(f"analysis missing {label}")
        return
    if comparison.get("paired_query_count") != expected_n:
        errors.append(f"{label} paired_query_count does not match raw rows")
    delta = comparison.get("gold_recall")
    if not isinstance(delta, dict) or not {"n", "mean_delta", "ci95_low", "ci95_high"} <= set(delta):
        errors.append(f"{label}.gold_recall lacks paired delta and confidence interval")


def recompute_part1_primary_analysis(
    manifest: dict[str, Any], full_results: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Recalculate the focused Part 1 primary endpoint from its raw rows.

    Approval must not trust a handwritten aggregate or decision.  The analysis
    seed, bootstrap iterations, and effect threshold are all taken from the
    result manifest so the saved artifact is its own reproducibility contract.
    """
    controls = manifest["controls"]
    decision_rule = manifest["decision_rule"]
    iterations = controls["analysis_bootstrap_iterations"]
    if iterations != DEFAULT_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            "analysis_bootstrap_iterations must equal "
            f"{DEFAULT_BOOTSTRAP_ITERATIONS}"
        )
    comparison = compare_result_rows(
        full_results["baseline"]["results"],
        full_results["taxonomy_only"]["results"],
        seed=controls["analysis_bootstrap_seed"],
        bootstrap_iterations=iterations,
    )
    decision = classify_practical_effect(
        comparison["gold_recall"],
        minimum_effect_size=decision_rule["minimum_practical_effect_size"],
    )
    return comparison, decision


def _validate_part1_raw_analysis(
    manifest: dict[str, Any], full_results: dict[str, Any], analysis: dict[str, Any],
    errors: list[str],
) -> None:
    """Require stored Part 1 primary statistics and decision to match raw rows."""
    label = "taxonomy_only_minus_baseline"
    stored = analysis.get(label)
    if not isinstance(stored, dict):
        return
    try:
        recalculated, recalculated_decision = recompute_part1_primary_analysis(
            manifest, full_results
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"{label} cannot be recalculated from raw rows: {error}")
        return

    stored_delta = stored.get("gold_recall")
    recalculated_delta = recalculated["gold_recall"]
    expected_delta_keys = (
        "n", "mean_delta", "ci95_low", "ci95_high", "iterations", "seed",
    )
    if (
        not isinstance(stored_delta, dict)
        or any(stored_delta.get(key) != recalculated_delta[key] for key in expected_delta_keys)
        or stored.get("paired_query_count") != recalculated["paired_query_count"]
    ):
        errors.append(
            f"{label}.gold_recall does not match raw rows under manifest bootstrap controls"
        )
    if stored.get("document_gold_recall_decision") != recalculated_decision:
        errors.append(
            f"{label} decision does not match raw rows under manifest decision rule"
        )


def validate_focused_part12_result(
    manifest: dict[str, Any],
    full_results: dict[str, Any],
    analysis: dict[str, Any],
) -> list[str]:
    """Return all focused-result contract failures without mutating a result."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if not isinstance(full_results, dict):
        return ["full_results must be an object"]
    if not isinstance(analysis, dict):
        return ["analysis must be an object"]

    _validate_manifest_common(manifest, errors)
    schema = manifest.get("schema_version")

    if schema == "dr-dci.part1-taxonomy.v2":
        if manifest.get("primary_endpoint") != "workspace_document_gold_recall":
            errors.append("Part 1 primary_endpoint must be workspace_document_gold_recall")
        decision_rule = manifest.get("decision_rule")
        minimum_effect = (
            decision_rule.get("minimum_practical_effect_size")
            if isinstance(decision_rule, dict) else None
        )
        if not isinstance(minimum_effect, (int, float)) or minimum_effect < 0:
            errors.append("Part 1 decision_rule lacks a non-negative minimum_practical_effect_size")
        if not isinstance(decision_rule, dict) or not isinstance(
            decision_rule.get("minimum_practical_effect_version"), str
        ) or not decision_rule["minimum_practical_effect_version"]:
            errors.append("Part 1 decision_rule lacks minimum_practical_effect_version")
        if not isinstance(decision_rule, dict) or decision_rule.get("status") != "approved":
            errors.append("Part 1 decision_rule must record approved threshold status")
        baseline_ids = _validate_rows("baseline", full_results.get("baseline"), errors)
        treatment_ids = _validate_rows("taxonomy_only", full_results.get("taxonomy_only"), errors)
        if baseline_ids and treatment_ids and baseline_ids != treatment_ids:
            errors.append("Part 1 arms must contain identical query IDs")
        comparison_label = "taxonomy_only_minus_baseline"
        _validate_comparison(comparison_label, analysis, len(baseline_ids & treatment_ids), errors)
        comparison = analysis.get(comparison_label, {})
        if comparison.get("document_gold_recall_decision") not in {
            "positive_practical_signal", "negative_practical_signal", "inconclusive",
        }:
            errors.append(f"{comparison_label} lacks document_gold_recall_decision")
        if (
            isinstance(manifest.get("controls"), dict)
            and isinstance(decision_rule, dict)
            and manifest["controls"].get("minimum_practical_effect_size")
            != decision_rule.get("minimum_practical_effect_size")
        ):
            errors.append("Part 1 controls and decision_rule minimum effect size differ")
        if (
            isinstance(manifest.get("controls"), dict)
            and isinstance(decision_rule, dict)
            and manifest["controls"].get("minimum_practical_effect_version")
            != decision_rule.get("minimum_practical_effect_version")
        ):
            errors.append("Part 1 controls and decision_rule minimum effect version differ")
        _validate_part1_raw_analysis(manifest, full_results, analysis, errors)
        return errors

    if schema == "dr-dci.part2-taxonomy-scaling.v1":
        part1_gate = manifest.get("part1_approval_gate")
        if not isinstance(part1_gate, dict) or part1_gate.get("status") != "approved":
            errors.append("Part 2 manifest lacks an approved Part 1 result gate")
        elif (
            not isinstance(part1_gate.get("configured_path"), str)
            or not isinstance(part1_gate.get("sha256"), str)
            or len(part1_gate["sha256"]) != 64
            or part1_gate.get("decision") != "positive_practical_signal"
            or not isinstance(part1_gate.get("compatibility"), dict)
        ):
            errors.append("Part 2 approved Part 1 result gate is incomplete")
        else:
            compatibility = part1_gate["compatibility"]
            required_compatibility = (
                "minimum_practical_effect_size",
                "minimum_practical_effect_version",
                "experiment_contract_sha256",
                "part1_primary_analysis",
                "runtime_dependencies",
            )
            if any(key not in compatibility for key in required_compatibility):
                errors.append("Part 2 approved Part 1 result gate is incomplete")
        sizes = manifest.get("subsets")
        if not isinstance(sizes, list) or len(sizes) < 2 or any(
            not isinstance(size, int) for size in sizes
        ):
            errors.append("Part 2 manifest subsets must contain at least two integer scales")
            return errors
        expected_primary = f"{max(sizes) // 1000}k_minus_{min(sizes) // 1000}k within each arm"
        if manifest.get("primary_scale_comparison") != expected_primary:
            errors.append(f"Part 2 primary_scale_comparison must be {expected_primary}")
        retrieval_component = manifest.get("retrieval_only_component")
        if not isinstance(retrieval_component, dict) or retrieval_component.get(
            "taxonomy_arm_comparison"
        ) is not False:
            errors.append("Part 2 retrieval_only_component must declare taxonomy_arm_comparison false")
        agent_component = manifest.get("agent_component")
        if not isinstance(agent_component, dict) or not agent_component.get("independence"):
            errors.append("Part 2 agent_component must state non-independent query reuse")
        single_pull_comparison = manifest.get("single_pull_comparison")
        if not isinstance(single_pull_comparison, dict):
            errors.append("Part 2 single-pull comparison must be labeled an interface ablation")
        elif single_pull_comparison.get(
            "classification"
        ) != "exploratory_interface_ablation_not_pull_count_only":
            errors.append("Part 2 single-pull comparison must be labeled an interface ablation")
        elif not single_pull_comparison.get("within_dynamic_diagnostic"):
            errors.append("Part 2 single-pull comparison lacks within-dynamic diagnostic")

        arm_ids: dict[str, set[str]] = {}
        for arm in ("baseline", "taxonomy_only"):
            for size in sizes:
                size_key = f"{size // 1000}k"
                label = f"{arm}_{size_key}"
                arm_ids[label] = _validate_rows(label, full_results.get(label), errors)
                if manifest.get("include_single_pull", True):
                    single_label = f"single-pull_{arm}_{size_key}"
                    single_ids = _validate_rows(
                        single_label, full_results.get(single_label), errors
                    )
                    if arm_ids[label] and single_ids and arm_ids[label] != single_ids:
                        errors.append(f"{single_label} must contain the dynamic run query IDs")

        for size in sizes:
            size_key = f"{size // 1000}k"
            baseline_ids = arm_ids.get(f"baseline_{size_key}", set())
            treatment_ids = arm_ids.get(f"taxonomy_only_{size_key}", set())
            if baseline_ids and treatment_ids and baseline_ids != treatment_ids:
                errors.append(f"Part 2 arms at {size_key} must contain identical query IDs")
            _validate_comparison(
                f"taxonomy_minus_baseline_{size_key}", analysis,
                len(baseline_ids & treatment_ids), errors,
            )

        for arm in ("baseline", "taxonomy_only"):
            for control_size, treatment_size in combinations(sizes, 2):
                control_key = f"{control_size // 1000}k"
                treatment_key = f"{treatment_size // 1000}k"
                control_ids = arm_ids.get(f"{arm}_{control_key}", set())
                treatment_ids = arm_ids.get(f"{arm}_{treatment_key}", set())
                _validate_comparison(
                    f"{arm}_{treatment_key}_minus_{control_key}", analysis,
                    len(control_ids & treatment_ids), errors,
                )
            if manifest.get("include_single_pull", True):
                for size in sizes:
                    size_key = f"{size // 1000}k"
                    _validate_comparison(
                        f"dynamic_minus_single_{arm}_{size_key}", analysis,
                        len(arm_ids.get(f"{arm}_{size_key}", set())), errors,
                    )
        return errors

    errors.append("manifest schema_version is not a focused Part 1 or Part 2 schema")
    return errors
