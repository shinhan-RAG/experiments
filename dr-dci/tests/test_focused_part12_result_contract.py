import unittest

from src.eval.part12_result_contract import validate_focused_part12_result


def agent_row(query_id="q1", recall=0.2):
    return {
        "query_id": query_id,
        "gold_recall": recall,
        "latency_seconds": 1.0,
        "latency_without_taxonomy_boost_telemetry_seconds": 1.0,
        "taxonomy_boost_telemetry_seconds": 0.0,
        "pull_count": 1,
        "pull_queries": ["query"],
        "pull_traces": [{"workspace_document_ids_after": ["d1"]}],
        "first_pull_document_gold_recall": recall,
        "workspace_expansion_document_gold_recall": 0.0,
    }


def paired():
    return {
        "n": 1,
        "mean_delta": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "iterations": 10_000,
        "seed": 42,
    }


def part1_manifest():
    return {
        "schema_version": "dr-dci.part1-taxonomy.v2",
        "focused": True,
        "primary_endpoint": "workspace_document_gold_recall",
        "decision_rule": {
            "minimum_practical_effect_size": 0.01,
            "minimum_practical_effect_version": "v1",
            "status": "approved",
        },
        "arms": [{"name": "baseline"}, {"name": "taxonomy_only"}],
        "dataset_provenance": {},
        "experiment_config": {},
        "execution_environment": {
            "dependencies": {
                "numpy": "1.0",
                "requests": "1.0",
                "PyYAML": "1.0",
            },
        },
        "git_commit": "a" * 40,
        "experiment_contract_sha256": "c" * 64,
        "preflight": {"status": "ready"},
        "controls": {
            "analysis_bootstrap_seed": 42,
            "analysis_seed_purpose": "paired_bootstrap",
            "analysis_bootstrap_iterations": 10_000,
            "minimum_practical_effect_size": 0.01,
            "minimum_practical_effect_version": "v1",
            "embedding_model": "embed",
            "agent_model": "agent",
            "agent_temperature": 0,
            "agent_max_tokens": 2048,
            "agent_generation_seed": None,
            "judge_model": "judge",
            "judge_temperature": 0,
            "judge_max_tokens": 512,
            "judge_generation_seed": None,
            "pull_top_k": 20,
            "workspace_max_docs": 100,
            "max_turns": 10,
        },
    }


class FocusedPart12ResultContractTests(unittest.TestCase):
    def test_part1_accepts_complete_rows_and_prespecified_decision(self):
        errors = validate_focused_part12_result(
            part1_manifest(),
            {
                "baseline": {"results": [agent_row()]},
                "taxonomy_only": {"results": [agent_row()]},
            },
            {
                "taxonomy_only_minus_baseline": {
                    "paired_query_count": 1,
                    "gold_recall": paired(),
                    "document_gold_recall_decision": "inconclusive",
                },
            },
        )

        self.assertEqual(errors, [])

    def test_part1_rejects_missing_latency_split_or_decision(self):
        row = agent_row()
        del row["taxonomy_boost_telemetry_seconds"]
        errors = validate_focused_part12_result(
            part1_manifest(),
            {
                "baseline": {"results": [row]},
                "taxonomy_only": {"results": [agent_row()]},
            },
            {
                "taxonomy_only_minus_baseline": {
                    "paired_query_count": 1,
                    "gold_recall": paired(),
                },
            },
        )

        self.assertTrue(any("taxonomy_boost_telemetry_seconds" in error for error in errors))
        self.assertTrue(any("document_gold_recall_decision" in error for error in errors))

    def test_part1_rejects_invalid_latency_and_workspace_recall_invariants(self):
        row = agent_row()
        row["latency_without_taxonomy_boost_telemetry_seconds"] = 0.7
        row["taxonomy_boost_telemetry_seconds"] = 0.2
        row["first_pull_document_gold_recall"] = 0.1
        row["workspace_expansion_document_gold_recall"] = 0.2
        errors = validate_focused_part12_result(
            part1_manifest(),
            {
                "baseline": {"results": [row]},
                "taxonomy_only": {"results": [agent_row()]},
            },
            {
                "taxonomy_only_minus_baseline": {
                    "paired_query_count": 1,
                    "gold_recall": paired(),
                    "document_gold_recall_decision": "inconclusive",
                },
            },
        )

        self.assertTrue(any("latency_without" in error for error in errors))
        self.assertTrue(any("workspace expansion" in error for error in errors))

    def test_part1_rejects_positive_analysis_inconsistent_with_raw_rows(self):
        errors = validate_focused_part12_result(
            part1_manifest(),
            {
                "baseline": {"results": [agent_row(recall=0.9)]},
                "taxonomy_only": {"results": [agent_row(recall=0.1)]},
            },
            {
                "taxonomy_only_minus_baseline": {
                    "paired_query_count": 1,
                    "gold_recall": {
                        "n": 1,
                        "mean_delta": 0.02,
                        "ci95_low": 0.02,
                        "ci95_high": 0.03,
                        "iterations": 10_000,
                        "seed": 42,
                    },
                    "document_gold_recall_decision": "positive_practical_signal",
                },
            },
        )

        self.assertTrue(any("does not match raw rows" in error for error in errors))
        self.assertTrue(any("decision does not match raw rows" in error for error in errors))

    def test_part1_rejects_noncanonical_bootstrap_iterations(self):
        manifest = part1_manifest()
        manifest["controls"]["analysis_bootstrap_iterations"] = 1_000
        errors = validate_focused_part12_result(
            manifest,
            {
                "baseline": {"results": [agent_row()]},
                "taxonomy_only": {"results": [agent_row()]},
            },
            {
                "taxonomy_only_minus_baseline": {
                    "paired_query_count": 1,
                    "gold_recall": {
                        **paired(),
                        "iterations": 1_000,
                    },
                    "document_gold_recall_decision": "inconclusive",
                },
            },
        )

        self.assertTrue(any("must equal 10000" in error for error in errors))

    def test_part2_requires_primary_scale_and_dynamic_single_comparisons(self):
        manifest = {
            **part1_manifest(),
            "schema_version": "dr-dci.part2-taxonomy-scaling.v1",
            "subsets": [1000, 2000],
            "include_single_pull": True,
            "primary_scale_comparison": "2k_minus_1k within each arm",
            "retrieval_only_component": {
                "scope": "common dense original-query scale probe",
                "taxonomy_arm_comparison": False,
            },
            "agent_component": {
                "scope": "baseline and taxonomy-only dynamic/single-pull comparisons",
                "independence": "not independent",
            },
            "single_pull_comparison": {
                "classification": "exploratory_interface_ablation_not_pull_count_only",
                "within_dynamic_diagnostic": "first versus final workspace",
            },
            "part1_approval_gate": {
                "status": "approved",
                "configured_path": "/results/approved-part1.json",
                "sha256": "b" * 64,
                "decision": "positive_practical_signal",
                "compatibility": {
                    "model_and_retrieval_controls": "matched",
                    "minimum_practical_effect_size": 0.01,
                    "minimum_practical_effect_version": "v1",
                    "experiment_contract_sha256": "c" * 64,
                    "part1_primary_analysis": "recomputed_matched",
                    "runtime_dependencies": "matched",
                },
            },
        }
        full_results = {
            f"{prefix}_{size}k": {"results": [agent_row()]}
            for prefix in ("baseline", "taxonomy_only", "single-pull_baseline", "single-pull_taxonomy_only")
            for size in (1, 2)
        }
        analysis = {
            "taxonomy_minus_baseline_1k": {"paired_query_count": 1, "gold_recall": paired()},
            "taxonomy_minus_baseline_2k": {"paired_query_count": 1, "gold_recall": paired()},
            "baseline_2k_minus_1k": {"paired_query_count": 1, "gold_recall": paired()},
            "taxonomy_only_2k_minus_1k": {"paired_query_count": 1, "gold_recall": paired()},
            "dynamic_minus_single_baseline_1k": {"paired_query_count": 1, "gold_recall": paired()},
            "dynamic_minus_single_baseline_2k": {"paired_query_count": 1, "gold_recall": paired()},
            "dynamic_minus_single_taxonomy_only_1k": {"paired_query_count": 1, "gold_recall": paired()},
            "dynamic_minus_single_taxonomy_only_2k": {"paired_query_count": 1, "gold_recall": paired()},
        }

        self.assertEqual(validate_focused_part12_result(manifest, full_results, analysis), [])
        del analysis["baseline_2k_minus_1k"]
        errors = validate_focused_part12_result(manifest, full_results, analysis)
        self.assertTrue(any("baseline_2k_minus_1k" in error for error in errors))
        analysis["baseline_2k_minus_1k"] = {"paired_query_count": 1, "gold_recall": paired()}
        del manifest["part1_approval_gate"]
        errors = validate_focused_part12_result(manifest, full_results, analysis)
        self.assertTrue(any("approved Part 1 result gate" in error for error in errors))
