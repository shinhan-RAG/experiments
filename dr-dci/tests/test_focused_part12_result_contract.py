import unittest

from src.eval.part12_result_contract import validate_focused_part12_result


def agent_row(query_id="q1"):
    return {
        "query_id": query_id,
        "gold_recall": 0.2,
        "latency_seconds": 1.0,
        "latency_without_taxonomy_boost_telemetry_seconds": 1.0,
        "taxonomy_boost_telemetry_seconds": 0.0,
        "pull_count": 1,
        "pull_queries": ["query"],
        "pull_traces": [{"workspace_document_ids_after": ["d1"]}],
        "first_pull_document_gold_recall": 0.2,
        "workspace_expansion_document_gold_recall": 0.0,
    }


def paired():
    return {"n": 1, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}


def part1_manifest():
    return {
        "schema_version": "dr-dci.part1-taxonomy.v2",
        "focused": True,
        "primary_endpoint": "workspace_document_gold_recall",
        "decision_rule": {
            "minimum_practical_effect_size": 0.01,
            "status": "approved",
        },
        "arms": [{"name": "baseline"}, {"name": "taxonomy_only"}],
        "dataset_provenance": {},
        "experiment_config": {},
        "execution_environment": {},
        "git_commit": "a" * 40,
        "preflight": {"status": "ready"},
        "controls": {
            "analysis_bootstrap_seed": 42,
            "analysis_seed_purpose": "paired_bootstrap_and_sign_flip",
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
