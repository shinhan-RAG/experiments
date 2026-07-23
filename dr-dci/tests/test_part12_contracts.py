import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment
from src.eval.comparison import classify_practical_effect, compare_result_rows
from src.eval.part12_contracts import (
    audit_part12,
    duplicate_arms,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Part12ContractTests(unittest.TestCase):
    def test_duplicate_arms_ignore_labels(self):
        duplicates = duplicate_arms([
            {"name": "taxonomy_only", "taxonomy": True},
            {"name": "stack_tax", "description": "same treatment", "taxonomy": True},
            {"name": "baseline", "taxonomy": False},
        ])
        self.assertEqual(duplicates[0]["arms"], ["taxonomy_only", "stack_tax"])

    def test_audit_blocks_duplicate_arms_selected_for_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_jsonl(data / "raw" / "fixture" / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 1},
            ])
            ids = ["gold", *[f"n{i}" for i in range(999)]]
            write_json(data / "subsets" / "fixture" / "1k.json", {
                "subset_size": 1000, "doc_ids": ids,
            })
            write_json(data / "taxonomy" / "fixture_1k.json", {
                doc_id: {"L1": "A"} for doc_id in ids
            })
            config = {"parts": {
                "part1_stacking": {"dataset": "fixture", "steps": [
                    {"name": "baseline"},
                    {"name": "taxonomy_only", "taxonomy": True},
                    {"name": "stack_tax", "taxonomy": True},
                ]},
                "part2_scaling": {"dataset": "fixture", "subsets": [1000]},
            }}

            report = audit_part12(
                config, data, step_names={"taxonomy_only", "stack_tax"}, sizes=[1000]
            )

            self.assertEqual(report["status"], "blocked")
            self.assertTrue(any("duplicate treatment arms" in item for item in report["blockers"]))

    def test_model_run_preflight_requires_immutable_source_revisions(self):
        config = {
            "data_provenance": {
                "fixture": {
                    "corpus": {"dataset": "source", "split": "corpus", "revision": None},
                    "queries": {"dataset": "source", "split": "queries", "revision": "abc"},
                    "qrels": {"dataset": "source-qrels", "split": "test", "revision": "def"},
                },
            },
        }

        blockers = run_experiment.dataset_provenance_blockers(config, "fixture")

        self.assertEqual(len(blockers), 1)
        self.assertIn("fixture.corpus", blockers[0])

    def test_focused_execution_requires_approved_practical_effect_threshold(self):
        provisional = {"parts": {"part1_stacking": {
            "minimum_practical_effect_status": "provisional_pending_approval",
        }}}

        self.assertTrue(run_experiment.focused_decision_rule_blockers(provisional))
        approved = {"parts": {"part1_stacking": {
            "minimum_practical_effect_status": "approved",
        }}}
        self.assertEqual(run_experiment.focused_decision_rule_blockers(approved), [])

    def test_audit_blocks_missing_artifacts_but_accepts_nested_gold_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = data / "raw" / "fixture"
            write_jsonl(raw / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 1},
            ])
            first_ids = ["gold", *[f"n{i}" for i in range(999)]]
            second_ids = [*first_ids, *[f"x{i}" for i in range(1000)]]
            for size, ids in ((1000, first_ids), (2000, second_ids)):
                write_json(data / "subsets" / "fixture" / f"{size // 1000}k.json", {
                    "subset_size": size,
                    "doc_ids": ids,
                })

            config = {"parts": {
                "part1_stacking": {
                    "dataset": "fixture",
                    "steps": [
                        {"name": "baseline"},
                        {"name": "taxonomy_only", "taxonomy": True},
                    ],
                },
                "part2_scaling": {"dataset": "fixture", "subsets": [1000, 2000]},
            }}
            report = audit_part12(config, data)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["subsets"]["blockers"])
            self.assertTrue(any("missing taxonomy artifact" in item
                                for item in report["blockers"]))

    def test_audit_rejects_changed_shared_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = data / "raw" / "fixture"
            write_jsonl(raw / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 1},
            ])
            first_ids = ["gold", *[f"n{i}" for i in range(999)]]
            second_ids = [*first_ids, *[f"x{i}" for i in range(1000)]]
            for size, ids in ((1000, first_ids), (2000, second_ids)):
                write_json(data / "subsets" / "fixture" / f"{size // 1000}k.json", {
                    "subset_size": size, "doc_ids": ids,
                })
            first_taxonomy = {doc_id: {"L1": "A"} for doc_id in first_ids}
            second_taxonomy = {doc_id: {"L1": "A"} for doc_id in second_ids}
            second_taxonomy["gold"] = {"L1": "B"}
            write_json(data / "taxonomy" / "fixture_1k.json", first_taxonomy)
            write_json(data / "taxonomy" / "fixture_2k.json", second_taxonomy)
            config = {"parts": {
                "part1_stacking": {"dataset": "fixture", "steps": [
                    {"name": "taxonomy_only", "taxonomy": True},
                ]},
                "part2_scaling": {"dataset": "fixture", "subsets": [1000, 2000]},
            }}
            report = audit_part12(config, data)
            self.assertTrue(any("scale is not the only variable" in item
                                for item in report["blockers"]))

    def test_taxonomy_audit_reports_gold_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = data / "raw" / "fixture"
            write_jsonl(raw / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 2},
                {"query-id": "q1", "corpus-id": "nonpositive", "score": 0},
            ])
            ids = ["gold", "nonpositive", *[f"n{i}" for i in range(998)]]
            write_json(data / "subsets" / "fixture" / "1k.json", {
                "subset_size": 1000, "doc_ids": ids,
            })
            write_json(data / "taxonomy" / "fixture_1k.json", {
                doc_id: {"L1": "A", "L2": "B"} for doc_id in ids
            })
            config = {"parts": {
                "part1_stacking": {"dataset": "fixture", "steps": [
                    {"name": "taxonomy_only", "taxonomy": True},
                ]},
                "part2_scaling": {"dataset": "fixture", "subsets": [1000]},
            }}

            report = audit_part12(config, data)

            artifact = report["augmentations"]["artifacts"][0]
            self.assertEqual(artifact["positive_gold_document_count"], 1)
            self.assertEqual(artifact["positive_gold_covered_document_count"], 1)
            self.assertEqual(artifact["positive_gold_coverage_rate"], 1.0)

    def test_paired_agent_comparison_keeps_quality_and_cost_separate(self):
        control = [{"query_id": "q1", "gold_recall": 0.0, "pull_count": 1,
                    "latency_seconds": 1.0, "judgment": "incorrect"}]
        treatment = [{"query_id": "q1", "gold_recall": 1.0, "pull_count": 2,
                      "latency_seconds": 2.0, "judgment": "correct"}]
        report = compare_result_rows(control, treatment, seed=42)
        self.assertEqual(report["gold_recall"]["mean_delta"], 1.0)
        self.assertEqual(report["pull_count"]["mean_delta"], 1.0)
        self.assertEqual(report["accuracy"]["mean_delta"], 1.0)

    def test_practical_effect_rule_requires_more_than_a_ci_above_zero(self):
        self.assertEqual(
            classify_practical_effect(
                {"ci95_low": 0.001, "ci95_high": 0.003},
                minimum_effect_size=0.01,
            ),
            "inconclusive",
        )
        self.assertEqual(
            classify_practical_effect(
                {"ci95_low": 0.011, "ci95_high": 0.020},
                minimum_effect_size=0.01,
            ),
            "positive_practical_signal",
        )

    def test_focused_part1_preflight_runs_before_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_jsonl(data / "raw" / "fixture" / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 1},
            ])
            ids = ["gold", *[f"n{i}" for i in range(999)]]
            write_json(data / "subsets" / "fixture" / "1k.json", {
                "subset_size": 1000, "doc_ids": ids,
            })
            config = {
                "seed": 42,
                "parts": {
                    "part1_stacking": {
                        "dataset": "fixture", "subset": 1000,
                        "steps": [
                            {"name": "baseline", "description": "control"},
                            {"name": "taxonomy_only", "description": "treatment",
                             "taxonomy": True},
                        ],
                    },
                    "part2_scaling": {"dataset": "fixture", "subsets": [1000]},
                },
            }
            with patch.object(run_experiment, "DATA_DIR", data), \
                    patch.object(run_experiment.PullRetriever, "index") as index:
                with self.assertRaisesRegex(RuntimeError, "missing taxonomy artifact"):
                    run_experiment.run_part1(config, focused=True)
                index.assert_not_called()

    def test_requested_augmentation_missing_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_experiment, "DATA_DIR", Path(directory)):
                with self.assertRaisesRegex(FileNotFoundError, "requested augmentation 'taxonomy' is missing"):
                    run_experiment.load_augmentations(
                        "fixture", 1000, {"taxonomy": True}
                    )

    def test_focused_taxonomy_arms_hold_the_schema_prompt_constant(self):
        steps = [
            {"name": "baseline", "taxonomy": False},
            {"name": "taxonomy_only", "taxonomy": True},
            {"name": "stack_all", "taxonomy": True, "prefix": True},
        ]

        focused = run_experiment.focused_taxonomy_steps(steps)

        self.assertEqual([step["name"] for step in focused], [
                         "baseline", "taxonomy_only"])
        self.assertTrue(all(step["taxonomy_prompt_schema"] for step in focused))
        self.assertTrue(all(not step["workspace_taxonomy"] for step in focused))
        self.assertFalse(focused[0]["taxonomy"])
        self.assertTrue(focused[1]["taxonomy"])

    def test_focused_part2_records_dynamic_minus_single_per_arm_and_scale(self):
        config = {
            "seed": 42,
            "parts": {
                "part2_scaling": {
                    "dataset": "fixture",
                    "subsets": [1000, 2000, 3000],
                    "include_single_pull": True,
                },
            },
        }
        saved = {}

        def fake_run(*args, **kwargs):
            single_pull = kwargs.get("single_pull", False)
            return [{
                "query_id": "q1",
                "gold_recall": 0.2 if single_pull else 0.4,
                "pull_count": 1 if single_pull else 2,
                "judgment": "correct",
            }]

        with patch.object(run_experiment, "_part12_preflight", return_value={"status": "ready"}), \
                patch.object(run_experiment, "load_corpus", return_value=[]), \
                patch.object(run_experiment, "load_queries", return_value=([], [])), \
                patch.object(run_experiment, "run_dr_dci", side_effect=fake_run) as run_agent, \
                patch.object(run_experiment, "build_part12_manifest", return_value={}), \
                patch.object(run_experiment, "validate_focused_part12_result", return_value=[]), \
                patch.object(run_experiment, "save_results", side_effect=lambda *args, **kwargs: saved.update(kwargs)):
            run_experiment.run_part2(config, focused=True)

        self.assertEqual(run_agent.call_count, 12)  # 2 arms × 3 scales × (dynamic + single)
        self.assertEqual(saved["analysis"]["dynamic_minus_single_baseline_1k"]["gold_recall"]["mean_delta"], 0.2)
        self.assertEqual(saved["analysis"]["dynamic_minus_single_taxonomy_only_2k"]["pull_count"]["mean_delta"], 1.0)
        self.assertIn("baseline_3k_minus_2k", saved["analysis"])
        self.assertTrue(saved["manifest"]["include_single_pull"])


if __name__ == "__main__":
    unittest.main()
