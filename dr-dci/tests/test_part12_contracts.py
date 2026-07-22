import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment
from src.eval.comparison import compare_result_rows
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

    def test_paired_agent_comparison_keeps_quality_and_cost_separate(self):
        control = [{"query_id": "q1", "gold_recall": 0.0, "pull_count": 1,
                    "latency_seconds": 1.0, "judgment": "incorrect"}]
        treatment = [{"query_id": "q1", "gold_recall": 1.0, "pull_count": 2,
                      "latency_seconds": 2.0, "judgment": "correct"}]
        report = compare_result_rows(control, treatment, seed=42)
        self.assertEqual(report["gold_recall"]["mean_delta"], 1.0)
        self.assertEqual(report["pull_count"]["mean_delta"], 1.0)
        self.assertEqual(report["accuracy"]["mean_delta"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
