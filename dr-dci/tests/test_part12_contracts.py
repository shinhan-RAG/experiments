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

    def test_repository_part1_config_has_unique_treatments(self):
        import yaml

        config_path = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
        with config_path.open(encoding="utf-8") as stream:
            steps = yaml.safe_load(stream)["parts"]["part1_stacking"]["steps"]
        self.assertEqual(duplicate_arms(steps), [])

    def test_requested_augmentation_missing_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_experiment, "DATA_DIR", Path(tmp)
        ):
            with self.assertRaises(FileNotFoundError):
                run_experiment.load_augmentations(
                    "dataset", 20_000, {"taxonomy": True}
                )

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

    def test_metadata_variant_is_a_distinct_treatment(self):
        """variant 만 다른 arm 은 서로 다른 처치다.

        FEATURE_KEYS 에 metadata_variant 가 빠지면 두 arm 이 같은 처치로
        오판돼 duplicate_arms 가 실행을 막는다.
        """
        duplicates = duplicate_arms([
            {"name": "metadata_only", "metadata": True},
            {"name": "parser_meta_only", "metadata": True,
             "metadata_variant": "parser"},
        ])
        self.assertEqual(duplicates, [])

    def test_metadata_variant_audits_its_own_artifact_file(self):
        """variant 별로 다른 파일을 확인해야 한다.

        LLM 생성분(fixture_1k.json)이 있어도 variant 파일
        (fixture-parser_1k.json)이 없으면 블록해야 한다 — 그러지 않으면
        parser arm 이 라벨만 남고 처치 없이 돌아간다.
        """
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_jsonl(data / "raw" / "fixture" / "qrels.jsonl", [
                {"query-id": "q1", "corpus-id": "gold", "score": 1},
            ])
            ids = ["gold", *[f"n{i}" for i in range(999)]]
            write_json(data / "subsets" / "fixture" / "1k.json", {
                "subset_size": 1000, "doc_ids": ids,
            })
            write_json(data / "metadata" / "fixture_1k.json",
                       {doc_id: {"doc_type": "x"} for doc_id in ids})
            config = {"parts": {"part1_stacking": {
                "dataset": "fixture", "subset": 1000, "steps": [
                    {"name": "metadata_only", "metadata": True},
                    {"name": "parser_meta_only", "metadata": True,
                     "metadata_variant": "parser"},
                ]}}}

            report = audit_part12(config, data)
            self.assertEqual(report["status"], "blocked")
            self.assertTrue(any("fixture-parser_1k.json" in item
                                for item in report["blockers"]),
                            report["blockers"])

            # variant 파일을 채우면 통과한다
            write_json(data / "metadata" / "fixture-parser_1k.json",
                       {doc_id: {"element_type": "table"} for doc_id in ids})
            report = audit_part12(config, data)
            self.assertEqual(report["status"], "ready", report["blockers"])

    def test_load_augmentations_reads_metadata_variant_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_experiment, "DATA_DIR", Path(tmp)
        ):
            write_json(Path(tmp) / "metadata" / "ds-parser_1k.json",
                       {"c1": {"element_type": "table"}})
            # variant 요청 시 variant 파일을 읽는다
            _, _, _, metadata = run_experiment.load_augmentations(
                "ds", 1000, {"metadata": True, "metadata_variant": "parser"})
            self.assertEqual(metadata, {"c1": {"element_type": "table"}})
            # variant 없는 요청은 기존 경로를 쓰고, 없으면 하드 실패한다
            with self.assertRaises(FileNotFoundError):
                run_experiment.load_augmentations("ds", 1000, {"metadata": True})

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
