import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment
from src.eval.scale_probe_contract import validate_scale_probe_result


def valid_payload():
    rows = [
        {
            "query_id": "q1",
            "recall_at_5": 0.1,
            "recall_at_20": 0.2,
            "hit_at_5": 1.0,
            "hit_at_10": 1.0,
            "precision_at_20": 0.1,
            "ndcg_at_10": 0.2,
            "probe_latency_seconds": 0.01,
            "ranked_top20": ["d1"],
        }
    ]
    paired = {"n": 1, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    comparison = {
        "paired_query_count": 1,
        **{key: dict(paired) for key in (
            "recall_at_5", "recall_at_20", "hit_at_5", "hit_at_10",
            "precision_at_20", "ndcg_at_10", "probe_latency_seconds",
        )},
    }
    return {
        "manifest": {
            "schema_version": "dr-dci.part2-scale-probe.v1",
            "single_variable": "distractor_count",
            "design": "controlled distractor scaling (nested subsets, gold preserved)",
            "backend": "dense",
            "dataset": "trec-covid",
            "subset_sizes": [20000, 50000],
            "primary_metrics": ["ndcg_at_10", "precision_at_20"],
            "secondary_metrics": ["recall_at_5", "recall_at_20", "hit_at_5", "hit_at_10", "probe_latency_seconds"],
            "seed": 42,
            "git_commit": "a" * 40,
            "dataset_provenance": {"files": {"corpus": "a", "queries": "b", "qrels": "c"}},
            "experiment_config": {"sha256": "d"},
            "execution_environment": {"python": "3.13"},
        },
        "full_results": {
            "dense_20k": {"probe_rows": list(rows)},
            "dense_50k": {"probe_rows": list(rows)},
        },
        "analysis": {"dense_50k_minus_20k": comparison},
    }


class ScaleProbeResultContractTests(unittest.TestCase):
    def test_accepts_complete_reproducible_result(self):
        self.assertEqual(validate_scale_probe_result(valid_payload()), [])

    def test_rejects_missing_raw_rows_or_provenance(self):
        payload = valid_payload()
        del payload["full_results"]["dense_50k"]
        del payload["manifest"]["dataset_provenance"]

        errors = validate_scale_probe_result(payload)

        self.assertTrue(any("dataset_provenance" in error for error in errors))
        self.assertTrue(any("dense_50k" in error for error in errors))

    def test_manifest_fingerprints_raw_inputs_and_effective_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "fixture"
            raw.mkdir(parents=True)
            (raw / "corpus.jsonl").write_text('{"_id": "d1"}\n')
            (raw / "queries.jsonl").write_text('{"_id": "q1"}\n')
            (raw / "qrels.jsonl").write_text(
                '{"query-id": "q1", "corpus-id": "d1", "score": 1}\n'
            )
            subsets = root / "subsets" / "fixture"
            subsets.mkdir(parents=True)
            (subsets / "1k.json").write_text(json.dumps({"doc_ids": ["d1"]}))
            config_path = root / "experiment.yaml"
            config_path.write_text("seed: 42\n")
            config = {
                "seed": 42,
                "models": {"embedding": {"name": "embed", "url": "http://embed"},
                           "agent_llm": {"name": "agent", "url": "http://agent",
                                         "temperature": 0.2, "max_tokens": 2048, "seed": 17},
                           "judge_llm": {"name": "judge", "url": "http://judge",
                                         "temperature": 0.0, "max_tokens": 512, "seed": 19}},
                "agent": {"pull_top_k": 20, "workspace_max_docs": 100, "max_turns": 10},
                "evaluation": {"metrics": ["gold_recall_at_workspace"]},
            }
            with patch.object(run_experiment, "DATA_DIR", root), \
                    patch.object(run_experiment, "CONFIG_DIR", root):
                manifest = run_experiment.build_part12_manifest(
                    config, "fixture", [1000], config_path=config_path
                )

            self.assertEqual(manifest["dataset_provenance"]["counts"]["corpus_documents"], 1)
            self.assertEqual(manifest["dataset_provenance"]["counts"]["positive_gold_documents"], 1)
            self.assertEqual(manifest["dataset_provenance"]["subsets"][0]["sha256"].__len__(), 64)
            self.assertEqual(manifest["experiment_config"]["sha256"].__len__(), 64)
            self.assertEqual(manifest["controls"]["query_instruction"], None)
            self.assertEqual(manifest["controls"]["analysis_bootstrap_seed"], 42)
            self.assertEqual(
                manifest["controls"]["analysis_seed_purpose"],
                "paired_bootstrap",
            )
            self.assertEqual(manifest["controls"]["agent_max_tokens"], 2048)
            self.assertEqual(manifest["controls"]["judge_max_tokens"], 512)
            self.assertEqual(manifest["controls"]["agent_generation_seed"], 17)


if __name__ == "__main__":
    unittest.main()
