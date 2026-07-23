import importlib.util
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.miracl_ko.lexical_smoke import (
    compare_smoke_scales,
    evaluate_passage_rankings,
    parse_anserini_trec_run,
    sha256_json,
    SmokeResultValidationInputs,
    validate_lexical_smoke_config,
    validate_source_archive_manifest,
    validate_standalone_smoke_result,
)


RUNNER_SPEC = importlib.util.spec_from_file_location(
    "miracl_ko_lexical_smoke_runner_for_test",
    Path(__file__).resolve().parents[1] / "scripts" / "run_miracl_ko_lexical_smoke.py",
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def smoke_config():
    return {
        "schema_version": "dr-dci.miracl-ko-lexical-smoke.v1",
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": "passage",
        "scope": "standalone_lexical_plumbing_smoke_only",
        "backend": {
            "name": "anserini_lucene_cjk",
            "package": "anserini",
            "version": "2.1.1",
            "distribution_package": "pyserini",
            "distribution_version": "2.1.0",
            "analyzer_language": "ko",
            "analyzer_class": "org.apache.lucene.analysis.cjk.CJKAnalyzer",
            "distribution_source_archive_sha256": "384fb783c52ac1605caabe8a75f520323dfed2b5595072911c87f6cfca8bf15f",
            "jar_relative_path": "pyserini/resources/jars/anserini-2.1.1-fatjar.jar",
            "jar_path": "/opt/anserini/anserini-2.1.1-fatjar.jar",
            "jar_sha256": "3c83883246d0fb2326c8a9291572b969467cf478d1fc65f517cbf37fd9b0d914",
        },
        "runtime": {
            "container_base_image": "python:3.12-slim-trixie",
            "container_base_image_sha256": "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de",
            "java_runtime_version": "21.0.11+10-1~deb13u2",
            "execution_mode": "anserini_java_cli_via_pyserini_distribution",
            "supporting_runtime_packages": {"numpy": "2.4.2"},
        },
        "retrieval": {
            "top_k": 100,
            "bm25_k1": 0.9,
            "bm25_b": 0.4,
            "index_threads": 1,
            "index_memory_buffer_mb": 256,
            "search_threads": 1,
        },
        "evaluation": {"split": "dev", "bootstrap_seed": 42, "bootstrap_iterations": 10_000},
    }


class MiraclKoLexicalSmokeTests(unittest.TestCase):
    def test_config_rejects_non_passage_or_unapproved_backend(self):
        config = smoke_config()
        validate_lexical_smoke_config(config)

        wrong_unit = {**config, "retrieval_unit": "document"}
        with self.assertRaisesRegex(ValueError, "passage"):
            validate_lexical_smoke_config(wrong_unit)

        wrong_backend = {**config, "backend": {**config["backend"], "name": "regex_bm25"}}
        with self.assertRaisesRegex(ValueError, "Anserini"):
            validate_lexical_smoke_config(wrong_backend)

        wrong_runtime = {**config, "runtime": {**config["runtime"], "execution_mode": "python_api"}}
        with self.assertRaisesRegex(ValueError, "Anserini Java CLI"):
            validate_lexical_smoke_config(wrong_runtime)

    def test_evaluation_keeps_passage_ids_and_uses_positive_qrels_only_for_scoring(self):
        queries = [{"qid": "q1", "query": "질문"}, {"qid": "q2", "query": "다른 질문"}]
        qrels = {
            "q1": {"a#0": 2, "a#1": 0, "b#0": 1},
            "q2": {"c#0": 1},
        }
        rankings = {
            "q1": [("a#1", 3.0), ("b#0", 2.0), ("a#0", 1.0)],
            "q2": [("c#0", 4.0)],
        }
        rows, aggregate = evaluate_passage_rankings(queries, qrels, rankings, corpus_ids={
            "a#0", "a#1", "b#0", "c#0",
        })

        self.assertEqual([row["query_id"] for row in rows], ["q1", "q2"])
        self.assertEqual(rows[0]["ranked_passage_ids"][:3], ["a#1", "b#0", "a#0"])
        self.assertEqual(rows[0]["passage_recall_at_5"], 1.0)
        self.assertEqual(rows[0]["passage_hit_at_5"], 1.0)
        self.assertEqual(rows[0]["passage_mrr"], 0.5)
        self.assertEqual(rows[0]["passage_precision_at_20"], 0.1)
        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(aggregate["precision_at_20_denominator"], 20)

    def test_evaluation_rejects_orphan_retrieval_id_and_missing_query_row(self):
        queries = [{"qid": "q1", "query": "질문"}]
        qrels = {"q1": {"a#0": 1}}
        with self.assertRaisesRegex(ValueError, "orphan"):
            evaluate_passage_rankings(
                queries, qrels, {"q1": [("unknown#0", 1.0)]}, corpus_ids={"a#0"}
            )
        with self.assertRaisesRegex(ValueError, "missing ranking"):
            evaluate_passage_rankings(queries, qrels, {}, corpus_ids={"a#0"})

    def test_anserini_trec_parser_preserves_passage_ids_and_rejects_foreign_qids(self):
        with TemporaryDirectory() as directory:
            run_path = Path(directory) / "run.txt"
            run_path.write_text(
                "q1 Q0 a#0 1 2.5 Anserini\nq1 Q0 a#1 2 1.5 Anserini\n",
                encoding="utf-8",
            )
            rankings = parse_anserini_trec_run(run_path, query_ids={"q1", "q2"})
            self.assertEqual(rankings["q1"], [("a#0", 2.5), ("a#1", 1.5)])
            self.assertEqual(rankings["q2"], [])

            run_path.write_text("foreign Q0 a#0 1 2.5 Anserini\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown query ID"):
                parse_anserini_trec_run(run_path, query_ids={"q1"})

    def test_result_contract_requires_raw_rows_and_pinned_backend_provenance(self):
        queries = [{"qid": "q1", "query": "질문"}]
        qrels = {"q1": {"a#0": 1}}
        corpus_ids = {"a#0"}
        rows, metrics = evaluate_passage_rankings(
            queries, qrels, {"q1": [("a#0", 1.0)]}, corpus_ids=corpus_ids
        )
        runtime = {"runtime": "fixed"}
        provenance = {
            "subset_sha256": "a" * 64,
            "query_qrel_sha256": "b" * 64,
            "backend_config_sha256": sha256_json(smoke_config()),
            "backend_runtime_sha256": sha256_json(runtime),
            "raw_rows_sha256": sha256_json(rows),
            "subset_manifest_sha256": "c" * 64,
            "runner_code_sha256": "d" * 64,
            "contract_code_sha256": "e" * 64,
            "container_recipe_sha256": "f" * 64,
        }
        inputs = SmokeResultValidationInputs(
            queries=queries,
            qrels_by_qid=qrels,
            corpus_ids=corpus_ids,
            expected_provenance=provenance,
            expected_experiment_contract_sha256="1" * 64,
        )
        result = {
            "schema_version": "dr-dci.miracl-ko-lexical-smoke-result.v1",
            "dataset": "MIRACL",
            "language": "ko",
            "retrieval_unit": "passage",
            "scale": 20_000,
            "source_git_commit": "e" * 40,
            "source_provenance": {
                "mode": "git_checkout",
                "source_git_commit": "e" * 40,
                "git_clean": True,
            },
            "experiment_contract_sha256": "1" * 64,
            "runtime": runtime,
            "raw_rows": rows,
            "metrics": metrics,
            "latency": {
                "measurement": "search_batch_elapsed_seconds_only",
                "search_batch_seconds": 1.0,
                "batch_mean_per_query_seconds": 1.0,
                "query_count": 1,
                "paired_bootstrap_allowed": False,
            },
            "provenance": provenance,
        }
        validate_standalone_smoke_result(result, config=smoke_config(), inputs=inputs)
        broken = {**result, "provenance": {**result["provenance"], "backend_config_sha256": "bad"}}
        with self.assertRaisesRegex(ValueError, "backend_config_sha256"):
            validate_standalone_smoke_result(broken, config=smoke_config(), inputs=inputs)
        missing_commit = {key: value for key, value in result.items() if key != "source_git_commit"}
        with self.assertRaisesRegex(ValueError, "source_git_commit"):
            validate_standalone_smoke_result(missing_commit, config=smoke_config(), inputs=inputs)
        missing_source_provenance = {key: value for key, value in result.items() if key != "source_provenance"}
        with self.assertRaisesRegex(ValueError, "source provenance"):
            validate_standalone_smoke_result(missing_source_provenance, config=smoke_config(), inputs=inputs)

        metric_tamper = {**result, "metrics": {**result["metrics"], "passage_recall_at_20": 0.0}}
        with self.assertRaisesRegex(ValueError, "aggregate passage_recall_at_20"):
            validate_standalone_smoke_result(metric_tamper, config=smoke_config(), inputs=inputs)
        tampered_rows = [{**rows[0], "passage_recall_at_20": 0.0}]
        tampered_provenance = {**provenance, "raw_rows_sha256": sha256_json(tampered_rows)}
        raw_row_tamper = {
            **result,
            "raw_rows": tampered_rows,
            "provenance": tampered_provenance,
        }
        tampered_inputs = SmokeResultValidationInputs(
            queries=queries,
            qrels_by_qid=qrels,
            corpus_ids=corpus_ids,
            expected_provenance=tampered_provenance,
            expected_experiment_contract_sha256="1" * 64,
        )
        with self.assertRaisesRegex(ValueError, "raw row passage_recall_at_20"):
            validate_standalone_smoke_result(raw_row_tamper, config=smoke_config(), inputs=tampered_inputs)
        runtime_hash_tamper = {**result, "provenance": {**provenance, "backend_runtime_sha256": "2" * 64}}
        with self.assertRaisesRegex(ValueError, "backend_runtime_sha256"):
            validate_standalone_smoke_result(runtime_hash_tamper, config=smoke_config(), inputs=inputs)
        runner_hash_tamper = {**result, "provenance": {**provenance, "runner_code_sha256": "3" * 64}}
        with self.assertRaisesRegex(ValueError, "runner_code_sha256"):
            validate_standalone_smoke_result(runner_hash_tamper, config=smoke_config(), inputs=inputs)

    def test_source_archive_manifest_requires_matching_archive_and_contract(self):
        manifest = {
            "schema_version": "dr-dci.miracl-ko-lexical-smoke-source-manifest.v1",
            "source_git_commit": "a" * 40,
            "source_archive_sha256": "b" * 64,
            "experiment_contract_sha256": "c" * 64,
        }
        self.assertEqual(
            validate_source_archive_manifest(
                manifest,
                source_archive_sha256="b" * 64,
                expected_experiment_contract_sha256="c" * 64,
            ),
            "a" * 40,
        )
        with self.assertRaisesRegex(ValueError, "source archive"):
            validate_source_archive_manifest(
                manifest,
                source_archive_sha256="d" * 64,
                expected_experiment_contract_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "experiment contract"):
            validate_source_archive_manifest(
                manifest,
                source_archive_sha256="b" * 64,
                expected_experiment_contract_sha256="d" * 64,
            )

    def test_runner_requires_clean_checkout_and_exact_source_commit(self):
        head = "a" * 40
        with patch.object(RUNNER_MODULE.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=f"{head}\n"),
            subprocess.CompletedProcess([], 0, stdout=""),
        ]):
            self.assertEqual(
                RUNNER_MODULE.resolve_source_provenance(
                    head,
                    source_manifest_path=None,
                    source_archive_path=None,
                ),
                {"mode": "git_checkout", "source_git_commit": head, "git_clean": True},
            )
        with patch.object(RUNNER_MODULE.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=f"{head}\n"),
            subprocess.CompletedProcess([], 0, stdout=" M src/miracl_ko/lexical_smoke.py\n"),
        ]):
            with self.assertRaisesRegex(RuntimeError, "dirty"):
                RUNNER_MODULE.resolve_source_provenance(
                    head,
                    source_manifest_path=None,
                    source_archive_path=None,
                )
        with patch.object(RUNNER_MODULE.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=f"{head}\n"),
            subprocess.CompletedProcess([], 0, stdout=""),
        ]):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                RUNNER_MODULE.resolve_source_provenance(
                    "b" * 40,
                    source_manifest_path=None,
                    source_archive_path=None,
                )

    def test_scale_comparison_is_paired_and_has_no_practical_effect_decision(self):
        def row(qid, recall):
            return {
                "query_id": qid,
                "passage_ndcg_at_10": 0.0,
                "passage_recall_at_5": 0.0,
                "passage_recall_at_20": recall,
                "passage_recall_at_100": recall,
                "passage_hit_at_5": 0.0,
                "passage_hit_at_10": 0.0,
                "passage_precision_at_20": 0.0,
                "passage_mrr": 0.0,
                "query_latency_seconds": 0.0,
            }
        control = [
            row("q1", 1.0),
            row("q2", 0.0),
        ]
        treatment = [
            row("q1", 1.0),
            row("q2", 1.0),
        ]
        comparison = compare_smoke_scales(control, treatment, seed=42, iterations=100)
        self.assertEqual(comparison["delta_direction"], "110k_minus_20k")
        self.assertEqual(comparison["paired_query_count"], 2)
        self.assertNotIn("minimum_practical_effect", comparison)
        self.assertNotIn("query_latency_seconds", comparison["metrics"])

    def test_batch_latency_is_not_emitted_as_a_query_level_retrieval_metric(self):
        queries = [{"qid": "q1", "query": "질문"}]
        rows, aggregate = evaluate_passage_rankings(
            queries,
            {"q1": {"a#0": 1}},
            {"q1": [("a#0", 1.0)]},
            corpus_ids={"a#0"},
        )
        self.assertNotIn("query_latency_seconds", rows[0])
        self.assertNotIn("query_latency_seconds", aggregate)

    def test_legacy_copied_batch_latency_is_descriptive_only_not_a_bootstrap_sample(self):
        queries = [{"qid": "q1", "query": "질문"}, {"qid": "q2", "query": "다른 질문"}]
        qrels = {"q1": {"a#0": 1}, "q2": {"b#0": 1}}
        corpus_ids = {"a#0", "b#0"}
        rows, metrics = evaluate_passage_rankings(
            queries,
            qrels,
            {"q1": [("a#0", 1.0)], "q2": [("b#0", 1.0)]},
            corpus_ids=corpus_ids,
            latencies_by_qid={"q1": 0.5, "q2": 0.5},
        )
        metrics.update({
            "search_batch_seconds": 1.0,
            "query_latency_measurement": "batch_elapsed_seconds_divided_by_dev_query_count",
        })
        runtime = {"runtime": "legacy"}
        provenance = {
            "subset_sha256": "a" * 64,
            "query_qrel_sha256": "b" * 64,
            "backend_config_sha256": sha256_json(smoke_config()),
            "backend_runtime_sha256": sha256_json(runtime),
            "raw_rows_sha256": sha256_json(rows),
            "subset_manifest_sha256": "c" * 64,
            "runner_code_sha256": "d" * 64,
            "contract_code_sha256": "e" * 64,
            "container_recipe_sha256": "f" * 64,
        }
        legacy_result = {
            "schema_version": "dr-dci.miracl-ko-lexical-smoke-result.v1",
            "dataset": "MIRACL",
            "language": "ko",
            "retrieval_unit": "passage",
            "scale": 20_000,
            "source_git_commit": "a" * 40,
            "runtime": runtime,
            "raw_rows": rows,
            "metrics": metrics,
            "provenance": provenance,
        }
        validate_standalone_smoke_result(
            legacy_result,
            config=smoke_config(),
            inputs=SmokeResultValidationInputs(
                queries=queries,
                qrels_by_qid=qrels,
                corpus_ids=corpus_ids,
                expected_provenance=provenance,
                expected_experiment_contract_sha256="1" * 64,
                allow_legacy_contract=True,
            ),
        )
        comparison = compare_smoke_scales(rows, rows, seed=42, iterations=100)
        self.assertNotIn("query_latency_seconds", comparison["metrics"])


if __name__ == "__main__":
    unittest.main()
