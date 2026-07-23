import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.miracl_ko.lexical_smoke import (
    compare_smoke_scales,
    evaluate_passage_rankings,
    parse_anserini_trec_run,
    sha256_json,
    validate_lexical_smoke_config,
    validate_standalone_smoke_result,
)


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
        rows = [{"query_id": "q1", "retrieval_unit": "passage", "ranked_passage_ids": ["a#0"]}]
        metrics = {
            "passage_ndcg_at_10": 1.0,
            "passage_recall_at_5": 1.0,
            "passage_recall_at_20": 1.0,
            "passage_recall_at_100": 1.0,
            "passage_hit_at_5": 1.0,
            "passage_hit_at_10": 1.0,
            "passage_precision_at_20": 0.05,
            "passage_mrr": 1.0,
        }
        result = {
            "schema_version": "dr-dci.miracl-ko-lexical-smoke-result.v1",
            "dataset": "MIRACL",
            "language": "ko",
            "retrieval_unit": "passage",
            "scale": 20_000,
            "raw_rows": rows,
            "metrics": metrics,
            "provenance": {
                "subset_sha256": "a" * 64,
                "query_qrel_sha256": "b" * 64,
                "backend_config_sha256": sha256_json(smoke_config()),
                "backend_runtime_sha256": "d" * 64,
                "raw_rows_sha256": sha256_json(rows),
            },
        }
        validate_standalone_smoke_result(result, config=smoke_config())
        broken = {**result, "provenance": {**result["provenance"], "backend_config_sha256": "bad"}}
        with self.assertRaisesRegex(ValueError, "backend_config_sha256"):
            validate_standalone_smoke_result(broken, config=smoke_config())

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


if __name__ == "__main__":
    unittest.main()
