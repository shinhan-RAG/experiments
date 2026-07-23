import unittest
from pathlib import Path

from src.miracl_ko.preparation import (
    SCALE_SIZES,
    build_nested_subset_ids,
    derive_passage_fields,
    positive_passage_ids,
    raw_data_paths_are_ignored,
    validate_acquisition_manifest,
    validate_leakage_inputs,
    validate_miracl_result_unit,
    validate_miracl_smoke_payload,
    validate_nested_subset_manifest,
    validate_nested_subset_payloads,
)


class MiraclKoPreparationTests(unittest.TestCase):
    def test_passage_ids_preserve_article_and_index(self):
        self.assertEqual(
            derive_passage_fields("123#4"),
            {"corpus_id": "123#4", "article_id": "123", "passage_index": 4},
        )
        with self.assertRaisesRegex(ValueError, "article#passage"):
            derive_passage_fields("123")

    def test_positive_set_excludes_zero_and_negative_judgments(self):
        qrels = [
            {"qid": "q1", "corpus_id": "a#0", "relevance": 1},
            {"qid": "q1", "corpus_id": "a#1", "relevance": 0},
            {"qid": "q2", "corpus_id": "b#0", "relevance": -1},
            {"qid": "q2", "corpus_id": "b#1", "relevance": 2},
        ]
        self.assertEqual(positive_passage_ids(qrels), {"a#0", "b#1"})

    def test_nested_subset_ids_are_deterministic_and_gold_preserving(self):
        corpus_ids = {f"a#{index}" for index in range(120_000)}
        positives = {"a#1", "a#2", "a#3"}
        first = build_nested_subset_ids(corpus_ids, positives)
        second = build_nested_subset_ids(corpus_ids, positives)

        self.assertEqual(first, second)
        self.assertEqual(set(first[20_000]) & positives, positives)
        self.assertEqual(first[20_000], first[50_000][:20_000])
        self.assertEqual(first[50_000], first[110_000][:50_000])
        self.assertEqual({len(rows) for rows in first.values()}, set(SCALE_SIZES))

    def test_subset_builder_fails_loudly_when_positive_set_exceeds_20k(self):
        corpus_ids = {f"a#{index}" for index in range(25_000)}
        positives = {f"a#{index}" for index in range(20_001)}
        with self.assertRaisesRegex(ValueError, r"M\+ has 20,001"):
            build_nested_subset_ids(corpus_ids, positives)

    def test_subset_manifest_requires_scale_invariant_query_qrel_inputs(self):
        inputs = {"queries_dev": {"sha256": "a" * 64}}
        manifest = {
            "retrieval_unit": "passage",
            "subsets": {
                str(size): {
                    "passage_count": size,
                    "positive_passages_preserved": True,
                    "query_qrel_inputs": inputs,
                }
                for size in SCALE_SIZES
            },
        }
        validate_nested_subset_manifest(manifest)
        manifest["subsets"]["50000"]["query_qrel_inputs"] = {"queries_dev": {"sha256": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "invariant"):
            validate_nested_subset_manifest(manifest)

    def test_nested_payload_validator_rejects_changed_shared_passage(self):
        payloads = {
            20_000: [{"corpus_id": "a#0", "article_id": "a", "passage_index": 0, "title": "A", "text": "one"}],
            50_000: [{"corpus_id": "a#0", "article_id": "a", "passage_index": 0, "title": "A", "text": "changed"}],
            110_000: [{"corpus_id": "a#0", "article_id": "a", "passage_index": 0, "title": "A", "text": "one"}],
        }
        with self.assertRaisesRegex(ValueError, "shared passage"):
            validate_nested_subset_payloads(payloads, positive_ids={"a#0"})

    def test_acquisition_manifest_requires_revision_and_sha256(self):
        base_manifest = {
            "dataset": "MIRACL",
            "language": "ko",
            "source_urls": {"topics_qrels": "https://example.test/topics", "corpus": "https://example.test/corpus"},
            "downloaded_at": "2026-07-23T00:00:00+00:00",
            "license": "Apache-2.0",
            "underlying_content_license": {"content": "Wikipedia"},
            "acquisition_script_version": "test",
            "acquisition_script_sha256": "b" * 64,
            "python_version": "test",
            "files": [],
        }
        with self.assertRaisesRegex(ValueError, "resolved_revision"):
            validate_acquisition_manifest(base_manifest)
        manifest = {
            **base_manifest,
            "resolved_revision": {"topics_qrels": "a", "corpus": "b"},
            "files": [{
                "source_url": "https://example.test/topics",
                "relative_path": "raw/file",
                "byte_size": 1,
                "sha256": "a" * 64,
            }],
        }
        validate_acquisition_manifest(manifest)

    def test_raw_miracl_paths_are_git_ignored(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertTrue(raw_data_paths_are_ignored(repo_root, repo_root / "data" / "miracl-ko"))
        self.assertFalse(raw_data_paths_are_ignored(repo_root, repo_root / "docs" / "miracl-ko"))

    def test_leakage_input_contract_rejects_qrel_and_relevance(self):
        with self.assertRaisesRegex(ValueError, "qrel"):
            validate_leakage_inputs({"taxonomy": ["title", "text", "qrels"]})
        with self.assertRaisesRegex(ValueError, "relevance"):
            validate_leakage_inputs({"retriever": ["corpus_id", "title", "text", "relevance"]})
        validate_leakage_inputs({
            "taxonomy": ["corpus_id", "title", "text"],
            "retriever": ["corpus_id", "title", "text"],
            "query_rewrite": ["qid", "query"],
        })

    def test_miracl_result_unit_rejects_document_metric_names(self):
        with self.assertRaisesRegex(ValueError, "passage"):
            validate_miracl_result_unit({"retrieval_unit": "document"})
        with self.assertRaisesRegex(ValueError, "document_"):
            validate_miracl_result_unit({
                "retrieval_unit": "passage",
                "metrics": {"document_recall_at_20": 1.0},
            })
        validate_miracl_result_unit({
            "retrieval_unit": "passage",
            "metrics": {"passage_recall_at_20": 1.0},
        })

    def test_smoke_payload_requires_passage_raw_rows_and_provenance(self):
        with self.assertRaisesRegex(ValueError, "raw per-query"):
            validate_miracl_smoke_payload({
                "retrieval_unit": "passage",
                "metrics": {"passage_ndcg_at_10": 0.1},
                "provenance": {"subset_sha256": "a" * 64},
            })
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_miracl_smoke_payload({
                "retrieval_unit": "passage",
                "raw_rows": [{"qid": "q1", "passage_ndcg_at_10": 0.1}],
                "metrics": {"passage_ndcg_at_10": 0.1},
            })
        validate_miracl_smoke_payload({
            "retrieval_unit": "passage",
            "raw_rows": [{"qid": "q1", "passage_ndcg_at_10": 0.1}],
            "metrics": {"passage_ndcg_at_10": 0.1},
            "provenance": {"subset_sha256": "a" * 64, "query_qrel_sha256": "b" * 64},
        })


if __name__ == "__main__":
    unittest.main()
