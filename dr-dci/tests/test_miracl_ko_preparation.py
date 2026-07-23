import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.miracl_ko.preparation import (
    PREPARATION_CONTRACT_FILES,
    SCALE_SIZES,
    SCALE_VERSION,
    build_preparation_contract,
    build_nested_subset_ids,
    derive_passage_fields,
    judged_passage_ids,
    positive_passage_ids,
    raw_data_paths_are_ignored,
    require_clean_git_source,
    validate_acquisition_manifest,
    validate_eda_blocking_conditions,
    validate_leakage_inputs,
    validate_miracl_result_unit,
    validate_miracl_smoke_payload,
    validate_nested_subset_manifest,
    validate_nested_subset_payloads,
    validate_normalization_manifest,
    validate_revision_lock,
    scale_rank,
    sha256_file,
    sha256_json,
    stream_nested_subset_ids,
    validate_preparation_contract,
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
        self.assertEqual(judged_passage_ids(qrels), {"a#0", "a#1", "b#0", "b#1"})

    def test_nested_subset_ids_are_deterministic_and_gold_preserving(self):
        sizes = (5, 8, 12)
        corpus_ids = {f"a#{index}" for index in range(20)}
        judged = {"a#1", "a#2", "a#3"}
        first = build_nested_subset_ids(corpus_ids, judged, sizes=sizes)
        second = build_nested_subset_ids(corpus_ids, judged, sizes=sizes)

        self.assertEqual(first, second)
        self.assertTrue(judged.issubset(first[5]))
        self.assertTrue(set(first[5]) < set(first[8]) < set(first[12]))
        self.assertEqual(first[5], sorted(first[5], key=scale_rank))
        self.assertEqual(first[8], sorted(first[8], key=scale_rank))
        self.assertEqual(
            first[5],
            [corpus_id for corpus_id in first[8] if corpus_id in set(first[5])],
        )
        self.assertEqual({len(rows) for rows in first.values()}, set(sizes))
        with tempfile.TemporaryDirectory() as temporary:
            corpus_path = Path(temporary) / "corpus.jsonl"
            corpus_path.write_text(
                "".join(json.dumps({"corpus_id": corpus_id}) + "\n" for corpus_id in sorted(corpus_ids)),
                encoding="utf-8",
            )
            self.assertEqual(stream_nested_subset_ids(corpus_path, judged, sizes=sizes), first)

    def test_subset_builder_fails_loudly_when_positive_set_exceeds_20k(self):
        corpus_ids = {f"a#{index}" for index in range(25_000)}
        judged = {f"a#{index}" for index in range(20_001)}
        with self.assertRaisesRegex(ValueError, "mandatory judged set has 20,001"):
            build_nested_subset_ids(corpus_ids, judged)

    def test_subset_payload_validator_rejects_relevance_ordered_rows(self):
        reversed_ids = list(reversed(sorted(["a#1", "a#2"], key=scale_rank)))
        payloads = {
            20_000: [
                {
                    "corpus_id": corpus_id,
                    "article_id": "a",
                    "passage_index": int(corpus_id.rsplit("#", 1)[1]),
                    "title": "A",
                    "text": corpus_id,
                }
                for corpus_id in reversed_ids
            ],
        }
        with self.assertRaisesRegex(ValueError, "global hash rank"):
            validate_nested_subset_payloads(payloads, mandatory_ids={"a#1"})

    def test_subset_manifest_requires_scale_invariant_query_qrel_inputs(self):
        inputs = {"queries_dev": {"sha256": "a" * 64}}
        manifest = {
            "retrieval_unit": "passage",
            "scale_version": SCALE_VERSION,
            "preparation_contract_sha256": "c" * 64,
            "source_git_commit": "d" * 40,
            "output_order_definition": f"SHA256({SCALE_VERSION}\\0 + corpus_id), independent of relevance",
            "all_judged_qrels_forced_into_mandatory_set": True,
            "subsets": {
                str(size): {
                    "passage_count": size,
                    "positive_passages_preserved": True,
                    "judged_passages_preserved": True,
                    "query_qrel_inputs": inputs,
                }
                for size in SCALE_SIZES
            },
        }
        validate_nested_subset_manifest(manifest)
        manifest["subsets"]["50000"]["query_qrel_inputs"] = {"queries_dev": {"sha256": "b" * 64}}
        with self.assertRaisesRegex(ValueError, "invariant"):
            validate_nested_subset_manifest(manifest)
        manifest["subsets"]["50000"]["query_qrel_inputs"] = inputs
        manifest["all_judged_qrels_forced_into_mandatory_set"] = False
        with self.assertRaisesRegex(ValueError, "all judged"):
            validate_nested_subset_manifest(manifest)

    def test_v1_subset_manifest_cannot_be_reused_under_v2_contract(self):
        manifest = {
            "retrieval_unit": "passage",
            "scale_version": "miracl-ko-scale-v1",
            "preparation_contract_sha256": "c" * 64,
            "all_judged_qrels_forced_into_mandatory_set": True,
            "subsets": {
                str(size): {
                    "passage_count": size,
                    "positive_passages_preserved": True,
                    "judged_passages_preserved": True,
                    "query_qrel_inputs": {"queries_dev": {"sha256": "a" * 64}},
                }
                for size in SCALE_SIZES
            },
        }
        with self.assertRaisesRegex(ValueError, "scale-v2"):
            validate_nested_subset_manifest(manifest)

    def test_nested_payload_validator_rejects_changed_shared_passage(self):
        ids = sorted(["a#0", "a#1", "a#2"], key=scale_rank)
        shared = ids[0]

        def row(corpus_id, text):
            return {
                "corpus_id": corpus_id,
                "article_id": "a",
                "passage_index": int(corpus_id.rsplit("#", 1)[1]),
                "title": "A",
                "text": text,
            }

        payloads = {
            20_000: [row(shared, "one")],
            50_000: [row(corpus_id, "changed" if corpus_id == shared else "two") for corpus_id in ids[:2]],
            110_000: [row(corpus_id, "one" if corpus_id == shared else "three") for corpus_id in ids],
        }
        with self.assertRaisesRegex(ValueError, "shared passage"):
            validate_nested_subset_payloads(payloads, mandatory_ids={shared})

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
            "preparation_contract_sha256": "c" * 64,
            "source_git_commit": "d" * 40,
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
                "source_key": "topics_qrels",
                "resolved_revision": "a",
                "relative_path": "raw/file",
                "byte_size": 1,
                "sha256": "a" * 64,
                "hash_verification": {"kind": "local_sha256_at_pinned_revision"},
            }],
        }
        validate_acquisition_manifest(manifest)

    def test_revision_lock_and_raw_manifest_hash_are_rechecked(self):
        lock = {
            "schema_version": "dr-dci.miracl-ko-revision-lock.v1",
            "dataset": "MIRACL",
            "language": "ko",
            "change_control": "test",
            "approval_record": {
                "status": "test", "reviewed_at": "2026-07-23",
                "evidence_path": "docs/evidence.md", "evidence_sha256": "c" * 64,
                "scope": "test",
            },
            "sources": {
                "topics_qrels": {"repository": "miracl/miracl", "revision": "a" * 40},
                "corpus": {"repository": "miracl/miracl-corpus", "revision": "b" * 40},
            },
        }
        validate_revision_lock(lock)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw" / "artifact"
            raw.parent.mkdir()
            raw.write_text("fixed", encoding="utf-8")
            manifest = {
                "dataset": "MIRACL",
                "language": "ko",
                "source_urls": {"topics_qrels": "https://example.test/topics", "corpus": "https://example.test/corpus"},
                "resolved_revision": {"topics_qrels": "a" * 40, "corpus": "b" * 40},
                "revision_lock_sha256": sha256_json(lock),
                "downloaded_at": "2026-07-23T00:00:00+00:00",
                "license": "Apache-2.0",
                "underlying_content_license": {"content": "Wikipedia"},
                "acquisition_script_version": "test",
                "acquisition_script_sha256": "c" * 64,
                "preparation_contract_sha256": "d" * 64,
                "source_git_commit": "e" * 40,
                "python_version": "test",
                "files": [{
                    "source_url": "https://example.test/topics",
                    "source_key": "topics_qrels",
                    "resolved_revision": "a" * 40,
                    "relative_path": "raw/artifact",
                    "byte_size": raw.stat().st_size,
                    "sha256": sha256_file(raw),
                    "hash_verification": {"kind": "local_sha256_at_pinned_revision"},
                }],
            }
            validate_acquisition_manifest(manifest, data_dir=root, revision_lock=lock)
            raw.write_text("alter", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                validate_acquisition_manifest(manifest, data_dir=root, revision_lock=lock)

    def test_revision_lock_rechecks_repository_evidence_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "docs" / "revision-evidence.md"
            evidence.parent.mkdir()
            evidence.write_text("approved revision evidence\n", encoding="utf-8")
            lock = {
                "schema_version": "dr-dci.miracl-ko-revision-lock.v1",
                "dataset": "MIRACL",
                "language": "ko",
                "change_control": "test",
                "approval_record": {
                    "status": "test",
                    "reviewed_at": "2026-07-23",
                    "evidence_path": "docs/revision-evidence.md",
                    "evidence_sha256": sha256_file(evidence),
                    "scope": "test",
                },
                "sources": {
                    "topics_qrels": {"repository": "miracl/miracl", "revision": "a" * 40},
                    "corpus": {"repository": "miracl/miracl-corpus", "revision": "b" * 40},
                },
            }
            validate_revision_lock(lock, repo_root=root)
            evidence.write_text("altered evidence\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence sha256"):
                validate_revision_lock(lock, repo_root=root)

    def test_preparation_contract_hashes_wrapper_module_and_revision_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative_path in PREPARATION_CONTRACT_FILES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path + "\n", encoding="utf-8")
            contract = build_preparation_contract(root)
            validate_preparation_contract(contract, repo_root=root)
            module_path = root / "src/miracl_ko/preparation.py"
            original = module_path.read_text(encoding="utf-8")
            module_path.write_text("X" + original[1:], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preparation contract file sha256"):
                validate_preparation_contract(contract, repo_root=root)

    def test_preparation_execution_rejects_dirty_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "source.txt"
            source.write_text("committed\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "source.txt"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(root), "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture",
                ],
                check=True,
            )
            self.assertRegex(require_clean_git_source(root), r"^[0-9a-f]{40}$")
            (root / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean Git worktree"):
                require_clean_git_source(root)

    def test_normalized_manifest_hash_is_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw" / "source"
            output = root / "normalized" / "output.jsonl"
            raw.parent.mkdir()
            output.parent.mkdir()
            raw.write_text("source", encoding="utf-8")
            output.write_text('{"corpus_id":"a#0"}\n', encoding="utf-8")
            manifest = {
                "dataset": "MIRACL", "language": "ko", "retrieval_unit": "passage",
                "transformation_version": "test", "generated_at": "2026-07-23T00:00:00+00:00",
                "acquisition_manifest_sha256": "a" * 64,
                "revision_lock_sha256": "b" * 64,
                "preparation_contract_sha256": "c" * 64,
                "source_git_commit": "d" * 40,
                "inputs": {"source": {"relative_path": "raw/source", "byte_size": raw.stat().st_size, "sha256": sha256_file(raw)}},
                "outputs": {"corpus": {"relative_path": "normalized/output.jsonl", "byte_size": output.stat().st_size, "sha256": sha256_file(output), "rows": 1}},
                "unicode_nfc_changed_fields": {"title": 0, "text": 0, "query": 0},
            }
            validate_normalization_manifest(manifest, data_dir=root)
            output.write_text('{"corpus_id":"b#0"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                validate_normalization_manifest(manifest, data_dir=root)

    def test_eda_blocking_contract_rejects_empty_text_query_and_release_drift(self):
        valid = {
            "corpus": {"empty_text_count": 0},
            "queries": {"empty_query_count": {"train": 0, "dev": 0}},
            "official_reference_count_comparison": {"status": "matches"},
        }
        validate_eda_blocking_conditions(valid)
        for key, value in (
            ("corpus", {"empty_text_count": 1}),
            ("queries", {"empty_query_count": {"train": 1, "dev": 0}}),
            ("official_reference_count_comparison", {"status": "release_drift_or_transform_difference"}),
        ):
            candidate = {**valid, key: value}
            with self.assertRaises(ValueError):
                validate_eda_blocking_conditions(candidate)

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
