import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

import run_experiment
from scripts import build_aihub_corpus
from scripts import build_aihub_element_corpus
from scripts import build_aihub_qa
from scripts import derive_qrels


class AIHubDatasetRoutingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.original_data_dir = run_experiment.DATA_DIR
        run_experiment.DATA_DIR = self.data_dir

    def tearDown(self):
        run_experiment.DATA_DIR = self.original_data_dir
        self._tmp.cleanup()

    def test_new_variants_resolve_to_generated_directories(self):
        self.assertEqual(
            run_experiment.dataset_dir("aihub-smoke"),
            self.data_dir / "aihub" / "smoke",
        )
        self.assertEqual(
            run_experiment.dataset_dir("aihub-element"),
            self.data_dir / "aihub" / "element",
        )

    def test_shared_spans_are_limited_to_declared_aihub_variants(self):
        qa_dir = self.data_dir / "aihub" / "qa"
        qa_dir.mkdir(parents=True)
        (qa_dir / "qa_meta.jsonl").write_text(
            json.dumps(
                {
                    "qid": "1",
                    "supporting_spans": [
                        {"text": "근거 문자열", "char_start": 0, "char_end": 6}
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertIn("1", run_experiment.load_supporting_spans("aihub-smoke"))
        self.assertIn("1", run_experiment.load_supporting_spans("aihub-element"))
        self.assertEqual(run_experiment.load_supporting_spans("trec-covid"), {})


class AIHubChunkBuilderTests(unittest.TestCase):
    def test_short_tail_is_retained_without_exceeding_chunk_limit(self):
        raw = ("가" * 500) + ". " + ("나" * 50)
        offsets = build_aihub_corpus.chunk_offsets(raw)
        recovered = "".join(raw[start:end] for start, end in offsets)
        self.assertEqual(recovered.replace(" ", ""), raw.replace(" ", ""))
        self.assertEqual(offsets[-1][1], len(raw))
        self.assertTrue(all(end - start <= 512 for start, end in offsets))
        self.assertLess(offsets[-1][1] - offsets[-1][0], 80)

    def test_text_mediated_element_offsets_preserve_short_document(self):
        raw = "짧지만 보존되어야 하는 법률 텍스트"
        self.assertEqual(
            build_aihub_element_corpus.legal_offsets(raw),
            [(0, len(raw))],
        )

    def test_clean_builder_uses_explicit_archives_and_records_hashes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "input.zip"
            payload = {
                "totalcount": 1,
                "data": [
                    {
                        "book_id": "book",
                        "text": "가" * 100,
                        "category": "cat",
                        "keyword": [],
                        "NE": [],
                    }
                ],
            }
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("labels.json", json.dumps(payload, ensure_ascii=False))
            output = root / "smoke"
            parents = root / "parents.jsonl"
            manifest = build_aihub_corpus.build(
                {"의료": archive, "법률": archive},
                output_dir=output,
                parents_output=parents,
                n_per_domain=1,
                chunk_chars=512,
                min_chars=80,
            )
            rows = [
                json.loads(line)
                for line in (output / "corpus.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["parent_id"] for row in rows}), 2)
            self.assertEqual(manifest["dropped_non_whitespace_chars"], 0)
            self.assertEqual(
                manifest["sources"]["법률"]["sha256"],
                build_aihub_corpus.sha256_file(archive),
            )


class AIHubQrelProjectionTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {
                "_id": "p:c0",
                "parent_id": "p",
                "text": "가" * 10,
                "char_start": 0,
                "char_end": 10,
            },
            {
                "_id": "p:c1",
                "parent_id": "p",
                "text": "나" * 10,
                "char_start": 10,
                "char_end": 20,
            },
        ]
        self.queries = [{"_id": "1", "text": "질문"}]

    def test_parent_and_offset_bound_chunk_qrels_are_both_emitted(self):
        qa = [
            {
                "qid": "1",
                "parent_id": "p",
                "supporting_spans": [
                    {"text": "근거문자열입니다", "char_start": 2, "char_end": 10},
                    {"text": "두번째근거문자열", "char_start": 10, "char_end": 20},
                ],
            }
        ]
        queries, parent_qrels, chunk_qrels = derive_qrels.derive_qrels(
            self.corpus,
            qa,
            self.queries,
        )
        self.assertEqual(queries, self.queries)
        self.assertEqual(
            parent_qrels,
            [{"query-id": "1", "corpus-id": "p", "score": 2}],
        )
        self.assertEqual(
            chunk_qrels,
            [
                {"query-id": "1", "corpus-id": "p:c0", "score": 2},
                {"query-id": "1", "corpus-id": "p:c1", "score": 2},
            ],
        )

    def test_absent_parent_fails_instead_of_leaving_orphan_query(self):
        qa = [
            {
                "qid": "1",
                "parent_id": "missing",
                "supporting_spans": [
                    {"text": "근거문자열입니다", "char_start": 0, "char_end": 8}
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "absent parent"):
            derive_qrels.derive_qrels(self.corpus, qa, self.queries)

    def test_one_character_overlap_is_not_chunk_gold(self):
        qa = [
            {
                "qid": "1",
                "parent_id": "p",
                "supporting_spans": [
                    {"text": "경계근거문자열", "char_start": 9, "char_end": 17}
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "not represented"):
            derive_qrels.derive_qrels(self.corpus, qa, self.queries)


class AIHubQABuilderImportTests(unittest.TestCase):
    def test_builder_import_does_not_require_dotenv_file(self):
        self.assertTrue(callable(build_aihub_qa.build))
        self.assertEqual(build_aihub_qa.locate_all("가나다라마바사", "가나다라마바사"), [])
        self.assertEqual(
            build_aihub_qa.locate_all("12345678 xx 12345678", "12345678"),
            [(0, 8), (12, 20)],
        )


if __name__ == "__main__":
    unittest.main()
