"""청크형(법률) 데이터셋의 parent-aware 로딩·평가 계약.

법률 corpus는 `_id`가 청크 ID(`precedent:X#chunk-0000`)이고 qrels의
corpus-id는 parent 문서 ID(`precedent:X`)를 가리킨다. 평가는 README가
명시한 대로 parent 기준으로 매칭되어야 한다.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import run_experiment
from run_experiment import (
    dataset_dir,
    load_corpus,
    load_queries,
    parent_map_from_corpus,
    run_pull_probe,
    to_parent_ids,
)


CHUNKED_CORPUS = [
    {"_id": "precedent:A#chunk-0000", "parent_id": "precedent:A",
     "chunk_index": 0, "title": "", "text": "가"},
    {"_id": "precedent:A#chunk-0001", "parent_id": "precedent:A",
     "chunk_index": 1, "title": "", "text": "나"},
    {"_id": "decision:B#chunk-0000", "parent_id": "decision:B",
     "chunk_index": 0, "title": "", "text": "다"},
]

BEIR_CORPUS = [
    {"_id": "10", "title": "t", "text": "x"},
    {"_id": "11", "title": "t", "text": "y"},
]


class ParentMapTests(unittest.TestCase):
    def test_chunked_corpus_maps_chunk_to_parent(self):
        mapping = parent_map_from_corpus(CHUNKED_CORPUS)
        self.assertEqual(mapping["precedent:A#chunk-0001"], "precedent:A")
        self.assertEqual(mapping["decision:B#chunk-0000"], "decision:B")

    def test_beir_corpus_yields_empty_map(self):
        self.assertEqual(parent_map_from_corpus(BEIR_CORPUS), {})

    def test_to_parent_ids_maps_and_dedupes_preserving_order(self):
        mapping = parent_map_from_corpus(CHUNKED_CORPUS)
        ranked = ["precedent:A#chunk-0001", "decision:B#chunk-0000",
                  "precedent:A#chunk-0000", "unknown-id"]
        self.assertEqual(
            to_parent_ids(ranked, mapping),
            ["precedent:A", "decision:B", "unknown-id"],
        )

    def test_to_parent_ids_with_empty_map_is_identity(self):
        self.assertEqual(to_parent_ids(["10", "11"], {}), ["10", "11"])


class DatasetDirTests(unittest.TestCase):
    def test_beir_dataset_resolves_to_raw(self):
        self.assertEqual(dataset_dir("fiqa"),
                         run_experiment.DATA_DIR / "raw" / "fiqa")

    def test_aihub_variants_resolve_to_aihub_tree(self):
        self.assertEqual(dataset_dir("aihub-smoke20k"),
                         run_experiment.DATA_DIR / "aihub" / "smoke20k")
        self.assertEqual(dataset_dir("aihub-full"),
                         run_experiment.DATA_DIR / "aihub" / "full")


class ChunkedDatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self._orig_data_dir = run_experiment.DATA_DIR
        run_experiment.DATA_DIR = self.data_dir

        ds_dir = self.data_dir / "aihub" / "smoke20k"
        ds_dir.mkdir(parents=True)
        with open(ds_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
            for doc in CHUNKED_CORPUS:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        with open(ds_dir / "queries.jsonl", "w", encoding="utf-8") as f:
            for q in [{"_id": "q:1", "text": "질의1"}, {"_id": "q:2", "text": "질의2"}]:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        with open(ds_dir / "qrels.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"query-id": "q:1", "corpus-id": "precedent:A",
                                "score": 1}) + "\n")
        with open(ds_dir / "1k_parent_ids.json", "w", encoding="utf-8") as f:
            json.dump(["precedent:A"], f)
        self.ds_dir = ds_dir

    def tearDown(self):
        run_experiment.DATA_DIR = self._orig_data_dir
        self._tmp.cleanup()

    def test_load_corpus_from_aihub_dir(self):
        corpus = load_corpus("aihub-smoke20k")
        self.assertEqual(len(corpus), 3)

    def test_load_corpus_subset_filters_by_parent_id(self):
        corpus = load_corpus("aihub-smoke20k", subset_size=1000)
        self.assertEqual(
            sorted(d["_id"] for d in corpus),
            ["precedent:A#chunk-0000", "precedent:A#chunk-0001"],
        )

    def test_load_queries_uses_agent_sample_when_present(self):
        with open(self.ds_dir / "agent_queries_50.jsonl", "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"_id": "q:1", "text": "질의1",
                                "stratum": "가사"}, ensure_ascii=False) + "\n")
        queries, qrels = load_queries("aihub-smoke20k")
        self.assertEqual([q["_id"] for q in queries], ["q:1"])
        self.assertEqual(len(qrels), 1)

    def test_load_queries_falls_back_to_full_queries(self):
        queries, _ = load_queries("aihub-smoke20k")
        self.assertEqual(len(queries), 2)


class _StaticChunkRetriever:
    """probe 계약 테스트용: 청크 ID를 반환하는 고정 pull backend."""

    def pull(self, query, **kwargs):
        return {"results": [
            {"doc_id": "precedent:A#chunk-0001", "score": 1.0},
            {"doc_id": "precedent:A#chunk-0000", "score": 0.9},
            {"doc_id": "decision:B#chunk-0000", "score": 0.5},
        ]}


class ParentAwareProbeTests(unittest.TestCase):
    def test_probe_scores_at_parent_level(self):
        parent_map = parent_map_from_corpus(CHUNKED_CORPUS)
        queries = [{"_id": "q:1", "text": "질의"}]
        query_gold = {"q:1": {"precedent:A"}}
        rows = run_pull_probe(_StaticChunkRetriever(), queries, query_gold,
                              parent_map=parent_map)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recall_at_5"], 1.0)
        # 같은 parent의 두 청크는 rank 리스트에서 한 번만 세어져야 한다
        self.assertEqual(rows[0]["ranked_top20"],
                         ["precedent:A", "decision:B"])


if __name__ == "__main__":
    unittest.main()
