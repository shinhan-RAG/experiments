"""Part 6 문서 내부 검색과 결과 계측 계약."""

import numpy as np

from run_experiment import (
    part6_build_aug_texts,
    part6_rank_metrics,
    summarize_part6_rows,
)
from src.agent.retriever import PullRetriever, RetrieverConfig
from src.retrieval.bm25 import BM25


class FixedRetriever(PullRetriever):
    def _embed_batch(self, texts, batch_size=256):
        vectors = {
            "d0": np.array([1.0, 0.0]),
            "d1": np.array([0.8, 0.2]),
            "d2": np.array([0.0, 1.0]),
        }
        return [vectors[next(key for key in vectors if key in text)] for text in texts]


def _retriever(backend="dense"):
    retriever = FixedRetriever(RetrieverConfig(
        embedding_url="unused",
        embedding_model="fixed",
        backend=backend,
        query_instruction=None,
    ))
    retriever.index([
        {"_id": "d0", "title": "d0", "text": "outside keyword"},
        {"_id": "d1", "title": "d1", "text": "inside keyword"},
        {"_id": "d2", "title": "d2", "text": "inside other"},
    ])
    return retriever


def test_rank_candidates_never_returns_chunks_outside_selected_document():
    retriever = _retriever()
    retriever._embed_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("precomputed query embedding must be reused")
    )
    ranked = retriever.rank_candidates(
        "keyword", ["d1", "d2"], query_embedding=np.array([1.0, 0.0])
    )
    assert [row["doc_id"] for row in ranked] == ["d1", "d2"]


def test_hybrid_candidate_ranking_excludes_external_lexical_hit():
    retriever = _retriever(backend="hybrid_rrf")
    ranked = retriever.rank_candidates(
        "keyword", {"d1", "d2"}, query_embedding=np.array([0.0, 1.0])
    )
    assert {row["doc_id"] for row in ranked} == {"d1", "d2"}
    assert all(row["doc_id"] != "d0" for row in ranked)


def test_bm25_subset_statistics_match_an_independently_fitted_subset():
    docs = [
        {"_id": "outside", "title": "", "text": "term term term"},
        {"_id": "a", "title": "", "text": "term alpha"},
        {"_id": "b", "title": "", "text": "beta"},
    ]
    global_index = BM25()
    global_index.fit(docs)
    subset_index = BM25()
    subset_index.fit(docs[1:])
    restricted = global_index.search(
        "term", top_k=2, allowed_ids={"a", "b"}, subset_statistics=True
    )
    expected = subset_index.search("term", top_k=2)
    assert restricted == expected


def test_part6_metrics_and_latency_summary():
    row = {
        **part6_rank_metrics(["x", "gold", "z"], {"gold"}),
        "query_embedding_seconds": 2.0,
        "ranking_seconds": 0.25,
        "probe_latency_seconds": 2.25,
    }
    assert row["gold_rank"] == 2
    assert row["mrr"] == 0.5
    assert row["recall_at_1"] == 0.0
    assert row["recall_at_3"] == 1.0
    summary = summarize_part6_rows([row])
    assert summary["p50_retrieval_seconds"] == 2.25
    assert summary["avg_query_embedding_seconds"] == 2.0
    assert summary["avg_ranking_seconds"] == 0.25


def test_part6_condition_can_select_a_distinct_metadata_source():
    corpus = [{"_id": "c1"}, {"_id": "c2"}]
    maps = {
        "metadata": {"c1": {"문서유형": "약관"}},
        "chunk_metadata": {
            "c1": {"핵심주제": "보험금 청구", "검색키워드": ["청구", "서류"]},
            "c2": {"핵심주제": "해지환급금"},
        },
    }
    rendered = part6_build_aug_texts(
        corpus,
        {"index_metadata": True, "metadata_source": "chunk_metadata"},
        maps,
    )
    assert rendered == {
        "c1": "핵심주제:보험금 청구 검색키워드:청구 서류",
        "c2": "핵심주제:해지환급금",
    }
