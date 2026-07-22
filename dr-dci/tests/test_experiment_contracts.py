import unittest

import numpy as np

from src.agent.dci_agent import DCIAgent
from src.agent.retriever import PullRetriever, RetrieverConfig
from src.eval.comparison import compare_paired_results
from src.eval.judge import Judge, compute_metrics
from src.retrieval import BM25, embedding_cache_key


class StaticEmbeddingRetriever(PullRetriever):
    def __init__(self, config, query_embedding):
        super().__init__(config)
        self.query_embedding = np.asarray(query_embedding, dtype=float)

    def _embed_batch(self, texts, batch_size=256):
        return [self.query_embedding.copy() for _ in texts]


class StaticPullRetriever:
    def pull(self, query, taxonomy_filter=None):
        return [
            {"doc_id": "d1", "score": 1.0},
            {"doc_id": "d2", "score": 0.5},
        ]


class ExperimentContractTests(unittest.TestCase):
    def test_judge_does_not_match_incorrect_as_correct(self):
        self.assertEqual(Judge.parse_judgment("correct"), "correct")
        self.assertEqual(Judge.parse_judgment("incorrect"), "incorrect")
        self.assertEqual(Judge.parse_judgment("probably correct"), "error")

    def test_embedding_cache_key_covers_content_and_model(self):
        base = embedding_cache_key(
            namespace="pull",
            model="model-a",
            use_prefix=False,
            doc_ids=["d1"],
            texts=["alpha"],
        )
        changed_text = embedding_cache_key(
            namespace="pull",
            model="model-a",
            use_prefix=False,
            doc_ids=["d1"],
            texts=["beta"],
        )
        changed_model = embedding_cache_key(
            namespace="pull",
            model="model-b",
            use_prefix=False,
            doc_ids=["d1"],
            texts=["alpha"],
        )
        self.assertNotEqual(base, changed_text)
        self.assertNotEqual(base, changed_model)

    def test_bm25_matches_exact_korean_term(self):
        bm25 = BM25()
        bm25.fit([
            {"_id": "d1", "title": "", "text": "사망보험금 지급 조건"},
            {"_id": "d2", "title": "", "text": "보험 계약 일반 안내"},
        ])
        self.assertEqual(bm25.search("사망보험금", top_k=1)[0]["doc_id"], "d1")

    def test_hybrid_pull_adds_lexical_candidate_without_changing_top_k(self):
        documents = [
            {"_id": "lexical", "title": "", "text": "ZXQ991 exact clause"},
            {"_id": "dense", "title": "", "text": "general semantic match"},
        ]

        dense = StaticEmbeddingRetriever(
            RetrieverConfig("unused", "model", top_k=1, backend="dense", bm25_top_k=2),
            query_embedding=[1.0, 0.0],
        )
        dense.doc_ids = ["lexical", "dense"]
        dense.embedding_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(dense.pull("ZXQ991")[0]["doc_id"], "dense")

        hybrid = StaticEmbeddingRetriever(
            RetrieverConfig("unused", "model", top_k=1, backend="hybrid_rrf", bm25_top_k=2),
            query_embedding=[1.0, 0.0],
        )
        hybrid.doc_ids = ["lexical", "dense"]
        hybrid.embedding_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        hybrid.bm25.fit(documents)
        results = hybrid.pull("ZXQ991")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "lexical")

    def test_workspace_limit_is_applied_by_agent(self):
        corpus = {
            "d1": {"title": "one", "text": "first"},
            "d2": {"title": "two", "text": "second"},
        }
        agent = DCIAgent(
            llm_url="unused",
            model_name="unused",
            retriever=StaticPullRetriever(),
            corpus=corpus,
            max_turns=2,
            workspace_max_docs=1,
        )
        responses = iter([
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "pull",
                        "arguments": '{"query": "test"}',
                    },
                }],
            },
            {"content": "done", "tool_calls": None},
        ])
        agent._call_llm = lambda messages: next(responses)
        result = agent.run("question")
        self.assertEqual(result["workspace_docs"], ["d1"])
        self.assertEqual(result["retrieved_candidates"], 2)
        self.assertEqual(result["added_documents"], 1)
        self.assertEqual(result["turns"], 2)

    def test_metrics_exclude_judge_transport_errors_from_accuracy(self):
        metrics = compute_metrics([
            {
                "judgment": "correct",
                "gold_recall": 1.0,
                "pull_count": 2,
                "retrieved_candidates": 40,
                "workspace_docs": ["d1"],
                "turns": 3,
                "latency_seconds": 1.0,
            },
            {
                "judgment": "error",
                "gold_recall": 0.0,
                "pull_count": 1,
                "retrieved_candidates": 20,
                "workspace_docs": [],
                "turns": 2,
                "latency_seconds": 2.0,
            },
        ])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["judged_n"], 1)
        self.assertEqual(metrics["judge_error_n"], 1)

    def test_paired_comparison_aligns_queries_and_is_deterministic(self):
        control = [
            {"query_id": "q2", "gold_recall": 0.0},
            {"query_id": "q1", "gold_recall": 0.5},
        ]
        treatment = [
            {"query_id": "q1", "gold_recall": 1.0},
            {"query_id": "q2", "gold_recall": 0.5},
        ]
        first = compare_paired_results(control, treatment, seed=42)
        second = compare_paired_results(control, treatment, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_query_count"], 2)
        self.assertEqual(
            first["treatment_minus_control"]["mean_delta"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
