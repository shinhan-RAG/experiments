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


class RankMetricTests(unittest.TestCase):
    def test_rank_metrics_values_and_truncation(self):
        from src.eval.retrieval_metrics import rank_metrics

        ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "g1"]
        gold = {"b", "g1", "zz"}
        m = rank_metrics(ranked, gold)
        self.assertAlmostEqual(m["recall_at_5"], 1 / 3)     # b만 top-5
        self.assertAlmostEqual(m["recall_at_20"], 2 / 3)    # b·g1, zz는 코퍼스 밖
        self.assertEqual(m["hit_at_5"], 1.0)
        self.assertEqual(m["hit_at_10"], 1.0)

    def test_rank_metrics_reject_empty_gold(self):
        from src.eval.retrieval_metrics import rank_metrics

        with self.assertRaises(ValueError):
            rank_metrics(["a"], set())


class PullProbeTests(unittest.TestCase):
    def test_probe_excludes_queries_without_gold_and_is_deterministic(self):
        from run_experiment import run_pull_probe

        retriever = StaticPullRetriever()
        queries = [
            {"_id": "q1", "text": "판정 질의"},
            {"_id": "q2", "text": "gold 없는 질의"},
        ]
        query_gold = {"q1": {"d2"}}
        rows = run_pull_probe(retriever, queries, query_gold)
        self.assertEqual([r["query_id"] for r in rows], ["q1"])  # 분모=judged만
        self.assertAlmostEqual(rows[0]["recall_at_5"], 1.0)
        self.assertEqual(rows[0]["hit_at_5"], 1.0)
        self.assertEqual(rows[0]["ranked_top20"], ["d1", "d2"])
        again = run_pull_probe(retriever, queries, query_gold)
        self.assertEqual(
            [r["ranked_top20"] for r in rows],
            [r["ranked_top20"] for r in again],
        )

    def test_probe_paired_comparison_shape(self):
        from src.eval.comparison import compare_probe_rows

        control = [{"query_id": "q1", "recall_at_5": 0.0, "recall_at_20": 0.5,
                    "hit_at_5": 0.0, "hit_at_10": 1.0,
                    "probe_latency_seconds": 0.01}]
        treatment = [{"query_id": "q1", "recall_at_5": 0.5, "recall_at_20": 0.5,
                      "hit_at_5": 1.0, "hit_at_10": 1.0,
                      "probe_latency_seconds": 0.02}]
        out = compare_probe_rows(control, treatment, seed=42)
        self.assertEqual(out["paired_query_count"], 1)
        self.assertAlmostEqual(out["recall_at_5"]["mean_delta"], 0.5)
        self.assertIn("ci95_low", out["recall_at_5"])


class AgentAccountingTests(unittest.TestCase):
    def test_agent_counts_tool_calls_and_tokens(self):
        import json as _json

        from src.agent.dci_agent import DCIAgent

        agent = DCIAgent(
            llm_url="http://unused", model_name="stub",
            retriever=StaticPullRetriever(),
            corpus={"d1": {"title": "t", "text": "본문"},
                    "d2": {"title": "t2", "text": "본문2"}},
            max_turns=5, workspace_max_docs=10,
        )
        scripted = [
            {"content": None, "_usage": {"prompt_tokens": 10, "completion_tokens": 3},
             "tool_calls": [
                 {"id": "1", "function": {"name": "pull",
                                          "arguments": _json.dumps({"query": "q"})}},
                 {"id": "2", "function": {"name": "grep",
                                          "arguments": _json.dumps({"pattern": "본문"})}},
             ]},
            {"content": None, "_usage": {"prompt_tokens": 20, "completion_tokens": 7},
             "tool_calls": [
                 {"id": "3", "function": {"name": "answer",
                                          "arguments": _json.dumps({"text": "답"})}},
             ]},
        ]
        agent._call_llm = lambda messages: scripted.pop(0)

        result = agent.run("질의")
        self.assertEqual(result["tool_call_counts"]["pull"], 1)
        self.assertEqual(result["tool_call_counts"]["grep"], 1)
        self.assertEqual(result["tool_call_counts"]["answer"], 1)
        self.assertEqual(result["tool_calls_total"], 3)      # turns(2)와 분리 계측
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["llm_prompt_tokens"], 30)
        self.assertEqual(result["llm_completion_tokens"], 10)
        self.assertEqual(result["pull_count"], 1)
        self.assertEqual(result["answer"], "답")
