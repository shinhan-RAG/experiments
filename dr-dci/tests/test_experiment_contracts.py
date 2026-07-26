import math
import unittest
from unittest.mock import patch

import numpy as np

from src.agent.dci_agent import DCIAgent
from src.agent.retriever import PullRetriever, RetrieverConfig, select_top_indices
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
    def pull(self, query, taxonomy_filter=None, top_k=None, exclude_ids=None):
        rows = [
            {"doc_id": "d1", "score": 1.0},
            {"doc_id": "d2", "score": 0.5},
        ]
        excluded = exclude_ids or set()
        rows = [row for row in rows if row["doc_id"] not in excluded]
        limit = top_k or len(rows)
        return {"results": rows[:limit], "requested": limit,
                "duplicates_excluded": len(excluded & {"d1", "d2"})}


class TelemetryPullRetriever:
    def pull(self, query, taxonomy_filter=None):
        from src.agent.retriever import PullResult

        return PullResult(
            [{"doc_id": "d1", "score": 1.0}],
            {
                "taxonomy_boost_eligible_documents": 1,
                "taxonomy_boosted_positive_score_documents": 1,
                "taxonomy_boosted_returned_documents": 1,
                "taxonomy_boost_rank_changed": True,
                "taxonomy_boost_top_k_entered_documents": 1,
                "taxonomy_boost_top_k_exited_documents": 1,
                "taxonomy_boost_target_score_count": 1,
                "taxonomy_boost_target_negative_score_count": 0,
                "taxonomy_boost_target_score_min": 0.4,
                "taxonomy_boost_target_score_max": 0.4,
                "taxonomy_boost_rank_changes": [{
                    "doc_id": "d1", "rank_before": 2, "rank_after": 1,
                    "score_before": 0.4, "score_after": 0.6,
                }],
            },
        )


class ExperimentContractTests(unittest.TestCase):
    def test_agent_and_judge_use_configured_generation_controls(self):
        class Response:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {
                    "choices": [{"message": {"content": "correct", "tool_calls": None}}],
                    "usage": {},
                }

        agent = DCIAgent(
            llm_url="https://example.test/v1/chat/completions",
            model_name="agent-model",
            retriever=StaticPullRetriever(),
            corpus={},
            temperature=0.2,
            llm_max_tokens=2048,
            llm_seed=17,
        )
        with patch("src.agent.dci_agent.requests.post", return_value=Response()) as post:
            agent._call_llm([])
        agent_payload = post.call_args.kwargs["json"]
        self.assertEqual(agent_payload["temperature"], 0.2)
        self.assertEqual(agent_payload["max_tokens"], 2048)
        self.assertEqual(agent_payload["seed"], 17)

        judge = Judge(
            llm_url="https://example.test/v1/chat/completions",
            model_name="judge-model",
            prompt_template="{query} {reference_answer} {candidate_answer}",
            temperature=0.3,
            max_tokens=512,
            llm_seed=19,
        )
        with patch("src.eval.judge.requests.post", return_value=Response()) as post:
            self.assertEqual(judge.evaluate_accuracy("q", "r", "a"), "correct")
        judge_payload = post.call_args.kwargs["json"]
        self.assertEqual(judge_payload["temperature"], 0.3)
        self.assertEqual(judge_payload["max_tokens"], 512)
        self.assertEqual(judge_payload["seed"], 19)

    def test_judge_does_not_match_incorrect_as_correct(self):
        self.assertEqual(Judge.parse_judgment("correct"), "correct")
        self.assertEqual(Judge.parse_judgment("incorrect"), "incorrect")
        self.assertEqual(Judge.parse_judgment("probably correct"), "format_error")

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
        self.assertEqual(dense.pull("ZXQ991")["results"][0]["doc_id"], "dense")

        hybrid = StaticEmbeddingRetriever(
            RetrieverConfig("unused", "model", top_k=1, backend="hybrid_rrf", bm25_top_k=2),
            query_embedding=[1.0, 0.0],
        )
        hybrid.doc_ids = ["lexical", "dense"]
        hybrid.embedding_matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        hybrid.bm25.fit(documents)
        results = hybrid.pull("ZXQ991")
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0]["doc_id"], "lexical")

    def test_hybrid_baseline_uses_the_same_query_instruction_contract(self):
        from src.hybrid.pipeline import HybridRAG

        pipeline = HybridRAG(
            embedding_url="unused", embedding_model="model",
            reranker_url="", reranker_model="",
            llm_url="unused", llm_model="model",
            dense_top_k=1, bm25_top_k=1, rerank_top_k=1,
            query_instruction="PREFIX: ",
        )
        pipeline.doc_ids = ["d1"]
        pipeline.embedding_matrix = np.asarray([[1.0, 0.0]])
        pipeline.corpus = {"d1": {"title": "t", "text": "body"}}
        pipeline.bm25.fit([{"_id": "d1", "title": "t", "text": "body"}])
        seen = []

        def embed(texts, batch_size=256):
            seen.extend(texts)
            return [np.asarray([1.0, 0.0]) for _ in texts]

        pipeline._embed_batch = embed
        pipeline._rerank = lambda query, candidates: candidates
        pipeline._generate_answer = lambda query, context: "answer"
        pipeline.run("question")
        self.assertEqual(seen, ["PREFIX: question"])

    def test_taxonomy_boost_only_increases_positive_scores_and_reports_rank_effect(self):
        retriever = StaticEmbeddingRetriever(
            RetrieverConfig("unused", "model", top_k=1, backend="dense", taxonomy_boost=1.5),
            query_embedding=[1.0, 0.0],
        )
        retriever.doc_ids = ["boosted", "plain", "negative"]
        retriever.embedding_matrix = np.asarray([
            [0.4, np.sqrt(1 - 0.4 ** 2)],
            [0.5, np.sqrt(1 - 0.5 ** 2)],
            [-0.2, np.sqrt(1 - 0.2 ** 2)],
        ])
        retriever.doc_taxonomy = {
            "boosted": {"L1": "Treatment"},
            "negative": {"L1": "Treatment"},
        }

        results = retriever.pull("query", taxonomy_filter={"L1": "Treatment"})

        self.assertEqual([row["doc_id"] for row in results], ["boosted"])
        telemetry = results.telemetry
        self.assertEqual(telemetry["taxonomy_boost_target_score_count"], 2)
        self.assertEqual(telemetry["taxonomy_boost_target_negative_score_count"], 1)
        self.assertEqual(telemetry["taxonomy_boosted_positive_score_documents"], 1)
        self.assertEqual(telemetry["taxonomy_boost_target_score_min"], -0.2)
        self.assertEqual(telemetry["taxonomy_boost_target_score_max"], 0.4)
        self.assertEqual(telemetry["taxonomy_boost_top_k_entered_documents"], 1)
        self.assertEqual(telemetry["taxonomy_boost_top_k_exited_documents"], 1)
        self.assertTrue(telemetry["taxonomy_boost_rank_changed"])
        self.assertEqual(telemetry["taxonomy_boost_rank_changes"], [
            {
                "doc_id": "boosted", "rank_before": 2, "rank_after": 1,
                "score_before": 0.4, "score_after": 0.6,
            },
            {
                "doc_id": "plain", "rank_before": 1, "rank_after": 2,
                "score_before": 0.5, "score_after": 0.5,
            },
        ])

    def test_baseline_pull_does_not_duplicate_ranking_for_taxonomy_telemetry(self):
        retriever = StaticEmbeddingRetriever(
            RetrieverConfig("unused", "model", top_k=1, backend="dense"),
            query_embedding=[1.0, 0.0],
        )
        retriever.doc_ids = ["d1", "d2", "d3"]
        retriever.embedding_matrix = np.asarray([
            [0.8, 0.6], [0.5, np.sqrt(1 - 0.5 ** 2)], [0.2, np.sqrt(1 - 0.2 ** 2)],
        ])

        with patch("src.agent.retriever.np.argsort", wraps=np.argsort) as argsort:
            results = retriever.pull("query")

        self.assertEqual(argsort.call_count, 1)
        self.assertEqual(results.telemetry["taxonomy_boost_telemetry_seconds"], 0.0)

    def test_bounded_top_selection_matches_stable_full_ranking(self):
        scores = np.asarray([0.2, 0.9, 0.9, -0.1, 0.5, 0.5, 0.3])

        selected = select_top_indices(scores, 5)

        expected = np.argsort(-scores, kind="stable")[:5]
        np.testing.assert_array_equal(selected, expected)

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
            min_pulls=1,
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

    def test_taxonomy_telemetry_records_an_effectual_boost_population(self):
        import json as _json

        agent = DCIAgent(
            llm_url="unused",
            model_name="unused",
            retriever=StaticPullRetriever(),
            corpus={
                "d1": {"title": "one", "text": "first"},
                "d2": {"title": "two", "text": "second"},
            },
            taxonomy_data={
                "d1": {"L1": "Treatment"},
                "d2": {"L1": "Diagnosis"},
            },
            max_turns=2,
        )
        responses = iter([
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "pull",
                        "arguments": _json.dumps({
                            "query": "test", "taxonomy_filter": {"L1": "Treatment"},
                        }),
                    },
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-2",
                    "function": {"name": "answer", "arguments": '{"text": "done"}'},
                }],
            },
        ])
        agent._call_llm = lambda messages: next(responses)

        result = agent.run("question")

        self.assertEqual(result["taxonomy_filtered_pulls"], 1)
        self.assertEqual(result["taxonomy_boost_eligible_documents"], 1)
        self.assertEqual(result["taxonomy_boosted_returned_documents"], 1)

    def test_agent_keeps_rank_effect_trace_for_each_executed_pull(self):
        import json as _json

        agent = DCIAgent(
            llm_url="unused",
            model_name="unused",
            retriever=TelemetryPullRetriever(),
            corpus={"d1": {"title": "one", "text": "first"}},
            taxonomy_data={"d1": {"L1": "Treatment"}},
            max_turns=2,
        )
        responses = iter([
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "pull",
                        "arguments": _json.dumps({
                            "query": "test", "taxonomy_filter": {"L1": "Treatment"},
                        }),
                    },
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-2",
                    "function": {"name": "answer", "arguments": '{"text": "done"}'},
                }],
            },
        ])
        agent._call_llm = lambda messages: next(responses)

        result = agent.run("question")

        self.assertEqual(result["taxonomy_boost_rank_changed_pulls"], 1)
        self.assertEqual(result["taxonomy_boosted_positive_score_documents"], 1)
        self.assertEqual(result["taxonomy_boost_top_k_entered_documents"], 1)
        self.assertEqual(result["taxonomy_boost_target_negative_score_count"], 0)
        self.assertEqual(result["pull_traces"][0]["taxonomy_boost_rank_changes"], [{
            "doc_id": "d1", "rank_before": 2, "rank_after": 1,
            "score_before": 0.4, "score_after": 0.6,
        }])
        self.assertEqual(result["pull_traces"][0]["workspace_document_ids_after"], ["d1"])

    def test_single_pull_ablation_keeps_only_the_first_agent_query(self):
        import json as _json

        agent = DCIAgent(
            llm_url="unused",
            model_name="unused",
            retriever=StaticPullRetriever(),
            corpus={
                "d1": {"title": "one", "text": "first"},
                "d2": {"title": "two", "text": "second"},
            },
            max_turns=3,
            single_pull=True,
        )
        responses = iter([
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "function": {"name": "pull", "arguments": _json.dumps({"query": "first"})},
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-2",
                    "function": {"name": "pull", "arguments": _json.dumps({"query": "second"})},
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call-3",
                    "function": {"name": "answer", "arguments": '{"text": "done"}'},
                }],
            },
        ])
        agent._call_llm = lambda messages: next(responses)

        result = agent.run("question")

        self.assertEqual(result["pull_count"], 1)
        self.assertEqual(result["tool_call_counts"]["pull"], 2)
        self.assertEqual(result["pull_queries"], ["first"])

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

    def test_ndcg_uses_graded_gains_with_log2_discount(self):
        from src.eval.retrieval_metrics import ndcg_at_k, precision_at_k

        ranked = ["a", "b", "c"]
        gains = {"a": 2.0, "c": 1.0, "x": 2.0}
        dcg = 2.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 2.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertAlmostEqual(ndcg_at_k(ranked, gains, 10), dcg / idcg)
        # 완전 순위는 1.0, 미검색은 0.0
        self.assertAlmostEqual(ndcg_at_k(["a"], {"a": 1.0}, 10), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["z"], {"a": 1.0}, 10), 0.0)
        self.assertAlmostEqual(precision_at_k(ranked, {"a", "c", "x"}, 20), 2 / 20)

    def test_rank_metrics_binary_fallback_matches_graded_keys(self):
        from src.eval.retrieval_metrics import rank_metrics

        binary = rank_metrics(["a", "b"], {"a"})
        graded = rank_metrics(["a", "b"], {"a"}, gains={"a": 2.0})
        self.assertEqual(set(binary), set(graded))
        self.assertAlmostEqual(binary["ndcg_at_10"], 1.0)   # 1위 완전 순위
        self.assertAlmostEqual(graded["ndcg_at_10"], 1.0)   # 배율 불변성
        self.assertAlmostEqual(binary["precision_at_20"], 1 / 20)


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

    def test_probe_records_graded_ndcg_when_gains_given(self):
        from run_experiment import run_pull_probe

        retriever = StaticPullRetriever()
        queries = [{"_id": "q1", "text": "판정 질의"}]
        rows = run_pull_probe(retriever, queries, {"q1": {"d2"}},
                              query_gains={"q1": {"d2": 2.0}})
        # d2가 2위: DCG=2/log2(3), IDCG=2/log2(2)
        self.assertAlmostEqual(rows[0]["ndcg_at_10"],
                               (2.0 / math.log2(3)) / 2.0)
        self.assertAlmostEqual(rows[0]["precision_at_20"], 1 / 20)

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
            min_pulls=1,
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
        self.assertEqual(result["taxonomy_filtered_pulls"], 0)
        self.assertEqual(result["system_fingerprints"], [])
        self.assertEqual(result["tool_calls_total"], 3)      # turns(2)와 분리 계측
        self.assertEqual(result["turns"], 2)
        self.assertEqual(result["llm_prompt_tokens"], 30)
        self.assertEqual(result["llm_completion_tokens"], 10)
        self.assertEqual(result["pull_count"], 1)
        self.assertEqual(result["answer"], "답")


class FailureHandlingTests(unittest.TestCase):
    def test_failed_query_row_has_metric_defaults(self):
        from run_experiment import failed_query_row

        row = failed_query_row(
            {"_id": "q9", "title": "질의"}, RuntimeError("boom"),
            requested_features={"taxonomy": True}, single_pull=False,
        )
        self.assertEqual(row["query_id"], "q9")
        self.assertTrue(row["failed"])
        self.assertEqual(row["termination_reason"], "harness_error")
        self.assertIn("boom", row["error"])
        self.assertEqual(row["answer"], "")
        # 집계 함수가 기대하는 필드가 기본값으로 존재해야 한다
        for key in ("gold_recall", "pull_count", "retrieved_candidates",
                    "workspace_docs", "turns", "latency_seconds",
                    "rule_violations", "trace"):
            self.assertIn(key, row)
        compute_metrics([row])  # 예외 없이 집계 가능

    def test_arm_aborts_when_failure_rate_exceeds_threshold(self):
        from run_experiment import check_arm_failure_rate

        ok = {"termination_reason": "answered"}
        bad = {"termination_reason": "llm_error"}
        check_arm_failure_rate([ok, ok, ok, ok, bad], threshold=0.2)  # 20%까지 허용
        with self.assertRaises(RuntimeError):
            check_arm_failure_rate([ok, ok, bad, bad], threshold=0.2)

    def test_empty_answer_is_not_sent_to_judge(self):
        from run_experiment import assign_judgment

        class ExplodingJudge:
            def evaluate_accuracy(self, *a, **k):
                raise AssertionError("judge must not be called for empty answers")

        row = {"query_id": "q1", "query_text": "질의", "answer": ""}
        assign_judgment(row, {"q1": "ref"}, ExplodingJudge())
        self.assertEqual(row["judgment"], "agent_error")

        answered = {"query_id": "q2", "query_text": "질의", "answer": "정상"}
        assign_judgment(answered, {}, ExplodingJudge())
        self.assertEqual(answered["judgment"], "n/a")


class RerankerVisibilityTests(unittest.TestCase):
    @staticmethod
    def _pipeline(reranker_url="http://mock-rerank", allow=False):
        from src.hybrid.pipeline import HybridRAG

        p = HybridRAG(
            embedding_url="u", embedding_model="m",
            reranker_url=reranker_url, reranker_model="rr",
            llm_url="u", llm_model="m",
            dense_top_k=1, bm25_top_k=1, rerank_top_k=1,
            allow_reranker_fallback=allow,
        )
        p.corpus = {"d1": {"title": "t", "text": "body"}}
        return p

    def test_hybrid_rerank_failure_raises_by_default(self):
        import requests
        from unittest.mock import patch

        from src.retrieval import RerankerError

        p = self._pipeline()
        with patch("src.hybrid.pipeline.requests.post",
                   side_effect=requests.exceptions.ConnectionError("down")):
            with self.assertRaises(RerankerError):
                p._rerank("q", [("d1", 1.0)])

    def test_hybrid_rerank_fallback_records_error_when_allowed(self):
        import requests
        from unittest.mock import patch

        p = self._pipeline(allow=True)
        with patch("src.hybrid.pipeline.requests.post",
                   side_effect=requests.exceptions.ConnectionError("down")):
            out = p._rerank("q", [("d1", 1.0)])
        self.assertEqual(out, [("d1", 1.0)])
        self.assertIn("ConnectionError", p.last_rerank_error)

    def test_hybrid_run_skips_rerank_without_url_and_reports_status(self):
        from unittest.mock import patch

        p = self._pipeline(reranker_url="")
        p.doc_ids = ["d1"]
        p.embedding_matrix = np.asarray([[1.0, 0.0]])
        p.bm25.fit([{"_id": "d1", "title": "t", "text": "body"}])
        p._embed_batch = lambda texts, batch_size=256: [np.asarray([1.0, 0.0]) for _ in texts]
        p._generate_answer = lambda query, context: "answer"
        with patch("src.hybrid.pipeline.requests.post",
                   side_effect=AssertionError("no network call expected")):
            result = p.run("question")
        self.assertFalse(result["reranker_used"])
        self.assertIsNone(result["reranker_error"])


class EmbeddingAuthTests(unittest.TestCase):
    def test_embed_batch_sends_bearer_only_when_key_set(self):
        from unittest.mock import patch

        from src.agent.retriever import PullRetriever, RetrieverConfig

        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"data": [{"index": 0, "embedding": [0.0, 1.0]}]}
            return R()

        with patch("src.agent.retriever.requests.post", side_effect=fake_post):
            r = PullRetriever(RetrieverConfig(
                embedding_url="https://api.openai.com/v1/embeddings",
                embedding_model="m", api_key="sk-test"))
            r._embed_batch(["x"])
            self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")

            r2 = PullRetriever(RetrieverConfig(
                embedding_url="http://localhost:8101/v1/embeddings",
                embedding_model="m"))
            r2._embed_batch(["x"])
            self.assertNotIn("Authorization", captured["headers"])
