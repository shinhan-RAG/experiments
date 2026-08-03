"""측정 설계 교정 B/E: 단계 분리 계측(doc_id)과 taxonomy_bonus 배선 검증"""
import numpy as np
from src.agent.dci_agent import DCIAgent
from src.agent.retriever import PullRetriever, RetrieverConfig
from src.agent.workspace import Workspace


class MockRetriever(PullRetriever):
    def __init__(self, n_docs, config):
        super().__init__(config)
        self.n_docs = n_docs

    def _embed_batch(self, texts, batch_size=256):
        out = []
        for t in texts:
            idx = int("".join(ch for ch in t if ch.isdigit()) or 0)
            vec = np.array([1.0 / (1 + abs(j - idx)) for j in range(self.n_docs)])
            out.append(vec)
        return out


def build_agent(n=6):
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock-stage",
                          top_k=3, query_instruction=None)
    r = MockRetriever(n, cfg)
    docs = [{"_id": f"d{i}", "title": f"d{i}", "text": f"doc {i}"} for i in range(n)]
    r.index(docs)
    corpus = {d["_id"]: d for d in docs}
    return DCIAgent(llm_url="mock", model_name="mock", retriever=r, corpus=corpus)


def test_pull_stats_record_result_doc_ids():
    agent = build_agent()
    ws = Workspace()
    state = {"pull_count": 0, "pull_queries": [], "pull_stats": [],
             "rule_violations": [], "retrieved_candidates": 0,
             "added_documents": 0, "taxonomy_filtered_pulls": 0}
    agent._execute_tool("pull", {"query": "q0"}, ws, state)
    assert len(state["pull_stats"]) == 1
    ids = state["pull_stats"][0]["result_doc_ids"]
    # pull이 가져온 문서가 doc_id로 남는다 — funnel의 ① 단계 복원 근거
    assert ids and all(i.startswith("d") for i in ids)
    assert set(ids) <= set(ws.docs.keys()) | set(ids)


def test_grep_find_summaries_expose_doc_ids():
    grep_summary = DCIAgent._summarize_result(
        "grep", {"matches": [{"doc_id": "d1", "match": "x"},
                             {"doc_id": "d2", "match": "y"}]})
    assert grep_summary["match_doc_ids"] == ["d1", "d2"]
    find_summary = DCIAgent._summarize_result(
        "find", {"matched": 2, "doc_ids": ["d3", "d4"]})
    assert find_summary["doc_ids"] == ["d3", "d4"]


def test_taxonomy_bonus_value_changes_ranking():
    # bonus가 config로 배선되어 실제 순위를 바꾸는지 — E의 스윕 전제 검증
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock-bonus",
                          top_k=3, query_instruction=None, taxonomy_bonus=0.0)
    r = MockRetriever(4, cfg)
    docs = [{"_id": f"d{i}", "title": f"d{i}", "text": f"doc {i}"} for i in range(4)]
    taxonomy = {"d3": {"L1": "cat"}}   # 질의 q0에서 가장 먼 문서를 우대
    r.index(docs, taxonomy=taxonomy)

    top_no_bonus = r.pull("q0", taxonomy_filter={"L1": "cat"})["results"][0]["doc_id"]
    assert top_no_bonus == "d0"        # bonus 0 → 순수 cosine 순위

    r.config.taxonomy_bonus = 10.0     # 재인덱싱 없이 값만 변경(스윕 경로)
    top_big_bonus = r.pull("q0", taxonomy_filter={"L1": "cat"})["results"][0]["doc_id"]
    assert top_big_bonus == "d3"       # 큰 bonus → 매치 문서가 1위로
