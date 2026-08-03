"""P0-2: pull dedup / backfill / ranked preview 검증 (mock embedder)"""
import numpy as np
import pytest
from src.agent.retriever import PullRetriever, RetrieverConfig


class MockRetriever(PullRetriever):
    """임베딩 endpoint 없이 동작하도록 _embed_batch를 오버라이드.

    각 문서 d{i}는 축 i의 one-hot 벡터. query 'q{i}'는 축 i를 향한다.
    -> similarity 순서가 결정적이라 rank/backfill을 검증할 수 있다.
    """

    def __init__(self, n_docs, config):
        super().__init__(config)
        self.n_docs = n_docs

    def _embed_batch(self, texts, batch_size=256):
        out = []
        for t in texts:
            vec = np.zeros(self.n_docs)
            # 'q3' 또는 'd3' 형태에서 인덱스 추출
            idx = int("".join(ch for ch in t if ch.isdigit()) or 0)
            # query는 해당 축을 강하게, 인접 축을 약하게 향하도록 구성
            for j in range(self.n_docs):
                vec[j] = 1.0 / (1 + abs(j - idx))
            out.append(vec)
        return out


def build(n=10):
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock", top_k=3,
                          query_instruction=None)
    r = MockRetriever(n, cfg)
    docs = [{"_id": f"d{i}", "title": f"d{i}", "text": f"doc {i}"} for i in range(n)]
    r.index(docs)
    return r


def test_pull_returns_ranked_preview():
    r = build()
    out = r.pull("q0", top_k=3)
    assert out["requested"] == 3
    assert len(out["results"]) == 3
    ranks = [x["rank"] for x in out["results"]]
    assert ranks == sorted(ranks)  # rank 오름차순
    assert all("doc_id" in x and "score" in x for x in out["results"])


def test_pull_excludes_workspace_and_backfills():
    r = build()
    first = r.pull("q0", top_k=3)
    got_ids = {x["doc_id"] for x in first["results"]}
    # 이미 가진 문서를 제외하고 다시 pull → 겹치지 않는 새 문서로 채움
    second = r.pull("q0", top_k=3, exclude_ids=got_ids)
    second_ids = {x["doc_id"] for x in second["results"]}
    assert got_ids.isdisjoint(second_ids)      # 중복 없음
    assert len(second["results"]) == 3          # backfill로 여전히 3개
    assert second["duplicates_excluded"] >= 1   # top 구간 중복이 카운트됨


def test_pull_dynamic_top_k():
    r = build()
    out = r.pull("q0", top_k=5)
    assert len(out["results"]) == 5


@pytest.mark.parametrize("bad", [0, -1, 201, 1.5, True])
def test_pull_rejects_invalid_dynamic_top_k(bad):
    with pytest.raises(ValueError):
        build().pull("q0", top_k=bad)


def test_cache_key_changes_with_prefix():
    r = build()
    ids = ["d0", "d1"]
    texts = ["a", "b"]
    k1 = r._cache_key(ids, texts)
    r.config.use_prefix = True
    k2 = r._cache_key(ids, texts)
    assert k1 != k2  # prefix 여부가 cache key에 반영 (P2-3)


def _signed_retriever(cosines: dict, taxonomy: dict):
    """cosine 값을 직접 지정한 retriever (음수 score 검증용)."""
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock")
    r = PullRetriever(cfg)
    r.doc_ids = list(cosines)
    r.embedding_matrix = np.array(
        [[c, float(np.sqrt(1 - c * c))] for c in cosines.values()]
    )
    r.doc_taxonomy = taxonomy
    r._embed_batch = lambda texts, batch_size=256: [np.array([1.0, 0.0])]
    return r


@pytest.mark.parametrize("match_cos,other_cos", [
    (-0.15, -0.2),   # 음수 구간: 곱셈 boost면 일치 문서가 역전당한다
    (0.0, -0.05),
    (0.3, 0.25),
])
def test_taxonomy_boost_never_demotes_matching_doc(match_cos, other_cos):
    r = _signed_retriever(
        {"m": match_cos, "o": other_cos}, {"m": {"L1": "T"}}
    )
    base = r._dense_scores("q")
    boosted = r._dense_scores("q", taxonomy_filter={"L1": "T"})
    assert boosted[0] >= base[0] - 1e-9   # 일치 문서 점수는 감소하지 않는다
    assert boosted[0] > boosted[1]        # 원래 우위였던 일치 문서가 역전되지 않는다


def test_cache_key_changes_with_text():
    r = build()
    k1 = r._cache_key(["d0"], ["original text"])
    k2 = r._cache_key(["d0"], ["changed text"])
    assert k1 != k2  # 같은 doc_id라도 본문이 바뀌면 무효화


# ---------------------------------------------------------------- 무추론 라우팅 (self_routed / doc_first)

# 2D 합성 코퍼스 — 문서 A(3청크, x축), 문서 B(2청크, y축), Legal(2청크, 대각)
_ROUTE_DOCS = {
    "a0": [1.0, 0.0], "a1": [0.95, 0.1], "a2": [0.9, 0.2],
    "b0": [0.0, 1.0], "b1": [0.1, 0.95],
    "l0": [-0.6, 0.6], "l1": [-0.55, 0.65],   # 이질 도메인 distractor — 별도 방향
}
_ROUTE_TAX = {
    "a0": {"L1": "A"}, "a1": {"L1": "A"}, "a2": {"L1": "A"},
    "b0": {"L1": "B"}, "b1": {"L1": "B"},
    "l0": {"L1": "Legal"}, "l1": {"L1": "Legal"},
}
# 질의 텍스트 → 임베딩 (mock)
_ROUTE_QUERIES = {
    "qa": [1.0, 0.0],        # 문서 A 방향
    "q_mixed": [0.6, 0.8],   # top1은 B지만 다수(3/5)는 A — 투표 복구 검증용
    "q_diag": [-0.5, 0.5],   # Legal centroid 방향
}


def _routing_retriever(backend, **overrides):
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock", top_k=5,
                          backend=backend, query_instruction=None, **overrides)
    r = PullRetriever(cfg)
    r.doc_ids = list(_ROUTE_DOCS)
    r.embedding_matrix = np.array(list(_ROUTE_DOCS.values()))
    r.doc_taxonomy = dict(_ROUTE_TAX)
    r._embed_batch = lambda texts, batch_size=256: [
        np.array(_ROUTE_QUERIES[t]) for t in texts
    ]
    return r


def test_self_route_votes_majority_and_reorders():
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=1)
    out = r.pull("qa", top_k=7)
    # top5 = a0,a1,a2,b1,b0 → A 3표 > B 2표 → A 파티션 우선
    assert r.last_self_route["picked"] == ["A"]
    assert [x["doc_id"] for x in out["results"][:3]] == ["a0", "a1", "a2"]
    # backfill: 비매치 문서도 소실되지 않는다
    assert {x["doc_id"] for x in out["results"]} == set(_ROUTE_DOCS)


def test_self_route_vote_recovers_from_wrong_top1():
    # dense rank-1(b1)이 오답 문서여도 다수 득표 문서(A)로 복구되는 핵심 시나리오
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=1,
                           self_route_weight="uniform")
    out = r.pull("q_mixed", top_k=3)
    assert r.last_self_route["picked"] == ["A"]
    assert all(x["doc_id"].startswith("a") for x in out["results"])


def test_self_route_rank_weight_follows_top_ranks():
    # rank 가중(1/rank)은 상위 rank를 우대 — 같은 질의에서 B가 이긴다
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=1,
                           self_route_weight="rank")
    r.pull("q_mixed", top_k=3)
    assert r.last_self_route["picked"] == ["B"]


def test_self_route_min_share_falls_back_to_dense():
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=1,
                           self_route_weight="uniform", self_route_min_share=0.7)
    out = r.pull("q_mixed", top_k=3)   # top_share = 3/5 = 0.6 < 0.7 → 폴백
    assert r.last_self_route["fallback"] is True
    assert out["results"][0]["doc_id"] == "b1"  # 순수 dense 순서 유지


def test_self_route_top2_or_match():
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=2)
    out = r.pull("qa", top_k=5)
    assert r.last_self_route["picked"] == ["A", "B"]
    # A∪B(5청크)가 앞, Legal이 backfill
    assert {x["doc_id"] for x in out["results"][:5]} == {"a0", "a1", "a2", "b0", "b1"}


def test_doc_first_centroid_picks_right_partition():
    r = _routing_retriever("doc_first", doc_first_top_n=1)
    out = r.pull("qa", top_k=7)
    assert r.last_doc_first["picked"][0] == "A"
    assert [x["doc_id"] for x in out["results"][:3]] == ["a0", "a1", "a2"]
    assert {x["doc_id"] for x in out["results"]} == set(_ROUTE_DOCS)  # gold 미소실


def test_doc_first_exclude_l1_skips_legal_centroid():
    r = _routing_retriever("doc_first", doc_first_top_n=1,
                           doc_first_exclude_l1=("Legal",))
    r.pull("q_diag", top_k=3)
    assert r.last_doc_first["picked"][0] != "Legal"
    # 제외 없이 돌리면 Legal centroid가 1위인 질의임을 교차 확인
    r2 = _routing_retriever("doc_first", doc_first_top_n=1)
    r2.pull("q_diag", top_k=3)
    assert r2.last_doc_first["picked"][0] == "Legal"


def test_doc_first_partitioned_falls_back_when_partition_small():
    # partitioned 의미론: 파티션(A=3) < gather_k(5) → dense 전역 폴백 + 진단 기록
    r = _routing_retriever("doc_first", doc_first_top_n=1, doc_first_backfill=False)
    out = r.pull("qa", top_k=5)
    assert r.last_doc_first.get("fallback") is True
    assert len(out["results"]) == 5


def test_doc_first_partitioned_slices_partition_only():
    r = _routing_retriever("doc_first", doc_first_top_n=1, doc_first_backfill=False)
    out = r.pull("qa", top_k=2)   # 파티션(3) >= gather_k(2) → 슬라이스 경로
    assert all(x["doc_id"].startswith("a") for x in out["results"])


def test_rank_all_applies_self_routing():
    r = _routing_retriever("self_routed", self_route_probe_k=5, self_route_top_n=1)
    ranked = r.rank_all("q_mixed")
    assert r.last_self_route["picked"] == ["A"]
    assert all(row["doc_id"].startswith("a") for row in ranked[:3])
    assert len(ranked) == len(_ROUTE_DOCS)


# ---------------------------------------------------------------- reranker 실패 가시화

def _reranking_retriever(allow_fallback=False, n=5):
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock", top_k=2,
                          reranker_url="http://mock-rerank", reranker_model="rr",
                          allow_reranker_fallback=allow_fallback,
                          query_instruction=None)
    r = MockRetriever(n, cfg)
    docs = [{"_id": f"d{i}", "title": f"d{i}", "text": f"doc {i}"} for i in range(n)]
    r.index(docs)
    return r


def test_rerank_failure_raises_by_default(monkeypatch):
    # 실험 실행에서 reranker 장애가 조용한 fallback으로 숨겨지면
    # arm 라벨(“+Reranker”)과 실제 구성이 달라진다 — 기본은 명시적 실패
    import requests
    from src.agent import retriever as mod
    from src.retrieval import RerankerError

    def post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("reranker down")

    monkeypatch.setattr(mod.requests, "post", post)
    with pytest.raises(RerankerError):
        _reranking_retriever().pull("q0")


def test_rerank_failure_fallback_only_when_allowed(monkeypatch):
    import requests
    from src.agent import retriever as mod

    def post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("reranker down")

    monkeypatch.setattr(mod.requests, "post", post)
    out = _reranking_retriever(allow_fallback=True).pull("q0")
    assert out["reranker_used"] is False
    assert "ConnectionError" in out["reranker_error"]
    assert len(out["results"]) == 2  # fallback이어도 결과는 유지


def test_pull_reports_reranker_success(monkeypatch):
    from src.agent import retriever as mod

    class FakeResp:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [
                {"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(self.n)
            ]}

    def post(url, json=None, timeout=None, **kwargs):
        return FakeResp(len(json["documents"]))

    monkeypatch.setattr(mod.requests, "post", post)
    out = _reranking_retriever().pull("q0")
    assert out["reranker_used"] is True
    assert out["reranker_error"] is None
