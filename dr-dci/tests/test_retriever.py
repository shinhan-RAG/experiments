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


def test_cache_key_changes_with_text():
    r = build()
    k1 = r._cache_key(["d0"], ["original text"])
    k2 = r._cache_key(["d0"], ["changed text"])
    assert k1 != k2  # 같은 doc_id라도 본문이 바뀌면 무효화
