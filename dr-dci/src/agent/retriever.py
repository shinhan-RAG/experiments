"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Prefix: 임베딩 보강 (자연어 요약)
- Taxonomy: pull 시 soft boost (네비게이션)
- Metadata/Tags: workspace 탐색 전용

Backend 4종:
- dense          : cosine 순위. taxonomy_filter가 오면 soft bonus(+0.15) — 기존 동작 유지
- hybrid_rrf     : dense + BM25 RRF fusion
- taxonomy_boosted: dense와 동일 경로. 호출자가 질의별 taxonomy_filter(라우팅)를 넘겨
                   soft bonus로 카테고리를 우대한다 — "부드러운 라우팅" arm
- taxonomy_routed: 하드 라우팅. filter에 맞는 카테고리 문서를 (raw cosine 순으로) 먼저,
                   나머지 문서를 그 뒤에 backfill — 오라우팅 시에도 gold가 사라지지 않고
                   순위만 밀린다. filter가 없거나 카테고리에 문서가 없으면 dense로 폴백.

가이드 반영 사항:
- P0-2: pull(query, top_k, exclude_ids) — workspace 기존 문서 제외 + 다음 rank 후보 backfill,
        rank/score 포함 결과와 pull 통계 반환
- P1-8: GTE instruction embedding — query 측에 retrieval instruction 적용
- P2-3: cache key에 model명 + 전체 doc text/prefix hash 포함
"""

import hashlib
import numpy as np
import requests
from dataclasses import dataclass
from pathlib import Path

from src.retrieval import BM25, RerankerError, reciprocal_rank_fusion


CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "embeddings"

# gte-Qwen2-1.5B-instruct 공식 model card 권장 형식 (query 측만 적용, 문서는 그대로)
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
)


@dataclass
class RetrieverConfig:
    embedding_url: str
    embedding_model: str
    top_k: int = 20
    use_prefix: bool = False
    # 덧셈 보너스: 곱셈 boost는 음수 cosine에서 penalty로 반전되므로 사용하지 않는다
    taxonomy_bonus: float = 0.15
    reranker_url: str = None
    reranker_model: str = None
    # reranker 실패 시 원래 순서로 계속할지 여부 — 기본은 명시적 실패
    allow_reranker_fallback: bool = False
    # 모델별 instruction은 비교 arm 모두에 동일하게 주입해야 한다.
    # 기본값을 숨은 GTE 전용 전처리로 두지 않는다.
    query_instruction: str = None
    api_key: str = None
    embedding_api_key: str = None  # 기존 pilot 호환 별칭
    backend: str = "dense"
    bm25_top_k: int = 20
    rrf_k: int = 60
    max_top_k: int = 200


class PullRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.doc_embeddings: dict[str, np.ndarray] = {}
        self.doc_ids: list[str] = []
        self.embedding_matrix: np.ndarray = None  # (N, D) for vectorized search
        self.doc_taxonomy: dict[str, dict] = {}
        self.doc_titles: dict[str, str] = {}
        self.doc_raw_texts: dict[str, str] = {}  # for reranker
        self.bm25 = BM25()
        self.last_rerank_error: str = None

    def _cache_key(self, doc_ids: list[str], texts: list[str]) -> str:
        """모델/전처리/본문이 바뀌면 무효화되는 cache key (P2-3)."""
        h = hashlib.md5()
        h.update(self.config.embedding_model.encode())
        h.update(f"prefix={self.config.use_prefix}".encode())
        h.update(f"n={len(doc_ids)}".encode())
        for did, text in zip(doc_ids, texts):
            h.update(did.encode())
            h.update(text.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    SUPPORTED_BACKENDS = {"dense", "hybrid_rrf", "taxonomy_routed", "taxonomy_boosted"}

    def index(self, documents: list[dict], prefixes: dict = None, taxonomy: dict = None):
        """문서를 인덱싱. 디스크 캐시 활용."""
        if self.config.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported retrieval backend: {self.config.backend}")
        self.doc_embeddings.clear()
        self.doc_taxonomy.clear()
        self.doc_titles.clear()
        self.doc_raw_texts.clear()
        doc_ids = []
        texts = []
        for doc in documents:
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            embed_text = f"{title} {text}"
            if self.config.use_prefix and prefixes and doc_id in prefixes:
                embed_text = f"{prefixes[doc_id]} {embed_text}"
            doc_ids.append(doc_id)
            texts.append(embed_text[:4096])
            self.doc_titles[doc_id] = title
            self.doc_raw_texts[doc_id] = f"{title} {text}"[:4096]
            self.doc_titles[doc_id] = title
            if taxonomy and doc_id in taxonomy:
                self.doc_taxonomy[doc_id] = taxonomy[doc_id]

        if self.config.backend == "hybrid_rrf":
            self.bm25.fit(documents)

        cache_key = self._cache_key(doc_ids, texts)
        cache_path = CACHE_DIR / f"{cache_key}.npz"

        if cache_path.exists():
            print(f"    [Cache HIT] Loading embeddings from {cache_path.name}")
            data = np.load(cache_path, allow_pickle=True)
            cached_ids = list(data["ids"])
            embeddings = data["embeddings"]
            if cached_ids == doc_ids:
                self.doc_ids = doc_ids
                self.embedding_matrix = embeddings
                for i, did in enumerate(doc_ids):
                    self.doc_embeddings[did] = embeddings[i]
                return

        print(f"    [Cache MISS] Embedding {len(texts)} documents (batch=256)...")
        embeddings = self._embed_batch(texts)
        emb_matrix = np.array(embeddings)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(doc_ids, dtype=object), embeddings=emb_matrix)
        print(f"    [Cache SAVED] {cache_path.name}")

        self.doc_ids = doc_ids
        self.embedding_matrix = emb_matrix
        for i, did in enumerate(doc_ids):
            self.doc_embeddings[did] = emb_matrix[i]

    def set_taxonomy(self, taxonomy: dict | None):
        """condition별 taxonomy state를 명시적으로 설정/해제 (P1-5 state 격리)."""
        self.doc_taxonomy = dict(taxonomy) if taxonomy else {}

    def rank_all(self, query: str, taxonomy_filter: dict = None) -> list[dict]:
        """전체 corpus에 대한 ranked list 반환 (retrieval-only 평가용).

        결과: [{doc_id, score, rank}] — rank는 1부터.
        """
        sims = self._dense_scores(query, taxonomy_filter)
        order = self._order_indices(sims, taxonomy_filter)
        dense = [
            {"doc_id": self.doc_ids[i], "score": float(sims[i]), "rank": r + 1}
            for r, i in enumerate(order)
        ]
        if self.config.backend != "hybrid_rrf":
            return dense

        lexical = self.bm25.search(query, self.config.bm25_top_k)
        fused = reciprocal_rank_fusion(
            [dense, lexical], k=self.config.rrf_k, top_k=len(dense)
        )
        for rank, row in enumerate(fused, 1):
            row["rank"] = rank
        return fused

    def _matches_filter(self, doc_id: str, taxonomy_filter: dict) -> bool:
        tax = self.doc_taxonomy.get(doc_id, {})
        return isinstance(tax, dict) and all(
            tax.get(k) == v for k, v in taxonomy_filter.items()
        )

    def _order_indices(self, sims: np.ndarray, taxonomy_filter: dict = None) -> np.ndarray:
        """dense 계열 backend의 최종 정렬 인덱스.

        taxonomy_routed + filter: 카테고리 매치 문서를 cosine 순으로 앞에, 나머지를
        뒤에 backfill. filter 없음/카테고리 무문서면 순수 dense 폴백(오라우팅 안전장치).
        """
        order = np.argsort(-sims, kind="stable")
        if (self.config.backend != "taxonomy_routed" or not taxonomy_filter
                or not self.doc_taxonomy):
            return order
        match = np.array(
            [self._matches_filter(did, taxonomy_filter) for did in self.doc_ids],
            dtype=bool,
        )
        if not match.any():
            return order
        in_cat = order[match[order]]
        out_cat = order[~match[order]]
        return np.concatenate([in_cat, out_cat])

    def _dense_scores(self, query: str, taxonomy_filter: dict = None) -> np.ndarray:
        """질의와 문서 행렬의 cosine score를 계산한다."""
        query_text = query
        if self.config.query_instruction:
            query_text = f"{self.config.query_instruction}{query}"
        query_emb = self._embed_batch([query_text])[0]

        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        sims = self.embedding_matrix @ query_emb / (norms * query_norm + 1e-8)

        # taxonomy_routed는 raw cosine 위에서 하드 재정렬하므로 bonus를 섞지 않는다
        if (taxonomy_filter and self.doc_taxonomy
                and self.config.backend != "taxonomy_routed"):
            for i, did in enumerate(self.doc_ids):
                if self._matches_filter(did, taxonomy_filter):
                    sims[i] += self.config.taxonomy_bonus
        return sims

    def pull(self, query: str, taxonomy_filter: dict = None,
             top_k: int = None, exclude_ids: set = None) -> dict:
        """Pull action (P0-2 충실 구현).

        - top_k: agent가 요청하는 동적 retrieval budget (없으면 config 기본값)
        - exclude_ids: workspace에 이미 있는 문서 — 후보에서 제외하고
          다음 rank 후보로 backfill하여 항상 새 문서 top_k개를 채운다.

        반환:
        {
          "results": [{doc_id, score, rank}],   # 신규 문서만, corpus 전체 기준 rank
          "requested": k,
          "duplicates_excluded": m,             # top 후보 중 workspace 중복으로 건너뛴 수
        }
        """
        k = self.config.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("top_k must be an integer")
        if k < 1 or k > self.config.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.config.max_top_k}")
        exclude_ids = exclude_ids or set()

        # reranker가 있으면 여유 있게 후보를 모은 뒤 rerank
        gather_k = k * 4 if self.config.reranker_url else k
        if self.config.backend != "hybrid_rrf":
            sims = self._dense_scores(query, taxonomy_filter)
            order = self._order_indices(sims, taxonomy_filter)
            ranked = (
                {"doc_id": self.doc_ids[i], "score": float(sims[i]), "rank": rank}
                for rank, i in enumerate(order, 1)
            )
        else:
            # Agent pull은 전량 JSON을 만들지 않고, workspace 중복을
            # backfill할 수 있는 범위만 fusion한다. 전체 순위가 필요한
            # retrieval-only 평가는 rank_all()을 사용한다.
            sims = self._dense_scores(query, taxonomy_filter)
            budget = min(
                len(self.doc_ids),
                gather_k + len(exclude_ids) + self.config.bm25_top_k,
            )
            dense_order = np.argsort(-sims, kind="stable")[:budget]
            dense = [
                {"doc_id": self.doc_ids[i], "score": float(sims[i])}
                for i in dense_order
            ]
            lexical = self.bm25.search(query, max(self.config.bm25_top_k, budget))
            fused = reciprocal_rank_fusion(
                [dense, lexical], k=self.config.rrf_k, top_k=budget
            )
            for rank, row in enumerate(fused, 1):
                row["rank"] = rank
            ranked = iter(fused)

        results = []
        duplicates = 0
        for item in ranked:
            if item["doc_id"] in exclude_ids:
                # 원래 top 구간에서의 중복만 카운트 (backfill 이전 기준)
                if item["rank"] <= k:
                    duplicates += 1
                continue
            results.append(item)
            if len(results) >= gather_k:
                break

        self.last_rerank_error = None
        reranker_used = False
        if self.config.reranker_url and results:
            results = self._rerank(query, results)
            reranker_used = self.last_rerank_error is None

        return {
            "results": results[:k],
            "requested": k,
            "duplicates_excluded": duplicates,
            "reranker_used": reranker_used,
            "reranker_error": self.last_rerank_error,
        }

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        docs = [self.doc_raw_texts.get(c["doc_id"], "") for c in candidates]
        try:
            payload = {
                "model": self.config.reranker_model,
                "query": query,
                "documents": docs,
            }
            resp = requests.post(self.config.reranker_url, json=payload, timeout=60)
            resp.raise_for_status()
            results = resp.json()["results"]
            scored = [
                {"doc_id": candidates[r["index"]]["doc_id"], "score": r["relevance_score"]}
                for r in results
            ]
            scored.sort(key=lambda x: -x["score"])
            for i, item in enumerate(scored):
                item["rank"] = i + 1
            return scored
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if self.config.allow_reranker_fallback:
                self.last_rerank_error = error
                return candidates
            raise RerankerError(f"reranker call failed: {error}") from exc

    def _embed_batch(self, texts: list[str], batch_size: int = 256) -> list[np.ndarray]:
        """vLLM embedding endpoint 호출 (batch=256)"""
        all_embeddings = []
        headers = {"Content-Type": "application/json"}
        api_key = self.config.api_key or self.config.embedding_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.config.embedding_model, "input": batch}
            resp = requests.post(self.config.embedding_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
