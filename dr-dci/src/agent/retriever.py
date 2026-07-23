"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Prefix: 임베딩 보강 (자연어 요약)
- Taxonomy: pull 시 soft boost (네비게이션)
- Metadata/Tags: workspace 탐색 전용

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
    taxonomy_boost: float = 1.5
    reranker_url: str = None
    reranker_model: str = None
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION  # None이면 instruction 미적용
    embedding_api_key: str = None  # OpenAI 등 인증이 필요한 임베딩 endpoint용


class PullRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.doc_embeddings: dict[str, np.ndarray] = {}
        self.doc_ids: list[str] = []
        self.embedding_matrix: np.ndarray = None  # (N, D) for vectorized search
        self.doc_taxonomy: dict[str, dict] = {}
        self.doc_titles: dict[str, str] = {}
        self.doc_raw_texts: dict[str, str] = {}  # for reranker

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

    def index(self, documents: list[dict], prefixes: dict = None, taxonomy: dict = None):
        """문서를 인덱싱. 디스크 캐시 활용."""
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
            if taxonomy and doc_id in taxonomy:
                self.doc_taxonomy[doc_id] = taxonomy[doc_id]

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
        query_text = query
        if self.config.query_instruction:
            query_text = f"{self.config.query_instruction}{query}"
        query_emb = self._embed_batch([query_text])[0]

        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        sims = self.embedding_matrix @ query_emb / (norms * query_norm + 1e-8)

        if taxonomy_filter and self.doc_taxonomy:
            for i, did in enumerate(self.doc_ids):
                tax = self.doc_taxonomy.get(did, {})
                if isinstance(tax, dict) and all(tax.get(k) == v for k, v in taxonomy_filter.items()):
                    sims[i] *= self.config.taxonomy_boost

        order = np.argsort(-sims)
        return [
            {"doc_id": self.doc_ids[i], "score": float(sims[i]), "rank": r + 1}
            for r, i in enumerate(order)
        ]

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
        k = top_k or self.config.top_k
        exclude_ids = exclude_ids or set()

        ranked = self.rank_all(query, taxonomy_filter=taxonomy_filter)

        results = []
        duplicates = 0
        # reranker가 있으면 여유 있게 후보를 모은 뒤 rerank
        gather_k = k * 4 if self.config.reranker_url else k
        for item in ranked:
            if item["doc_id"] in exclude_ids:
                # 원래 top 구간에서의 중복만 카운트 (backfill 이전 기준)
                if item["rank"] <= k:
                    duplicates += 1
                continue
            results.append(item)
            if len(results) >= gather_k:
                break

        if self.config.reranker_url and results:
            results = self._rerank(query, results)

        return {
            "results": results[:k],
            "requested": k,
            "duplicates_excluded": duplicates,
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
        except Exception:
            return candidates

    def _embed_batch(self, texts: list[str], batch_size: int = 256) -> list[np.ndarray]:
        """vLLM embedding endpoint 호출 (batch=256)"""
        all_embeddings = []
        headers = {"Content-Type": "application/json"}
        if self.config.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.config.embedding_api_key}"
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.config.embedding_model, "input": batch}
            resp = requests.post(self.config.embedding_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
