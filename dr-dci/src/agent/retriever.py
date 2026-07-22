"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Prefix: 임베딩 보강 (자연어 요약)
- Taxonomy: pull 시 soft boost (네비게이션)
- Metadata/Tags: workspace 탐색 전용
"""

import numpy as np
import requests
from dataclasses import dataclass
from pathlib import Path

from src.retrieval import BM25, embedding_cache_key, reciprocal_rank_fusion


CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "embeddings"
@dataclass
class RetrieverConfig:
    embedding_url: str
    embedding_model: str
    top_k: int = 20
    use_prefix: bool = False
    taxonomy_boost: float = 1.5
    reranker_url: str = None
    reranker_model: str = None
    backend: str = "dense"
    bm25_top_k: int = 20
    rrf_k: int = 60
    api_key: str = None            # OpenAI 호환 원격 endpoint용(로컬 vLLM은 불요)


class PullRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.doc_embeddings: dict[str, np.ndarray] = {}
        self.doc_ids: list[str] = []
        self.embedding_matrix: np.ndarray = None  # (N, D) for vectorized search
        self.doc_taxonomy: dict[str, dict] = {}
        self.doc_raw_texts: dict[str, str] = {}  # for reranker
        self.bm25 = BM25()

    def index(self, documents: list[dict], prefixes: dict = None, taxonomy: dict = None):
        """문서를 인덱싱. 디스크 캐시 활용."""
        if self.config.backend not in {"dense", "hybrid_rrf"}:
            raise ValueError(f"unsupported retrieval backend: {self.config.backend}")

        self.doc_embeddings.clear()
        self.doc_raw_texts.clear()
        self.doc_taxonomy.clear()
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
            self.doc_raw_texts[doc_id] = f"{title} {text}"[:4096]
            if taxonomy and doc_id in taxonomy:
                self.doc_taxonomy[doc_id] = taxonomy[doc_id]

        if self.config.backend == "hybrid_rrf":
            self.bm25.fit(documents)

        cache_key = embedding_cache_key(
            namespace="pull",
            model=self.config.embedding_model,
            use_prefix=self.config.use_prefix,
            doc_ids=doc_ids,
            texts=texts,
        )
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

        # 디스크 캐시 저장
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(doc_ids, dtype=object), embeddings=emb_matrix)
        print(f"    [Cache SAVED] {cache_path.name}")

        self.doc_ids = doc_ids
        self.embedding_matrix = emb_matrix
        for i, did in enumerate(doc_ids):
            self.doc_embeddings[did] = emb_matrix[i]

    def pull(self, query: str, taxonomy_filter: dict = None) -> list[dict]:
        """Pull action: vectorized cosine similarity + taxonomy soft boost."""
        query_emb = self._embed_batch([query])[0]

        # 벡터화 cosine similarity (행렬 연산)
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        sims = self.embedding_matrix @ query_emb / (norms * query_norm + 1e-8)

        # taxonomy soft boost
        if taxonomy_filter and self.doc_taxonomy:
            for i, did in enumerate(self.doc_ids):
                tax = self.doc_taxonomy.get(did, {})
                if isinstance(tax, dict) and all(tax.get(k) == v for k, v in taxonomy_filter.items()):
                    sims[i] *= self.config.taxonomy_boost

        # Dense candidates are always built so the hybrid arm changes only the backend.
        candidate_k = self.config.top_k * 4 if self.config.reranker_url else self.config.top_k
        dense_k = max(candidate_k, self.config.bm25_top_k)
        top_indices = np.argsort(-sims, kind="stable")[:dense_k]
        dense_candidates = [
            {"doc_id": self.doc_ids[i], "score": float(sims[i])}
            for i in top_indices
        ]

        if self.config.backend == "hybrid_rrf":
            lexical_candidates = self.bm25.search(query, self.config.bm25_top_k)
            candidates = reciprocal_rank_fusion(
                [dense_candidates, lexical_candidates],
                k=self.config.rrf_k,
                top_k=candidate_k,
            )
        else:
            candidates = dense_candidates[:candidate_k]

        if self.config.reranker_url:
            candidates = self._rerank(query, candidates)

        return candidates[:self.config.top_k]

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
            scored = [{"doc_id": candidates[r["index"]]["doc_id"], "score": r["relevance_score"]} for r in results]
            scored.sort(key=lambda x: -x["score"])
            return scored
        except Exception:
            return candidates

    def _embed_batch(self, texts: list[str], batch_size: int = 256) -> list[np.ndarray]:
        """vLLM embedding endpoint 호출 (batch=256)"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.config.embedding_model, "input": batch}
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            resp = requests.post(self.config.embedding_url, json=payload,
                                 headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
