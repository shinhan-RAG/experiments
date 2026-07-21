"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Prefix: 임베딩 보강 (자연어 요약)
- Taxonomy: pull 시 soft boost (네비게이션)
- Metadata/Tags: workspace 탐색 전용
"""

import hashlib
import numpy as np
import requests
from dataclasses import dataclass
from pathlib import Path


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


class PullRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.doc_embeddings: dict[str, np.ndarray] = {}
        self.doc_ids: list[str] = []
        self.embedding_matrix: np.ndarray = None  # (N, D) for vectorized search
        self.doc_taxonomy: dict[str, dict] = {}
        self.doc_raw_texts: dict[str, str] = {}  # for reranker

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
            self.doc_raw_texts[doc_id] = f"{title} {text}"[:4096]
            if taxonomy and doc_id in taxonomy:
                self.doc_taxonomy[doc_id] = taxonomy[doc_id]

        # 캐시 키: doc_ids 해시 + prefix 여부
        cache_key = hashlib.md5(
            f"{sorted(doc_ids)[:5]}_{len(doc_ids)}_{self.config.use_prefix}".encode()
        ).hexdigest()[:12]
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

        # dense top candidates (reranker가 있으면 더 많이 뽑아서 rerank)
        candidate_k = self.config.top_k * 4 if self.config.reranker_url else self.config.top_k
        top_indices = np.argsort(-sims)[:candidate_k]
        candidates = [{"doc_id": self.doc_ids[i], "score": float(sims[i])} for i in top_indices]

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
            resp = requests.post(self.config.embedding_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
