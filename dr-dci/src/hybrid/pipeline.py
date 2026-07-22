"""
Hybrid RAG Baseline: Dense + BM25 + RRF + Reranker + LLM
"""

import json
import math
import re
import numpy as np
import requests
from collections import defaultdict


class BM25:
    """Simple BM25 implementation"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lens = {}
        self.avg_dl = 0
        self.corpus_size = 0
        self.index = {}  # doc_id → term_freqs

    def fit(self, documents: list[dict]):
        self.corpus_size = len(documents)
        total_len = 0

        for doc in documents:
            doc_id = doc["_id"]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            terms = self._tokenize(text)
            self.doc_lens[doc_id] = len(terms)
            total_len += len(terms)

            term_freqs = defaultdict(int)
            seen = set()
            for term in terms:
                term_freqs[term] += 1
                if term not in seen:
                    self.doc_freqs[term] += 1
                    seen.add(term)

            self.index[doc_id] = dict(term_freqs)

        self.avg_dl = total_len / self.corpus_size if self.corpus_size else 1

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        query_terms = self._tokenize(query)
        scores = {}

        for doc_id, term_freqs in self.index.items():
            score = 0
            dl = self.doc_lens[doc_id]

            for term in query_terms:
                if term not in term_freqs:
                    continue
                tf = term_freqs[term]
                df = self.doc_freqs.get(term, 0)
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                score += idf * tf_norm

            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"doc_id": did, "score": s} for did, s in ranked[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


class HybridRAG:
    def __init__(self, embedding_url: str, embedding_model: str,
                 reranker_url: str, reranker_model: str,
                 llm_url: str, llm_model: str,
                 dense_top_k: int = 20, bm25_top_k: int = 20, rerank_top_k: int = 20,
                 api_key: str = None):
        self.embedding_url = embedding_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.reranker_url = reranker_url
        self.reranker_model = reranker_model
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k

        self.bm25 = BM25()
        self.doc_ids: list[str] = []
        self.embedding_matrix: np.ndarray = None
        self.corpus: dict[str, dict] = {}

    def index(self, documents: list[dict]):
        """문서 인덱싱 (BM25 + Dense), 디스크 캐시 활용"""
        import hashlib
        from pathlib import Path
        cache_dir = Path(__file__).parent.parent.parent / "cache" / "embeddings"

        self.corpus = {doc["_id"]: doc for doc in documents}
        self.bm25.fit(documents)

        # Dense embeddings with caching
        texts = [f"{doc.get('title', '')} {doc.get('text', '')}"[:4096] for doc in documents]
        ids = [doc["_id"] for doc in documents]

        cache_key = hashlib.md5(
            f"{sorted(ids)[:5]}_{len(ids)}_hybrid".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"{cache_key}.npz"

        if cache_path.exists():
            print(f"    [Cache HIT] Loading hybrid embeddings from {cache_path.name}")
            data = np.load(cache_path, allow_pickle=True)
            cached_ids = list(data["ids"])
            if cached_ids == ids:
                self.doc_ids = ids
                self.embedding_matrix = data["embeddings"]
                return

        print(f"    [Cache MISS] Embedding {len(texts)} documents for hybrid (batch=256)...")
        embeddings = self._embed_batch(texts)
        self.doc_ids = ids
        self.embedding_matrix = np.array(embeddings)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(ids, dtype=object), embeddings=self.embedding_matrix)
        print(f"    [Cache SAVED] {cache_path.name}")

    def run(self, query: str) -> dict:
        """쿼리 실행: Dense + BM25 → RRF → Rerank → LLM"""
        # Dense retrieval (vectorized)
        query_emb = self._embed_batch([query])[0]
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        sims = self.embedding_matrix @ query_emb / (norms * query_norm + 1e-8)
        top_indices = np.argsort(-sims)[:self.dense_top_k]
        dense_results = [(self.doc_ids[i], float(sims[i])) for i in top_indices]

        # BM25 retrieval
        bm25_results = self.bm25.search(query, self.bm25_top_k)

        # RRF fusion
        fused = self._rrf_fusion(
            [(did, s) for did, s in dense_results],
            [(r["doc_id"], r["score"]) for r in bm25_results],
        )

        # Reranking
        reranked = self._rerank(query, fused[:self.rerank_top_k * 2])
        top_docs = reranked[:self.rerank_top_k]

        # LLM 답변 생성
        context = self._build_context(top_docs)
        answer = self._generate_answer(query, context)

        return {
            "answer": answer,
            "retrieved_docs": [did for did, _ in top_docs],
            "pull_count": 1,  # hybrid는 항상 1회 검색
        }

    def _rrf_fusion(self, list_a: list, list_b: list, k: int = 60) -> list:
        """Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        for rank, (doc_id, _) in enumerate(list_a):
            scores[doc_id] += 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(list_b):
            scores[doc_id] += 1.0 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked

    def _rerank(self, query: str, candidates: list) -> list:
        """Reranker로 재정렬"""
        if not candidates:
            return []

        docs = []
        for doc_id, _ in candidates:
            if doc_id in self.corpus:
                text = f"{self.corpus[doc_id].get('title', '')} {self.corpus[doc_id].get('text', '')}"[:4096]
                docs.append(text)
            else:
                docs.append("")

        try:
            payload = {
                "model": self.reranker_model,
                "query": query,
                "documents": docs,
            }
            resp = requests.post(self.reranker_url, json=payload, timeout=60)
            resp.raise_for_status()
            results = resp.json()["results"]
            # reranker 결과로 재정렬
            scored = [(candidates[r["index"]][0], r["relevance_score"]) for r in results]
            scored.sort(key=lambda x: -x[1])
            return scored
        except Exception:
            # reranker 실패 시 원래 순서 유지
            return candidates

    def _build_context(self, docs: list, max_chars: int = 4000) -> str:
        context_parts = []
        total = 0
        for doc_id, _ in docs:
            if doc_id in self.corpus:
                text = f"[{doc_id}] {self.corpus[doc_id].get('title', '')}\n{self.corpus[doc_id].get('text', '')}"
                if total + len(text) > max_chars:
                    text = text[:max_chars - total]
                context_parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "Answer the question based on the provided context. Be concise and accurate."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        if "openai.com" not in self.llm_url:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(self.llm_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _embed_batch(self, texts: list[str], batch_size: int = 256) -> list[np.ndarray]:
        all_embeddings = []
        headers = {"Content-Type": "application/json"}
        if self.api_key and "openai.com" in self.embedding_url:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"model": self.embedding_model, "input": batch}
            resp = requests.post(self.embedding_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
