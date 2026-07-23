"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Prefix: 임베딩 보강 (자연어 요약)
- Taxonomy: pull 시 soft boost (네비게이션)
- Metadata/Tags: workspace 탐색 전용
"""

import numpy as np
import requests
import time
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


class PullResult(list):
    """Ranked pull candidates with bounded, per-pull treatment telemetry.

    It remains a list so existing callers can iterate over candidates unchanged.
    The telemetry describes the dense-score stage, before optional RRF fusion or
    reranking, where taxonomy soft boosting is applied.
    """

    def __init__(self, candidates: list[dict], telemetry: dict):
        super().__init__(candidates)
        self.telemetry = telemetry


def select_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Return an exact stable top-count ordering without sorting the full corpus."""
    count = min(int(count), len(scores))
    if count <= 0:
        return np.asarray([], dtype=int)
    if count == len(scores):
        return np.argsort(-scores, kind="stable")

    partition = np.argpartition(-scores, count - 1)[:count]
    boundary_score = scores[partition].min()
    strictly_above = np.flatnonzero(scores > boundary_score)
    boundary_ties = np.flatnonzero(scores == boundary_score)
    selected = np.concatenate((
        strictly_above,
        boundary_ties[:count - len(strictly_above)],
    ))
    return selected[np.argsort(-scores[selected], kind="stable")]


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

    def pull(self, query: str, taxonomy_filter: dict = None) -> PullResult:
        """Pull action: vectorized cosine similarity + taxonomy soft boost."""
        query_emb = self._embed_batch([query])[0]

        # 벡터화 cosine similarity (행렬 연산)
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_emb)
        raw_sims = self.embedding_matrix @ query_emb / (norms * query_norm + 1e-8)
        sims = raw_sims.copy()

        eligible_indices = np.asarray([], dtype=int)
        if taxonomy_filter and self.doc_taxonomy:
            eligible_indices = np.asarray([
                i for i, did in enumerate(self.doc_ids)
                if isinstance(self.doc_taxonomy.get(did), dict)
                and all(self.doc_taxonomy[did].get(key) == value
                        for key, value in taxonomy_filter.items())
            ], dtype=int)

        # Multiplying a negative cosine by a factor above one is a penalty, not
        # a boost.  Keep non-positive target scores unchanged and report their
        # prevalence so the treatment's operating population is observable.
        positive_eligible_indices = eligible_indices[raw_sims[eligible_indices] > 0]
        if len(positive_eligible_indices):
            sims[positive_eligible_indices] *= self.config.taxonomy_boost

        candidate_k = self.config.top_k * 4 if self.config.reranker_url else self.config.top_k
        dense_k = max(candidate_k, self.config.bm25_top_k)
        boosted_order = select_top_indices(sims, dense_k)
        boost_changed_scores = bool(len(positive_eligible_indices)) and (
            self.config.taxonomy_boost != 1.0
        )
        telemetry_seconds = 0.0
        if boost_changed_scores:
            telemetry_started = time.perf_counter()
            baseline_order = select_top_indices(raw_sims, dense_k)
            baseline_ranks = {
                int(index): rank + 1 for rank, index in enumerate(baseline_order)
            }
            boosted_ranks = {
                int(index): rank + 1 for rank, index in enumerate(boosted_order)
            }
            baseline_top_k = set(
                int(index) for index in baseline_order[:self.config.top_k]
            )
            boosted_top_k = set(
                int(index) for index in boosted_order[:self.config.top_k]
            )
            # Keep rank traces bounded to the union of candidate-stage documents.
            # A missing rank means that the document fell below the candidate
            # envelope; the lower bound makes that loss explicit without a
            # corpus-sized Python rank map.
            trace_indices = set(int(index) for index in baseline_order)
            trace_indices.update(int(index) for index in boosted_order)
            rank_changes = []
            for index in sorted(
                trace_indices,
                key=lambda item: (boosted_ranks.get(item, dense_k + 1), item),
            ):
                rank_before = baseline_ranks.get(index)
                rank_after = boosted_ranks.get(index)
                if rank_before == rank_after:
                    continue
                change = {
                    "doc_id": self.doc_ids[index],
                    "rank_before": rank_before,
                    "rank_after": rank_after,
                    "score_before": round(float(raw_sims[index]), 6),
                    "score_after": round(float(sims[index]), 6),
                }
                if rank_before is None:
                    change["rank_before_lower_bound"] = dense_k + 1
                if rank_after is None:
                    change["rank_after_lower_bound"] = dense_k + 1
                rank_changes.append(change)
            telemetry_seconds = time.perf_counter() - telemetry_started
        else:
            baseline_top_k = set(int(index) for index in boosted_order[:self.config.top_k])
            boosted_top_k = baseline_top_k
            rank_changes = []
        target_scores = raw_sims[eligible_indices]
        negative_target_scores = target_scores[target_scores < 0]
        telemetry = {
            "taxonomy_boost_stage": "dense_pre_backend_and_rerank",
            "taxonomy_boost_eligible_documents": int(len(eligible_indices)),
            "taxonomy_boosted_positive_score_documents": int(len(positive_eligible_indices)),
            "taxonomy_boosted_returned_documents": int(
                sum(index in boosted_top_k for index in eligible_indices)
            ),
            "taxonomy_boost_rank_changed": bool(rank_changes),
            "taxonomy_boost_top_k_entered_documents": int(
                len(boosted_top_k - baseline_top_k)
            ),
            "taxonomy_boost_top_k_exited_documents": int(
                len(baseline_top_k - boosted_top_k)
            ),
            "taxonomy_boost_target_score_count": int(len(target_scores)),
            "taxonomy_boost_target_negative_score_count": int(len(negative_target_scores)),
            "taxonomy_boost_target_score_min": (
                round(float(target_scores.min()), 6) if len(target_scores) else None
            ),
            "taxonomy_boost_target_score_max": (
                round(float(target_scores.max()), 6) if len(target_scores) else None
            ),
            "taxonomy_boost_rank_changes": rank_changes,
            "taxonomy_boost_telemetry_seconds": telemetry_seconds,
        }

        # Dense candidates are always built so the hybrid arm changes only the backend.
        top_indices = boosted_order
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

        return PullResult(candidates[:self.config.top_k], telemetry)

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
            # 일시 네트워크 장애 1회로 장시간 배치 전체(과금 포함)가 죽지 않게
            # 지수 백오프 재시도. 재시도 소진 시에만 전파(무음 강등 없음).
            for attempt in range(4):
                try:
                    resp = requests.post(self.config.embedding_url, json=payload,
                                         headers=headers, timeout=120)
                    resp.raise_for_status()
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 3:
                        raise
                    import time as _time
                    _time.sleep(2 ** attempt)
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))
        return all_embeddings
