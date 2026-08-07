"""Shared retrieval primitives used by experiment pipelines."""

from .bm25 import BM25
from .cache import embedding_cache_key
from .fusion import reciprocal_rank_fusion


class RerankerError(RuntimeError):
    """reranker 호출 실패. 실험 arm 라벨과 실제 구성이 어긋나지 않도록
    조용한 fallback 대신 기본적으로 이 예외를 던진다."""


__all__ = ["BM25", "embedding_cache_key", "reciprocal_rank_fusion", "RerankerError"]
