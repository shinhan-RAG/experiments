"""
Pull Retriever: 에이전트가 호출하는 검색 함수
- Dense retrieval (embedding similarity)
- Taxonomy path filtering (optional)
- Metadata pre-filtering (optional)
- Contextual prefix 적용 (optional)
"""

import numpy as np
import requests
from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    embedding_url: str
    embedding_model: str
    top_k: int = 20
    use_taxonomy: bool = False
    use_metadata: bool = False
    use_prefix: bool = False


class PullRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.doc_embeddings: dict[str, np.ndarray] = {}
        self.doc_texts: dict[str, str] = {}
        self.doc_taxonomy: dict[str, dict] = {}
        self.doc_metadata: dict[str, dict] = {}

    def index(self, documents: list[dict], taxonomy: dict = None,
              metadata: dict = None, prefixes: dict = None):
        """문서를 인덱싱 (임베딩 생성)"""
        for doc in documents:
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")

            # contextual prefix 적용
            if self.config.use_prefix and prefixes and doc_id in prefixes:
                embed_text = f"{prefixes[doc_id]} {title} {text}"
            else:
                embed_text = f"{title} {text}"

            self.doc_texts[doc_id] = embed_text[:512]  # 임베딩 입력 제한

            if taxonomy and doc_id in taxonomy:
                self.doc_taxonomy[doc_id] = taxonomy[doc_id]
            if metadata and doc_id in metadata:
                self.doc_metadata[doc_id] = metadata[doc_id]

        # 배치 임베딩
        texts = list(self.doc_texts.values())
        ids = list(self.doc_texts.keys())
        embeddings = self._embed_batch(texts)

        for doc_id, emb in zip(ids, embeddings):
            self.doc_embeddings[doc_id] = emb

    def pull(self, query: str, taxonomy_filter: dict = None,
             metadata_filter: dict = None) -> list[dict]:
        """Pull action: query로 top-k 문서 검색"""
        query_emb = self._embed_batch([query])[0]

        # 후보 필터링
        candidates = list(self.doc_embeddings.keys())

        if self.config.use_taxonomy and taxonomy_filter:
            candidates = [
                did for did in candidates
                if self._match_taxonomy(did, taxonomy_filter)
            ]

        if self.config.use_metadata and metadata_filter:
            candidates = [
                did for did in candidates
                if self._match_metadata(did, metadata_filter)
            ]

        # 유사도 계산
        scores = []
        for did in candidates:
            sim = np.dot(query_emb, self.doc_embeddings[did]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(self.doc_embeddings[did]) + 1e-8
            )
            scores.append((did, float(sim)))

        scores.sort(key=lambda x: -x[1])
        return [{"doc_id": did, "score": score} for did, score in scores[:self.config.top_k]]

    def _match_taxonomy(self, doc_id: str, filter: dict) -> bool:
        tax = self.doc_taxonomy.get(doc_id, {})
        for key, val in filter.items():
            if tax.get(key) != val:
                return False
        return True

    def _match_metadata(self, doc_id: str, filter: dict) -> bool:
        meta = self.doc_metadata.get(doc_id, {})
        if meta is None:
            return False
        for key, val in filter.items():
            if key == "entities":
                # entities: [{"name": ..., "category": ...}]
                doc_entities = meta.get("entities", [])
                if isinstance(val, list):
                    # val이 entity name 리스트인 경우
                    doc_names = {e.get("name", "").lower() for e in doc_entities if isinstance(e, dict)}
                    if not any(v.lower() in doc_names for v in val):
                        return False
                elif isinstance(val, str):
                    # val이 category인 경우
                    doc_cats = {e.get("category", "") for e in doc_entities if isinstance(e, dict)}
                    if val not in doc_cats:
                        return False
            elif meta.get(key) != val:
                return False
        return True

    def _embed_batch(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        """vLLM embedding endpoint 호출"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self.config.embedding_model,
                "input": batch,
            }
            resp = requests.post(self.config.embedding_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()["data"]
            for item in sorted(data, key=lambda x: x["index"]):
                all_embeddings.append(np.array(item["embedding"]))

        return all_embeddings
