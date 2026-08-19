#!/usr/bin/env python3
"""하이브리드 검색 모듈 — BM25 + Dense RRF 융합.

ChunkHybridSearch 클래스: BM25(bm25s) + Dense(BGE-m3-ko) 채널을
RRF(Reciprocal Rank Fusion)로 결합한다.

Usage (standalone test):
    EMBED_ENDPOINT=http://localhost:8101/v1 python hybrid_search.py \
        --query "보험료 납입면제 조건은?" --top-k 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import embedder

RRF_K = 60
CHANNEL_TOPK = 50
PREVIEW_CHARS = 240

_KO_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")
_JOSA = sorted(
    ["은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
     "와", "과", "도", "만", "까지", "부터", "에게", "한테"],
    key=len, reverse=True,
)


def _strip_josa(token: str) -> str | None:
    if len(token) <= 2:
        return None
    for suf in _JOSA:
        if token.endswith(suf) and len(token) - len(suf) >= 2:
            return token[:-len(suf)]
    return None


def _ko_stemmer(tokens: list[str]) -> list[str]:
    result = []
    for token in tokens:
        stripped = _strip_josa(token)
        result.append(stripped if stripped else token)
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class ChunkHybridSearch:
    """BM25 + Dense RRF fusion on chunk views."""

    def __init__(self, view: str = "V9", out_dir: Path | None = None):
        self.out = out_dir or OUT
        self.view = view
        self._bm25 = None
        self._dense = None
        self._chunks: dict[str, dict] | None = None

    @property
    def chunks(self) -> dict[str, dict]:
        if self._chunks is None:
            p = self.out / "chunks.jsonl"
            if p.exists():
                self._chunks = {r["chunk_id"]: r for r in read_jsonl(p)}
            else:
                self._chunks = {}
        return self._chunks

    def _load_bm25(self):
        if self._bm25 is not None:
            return self._bm25
        import bm25s
        vp = self.out / f"view_{self.view}.jsonl"
        if not vp.exists():
            raise FileNotFoundError(f"뷰 파일이 없습니다: {vp}")
        rows = read_jsonl(vp)
        texts = [r.get("text", "") for r in rows]
        ids = [r["chunk_id"] for r in rows]
        corpus_tokens = bm25s.tokenize(
            texts, stemmer=_ko_stemmer,
            stopwords=[], token_pattern=r"[가-힣a-zA-Z0-9]+",
            show_progress=False,
        )
        retriever = bm25s.BM25(k1=1.5, b=0.75, method="robertson")
        retriever.index(corpus_tokens, show_progress=False)
        self._bm25 = (retriever, ids)
        return self._bm25

    def _load_dense(self):
        if self._dense is not None:
            return self._dense
        npy = self.out / "emb" / f"chunk_{self.view}.npy"
        ids_path = self.out / "emb" / f"chunk_{self.view}_ids.json"
        if not npy.exists() or not ids_path.exists():
            raise FileNotFoundError(f"임베딩 파일이 없습니다: {npy}")
        matrix = np.load(npy, mmap_mode="r")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self._dense = (matrix, ids)
        return self._dense

    def bm25_search(self, query: str, top_k: int = CHANNEL_TOPK) -> list[str]:
        import bm25s
        retriever, ids = self._load_bm25()
        qtok = bm25s.tokenize(
            query, stemmer=_ko_stemmer, stopwords=[],
            token_pattern=r"[가-힣a-zA-Z0-9]+", show_progress=False,
        )
        k = min(top_k, len(ids))
        indices, scores = retriever.retrieve(qtok, k=k, show_progress=False)
        return [ids[int(indices[0, i])] for i in range(scores.shape[1])
                if float(scores[0, i]) > 0]

    def dense_search(self, query: str, top_k: int = CHANNEL_TOPK) -> list[tuple[str, float]]:
        matrix, ids = self._load_dense()
        qv = embedder.encode([query])[0]
        scores = np.asarray(matrix @ qv)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(ids[int(i)], float(scores[i])) for i in top if float(scores[i]) > 0]

    def hybrid_search(self, query: str, top_k: int = 20,
                      rrf_k: int = RRF_K) -> list[dict]:
        fused: dict[str, float] = defaultdict(float)
        provenance: dict[str, dict] = defaultdict(dict)

        bm25_hits = self.bm25_search(query, CHANNEL_TOPK)
        for rank, uid in enumerate(bm25_hits, 1):
            fused[uid] += 1.0 / (rrf_k + rank)
            provenance[uid]["bm25_rank"] = rank

        dense_hits = self.dense_search(query, CHANNEL_TOPK)
        for rank, (uid, sim) in enumerate(dense_hits, 1):
            fused[uid] += 1.0 / (rrf_k + rank)
            provenance[uid]["dense_rank"] = rank
            provenance[uid]["dense_sim"] = round(sim, 4)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        results = []
        for rank, (uid, score) in enumerate(ranked, 1):
            chunk = self.chunks.get(uid, {})
            results.append({
                "id": uid,
                "rank": rank,
                "rrf_score": round(score, 6),
                "preview": chunk.get("text", "")[:PREVIEW_CHARS],
                "provenance": provenance[uid],
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            })
        return results

    def search(self, query: str, strategy: str = "hybrid",
               top_k: int = 20) -> list[dict]:
        if strategy == "bm25":
            ids = self.bm25_search(query, top_k)
            return [{"id": uid, "rank": r + 1,
                     "preview": self.chunks.get(uid, {}).get("text", "")[:PREVIEW_CHARS],
                     "char_start": self.chunks.get(uid, {}).get("char_start"),
                     "char_end": self.chunks.get(uid, {}).get("char_end")}
                    for r, uid in enumerate(ids[:top_k])]
        if strategy == "dense":
            hits = self.dense_search(query, top_k)
            return [{"id": uid, "rank": r + 1, "dense_sim": round(sim, 4),
                     "preview": self.chunks.get(uid, {}).get("text", "")[:PREVIEW_CHARS],
                     "char_start": self.chunks.get(uid, {}).get("char_start"),
                     "char_end": self.chunks.get(uid, {}).get("char_end")}
                    for r, (uid, sim) in enumerate(hits[:top_k])]
        return self.hybrid_search(query, top_k)


def main():
    ap = argparse.ArgumentParser(description="하이브리드 검색 테스트")
    ap.add_argument("--query", required=True, help="검색 질의")
    ap.add_argument("--strategy", default="hybrid", choices=("bm25", "dense", "hybrid"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--view", default="V9", choices=("V9", "BASE"))
    args = ap.parse_args()

    hs = ChunkHybridSearch(view=args.view)
    results = hs.search(args.query, strategy=args.strategy, top_k=args.top_k)
    print(json.dumps({"query": args.query, "strategy": args.strategy,
                      "n_results": len(results), "results": results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
