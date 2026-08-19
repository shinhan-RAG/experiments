#!/usr/bin/env python3
"""임베딩 소스 추상화 — 결정론을 위해 한 실험 안에서는 한 소스만 쓴다.
  EMBED_ENDPOINT 가 있으면 OpenAI 호환 /v1/embeddings (예: KT 인스턴스 http://localhost:8101/v1) — 모델명 EMBED_MODEL(기본 dragonkue/BGE-m3-ko)
  없으면 로컬 sentence-transformers(MPS/CPU).
반환은 L2 정규화 float32 numpy."""
import json, os, urllib.request
import numpy as np
MODEL = os.environ.get("EMBED_MODEL", "dragonkue/BGE-m3-ko")
_local = None


def _remote(texts, endpoint, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": MODEL, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(endpoint.rstrip("/") + "/embeddings", data=body, headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["EMBED_API_KEY"]} if os.environ.get("EMBED_API_KEY") else {})})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        out.extend([x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])])
    V = np.asarray(out, dtype="float32")
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)


def encode(texts, batch=32, max_len=1024, progress=False):
    ep = os.environ.get("EMBED_ENDPOINT")
    if ep:
        return _remote(texts, ep, batch=max(8, batch))
    global _local
    if _local is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _local = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        _local.max_seq_length = max_len
    return _local.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=progress, convert_to_numpy=True).astype("float32")


def source():
    return {"source": os.environ.get("EMBED_ENDPOINT") or "local-sentence-transformers", "model": MODEL}
