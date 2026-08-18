#!/usr/bin/env python3
"""u2 element 임베딩 (BGE-m3-ko, MPS/CPU) → out/emb/<view>.npy + ids.json (+ 선택: Qdrant 업서트).

view:
  base   = element 원문만
  ctx    = "특약 > 조(제목) 원문" 좌표 헤더 + 원문 (Contextual Retrieval 방식의 최소 문맥; 설계서 2.4 (2)경로 — 메타를 본문과 함께 임베딩)
정규화 L2. 결정론: 배치·모델·revision 기록. QA 미접근.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
MODEL = "dragonkue/BGE-m3-ko"


def views(E, T, view):
    if view == "base":
        return [e["text"] for e in E]
    out = []
    for e in E:
        t = T.get(e["element_id"], {}); loc = t.get("locator") or {}
        head = " > ".join(x for x in [t.get("contract_key", "") or e["contract_scope"], (loc.get("article", "") + " " + loc.get("article_title", "")).strip()] if x)
        out.append((head + "\n" if head else "") + e["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(HERE / "out/elements_u2.jsonl"))
    ap.add_argument("--tags", default=str(HERE / "out/tags_u2_rules.jsonl"))
    ap.add_argument("--view", default="base", choices=("base", "ctx"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--qdrant", default="", help="예: http://localhost:6333 (지정 시 업서트)")
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    E = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    if a.limit > 0:
        E = E[: a.limit]
    T = {json.loads(l)["element_id"]: json.loads(l) for l in open(a.tags, encoding="utf-8")} if a.view == "ctx" else {}
    texts = views(E, T, a.view)
    m = SentenceTransformer(MODEL, device=dev)
    m.max_seq_length = a.max_len
    t0 = time.time()
    V = m.encode(texts, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    out = HERE / "out" / "emb"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"u2_{a.view}.npy", V)
    ids = [e["element_id"] for e in E]
    json.dump(ids, open(out / f"u2_{a.view}_ids.json", "w"))
    meta = {"model": MODEL, "device": dev, "view": a.view, "n": len(ids), "dim": int(V.shape[1]), "batch": a.batch, "max_len": a.max_len,
            "elapsed_s": round(time.time() - t0, 1), "elements_sha256": hashlib.sha256(open(a.elements, "rb").read()).hexdigest(),
            "vec_sha256": hashlib.sha256(V.tobytes()).hexdigest()}
    json.dump(meta, open(out / f"u2_{a.view}_meta.json", "w"), indent=1)
    print(json.dumps(meta))
    if a.qdrant:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        c = QdrantClient(url=a.qdrant, api_key=os.environ.get("QDRANT_API_KEY"))
        name = f"u2_{a.view}"
        if not c.collection_exists(name):
            c.create_collection(name, vectors_config=VectorParams(size=V.shape[1], distance=Distance.COSINE))
        pts = []
        for i, e in enumerate(E):
            t = T.get(e["element_id"], {})
            pts.append(PointStruct(id=int(e["element_id"][1:]), vector=V[i].tolist(), payload={
                "element_id": e["element_id"], "contract_scope": e["contract_scope"], "element_type": e["element_type"],
                "line_start": e["line_start"], "line_end": e["line_end"], "char_start": e["char_start"], "char_end": e["char_end"],
                "article": (t.get("locator") or {}).get("article", "")}))
            if len(pts) >= 512:
                c.upsert(name, pts); pts = []
        if pts:
            c.upsert(name, pts)
        print("qdrant upserted", name, c.count(name).count)


if __name__ == "__main__":
    main()
