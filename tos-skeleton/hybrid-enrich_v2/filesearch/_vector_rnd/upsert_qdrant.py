#!/usr/bin/env python3
"""out/emb/u2_<view>.npy → Qdrant 컬렉션 u2_<view> (payload: element_id·특약·유형·조·line/char span). 재실행 시 upsert(멱등)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
ap = argparse.ArgumentParser(); ap.add_argument("--view", default="base"); ap.add_argument("--url", default="http://localhost:6333"); a = ap.parse_args()
V = np.load(HERE / "out/emb" / f"u2_{a.view}.npy"); ids = json.load(open(HERE / "out/emb" / f"u2_{a.view}_ids.json"))
E = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out/elements_u2.jsonl", encoding="utf-8")}
T = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out/tags_u2_rules.jsonl", encoding="utf-8")}
J = [json.loads(l) for l in open(HERE / "out/elements_u2jo.jsonl", encoding="utf-8")]; m2j = {m: u["element_id"] for u in J for m in u["members"]}
c = QdrantClient(url=a.url, api_key=os.environ.get("QDRANT_API_KEY"))
name = f"u2_{a.view}"
if not c.collection_exists(name):
    c.create_collection(name, vectors_config=VectorParams(size=V.shape[1], distance=Distance.COSINE))
pts = []
for i, eid in enumerate(ids):
    e = E[eid]; t = T.get(eid, {}); loc = t.get("locator") or {}
    pts.append(PointStruct(id=int(eid[1:]), vector=V[i].tolist(), payload={
        "element_id": eid, "jo": m2j.get(eid, ""), "contract_scope": e["contract_scope"], "element_type": e["element_type"],
        "article": loc.get("article", ""), "article_title": loc.get("article_title", ""),
        "line_start": e["line_start"], "line_end": e["line_end"], "char_start": e["char_start"], "char_end": e["char_end"],
        "preview": " ".join(e["text"].split())[:200]}))
    if len(pts) >= 500:
        c.upsert(name, pts); pts = []
if pts:
    c.upsert(name, pts)
print(name, "count", c.count(name).count, "dim", V.shape[1])
