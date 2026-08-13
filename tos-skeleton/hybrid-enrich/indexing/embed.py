#!/usr/bin/env python3
"""ollama bge-m3로 vec_base / vec_meta 임베딩 → .npy 저장. 배치 임베딩, 재시작 가능."""
import json, os, sys, urllib.request, numpy as np, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
MODEL = "bge-m3"
BATCH = 64
MAX_CHARS = 6000  # bge-m3 8192 토큰이지만 안전 절단. 절단 수 기록. 전 인덱스 동일 규칙.

def embed_batch(texts):
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["embeddings"]

def run(name):
    src = os.path.join(OUT, f"{name}_texts.jsonl")
    dst = os.path.join(OUT, f"{name}.npy")
    ids_dst = os.path.join(OUT, f"{name}_ids.json")
    rows = [json.loads(l) for l in open(src)]
    ids = [r["chunk_id"] for r in rows]
    texts = [r["text"][:MAX_CHARS] for r in rows]
    truncated = sum(1 for r in rows if len(r["text"]) > MAX_CHARS)
    vecs = np.zeros((len(texts), 1024), dtype=np.float32)
    done = 0
    part = dst + ".part.npy"
    if os.path.exists(part):
        prev = np.load(part)
        done = int((np.abs(prev).sum(axis=1) > 0).sum())
        vecs[:done] = prev[:done]
        print(f"{name}: resume from {done}")
    t0 = time.time()
    for i in range(done, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for attempt in range(3):
            try:
                em = embed_batch(batch)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(5)
        vecs[i:i + len(em)] = np.array(em, dtype=np.float32)
        if (i // BATCH) % 20 == 0:
            np.save(part, vecs)
            rate = (i + len(batch) - done) / max(1e-9, time.time() - t0)
            print(f"{name}: {i + len(batch)}/{len(texts)} ({rate:.0f}/s)", flush=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms
    np.save(dst, vecs)
    json.dump({"ids": ids, "truncated": truncated, "model": MODEL, "max_chars": MAX_CHARS}, open(ids_dst, "w"))
    if os.path.exists(part):
        os.remove(part)
    print(f"{name}: done {len(texts)} truncated={truncated}")

if __name__ == "__main__":
    for name in sys.argv[1:] or ["vec_base", "vec_meta"]:
        run(name)
