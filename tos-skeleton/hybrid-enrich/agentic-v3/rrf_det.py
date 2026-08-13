#!/usr/bin/env python3
"""C0-RRF-DET: deterministic RRF baseline. LLM 0회, 비용 $0.

Dense (BGE-m3-ko) + BM25 lexical (태그 메타 view) → weighted RRF.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
HE = BASE.parent  # hybrid-enrich

# Add paths for imports
sys.path.insert(0, str(HE / "hybrid-v7"))
sys.path.insert(0, str(HE / "metajson-v6" / "meta-search-v4"))

from fuse import fuse, rrf_scores
from lex_index import LexIndex
from common import OUT as V4_OUT, read_jsonl


def load_dense(arm: str, chunkset: str = "fixed600"):
    """Load cached vectors and ids."""
    vecs = np.load(V4_OUT / f"vec_{chunkset}_{arm}.npy", mmap_mode="r")
    ids = json.loads((V4_OUT / f"vec_{chunkset}_{arm}_ids.json").read_text("utf-8"))["ids"]
    return vecs, ids


def dense_rank(query_vec: np.ndarray, vecs: np.ndarray, pool: int = 100) -> list[int]:
    """Top-pool by cosine sim. Returns indices into vecs."""
    sims = vecs @ query_vec
    return np.argsort(-sims)[:pool].tolist()


def embed_query(query: str, bge_url: str = "http://localhost:8378",
                model: str = "dragonkue/BGE-m3-ko") -> np.ndarray:
    """Embed a single query via BGE server."""
    import urllib.request
    body = json.dumps({"model": model, "input": [query]}).encode("utf-8")
    req = urllib.request.Request(
        f"{bge_url}/api/embed", data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        q = np.array(json.load(r)["embeddings"][0], dtype=np.float32)
    q /= (np.linalg.norm(q) or 1)
    return q


def run_rrf_det(questions: list[dict], dense_arm: str, lex_arm: str,
                weights: list[float] = [1.0, 0.35], k: int = 5,
                pool: int = 100, topk: int = 10, chunkset: str = "fixed600",
                bge_url: str = "http://localhost:8378") -> list[dict]:
    """Run deterministic RRF for all questions. Returns list of {qid, ranked_chunk_ids}."""
    vecs, ids = load_dense(dense_arm, chunkset)
    lex = LexIndex(chunkset, lex_arm, ids)

    results = []
    for i, q in enumerate(questions):
        qvec = embed_query(q["question"], bge_url)
        d_rank = dense_rank(qvec, vecs, pool)
        l_rank = lex.rank(q["question"], pool)
        fused = fuse(d_rank, l_rank, weights=weights, k=k, topk=topk)
        fused_ids = [ids[idx] for idx in fused]
        results.append({"qid": q["qid"], "ranked_chunk_ids": fused_ids,
                        "status": "ok", "arm": f"rrf-det_{dense_arm}_{lex_arm}"})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(questions)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-arm", default="V9_QSURF")
    ap.add_argument("--lex-arm", default="V9_QSURF")
    ap.add_argument("--weights", default="1.0,0.35", help="dense,lex weights")
    ap.add_argument("--k", type=int, default=5, help="RRF k parameter")
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--chunkset", default="fixed600")
    ap.add_argument("--gold", default=str(HE.parent.parent / "docs" / "QASet" / "레거시" / "v3_retrieval" / "qa_gold_v3.jsonl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--core-only", action="store_true", default=True)
    ap.add_argument("--output", default=str(BASE / "out" / "rrf_det_result.jsonl"))
    ap.add_argument("--bge-url", default="http://localhost:8378")
    a = ap.parse_args()

    # Load questions
    questions = []
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        if r["split"] != a.split:
            continue
        if a.core_only and not r.get("core_retrieval"):
            continue
        questions.append({"qid": r["qid"], "question": r["question"]})

    ws = [float(x) for x in a.weights.split(",")]
    print(f"C0-RRF-DET: {len(questions)}문항, dense={a.dense_arm}, lex={a.lex_arm}, w={ws}, k={a.k}", flush=True)

    results = run_rrf_det(questions, a.dense_arm, a.lex_arm, ws, a.k, a.pool, a.topk, a.chunkset, a.bge_url)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {a.output}", flush=True)


if __name__ == "__main__":
    main()
