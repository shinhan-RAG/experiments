#!/usr/bin/env python3
"""청크 임베딩 — BGE-m3-ko via embedder.py.

view_V9.jsonl 또는 view_BASE.jsonl의 텍스트를 임베딩하여
out/emb/chunk_{view}.npy + chunk_{view}_ids.json 으로 저장한다.

Usage:
    EMBED_ENDPOINT=http://localhost:8101/v1 python embed_chunks.py --view V9
    EMBED_ENDPOINT=http://localhost:8101/v1 python embed_chunks.py --view BASE
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))
import embedder


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="청크 임베딩 (BGE-m3-ko)")
    ap.add_argument("--view", default="V9", choices=("V9", "BASE", "RAW", "RULE", "RULE2", "RULE2A"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    out_base = Path(args.out_dir) if getattr(args, "out_dir", None) else OUT
    view_path = out_base / f"view_{args.view}.jsonl"
    if not view_path.exists():
        raise SystemExit(f"[오류] 뷰 파일이 없습니다: {view_path}. build_views.py를 먼저 실행하세요.")

    rows = read_jsonl(view_path)
    if args.limit > 0:
        rows = rows[:args.limit]

    texts = [r["text"] for r in rows]
    ids = [r["chunk_id"] for r in rows]

    print(f"임베딩 시작: {len(texts):,}건 / view={args.view} / batch={args.batch}")
    print(f"소스: {embedder.source()}")

    t0 = time.time()
    V = embedder.encode(texts, batch=args.batch, max_len=args.max_len, progress=True)

    emb_dir = out_base / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)

    npy_path = emb_dir / f"chunk_{args.view}.npy"
    ids_path = emb_dir / f"chunk_{args.view}_ids.json"
    meta_path = emb_dir / f"chunk_{args.view}_meta.json"

    np.save(npy_path, V)
    json.dump(ids, open(ids_path, "w"))

    meta = {
        "model": embedder.MODEL,
        "embed_source": embedder.source(),
        "view": args.view,
        "n": len(ids),
        "dim": int(V.shape[1]),
        "batch": args.batch,
        "max_len": args.max_len,
        "elapsed_s": round(time.time() - t0, 1),
        "view_sha256": hashlib.sha256(view_path.read_bytes()).hexdigest()[:16],
        "vec_sha256": hashlib.sha256(V.tobytes()).hexdigest()[:16],
    }
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
