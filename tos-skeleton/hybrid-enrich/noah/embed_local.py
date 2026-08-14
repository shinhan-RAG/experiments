#!/usr/bin/env python3
"""ollama bge-m3 로컬 임베딩 생성기.

입력: out/view_{BASE,V6,V9}.jsonl  ({"chunk_id": ..., "text": ...})
출력: out/vec_{VIEW}.npy        (float32, 행 L2 정규화 → 코사인 == 내적)
      out/vec_{VIEW}_ids.json   (list[str], 행 순서와 index-align)

중간 체크포인트: out/vec_{VIEW}.part.npy + out/vec_{VIEW}.part.json (done 개수)
→ 중단되면 다음 실행에서 이어서 진행한다.

사용:
  python embed_local.py --views BASE,V6,V9 --limit 0 --batch 32
  python embed_local.py --probe 50            # 처리량 프로브만
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
DIM = 1024
MAX_CHARS = 6000  # bge-m3 8192 토큰. 안전 절단, 전 뷰 동일 규칙.
CKPT_EVERY = 5    # N배치마다 체크포인트 저장

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def embed_batch(texts: list[str], timeout: int = 600) -> list[list[float]]:
    """배치 임베딩. /api/embed 사용 (배치 엔드포인트)."""
    payload = json.dumps({"model": MODEL, "input": texts, "keep_alive": "30m"}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["embeddings"]


def embed_batch_retry(texts: list[str], tries: int = 3) -> list[list[float]]:
    for attempt in range(tries):
        try:
            return embed_batch(texts)
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"  [retry {attempt + 1}] {exc}", file=sys.stderr, flush=True)
            time.sleep(5)
    raise RuntimeError("unreachable")


def load_view(view: str, limit: int = 0) -> tuple[list[str], list[str]] | None:
    """out/view_{VIEW}.jsonl 로드. 없으면 None."""
    path = OUT / f"view_{view}.jsonl"
    if not path.exists():
        print(f"[SKIP] {view}: 뷰 파일 없음 → {path}")
        return None
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(row["chunk_id"])
            texts.append((row.get("text") or "")[:MAX_CHARS])
            if limit and len(ids) >= limit:
                break
    return ids, texts


def run_view(view: str, limit: int, batch: int) -> bool:
    loaded = load_view(view, limit)
    if loaded is None:
        return False
    ids, texts = loaded
    total = len(texts)
    if total == 0:
        print(f"[SKIP] {view}: 빈 뷰")
        return False

    dst = OUT / f"vec_{view}.npy"
    ids_dst = OUT / f"vec_{view}_ids.json"
    part = OUT / f"vec_{view}.part.npy"
    part_meta = OUT / f"vec_{view}.part.json"

    vecs = np.zeros((total, DIM), dtype=np.float32)
    done = 0
    if part.exists() and part_meta.exists():
        meta = json.loads(part_meta.read_text(encoding="utf-8"))
        if meta.get("total") == total and meta.get("model") == MODEL:
            prev = np.load(part)
            done = min(int(meta.get("done", 0)), prev.shape[0], total)
            vecs[:done] = prev[:done]
            print(f"[{view}] 체크포인트 재개: {done}/{total}")
        else:
            print(f"[{view}] 체크포인트 불일치(total/model) → 처음부터")

    t0 = time.time()
    processed = 0
    for i in range(done, total, batch):
        chunk = texts[i:i + batch]
        emb = embed_batch_retry(chunk)
        vecs[i:i + len(emb)] = np.asarray(emb, dtype=np.float32)
        processed += len(chunk)
        bidx = (i - done) // batch
        if bidx % CKPT_EVERY == 0 or i + batch >= total:
            np.save(part, vecs)
            part_meta.write_text(
                json.dumps({"done": i + len(chunk), "total": total, "model": MODEL}),
                encoding="utf-8",
            )
            rate = processed / max(1e-9, time.time() - t0)
            print(f"[{view}] {i + len(chunk)}/{total}  {rate:.2f} docs/s", flush=True)

    elapsed = time.time() - t0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = (vecs / norms).astype(np.float32)

    np.save(dst, vecs)
    ids_dst.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    for tmp in (part, part_meta):
        if tmp.exists():
            tmp.unlink()

    rate = processed / max(1e-9, elapsed)
    print(f"[{view}] 완료: shape={vecs.shape} ids={len(ids)} "
          f"신규={processed} 경과={elapsed:.1f}s ({rate:.2f} docs/s)")
    print(f"[{view}] → {dst.name}, {ids_dst.name}")
    return True


def probe(n: int, batch: int) -> None:
    """처리량 프로브: 실제 문서 n건을 임베딩하고 docs/sec 보고."""
    src = None
    for cand in ("view_BASE.jsonl", "view_V6.jsonl", "view_V9.jsonl", "chunks.jsonl"):
        if (OUT / cand).exists():
            src = OUT / cand
            break
    if src is None:
        print("[ERR] 프로브용 소스 파일 없음")
        return
    texts: list[str] = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            texts.append((json.loads(line).get("text") or "")[:MAX_CHARS])
            if len(texts) >= n:
                break
    avg = sum(len(t) for t in texts) / max(1, len(texts))
    embed_batch_retry(texts[:2])  # 모델 워밍업
    t0 = time.time()
    got = 0
    for i in range(0, len(texts), batch):
        got += len(embed_batch_retry(texts[i:i + batch]))
    el = time.time() - t0
    print(f"[PROBE] src={src.name} n={got} batch={batch} avg_chars={avg:.0f} "
          f"elapsed={el:.2f}s → {got / max(1e-9, el):.2f} docs/s")
    print(f"[PROBE] 추정: 5797 chunks x 3 views = 17391 → "
          f"{17391 / max(1e-9, got / el) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description="ollama bge-m3 뷰 임베딩")
    ap.add_argument("--views", default="BASE,V6,V9", help="쉼표 구분 뷰 목록")
    ap.add_argument("--limit", type=int, default=0, help="뷰당 최대 문서 수 (0=전체)")
    ap.add_argument("--batch", type=int, default=32, help="요청당 텍스트 수")
    ap.add_argument("--probe", type=int, default=0, help="처리량 프로브만 실행 (문서 수)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.probe:
        probe(args.probe, args.batch)
        return

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    ok = 0
    t0 = time.time()
    for view in views:
        if run_view(view, args.limit, args.batch):
            ok += 1
    print(f"[ALL] {ok}/{len(views)} 뷰 처리, 총 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
