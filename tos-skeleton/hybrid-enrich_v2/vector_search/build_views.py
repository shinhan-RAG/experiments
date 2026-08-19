#!/usr/bin/env python3
"""검색 뷰(BASE / V9) 조립.

out/chunks.jsonl 전량을 기준으로, out/llm_meta_v9.jsonl의 메타데이터를
접두(prefix)로 붙인다. 메타데이터가 없는 청크는 접두 없이 본문만 나간다.

접두 예산 240자. 초과 시 접두만 자르고 본문은 절대 자르지 않는다.
행 스키마: {"chunk_id": "...", "text": "<prefix + body>"}

Usage:
    python build_views.py
    python build_views.py --views V9 --limit 100
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

META_BUDGET = 240
ALL_VIEWS = ("BASE", "V9")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_meta(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        cid = r.get("chunk_id")
        if cid and r.get("ok"):
            out[cid] = r
    return out


def v9_prefix(m: dict) -> str:
    wide = m.get("wide", "")
    narrow = m.get("narrow", "")
    terms = m.get("terms") or []
    if not wide and not narrow and not terms:
        return ""
    parts = []
    if wide:
        parts.append(f"[질문]{wide}")
    if narrow:
        parts.append(f"[세부]{narrow}")
    if terms:
        parts.append("[표현]" + " · ".join(terms))
    return "\n".join(parts) + "\n"


def clip_prefix(prefix: str) -> tuple[str, bool]:
    if len(prefix) <= META_BUDGET:
        return prefix, False
    return prefix[:META_BUDGET], True


def main() -> None:
    ap = argparse.ArgumentParser(description="BASE/V9 검색 뷰 생성")
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 청크 수 (0=전체)")
    ap.add_argument("--views", default="BASE,V9", help="생성할 뷰 (콤마 구분)")
    args = ap.parse_args()

    views = [v.strip().upper() for v in args.views.split(",") if v.strip()]
    bad = [v for v in views if v not in ALL_VIEWS]
    if bad:
        raise SystemExit(f"[오류] 알 수 없는 뷰: {bad} (가능: {ALL_VIEWS})")

    chunk_path = OUT / "chunks.jsonl"
    if not chunk_path.exists():
        raise SystemExit(f"[오류] 입력 파일이 없습니다: {chunk_path}")

    chunks = read_jsonl(chunk_path)
    if args.limit:
        chunks = chunks[:args.limit]

    meta_v9 = load_meta(OUT / "llm_meta_v9.jsonl")
    print(f"청크 {len(chunks):,} / v9메타 {len(meta_v9):,}")

    stats: list[tuple] = []

    for view in views:
        path = OUT / f"view_{view}.jsonl"
        n = 0
        n_meta = 0
        n_trunc = 0
        meta_chars = 0
        total_chars = 0
        with path.open("w", encoding="utf-8") as fh:
            for c in chunks:
                cid = c["chunk_id"]
                body = c["text"]
                prefix = ""
                if view == "V9":
                    m = meta_v9.get(cid)
                    if m:
                        prefix = v9_prefix(m)
                if prefix:
                    prefix, trunc = clip_prefix(prefix)
                    n_trunc += int(trunc)
                    n_meta += 1
                    meta_chars += len(prefix)
                text = prefix + body
                total_chars += len(text)
                fh.write(json.dumps({"chunk_id": cid, "text": text},
                                    ensure_ascii=False) + "\n")
                n += 1
        stats.append((view, n, n_meta, meta_chars, total_chars, n_trunc, path))

    print()
    print(f"{'view':6s} {'rows':>7s} {'coverage':>9s} {'avg_meta':>9s} "
          f"{'avg_total':>10s} {'trunc':>6s}")
    print("-" * 54)
    for view, n, n_meta, meta_chars, total_chars, n_trunc, path in stats:
        cov = 100 * n_meta / max(n, 1)
        avg_meta = meta_chars / max(n_meta, 1) if n_meta else 0.0
        avg_total = total_chars / max(n, 1)
        print(f"{view:6s} {n:7,d} {cov:8.1f}% {avg_meta:9.1f} "
              f"{avg_total:10.1f} {n_trunc:6,d}")
    print()
    for *_, path in stats:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
