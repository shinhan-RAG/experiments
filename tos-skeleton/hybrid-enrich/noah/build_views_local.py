#!/usr/bin/env python3
"""검색 뷰(BASE / V6 / V9) 조립.

out/chunks.jsonl 전량(5,797건)을 기준으로, 있으면 out/llm_meta_v6.jsonl /
out/llm_meta_v9.jsonl의 메타데이터를 **접두(prefix)** 로 붙인다. 메타데이터가
없는 청크는 접두 없이 본문만 나간다 — 검색기가 모든 청크에 도달할 수 있어야
하므로 어떤 뷰든 행 수는 항상 전체 청크 수와 같다.

접두 예산은 240자(build_views.py::v9_qsurf/_join 조립과 동일). 예산 초과 시
**접두만** 자르고 본문은 절대 자르지 않는다.

출력 행 스키마(고정): {"chunk_id": "...", "text": "<prefix + body>"}

사용법
    python build_views_local.py
    python build_views_local.py --views V9 --limit 100
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

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

META_BUDGET = 240
ALL_VIEWS = ("BASE", "V6", "V9")


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
    """chunk_id -> 메타 레코드. ok:false 행은 메타 없음으로 취급한다."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        cid = r.get("chunk_id")
        if cid and r.get("ok"):
            out[cid] = r
    return out


# ---------------------------------------------------------------- 접두 생성
def v6_prefix(m: dict) -> str:
    """v6 메타 객체를 compact JSON 문자열로 직렬화한다."""
    obj = {
        "rider": m.get("rider", ""),
        "article": m.get("article", ""),
        "nature": m.get("nature", ""),
        "topics": m.get("topics", []) or [],
        "keywords": m.get("keywords", []) or [],
    }
    if not any(obj.values()):
        return ""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def v9_prefix(m: dict) -> str:
    """[질문]{wide}\\n[세부]{narrow}\\n[표현]{t1 · t2 · t3}\\n"""
    wide = (m.get("wide") or "").strip()
    narrow = (m.get("narrow") or "").strip()
    terms = [t for t in (m.get("terms") or []) if t]
    if not (wide or narrow or terms):
        return ""
    return (f"[질문]{wide}\n"
            f"[세부]{narrow}\n"
            f"[표현]{' · '.join(terms)}\n")


def clip_prefix(prefix: str) -> tuple[str, bool]:
    """접두를 예산 안으로 자른다. (접두, 잘렸는지)"""
    if len(prefix) <= META_BUDGET:
        return prefix, False
    return prefix[:META_BUDGET], True


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="BASE/V6/V9 검색 뷰 생성")
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 청크 수 (0=전체)")
    ap.add_argument("--views", default="BASE,V6,V9", help="생성할 뷰 (콤마 구분)")
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

    meta_v6 = load_meta(OUT / "llm_meta_v6.jsonl")
    meta_v9 = load_meta(OUT / "llm_meta_v9.jsonl")
    print(f"청크 {len(chunks):,} / v6메타 {len(meta_v6):,} / v9메타 {len(meta_v9):,}")

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
                if view == "V6":
                    m = meta_v6.get(cid)
                    if m:
                        prefix = v6_prefix(m)
                elif view == "V9":
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

    # ---------------- 통계표
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
