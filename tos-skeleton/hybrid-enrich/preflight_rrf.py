#!/usr/bin/env python3
"""RRF preflight check."""
from __future__ import annotations

import json
import os
import sys as _sys

if _sys.stdout.encoding and _sys.stdout.encoding.lower().startswith("cp"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
DOC = BASE / "noah_qaset" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
QA_CSV = BASE / "noah_qaset" / "정답셋_348_train_v3_retrieval_250212.csv"
VLLM_GEN = "http://localhost:8201/v1/models"
VLLM_EMBED = "http://localhost:8101/v1/models"
OLLAMA_EMBED = "http://localhost:11434/api/tags"


def check(label: str, ok: bool, detail: str = ""):
    status = "OK" if ok else "FAIL"
    print(f"  [{status:4s}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def probe_url(url: str, timeout: int = 5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def main():
    ok_all = True
    print("=== 소스 문서 ===")
    ok_all &= check("250212 MD 존재", DOC.exists(), f"{DOC.stat().st_size / 1e6:.1f} MB" if DOC.exists() else "")
    ok_all &= check("QA CSV 존재", QA_CSV.exists())
    ok_all &= check("out/ 디렉토리", OUT.exists() or not OUT.exists(), "auto-create")

    print("\n=== 생성된 데이터 ===")
    for name in ("elements.jsonl", "elements_repaired_v1.jsonl", "chunks.jsonl",
                 "element_tags_llm_v1.jsonl", "element_tags_new_repaired_full_v1.jsonl"):
        p = OUT / name
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            check(name, True, f"{n:,} rows")
        else:
            check(name, False, "미생성")

    for prefix in ("vec_base", "vec_V6", "vec_V9"):
        npy = OUT / f"{prefix}.npy"
        ids = OUT / f"{prefix}_ids.json"
        if npy.exists() and ids.exists():
            n = json.loads(ids.read_text())
            check(f"{prefix} 벡터", True, f"{len(n.get('ids', []))} vecs")
        else:
            check(f"{prefix} 벡터", False, "미생성")

    print("\n=== LLM 서버 ===")
    gen = probe_url(VLLM_GEN)
    ok_all &= check("vLLM 생성 서버 (8201)", gen is not None, "Qwen2.5-7B-Instruct" if gen else "미구동")

    embed = probe_url(VLLM_EMBED)
    check("vLLM 임베딩 서버 (8101)", embed is not None, "gte-Qwen2-1.5B" if embed else "미구동 (Ollama 대체 가능)")

    ollama = probe_url(OLLAMA_EMBED)
    check("Ollama (11434)", ollama is not None, "bge-m3" if ollama else "미구동")

    if not embed and not ollama:
        print("  [WARN] 임베딩 서버가 하나도 없음 — embed.py 실행 불가")
        ok_all = False

    print("\n=== 골드셋 ===")
    gold = OUT / "unified_gold.jsonl"
    if gold.exists():
        rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines()]
        n_core = sum(1 for r in rows if r.get("core_retrieval"))
        check("unified_gold.jsonl", True, f"{len(rows)} total, {n_core} core")
    else:
        check("unified_gold.jsonl", False, "build_unified_gold.py 실행 필요")

    print(f"\n{'전체 OK' if ok_all else '일부 실패 — 위 FAIL 항목 해결 필요'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
