#!/usr/bin/env python3
"""문서 단위 summary 생성 — 9,417건 전부, haiku 1회/문서.

입력: 파일명 + 본문 앞 2,000자 (frontmatter 필드는 사용 안 함 — 실험 조건 독립성)
출력: out/doc_summary.jsonl (증분, 재실행 시 이어서)
"""
import json
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
DEST = OUT / "doc_summary.jsonl"

PROMPT = """다음은 보험 문서(md)의 파일명과 본문 앞부분이다.
이 문서가 어떤 문서이고 무엇을 다루는지 3~4문장으로 요약하라.
검색 색인으로 쓸 것이므로 상품·보장·문서 성격 키워드를 자연스럽게 포함하라. 요약만 출력.

파일명: {fn}
본문 앞부분:
{body}"""


def one(rel):
    p = ROOT / rel
    body = unicodedata.normalize(
        "NFC", p.read_text(encoding="utf-8", errors="ignore")[:2000])
    r = subprocess.run(["claude", "-p", "--model", "haiku"],
                       input=PROMPT.format(fn=Path(rel).name, body=body),
                       capture_output=True, text=True, timeout=120)
    return rel, r.stdout.strip()[:1000]


def main():
    docs = [json.loads(l)["file"] for l in open(OUT / "doc_frontmatter.jsonl")]
    done = set()
    if DEST.exists():
        done = {json.loads(l)["file"] for l in open(DEST)}
    todo = [d for d in docs if d not in done]
    print(f"요약 생성 {len(todo):,}건 (기존 {len(done):,})", flush=True)
    with open(DEST, "a") as out, ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, d) for d in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rel, summ = fu.result()
            except Exception:
                continue
            if summ:
                out.write(json.dumps({"file": rel, "summary": summ},
                                     ensure_ascii=False) + "\n")
                out.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("DOC_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    main()
