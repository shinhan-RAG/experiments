#!/usr/bin/env python3
"""요약 기반 컬렉션 frontmatter (비교 대조군) — haiku로 컬렉션당 요약 1건.

입력: 대표 문서 앞 2,000자 + 파일 목록. 출력: out/collection_summary.jsonl (증분)
"""
import json
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
DEST = OUT / "collection_summary.jsonl"

PROMPT = """다음은 한 보험 상품 폴더의 파일 목록과 대표 문서 일부다.
이 상품이 무엇이고(보종·특징), 어떤 문서들이 있는지 4~6문장으로 요약하라.
검색 색인으로 쓸 것이므로 상품 특징·보장 키워드를 자연스럽게 포함하라. 요약만 출력.

상품 폴더명: {name}
파일 목록: {files}
대표 문서 일부:
{body}"""


def one(fm):
    name = fm["collection"]
    d = ROOT / name
    rep = (fm["representative"].get("판매약관")
           or fm["representative"].get("사업방법서") or "")
    body = ""
    if rep and (d / rep).exists():
        body = unicodedata.normalize(
            "NFC", (d / rep).read_text(encoding="utf-8", errors="ignore")[:2000])
    files = ", ".join(iv["file"] for iv in fm["inventory"][:15])
    r = subprocess.run(["claude", "-p", "--model", "haiku"],
                       input=PROMPT.format(name=name, files=files, body=body),
                       capture_output=True, text=True, timeout=120)
    return name, r.stdout.strip()[:1200]


def main():
    fms = [json.loads(l) for l in open(OUT / "collection_fm.jsonl")]
    done = set()
    if DEST.exists():
        done = {json.loads(l)["collection"] for l in open(DEST)}
    todo = [f for f in fms if f["collection"] not in done and f["inventory"]]
    print(f"요약 생성 {len(todo)}건 (기존 {len(done)})", flush=True)
    with open(DEST, "a") as out, ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, f) for f in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                name, summ = fu.result()
            except Exception:
                continue
            out.write(json.dumps({"collection": name, "summary": summ},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    main()
