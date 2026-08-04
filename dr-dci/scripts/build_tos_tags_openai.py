"""shinhan-tos 청크 시맨틱 태그 생성 (OpenAI API, 재개 가능).

build_tos_tags.py(claude CLI)의 대체 경로. CLI는 10워커 동시 호출에서 rate limit에
걸려 성공률이 30%대로 무너졌고, 동시성을 낮추면 4,242건에 8시간이 걸린다.
OpenAI API는 병렬 처리와 재시도가 가능해 10분 내에 끝난다.

산출 형식은 build_tos_tags.py와 동일하므로 두 경로의 결과를 섞어 쓸 수 있다
(이미 생성된 1,350건은 그대로 재사용되고 실패분만 재시도된다).

출력: data/tags/shinhan-tos_chunk.jsonl (증분) → shinhan-tos_chunk.json (최종)
사용: python scripts/build_tos_tags_openai.py [--limit N] [--workers 16]
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")

RAW = BASE / "data" / "raw" / "shinhan-tos"
OUT_DIR = BASE / "data" / "tags"
JSONL = OUT_DIR / "shinhan-tos_chunk.jsonl"
FINAL = OUT_DIR / "shinhan-tos_chunk.json"

URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"

SYSTEM = (
    "너는 보험 문서에 시맨틱 태그를 붙이는 분류기다. "
    'JSON만 출력한다: {"tags": ["대분류>중분류>소분류", ...]}'
)
PROMPT = """다음 보험 문서 청크의 내용을 나타내는 3단계 계층 시맨틱 태그를 1~3개 부여하라.

형식: "대분류>중분류>소분류" (각 층은 한국어 명사구 2~8자, '>'는 정확히 2개)
대분류 예시: 계약, 보험금, 보험료, 해지환급, 면책, 특약, 상품구성, 공시, 운영

[문서] {doc}
[조항] {section}
[본문]
{text}
"""


def tag_one(chunk: dict, session: requests.Session, api_key: str) -> dict:
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(
                doc=chunk["doc"], section=chunk.get("section", ""),
                text=chunk["text"][:1500])},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(5):
        try:
            resp = session.post(URL, json=payload, headers=headers, timeout=90)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"status {resp.status_code}")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            tags = [
                t for t in json.loads(content).get("tags", [])
                if isinstance(t, str) and t.count(">") == 2
            ][:3]
            return {"chunk_id": chunk["_id"], "tags": tags, "ok": bool(tags)}
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt == 4:
                return {"chunk_id": chunk["_id"], "tags": [], "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"[:200]}
            time.sleep(2 ** attempt + random.random())
    return {"chunk_id": chunk["_id"], "tags": [], "ok": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY 없음 (.env 확인)")

    corpus = [json.loads(l) for l in open(RAW / "corpus.jsonl", encoding="utf-8")]

    done = set()
    if JSONL.exists():
        for line in open(JSONL, encoding="utf-8"):
            row = json.loads(line)
            if row.get("ok"):
                done.add(row["chunk_id"])
    todo = [c for c in corpus if c["_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"전체 {len(corpus)} / 기존 성공 {len(done)} / 이번 처리 {len(todo)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    started = time.time()
    session = requests.Session()
    with open(JSONL, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(tag_one, c, session, api_key) for c in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += row["ok"]
            if i % 200 == 0 or i == len(todo):
                rate = i / max(1e-6, time.time() - started)
                print(f"  {i}/{len(todo)} 성공 {n_ok} ({rate:.1f}/s, "
                      f"남은 {int((len(todo) - i) / max(rate, 1e-6))}s)", flush=True)

    tags = {}
    for line in open(JSONL, encoding="utf-8"):
        row = json.loads(line)
        if row.get("ok"):
            tags[row["chunk_id"]] = row["tags"]
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=1)
    cov = len(tags) / len(corpus)
    print(f"이번 성공 {n_ok}/{len(todo)} · 누적 {len(tags)}/{len(corpus)} "
          f"(커버리지 {cov:.1%}) → {FINAL}")


if __name__ == "__main__":
    main()
