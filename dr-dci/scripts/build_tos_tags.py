"""shinhan-tos 청크 시맨틱 태그 생성 (claude CLI haiku, 재개 가능).

회의(0804) §26.2의 엘리먼트/청크 시맨틱 태그 — 3단계 depth 계층 태그를
청크당 1~3개 부여한다. 예: "계약>주계약>가입조건".

- tos_llm.call_claude_json 재사용 (QA 생성과 동일한 로컬 claude CLI 경로)
- 진행분은 jsonl에 증분 저장 → 재실행 시 이어서 (KT Cloud/로컬 어디서든)
- 완료 후 {chunk_id: [tags]} 형태의 최종 json 산출

출력: data/tags/shinhan-tos_chunk.jsonl (증분) → shinhan-tos_chunk.json (최종)
사용: python scripts/build_tos_tags.py [--limit N] [--workers 8]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tos_llm import call_claude_json  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "shinhan-tos"
OUT_DIR = BASE / "data" / "tags"
JSONL = OUT_DIR / "shinhan-tos_chunk.jsonl"
FINAL = OUT_DIR / "shinhan-tos_chunk.json"

PROMPT = """다음은 보험 문서의 한 청크다. 이 청크의 내용을 나타내는 3단계 계층 시맨틱 태그를 1~3개 부여하라.

형식: "대분류>중분류>소분류" (각 층은 한국어 명사구, 2~8자)
대분류 예시: 계약, 보험금, 보험료, 해지환급, 면책, 특약, 상품구성, 공시, 운영
JSON만 출력: {{"tags": ["...>...>...", ...]}}

[문서] {doc}
[조항] {section}
[본문]
{text}
"""


def tag_one(chunk: dict) -> dict:
    prompt = PROMPT.format(
        doc=chunk["doc"], section=chunk.get("section", ""),
        text=chunk["text"][:1500],
    )
    try:
        res = call_claude_json(prompt, timeout=120)
    except OSError:   # 일시적 spawn 실패(WinError 2 등) — 해당 청크만 실패 처리, 재실행 시 재시도
        res = None
    tags = []
    if isinstance(res, dict):
        tags = [t for t in res.get("tags", [])
                if isinstance(t, str) and t.count(">") == 2][:3]
    return {"chunk_id": chunk["_id"], "tags": tags, "ok": bool(tags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 최대 청크 수 (0=전부)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

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
    print(f"전체 {len(corpus)} / 완료 {len(done)} / 이번 처리 {len(todo)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(JSONL, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(tag_one, c): c["_id"] for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += row["ok"]
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} (성공 {n_ok})", flush=True)

    # 최종 json 재구성 (성공분만, 최신 행 우선)
    tags = {}
    for line in open(JSONL, encoding="utf-8"):
        row = json.loads(line)
        if row.get("ok"):
            tags[row["chunk_id"]] = row["tags"]
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=1)
    print(f"이번 실행 성공 {n_ok}/{len(todo)} · 누적 {len(tags)} → {FINAL}")


if __name__ == "__main__":
    main()
