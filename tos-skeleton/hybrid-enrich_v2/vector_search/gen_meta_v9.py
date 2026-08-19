#!/usr/bin/env python3
"""V9 메타데이터 생성 — Qwen3-32B (OpenAI-compatible) 배치 호출.

hybrid-enrich/noah/gen_meta_codex.py에서 포팅. Codex CLI 대신
OpenAI-compatible HTTP API를 직접 호출한다.

Usage:
    python gen_meta_v9.py --dry-run --limit 5
    python gen_meta_v9.py --batch-size 5 --workers 2
    python gen_meta_v9.py --batch-size 5 --workers 2 --input out/chunks.jsonl
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

LLM_URL = os.environ.get("LLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-FP8")

WORKERS_DEFAULT = int(os.environ.get("META_WORKERS", "2"))
BATCH_DEFAULT = int(os.environ.get("META_BATCH", "5"))
CALL_TIMEOUT = int(os.environ.get("META_TIMEOUT", "300"))

BODY_CAP = 2400

# ================================================================ V9 프롬프트
SYSTEM_V9 = """너는 한국 보험 약관 청크를 색인하는 검색 엔지니어다.
주어진 약관 청크 하나를 읽고, 검색 색인에 붙일 필드 3개를 JSON으로 만든다.

이 작업의 전제를 먼저 이해해라 — 검색은 "조각끼리 구별하기"가 아니라 "질문과
조각을 잇기"다. 그래서 이 스키마는 조각을 **설명하지 않는다.** 조각을 읽고
이 조각에 **물어볼 수 있는 질문**만 적어라. 질문에는 층위가 있다 — 어떤 고객은
"보험료 안 내도 되는 경우"처럼 뭉툭하게 묻고, 어떤 고객은 "중증 갑상선암도
면제되냐"처럼 콕 집어 묻는다. 두 층위를 **각각** 적어야 둘 다 잡는다. 한쪽만
적으면 다른 층위의 질문을 놓친다.

출력 필드 (모두 필수):
  wide   [문자열]  이 조각이 답이 되는 범주 수준 질문 1개, 28자 이내. 조 번호·
         별표 번호·질병분류코드 같은 식별자를 쓰지 마라. **형제 청크(같은 조,
         같은 특약의 다른 조각)도 답할 수 있는 뭉툭한 질문이어도 된다 — 오히려
         뭉툭해야 주제형 질문과 붙는다.** 겹침을 걱정하지 말고 적어라.
         예: "보험료를 안 내도 되는 경우는?", "해약환급금은 어떻게 계산하나요?"
  narrow [문자열]  이 조각이 답이 되는 구체 질문 1개, **40자 이내**. 본문에 실재하는
         표기(질병코드·급여금 명칭·한도·횟수·부표 번호 등) 중 **최소 하나를
         그대로 포함**하되, **반드시 물음표로 끝나는 완결된 질문**이어야 한다.
         표기를 질문 안에 **감싸는** 것이지 본문 구절을 **옮겨 적는** 것이 아니다.
         본문에 그대로 있는 문자열은 폐기된다 — 본문에 있는 말은 이미 검색되기 때문이다.
         그런 표기가 본문에 없으면 억지로 만들지 말고 "".
         좋은 예: "중증 갑상선암도 납입면제 되나요?"
                  "C8591 갑상선 바늘생검은 어떤 조건에서 인정되나요?"
                  "30일한도형은 얼마까지 나오나요?"
         나쁜 예(폐기됨): "급여 갑상선 바늘생검 조직병리진단"   ← 본문 구절 복사, 질문 아님
                          "제2-3조 보험금 지급에 관한 세부규정"  ← 〃
  terms  [문자열 3~4개]  고객이 검색창에 칠 법한 **구어체** 표현. 각 16자 이내.
         오타·띄어쓰기 변형·축약형 허용.
         예: "암걸리면 보험료면제", "납입 안해도되나", "보험료 안냄"

절대 하지 말 것:
- 본문 문장을 그대로 복사하지 마라. 본문에 있는 말은 이미 검색된다.
- 본문에 없는 숫자·비율·금액·질병분류코드를 지어내지 마라.
- "보험", "약관", "보험금"처럼 아무 청크에나 붙는 일반어만으로 채우지 마라.

표·목차 청크에 대하여:
  표 머리글이나 목차 행도 검색 대상이다. 그 표가 무엇을 나열하는지 wide/narrow로
  적어라 — 통째로 비우지 마라. (실질 내용이 없는 구분선·페이지번호뿐인 청크는
  별도로 처리되어 이 프롬프트까지 오지 않는다.)"""

PURE_ADDENDUM_V9 = """

--- 지침 ---
문서 구조 정보는 전혀 주어지지 않는다. 특약명도 조 번호도 따로 주어지지 않는다.
청크 본문만 보고 세 필드를 판단하라. 본문에서 근거를 찾을 수 없으면 억지로
만들어내지 말고 narrow는 ""로, terms는 확실한 것만 남겨라.

/no_think"""


def build_system() -> str:
    return SYSTEM_V9 + PURE_ADDENDUM_V9


# ================================================================ 항법 탐지
SEP_ONLY = re.compile(r"^[\s\|\-:·．.]*$")


def is_navigation(chunk: dict) -> bool:
    text = chunk["text"]
    if SEP_ONLY.match(text):
        return True
    substantive = re.sub(r"[\s\|\-:·．.0-9]", "", text)
    return len(substantive) < 40


# ================================================================ sanitize
GENERIC_BLOCK = {"보험", "약관", "보험금", "특약", "보험계약"}

_JO_RX = re.compile(r"제\s?\d+(?:\s?-\s?\d+)?\s?조(?:\s?의\s?\d+)?")
_ANNEX_RX = re.compile(r"(?:별표|부표|별첨)\s?\d*")
_CODE_RX = re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{2}(?:\.\d{1,2})?(?![A-Za-z])")
_QTY_RX = re.compile(r"\d+\s*(?:일|년|회|%|세|만원|억원|배|개월|주|시간)")
_HANGUL4_RX = re.compile(r"[가-힣]{4,}")


def _clip_str(v, n: int) -> str:
    s = " ".join(str(v or "").split())
    return s[:n]


def _clip_list(v, max_items: int, max_chars: int) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, (int, float)):
            x = str(x)
        if not isinstance(x, str):
            continue
        s = " ".join(x.split())
        if s and s not in GENERIC_BLOCK:
            out.append(s[:max_chars])
    return out[:max_items]


def _narrow_grounded(narrow: str, body: str) -> bool:
    if not narrow or not body:
        return False
    for rx in (_CODE_RX, _QTY_RX, _ANNEX_RX, _HANGUL4_RX):
        for m in rx.findall(narrow):
            if m and m in body:
                return True
    return False


def sanitize_v9(raw: dict, nav: bool, body: str = "") -> dict:
    if nav:
        return {"wide": "", "narrow": "", "terms": [], "low_content": True,
                "narrow_grounded": False, "dropped_body": 0, "dropped_wide_id": 0}

    dropped_body = dropped_wide_id = 0

    wide = _clip_str(raw.get("wide"), 28)
    if wide and wide in body:
        wide, dropped_body = "", dropped_body + 1
    elif wide and (_JO_RX.search(wide) or _ANNEX_RX.search(wide) or _CODE_RX.search(wide)):
        wide, dropped_wide_id = "", dropped_wide_id + 1

    narrow = _clip_str(raw.get("narrow"), 40)
    if narrow and narrow in body:
        narrow, dropped_body = "", dropped_body + 1

    terms = _clip_list(raw.get("terms"), 4, 16)
    seen: set[str] = set()
    kept_terms = []
    for t in terms:
        if t == wide or t == narrow or t in seen:
            continue
        seen.add(t)
        kept_terms.append(t)

    return {
        "wide": wide,
        "narrow": narrow,
        "terms": kept_terms,
        "low_content": False,
        "narrow_grounded": _narrow_grounded(narrow, body),
        "dropped_body": dropped_body,
        "dropped_wide_id": dropped_wide_id,
    }


# ================================================================ 배치 프롬프트
def build_batch_prompt(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        body = c["text"][:BODY_CAP]
        parts.append(f"--- 청크 #{i} ---\n{body}")

    field_desc = '각 항목은 {"wide","narrow","terms"} 필드를 갖는다.'
    footer = (
        f"\n\n위 {len(chunks)}개 청크 각각에 대해 메타데이터를 만들어라.\n"
        f"반드시 JSON으로 응답하라. 최상위 키는 \"items\"이고, 값은 배열이다.\n"
        f"배열의 각 항목에는 \"index\" (1부터 시작하는 청크 번호)가 있어야 한다.\n"
        f"{field_desc}\n"
        f"예시 형식: {{\"items\": [{{\"index\": 1, ...}}, {{\"index\": 2, ...}}, ...]}}"
    )
    return "\n\n".join(parts) + footer


# ================================================================ LLM 호출
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_call_counter = itertools.count()


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.fails = 0
        self.tokens = 0

    def add(self, tokens: int = 0, fail: bool = False) -> None:
        with self.lock:
            self.calls += 1
            self.fails += int(fail)
            self.tokens += tokens

    def line(self) -> str:
        return f"calls={self.calls} fails={self.fails} tokens={self.tokens:,}"


def call_llm(system: str, user: str, stats: Stats,
             retries: int = 3) -> tuple[list | None, str | None]:
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                LLM_URL, data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            stats.add(fail=True)
            if exc.code == 429:
                wait = min(60, 2 ** (attempt + 2))
                print(f"  [429] rate limited, {wait}s 대기...", file=sys.stderr)
                time.sleep(wait)
                continue
            return None, f"http_{exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            stats.add(fail=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"network:{exc}"
        except json.JSONDecodeError as exc:
            stats.add(fail=True)
            return None, f"json_decode:{exc}"

        usage = data.get("usage", {})
        tok = usage.get("total_tokens", 0)

        content = data["choices"][0]["message"]["content"]
        content = _THINK_RE.sub("", content).strip()

        # JSON 추출
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    obj = json.loads(content[start:end])
                except json.JSONDecodeError:
                    stats.add(tokens=tok, fail=True)
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None, "json_parse_fail"
            else:
                stats.add(tokens=tok, fail=True)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, "no_json_in_response"

        items = obj.get("items")
        if not isinstance(items, list):
            stats.add(tokens=tok, fail=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, "no_items_key"
        stats.add(tokens=tok)
        return items, None

    return None, "all_retries_failed"


# ================================================================ 정렬
def align_items(items: list, n: int) -> list[dict] | None:
    if not isinstance(items, list) or len(items) != n:
        return None
    out: list[dict | None] = [None] * n
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        idx = item.get("index")
        slot = idx - 1 if isinstance(idx, int) else None
        if slot is None or not (0 <= slot < n) or out[slot] is not None:
            slot = pos
        if not (0 <= slot < n) or out[slot] is not None:
            return None
        out[slot] = item
    return None if any(x is None for x in out) else out  # type: ignore[return-value]


# ================================================================ 배치 처리
def process_batch(chunks: list[dict], system: str, stats: Stats,
                  max_retries: int = 3) -> list[dict]:
    n = len(chunks)
    user = build_batch_prompt(chunks)

    for attempt in range(max_retries):
        items, err = call_llm(system, user, stats)
        if err is None and items is not None:
            aligned = align_items(items, n)
            if aligned is not None:
                results = []
                for c, raw in zip(chunks, aligned):
                    rec = {"chunk_id": c["chunk_id"], "ok": True}
                    rec.update(sanitize_v9(raw, nav=False, body=c["text"]))
                    results.append(rec)
                return results
        time.sleep(2 ** attempt)

    if len(chunks) > 1:
        mid = len(chunks) // 2
        return (process_batch(chunks[:mid], system, stats, max_retries)
                + process_batch(chunks[mid:], system, stats, max_retries))

    return [{"chunk_id": chunks[0]["chunk_id"], "ok": False,
             "error": "all_retries_failed"}]


# ================================================================ 재개
def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and "chunk_id" in row:
                done.add(row["chunk_id"])
    return done


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


# ================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description="V9 메타데이터 생성 — Qwen3-32B")
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 청크 수 (0=전체)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM 호출 없이 프롬프트만 출력")
    ap.add_argument("--input", default=str(OUT / "chunks.jsonl"),
                    help="입력 경로")
    ap.add_argument("--output", default="",
                    help="출력 경로 (기본 out/llm_meta_v9.jsonl)")
    args = ap.parse_args()

    system = build_system()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"[오류] 입력 파일이 없습니다: {input_path}")

    output_path = Path(args.output) if args.output else OUT / "llm_meta_v9.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(input_path)
    if args.limit:
        chunks = chunks[:args.limit]

    nav_ids = {c["chunk_id"] for c in chunks if is_navigation(c)}
    non_nav = [c for c in chunks if c["chunk_id"] not in nav_ids]

    if args.dry_run:
        sample = non_nav[:min(args.batch_size, len(non_nav))] or chunks[:1]
        print("=" * 80)
        print(f"[DRY-RUN] batch={len(sample)}  model={LLM_MODEL}  "
              f"청크 {len(chunks):,} / 항법 {len(nav_ids):,} / API대상 {len(non_nav):,}")
        print(f"endpoint={LLM_URL}")
        print("=" * 80)
        print("\n--- SYSTEM ---")
        print(system)
        print("\n--- USER ---")
        print(build_batch_prompt(sample))
        print("\n[DRY-RUN] LLM 호출 없음")
        return

    done = load_done_ids(output_path)

    fh = output_path.open("a", encoding="utf-8")
    wlock = threading.Lock()

    def emit(rows: list[dict]) -> None:
        with wlock:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    nav_rows = []
    for c in chunks:
        if c["chunk_id"] in nav_ids and c["chunk_id"] not in done:
            rec = {"chunk_id": c["chunk_id"], "ok": True}
            rec.update(sanitize_v9({}, nav=True))
            nav_rows.append(rec)
            done.add(c["chunk_id"])
    if nav_rows:
        emit(nav_rows)
        print(f"[항법] {len(nav_rows):,}건 low_content 기록 (LLM 호출 없음)")

    todo = [c for c in non_nav if c["chunk_id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    print(f"[gen_meta_v9] 청크 {len(chunks):,} / 항법 {len(nav_ids):,} / "
          f"캐시 {len(done):,} / API대상 {len(todo):,} -> 배치 {len(batches):,}개 "
          f"(batch={args.batch_size}, workers={args.workers}, model={LLM_MODEL})",
          flush=True)
    print(f"출력: {output_path}")

    if not batches:
        print("처리할 청크가 없습니다.")
        fh.close()
        return

    stats = Stats()
    t0 = time.time()
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_batch, b, system, stats): len(b) for b in batches}
        for fut in as_completed(futs):
            rows = fut.result()
            emit(rows)
            n_done += len(rows)
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-9)
            remaining = (len(todo) - n_done) / max(rate, 1e-9)
            if n_done % 50 < args.batch_size or n_done == len(todo):
                print(f"   {n_done:,}/{len(todo):,}  {rate:.1f}/s  "
                      f"~{remaining / 60:.1f}분 남음  {stats.line()}", flush=True)

    fh.close()

    elapsed = time.time() - t0
    rows = read_jsonl(output_path)
    n = max(len(rows), 1)

    print(f"\n완료: {stats.calls:,}호출 / 실패 {stats.fails} / {elapsed:.0f}초 / "
          f"토큰 {stats.tokens:,}")
    print(f"\n필드 채움률 (n={len(rows):,}):")
    for k in ("wide", "narrow", "terms"):
        filled = sum(1 for r in rows if r.get(k))
        print(f"   {k:16s} {100 * filled / n:5.1f}%  ({filled:,}건)")
    low = sum(1 for r in rows if r.get("low_content"))
    print(f"   {'low_content':16s} {100 * low / n:5.1f}%  ({low:,}건)")
    g = sum(1 for r in rows if r.get("narrow_grounded"))
    print(f"   {'narrow_grounded':16s} {100 * g / n:5.1f}%  (게이트 지표)")
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_fail = sum(1 for r in rows if r.get("ok") is False)
    print(f"   ok:true {n_ok:,}건 / ok:false {n_fail:,}건")


if __name__ == "__main__":
    main()
