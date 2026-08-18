#!/usr/bin/env python3
"""로컬 vLLM 서버를 사용해 청크 메타데이터(V6/V9)를 생성한다.

gen_llm_meta_json.py의 프롬프트·sanitize 로직을 그대로 복사하되,
API 호출만 로컬 vLLM(OpenAI-compatible)으로 바꾼 자립형 스크립트다.

사용법:
    python gen_meta_local.py --schema v9
    python gen_meta_local.py --schema v6 --limit 100 --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request, error as urllib_error

# ---------------------------------------------------------------- 설정
VLLM_URL = "http://localhost:8201/v1/chat/completions"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

OUT = Path(__file__).resolve().parent / "out"

# ---------------------------------------------------------------- JSONL I/O
def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- 항법 탐지
SEP_ONLY = re.compile(r"^[\s\|\-:·．.]*$")


def is_navigation(chunk: dict) -> bool:
    """결정론적 low_content 판정 -- 모델 판단에 앞선다.

    구분선/페이지번호만 있는 청크, 문자다운 문자가 40자 미만인 청크만
    항법(navigation) 조각으로 본다.
    """
    text = chunk["text"]
    if SEP_ONLY.match(text):
        return True
    substantive = re.sub(r"[\s\|\-:·．.0-9]", "", text)
    return len(substantive) < 40


# ---------------------------------------------------------------- v6 프롬프트
SYSTEM_COMMON = """너는 한국 보험 약관 문서를 색인하는 검색 엔지니어다.
주어진 약관 청크 하나를 읽고, 검색 색인에 붙일 메타데이터 5개 필드를 JSON으로 만든다.

가장 중요한 규칙: **청크 본문에 이미 있는 표현을 그대로 복사하지 마라.**
본문에 있는 단어는 이미 검색된다. 특히 keywords는 고객이 실제로 검색창에
칠 법한 구어체·축약형 표현이어야 하고, 본문의 문어체 표현과는 달라야 의미가
있다.

출력 필드 (모두 필수, 값을 모르면 문자열은 "", 배열은 []):
  rider    특약명. 문서 전체 이름이 아니라 이 청크가 속한 특약 하나. 최대 30자.
  article  조 번호+제목. 예: "제3조(보험금의 지급사유)". 최대 30자.
  nature   이 조항의 성격(의미역) 한 구절. 예: "보험금 지급사유", "면책사유",
           "용어정의", "납입면제", "해지환급금", "청구절차", "갱신조건". 최대 20자.
  topics   [문자열 0~3개]  이 청크가 다루는 구체적 주제. 각 12자 이내.
  keywords [문자열 0~4개]  고객이 검색창에 칠 법한 구어체 표현. 각 16자 이내.

절대 하지 말 것:
- "보험", "약관", "보험금"처럼 그 자체로 아무 청크에나 붙는 일반어를
  topics/keywords의 단독 값으로 쓰지 마라.
- 이 문서 어느 청크에나 들어맞을 법한 범용 topics를 만들지 마라 — 색인을
  오염시킨다. topics는 이 청크에만 해당해야 한다.
- 본문에 없는 숫자·비율·금액·질병분류코드(C50 등)를 지어내지 마라.
- 확실하지 않으면 빈 문자열/빈 배열로 두어라. 지어내는 것보다 비우는 게 낫다.

표·목차 청크에 대하여:
  목차 행이나 표 머리글도 **검색 대상이다**. `| 제3조 보험금의 지급사유 | 65 |`
  같은 행은 "이 약관에 그 조항이 있다"는 정보를 담는다. 이런 청크에도 보이는
  라벨에서 rider/article/nature/topics/keywords를 뽑아라 — 절대 통째로
  비우지 마라(단, 실질 내용이 없는 구분선·페이지번호뿐인 청크는 별도로
  처리되어 이 프롬프트까지 오지 않는다)."""

PURE_ADDENDUM = """

--- 지침 ---
문서 구조 정보는 전혀 주어지지 않는다. 청크 본문만 보고 rider(특약명)와
article(조항)을 추정하라. 본문에 조 번호나 특약명이 명시돼 있지 않으면
억지로 만들어내지 말고 ""로 둬라."""


# ---------------------------------------------------------------- v9 프롬프트
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
만들어내지 말고 narrow는 ""로, terms는 확실한 것만 남겨라."""


def build_system_v6() -> str:
    return SYSTEM_COMMON + PURE_ADDENDUM


def build_system_v9() -> str:
    return SYSTEM_V9 + PURE_ADDENDUM_V9


# ---------------------------------------------------------------- sanitize (v6)
GENERIC_BLOCK = {"보험", "약관", "보험금", "특약", "보험계약"}


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


def sanitize(raw: dict, nav: bool) -> dict:
    """v6 길이 상한 / 범용어 제거를 코드로 강제한다."""
    if nav:
        return {"rider": "", "article": "", "nature": "", "topics": [], "keywords": [],
                "low_content": True}
    return {
        "rider": _clip_str(raw.get("rider"), 30),
        "article": _clip_str(raw.get("article"), 30),
        "nature": _clip_str(raw.get("nature"), 20),
        "topics": _clip_list(raw.get("topics"), 3, 12),
        "keywords": _clip_list(raw.get("keywords"), 4, 16),
        "low_content": False,
    }


# ---------------------------------------------------------------- sanitize (v9)
_JO_RX = re.compile(r"제\s?\d+(?:\s?-\s?\d+)?\s?조(?:\s?의\s?\d+)?")
_ANNEX_RX = re.compile(r"(?:별표|부표|별첨)\s?\d*")
_CODE_RX = re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{2}(?:\.\d{1,2})?(?![A-Za-z])")
_QTY_RX = re.compile(r"\d+\s*(?:일|년|회|%|세|만원|억원|배|개월|주|시간)")
_HANGUL4_RX = re.compile(r"[가-힣]{4,}")


def _narrow_grounded(narrow: str, body: str) -> bool:
    """narrow가 본문 실재 표기를 정말 담았는지 측정만 한다 (강제하지 않음)."""
    if not narrow or not body:
        return False
    for rx in (_CODE_RX, _QTY_RX, _ANNEX_RX, _HANGUL4_RX):
        for m in rx.findall(narrow):
            if m and m in body:
                return True
    return False


def sanitize_v9(raw: dict, nav: bool, body: str = "") -> dict:
    """v9 질의-층위 계약을 코드로 강제한다.

    강제하는 것:
      (1) 길이 상한 (프롬프트와 이중)
      (2) wide/narrow가 본문에 그대로 있으면 폐기
      (3) wide에 조번호/별표/질병코드가 들어가면 폐기 (narrow에는 걸지 않음)
      (4) terms: 범용어 제거 + 길이 클립 후 wide/narrow와 완전히 같은 항목,
          terms 내부 중복을 제거한다.
    """
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


# ---------------------------------------------------------------- 배치 프롬프트
def build_batch_prompt(chunks: list[dict], schema: str) -> str:
    """배치 프롬프트를 만든다. 청크마다 번호를 붙여 하나의 유저 메시지로 합친다.

    응답 형식: {"items": [{"index": 1, ...필드...}, ...]}
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        body = c["text"][:2400]
        parts.append(f"--- 청크 #{i} ---\n{body}")

    if schema == "v6":
        field_desc = '각 항목은 {"rider","article","nature","topics","keywords"} 필드를 갖는다.'
    else:
        field_desc = '각 항목은 {"wide","narrow","terms"} 필드를 갖는다.'

    footer = (
        f"\n\n위 {len(chunks)}개 청크 각각에 대해 메타데이터를 만들어라.\n"
        f"반드시 JSON으로 응답하라. 최상위 키는 \"items\"이고, 값은 배열이다.\n"
        f"배열의 각 항목에는 \"index\" (1부터 시작하는 청크 번호)가 있어야 한다.\n"
        f"{field_desc}\n"
        f"예시 형식: {{\"items\": [{{\"index\": 1, ...}}, {{\"index\": 2, ...}}, ...]}}"
    )
    return "\n\n".join(parts) + footer


# ---------------------------------------------------------------- vLLM API
class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.pin = self.pout = self.calls = self.fails = 0

    def add(self, pin: int, pout: int, fail: bool = False):
        with self.lock:
            self.pin += pin
            self.pout += pout
            self.calls += 1
            self.fails += int(fail)


def call_vllm(messages: list[dict], usage: Usage, retries: int = 4) -> tuple[dict | None, str | None]:
    """vLLM의 OpenAI-compatible 엔드포인트를 호출한다."""
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    last_err = "unknown"
    for attempt in range(retries):
        try:
            req = request.Request(
                VLLM_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            u = body.get("usage", {})
            usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed, None

        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            usage.add(0, 0, fail=True)
            return None, last_err

    return None, last_err


# ---------------------------------------------------------------- 배치 처리
def process_batch(
    batch: list[dict],
    system: str,
    schema: str,
    usage: Usage,
) -> list[dict]:
    """배치(최대 10개)를 하나의 API 호출로 처리하고 결과 리스트를 반환한다."""
    prompt = build_batch_prompt(batch, schema)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    parsed, err = call_vllm(messages, usage)

    results: list[dict] = []

    if parsed is None:
        # 전체 배치 실패
        for c in batch:
            results.append({"chunk_id": c["chunk_id"], "ok": False, "error": err})
        return results

    # items 배열에서 index로 매핑
    items_raw = parsed.get("items", [])
    if not isinstance(items_raw, list):
        items_raw = [parsed]  # 단일 항목이 바로 나온 경우 대비

    by_index: dict[int, dict] = {}
    for item in items_raw:
        if isinstance(item, dict):
            idx = item.get("index")
            if isinstance(idx, int):
                by_index[idx] = item

    for i, c in enumerate(batch, 1):
        raw = by_index.get(i)
        if raw is None:
            results.append({"chunk_id": c["chunk_id"], "ok": False,
                            "error": f"index {i} not found in response"})
            continue

        rec = {"chunk_id": c["chunk_id"], "ok": True}
        if schema == "v9":
            rec.update(sanitize_v9(raw, nav=False, body=c["text"]))
        else:
            rec.update(sanitize(raw, nav=False))
        results.append(rec)

    return results


# ---------------------------------------------------------------- 재개
def load_done_ids(path: Path) -> set[str]:
    """이미 완료된 chunk_id를 읽는다. 쓰다 만 마지막 줄은 무시한다."""
    if not path.exists():
        return set()
    done: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["chunk_id"])
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
    return done


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="로컬 vLLM 서버로 청크 메타데이터(V6/V9)를 생성한다.")
    ap.add_argument("--schema", required=True, choices=["v6", "v9"],
                    help="v6=5필드(rider/article/nature/topics/keywords), "
                         "v9=3필드(wide/narrow/terms, 질의 층위 분리)")
    ap.add_argument("--limit", type=int, default=0,
                    help="처리할 최대 청크 수 (0=전체)")
    ap.add_argument("--workers", type=int, default=2,
                    help="동시 API 호출 워커 수 (기본 2)")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="API 호출당 청크 수 (기본 10)")
    ap.add_argument("--dry-run", action="store_true",
                    help="API를 호출하지 않고 프롬프트 2개만 출력하고 종료")
    ap.add_argument("--out", default="",
                    help="출력 파일 경로 (기본: out/llm_meta_{v6,v9}_local.jsonl)")
    args = ap.parse_args()

    v9 = args.schema == "v9"
    system = build_system_v9() if v9 else build_system_v6()

    # 입력 청크 로드
    chunk_path = OUT / "chunks.jsonl"
    if not chunk_path.exists():
        raise SystemExit(f"[오류] 입력 파일이 없습니다: {chunk_path}")
    chunks = read_jsonl(chunk_path)
    if args.limit:
        chunks = chunks[:args.limit]

    nav_ids = {c["chunk_id"] for c in chunks if is_navigation(c)}

    # dry-run
    if args.dry_run:
        non_nav = [c for c in chunks if c["chunk_id"] not in nav_ids]
        sample = non_nav[:min(args.batch_size, len(non_nav))]
        if not sample:
            print("[DRY-RUN] 표시할 비-항법 청크가 없습니다 (limit을 늘려보세요)")
            return
        prompt = build_batch_prompt(sample, args.schema)
        print(f"{'=' * 80}")
        print(f"[DRY-RUN] schema={args.schema}  batch_size={len(sample)}")
        print(f"{'=' * 80}")
        print("--- SYSTEM ---")
        print(system)
        print("\n--- USER PROMPT ---")
        print(prompt)
        return

    # 출력 경로
    default_name = f"llm_meta_{args.schema}_local.jsonl"
    path = Path(args.out) if args.out else OUT / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(path)

    # 항법 조각 기록 (API 호출 없이)
    nav_to_write = [c for c in chunks if c["chunk_id"] in nav_ids and c["chunk_id"] not in done]
    if nav_to_write:
        with open(path, "a", encoding="utf-8") as fh:
            for c in nav_to_write:
                rec = {"chunk_id": c["chunk_id"], "ok": True}
                rec.update(sanitize_v9({}, nav=True) if v9 else sanitize({}, nav=True))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add(c["chunk_id"])

    # 신규 대상
    todo = [c for c in chunks if c["chunk_id"] not in done and c["chunk_id"] not in nav_ids]
    print(f"schema={args.schema}  대상 {len(chunks):,} / 항법조각 {len(nav_ids):,} "
          f"({100 * len(nav_ids) / max(len(chunks), 1):.1f}%) / 캐시 {len(done):,} / "
          f"신규 API호출 {len(todo):,}배치 / model={VLLM_MODEL}")
    print(f"배치크기={args.batch_size}  워커={args.workers}")
    print(f"출력: {path}")

    if not todo:
        print("신규 처리할 청크가 없습니다.")
        return

    # 배치 분할
    batches: list[list[dict]] = []
    for i in range(0, len(todo), args.batch_size):
        batches.append(todo[i:i + args.batch_size])

    usage = Usage()
    lock = threading.Lock()
    t0 = time.time()
    fh = open(path, "a", encoding="utf-8")

    processed = 0

    def work(batch: list[dict]):
        nonlocal processed
        results = process_batch(batch, system, args.schema, usage)
        with lock:
            for rec in results:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += len(batch)
            if processed % (args.batch_size * 10) == 0 or processed == len(todo):
                fh.flush()
                el = time.time() - t0
                rate = processed / max(el, 1e-9)
                remaining = (len(todo) - processed) / max(rate, 1e-9) / 60
                print(f"  {processed:,}/{len(todo):,}  {rate:.1f}청크/s  "
                      f"실패 {usage.fails}  ~{remaining:.0f}분 남음",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, batches))

    fh.close()

    el = time.time() - t0
    print(f"\n완료 {processed:,}건 / 실패 {usage.fails} / {el:.0f}초")
    print(f"토큰 in={usage.pin:,} out={usage.pout:,}")

    # 채움률 보고
    rows = read_jsonl(path)
    filled = collections.Counter()
    field_keys = ("wide", "narrow", "terms") if v9 else (
        "rider", "article", "nature", "topics", "keywords")
    for r in rows:
        for k in field_keys:
            if r.get(k):
                filled[k] += 1
        if r.get("low_content"):
            filled["low_content"] += 1
        if r.get("ok") is False:
            filled["error"] += 1
    n = max(len(rows), 1)
    print(f"\n필드 채움률 (n={len(rows):,}):")
    for k, v in filled.most_common():
        print(f"   {k:16s} {100 * v / n:5.1f}%")
    if v9:
        grounded = sum(1 for r in rows if r.get("narrow_grounded"))
        print(f"   {'narrow_grounded':16s} {100 * grounded / n:5.1f}%  (게이트 지표, 폐기 기준 아님)")


if __name__ == "__main__":
    main()
