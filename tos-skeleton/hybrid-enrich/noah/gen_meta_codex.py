#!/usr/bin/env python3
"""청크 메타데이터(v6/v9) 생성 — codex CLI 배치 호출.

프롬프트·sanitize는 gen_meta_local.py에서 **그대로 복사**했다(그 모듈은 import
시점에 vLLM 의존이 있어 import하지 않는다). 전송 계층만 gen_element_tags_llm.py의
codex 하네스를 이식했다: `codex exec --output-schema ... -o out.json -`,
ThreadPoolExecutor 배치, done-id 재개, 레이트리밋 감지, 실패 시 반으로 쪼개 재시도.

사용법
    python gen_meta_codex.py --schema v9 --dry-run --limit 5
    python gen_meta_codex.py --schema v9 --limit 5
    python gen_meta_codex.py --schema v6 --workers 4 --batch-size 20
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------- Windows stdout 인코딩
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

# ---------------------------------------------------------------- codex CLI 설정
CODEX = os.environ.get("CODEX_BIN") or str(
    Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")

WORK_DIR = OUT / "codex_meta_work"
WORKERS_DEFAULT = int(os.environ.get("META_WORKERS", "4"))
BATCH_DEFAULT = int(os.environ.get("CODEX_META_BATCH", "20"))
CALL_TIMEOUT = int(os.environ.get("CODEX_META_TIMEOUT", "600"))
MODEL_DEFAULT = os.environ.get("CODEX_META_MODEL", "gpt-5.6-luna")

BODY_CAP = 2400  # 원 레시피와 동일

# ================================================================ gen_meta_local.py 복사부
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


# ================================================================ /복사부 끝

# ---------------------------------------------------------------- JSON Schema (structured output)
_V6_ITEM = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "rider": {"type": "string"},
        "article": {"type": "string"},
        "nature": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["index", "rider", "article", "nature", "topics", "keywords"],
    "additionalProperties": False,
}

_V9_ITEM = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "wide": {"type": "string"},
        "narrow": {"type": "string"},
        "terms": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["index", "wide", "narrow", "terms"],
    "additionalProperties": False,
}


def envelope_schema(schema: str) -> dict:
    """codex --output-schema 용 봉투 스키마 (gen_element_tags_llm.py와 동일 스타일)."""
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": _V9_ITEM if schema == "v9" else _V6_ITEM},
        },
        "required": ["items"],
        "additionalProperties": False,
    }


FIELD_KEYS = {
    "v6": ("rider", "article", "nature", "topics", "keywords"),
    "v9": ("wide", "narrow", "terms"),
}


# ---------------------------------------------------------------- 배치 프롬프트
def build_batch_prompt(chunks: list[dict], schema: str) -> str:
    """유저 프롬프트. gen_meta_local.build_batch_prompt와 동일한 형식(1-based index)."""
    parts = []
    for i, c in enumerate(chunks, 1):
        body = c["text"][:BODY_CAP]
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


def build_full_prompt(chunks: list[dict], schema: str, system: str) -> str:
    """codex stdin으로 넘길 최종 프롬프트 = system + user."""
    return system + "\n\n" + build_batch_prompt(chunks, schema)


# ---------------------------------------------------------------- 레이트리밋 감지
# ⚠ 이 정규식은 **실패한 호출의 출력에만** 건다(gen_element_tags_llm.py 주석 참고).
_RATE_LIMIT_RX = re.compile(
    r"rate[ _-]?limit(?:ed|_exceeded)?\b"
    r"|429\s+too many requests"
    r"|quota exceeded"
    r"|usage limit reached"
    r"|you.{0,20}exceeded your current quota",
    re.I,
)

_call_counter = itertools.count()


class RateLimitDetected(Exception):
    """codex 응답에서 레이트리밋 신호를 발견했다 — 재시도하지 않고 즉시 전파한다."""


def _rl_snippet(combined: str) -> str:
    hits = [ln.strip() for ln in combined.splitlines() if _RATE_LIMIT_RX.search(ln)]
    return " | ".join(hits[:3])[:300] if hits else combined.strip()[-200:]


# ---------------------------------------------------------------- 통계
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


# ---------------------------------------------------------------- codex 호출
def call_codex(prompt: str, work: Path, call_id: int, model: str,
               stats: Stats) -> tuple[list | None, str | None]:
    """codex exec를 한 번 부른다. 프롬프트는 stdin, 결과는 `-o` 파일로 받는다."""
    o = work / f"out_{call_id}.json"
    cmd = [CODEX, "exec", "--model", model, "-c", "model_reasoning_effort=low",
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--output-schema", str(work / "schema.json"), "-o", str(o), "-"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run(cmd, input=prompt.encode("utf-8"),
                           capture_output=True, timeout=CALL_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        stats.add(fail=True)
        return None, "timeout"
    except OSError as exc:
        stats.add(fail=True)
        return None, f"spawn:{exc}"

    out_text = r.stdout.decode("utf-8", "replace")
    err_text = r.stderr.decode("utf-8", "replace")
    combined = out_text + "\n" + err_text

    m = re.search(r"tokens used\s*\n\s*([\d,]+)", combined)
    tok = int(m.group(1).replace(",", "")) if m else 0

    try:
        obj = json.loads(o.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        stats.add(fail=True)
        if r.returncode != 0 and _RATE_LIMIT_RX.search(combined):
            raise RateLimitDetected(_rl_snippet(combined))
        return None, f"output_parse:{type(exc).__name__}:{combined[-300:]}"
    finally:
        try:
            o.unlink()
        except OSError:
            pass

    items = obj.get("items")
    if not isinstance(items, list):
        stats.add(fail=True)
        return None, "no_items_key"
    stats.add(tokens=tok)
    return items, None


# ---------------------------------------------------------------- 정렬
def align_items(items: list, n: int) -> list[dict] | None:
    """응답 items를 1-based index로 정렬한다. 개수/범위가 안 맞으면 None."""
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


# ---------------------------------------------------------------- 배치 처리
def process_batch(chunks: list[dict], schema: str, system: str, stats: Stats,
                  work: Path, model: str, max_retries: int = 3) -> list[dict]:
    """배치 1개 처리. 실패 시 백오프 재시도 → 반으로 쪼개 재시도 → ok:false."""
    n = len(chunks)
    prompt = build_full_prompt(chunks, schema, system)

    for attempt in range(max_retries):
        call_id = next(_call_counter)
        items, err = call_codex(prompt, work, call_id, model, stats)
        if err is None:
            aligned = align_items(items, n)
            if aligned is not None:
                results = []
                for c, raw in zip(chunks, aligned):
                    rec = {"chunk_id": c["chunk_id"], "ok": True}
                    if schema == "v9":
                        rec.update(sanitize_v9(raw, nav=False, body=c["text"]))
                    else:
                        rec.update(sanitize(raw, nav=False))
                    results.append(rec)
                return results
        time.sleep(2 ** attempt)

    if len(chunks) > 1:
        mid = len(chunks) // 2
        return (process_batch(chunks[:mid], schema, system, stats, work, model, max_retries)
                + process_batch(chunks[mid:], schema, system, stats, work, model, max_retries))

    return [{"chunk_id": chunks[0]["chunk_id"], "ok": False,
             "error": "all_retries_failed"}]


# ---------------------------------------------------------------- 재개
def load_done_ids(path: Path) -> set[str]:
    """이미 성공(ok:true)한 chunk_id만 done으로 본다 — 실패는 재시도 대상."""
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


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="청크 메타데이터(v6/v9) 생성 — codex CLI 배치")
    ap.add_argument("--schema", required=True, choices=["v6", "v9"])
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 청크 수 (0=전체)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true",
                    help="codex 호출 없이 프롬프트+스키마만 출력")
    ap.add_argument("--input", default="out/chunks.jsonl", help="입력 경로 (BASE 기준)")
    ap.add_argument("--output", default="", help="출력 경로 (기본 out/llm_meta_{v6,v9}.jsonl)")
    args = ap.parse_args()

    schema = args.schema
    system = build_system_v9() if schema == "v9" else build_system_v6()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE / input_path
    if not input_path.exists():
        print(f"[오류] 입력 파일이 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else OUT / f"llm_meta_{schema}.jsonl"
    if not output_path.is_absolute():
        output_path = BASE / output_path

    chunks = read_jsonl(input_path)
    if args.limit:
        chunks = chunks[:args.limit]

    nav_ids = {c["chunk_id"] for c in chunks if is_navigation(c)}
    non_nav = [c for c in chunks if c["chunk_id"] not in nav_ids]

    # ---------------- dry-run
    if args.dry_run:
        sample = non_nav[:min(args.batch_size, len(non_nav))] or chunks[:1]
        print("=" * 80)
        print(f"[DRY-RUN] schema={schema}  batch={len(sample)}  model={args.model}  "
              f"청크 {len(chunks):,} / 항법 {len(nav_ids):,} / API대상 {len(non_nav):,}")
        print(f"codex={CODEX}")
        print("=" * 80)
        print("--- OUTPUT SCHEMA ---")
        print(json.dumps(envelope_schema(schema), ensure_ascii=False, indent=2))
        print("\n--- PROMPT (stdin) ---")
        print(build_full_prompt(sample, schema, system))
        print("\n[DRY-RUN] codex 호출 없음")
        return

    # ---------------- 준비
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "schema.json").write_text(
        json.dumps(envelope_schema(schema), ensure_ascii=False), encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(output_path)

    fh = output_path.open("a", encoding="utf-8")
    wlock = threading.Lock()

    def emit(rows: list[dict]) -> None:
        with wlock:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    # 항법 청크는 LLM 호출 없이 결정론적으로 기록
    nav_rows = []
    for c in chunks:
        if c["chunk_id"] in nav_ids and c["chunk_id"] not in done:
            rec = {"chunk_id": c["chunk_id"], "ok": True}
            rec.update(sanitize_v9({}, nav=True) if schema == "v9" else sanitize({}, nav=True))
            nav_rows.append(rec)
            done.add(c["chunk_id"])
    if nav_rows:
        emit(nav_rows)
        print(f"[항법] {len(nav_rows):,}건 low_content 기록 (codex 호출 없음)")

    todo = [c for c in non_nav if c["chunk_id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    print(f"[gen_meta_codex/{schema}] 청크 {len(chunks):,} / 항법 {len(nav_ids):,} / "
          f"캐시 {len(done):,} / API대상 {len(todo):,} -> 배치 {len(batches):,}개 "
          f"(batch={args.batch_size}, workers={args.workers}, model={args.model})",
          flush=True)
    print(f"출력: {output_path}")

    if not batches:
        print("처리할 청크가 없습니다.")
        fh.close()
        return

    stats = Stats()
    t0 = time.time()
    n_done = 0
    rate_limited: str | None = None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_batch, b, schema, system, stats, WORK_DIR, args.model): len(b)
                for b in batches}
        try:
            for fut in as_completed(futs):
                rows = fut.result()
                emit(rows)
                n_done += len(rows)
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                remaining = (len(todo) - n_done) / max(rate, 1e-9)
                if n_done % 100 < args.batch_size or n_done == len(todo):
                    print(f"   {n_done:,}/{len(todo):,}  {rate:.1f}/s  "
                          f"~{remaining / 60:.1f}분 남음  {stats.line()}", flush=True)
        except RateLimitDetected as exc:
            rate_limited = str(exc)
            ex.shutdown(wait=False, cancel_futures=True)

    fh.close()

    if rate_limited is not None:
        print(f"\n[중단] codex 레이트리밋으로 보인다. 즉시 중단한다.\n"
              f"부분 산출물: {output_path}\n감지된 신호: {rate_limited}\n", file=sys.stderr)
        sys.exit(1)

    # ---------------- 보고
    elapsed = time.time() - t0
    rows = read_jsonl(output_path)
    n = max(len(rows), 1)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_fail = sum(1 for r in rows if r.get("ok") is False)

    print(f"\n완료: {stats.calls:,}호출 / 실패 {stats.fails} / {elapsed:.0f}초 / "
          f"토큰 {stats.tokens:,}")
    print(f"\n필드 채움률 (n={len(rows):,}):")
    for k in FIELD_KEYS[schema]:
        filled = sum(1 for r in rows if r.get(k))
        print(f"   {k:16s} {100 * filled / n:5.1f}%  ({filled:,}건)")
    low = sum(1 for r in rows if r.get("low_content"))
    print(f"   {'low_content':16s} {100 * low / n:5.1f}%  ({low:,}건)")
    if schema == "v9":
        g = sum(1 for r in rows if r.get("narrow_grounded"))
        print(f"   {'narrow_grounded':16s} {100 * g / n:5.1f}%  (게이트 지표)")
    print(f"   ok:true {n_ok:,}건 / ok:false {n_fail:,}건")


if __name__ == "__main__":
    main()
