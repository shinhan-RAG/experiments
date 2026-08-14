#!/usr/bin/env python3
"""엘리먼트 8-슬롯 시멘틱 태그 생성 v2 — 구조 규칙 + LLM 분업.

v1(gen_element_tags_llm.py)의 문제
----------------------------------
LLM에게 8슬롯 전부를 맡겼더니 degenerate 응답이 나왔다. 49,929건 실측 채움률:
contract 96.9%(115종) / schema 100%(4종) / role 42.4%(11종) /
**subject 9.7%(단 8종)** / qualifier 8.9% / article 3.2% / table 2.0% / reference 5.2%.
subject는 가장 중요한 슬롯인데(ablation: subject+schema만으로 전체 성능의 86.3% 유지)
사실상 비어 있었다.

v2의 분업
---------
* **구조 규칙(LLM 없음)** — contract / article / schema.
  이미 96.9% / 100%로 잘 나오던 슬롯이라 규칙으로 고정해 회귀를 막는다.
* **LLM** — subject / role / qualifier / reference / table.
  프롬프트를 subject 중심으로 다시 썼다: few-shot 3개, "급여금·질병·보장 대상이
  본문에 이름으로 나오는데 subject가 비면 오류"라는 명시적 금지, 원문 축자 추출 강제.

전송 계층은 v1과 동일하다 — codex CLI(`codex exec --output-schema ... -o out.json -`),
프롬프트는 stdin, ThreadPoolExecutor 병렬, done-id 기반 resume, 레이트리밋 즉시 중단,
실패 시 배치 반분할 재시도.

사용법
    python gen_element_tags_v2.py --dry-run --limit 5
    python gen_element_tags_v2.py --limit 500 --workers 8 --batch-size 20
    python gen_element_tags_v2.py            # 전수
"""
from __future__ import annotations

import argparse
import collections
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

WORK_DIR = OUT / "codex_tag_work_v2"
WORKERS_DEFAULT = int(os.environ.get("TAG_WORKERS", "8"))
BATCH_DEFAULT = int(os.environ.get("CODEX_TAG_BATCH", "20"))
CALL_TIMEOUT = int(os.environ.get("CODEX_TAG_TIMEOUT", "600"))
MODEL_DEFAULT = os.environ.get("CODEX_TAG_MODEL", "gpt-5.6-luna")

MAX_BODY_CHARS = 2400          # 모델에 보내는 엘리먼트 본문 상한
SUBJECT_FILL_GATE = 0.60       # subject 채움률이 이보다 낮으면 non-zero exit

# ---------------------------------------------------------------- 8-슬롯
FIELDS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")
LLM_FIELDS = ("subject", "role", "qualifier", "reference", "table")   # LLM 담당
RULE_FIELDS = ("contract", "article", "schema")                        # 구조 규칙 담당

VALID_ROLES = (
    "exclusion_exception", "premium_waiver", "payment_trigger", "payment_amount",
    "limit_frequency", "timing_period", "definition", "criteria_rule",
    "contract_lifecycle", "claim_procedure", "code_reference",
)
VALID_ROLES_SET = set(VALID_ROLES)

VALID_SCHEMAS = {"paragraph", "heading", "table", "formula"}

# ---------------------------------------------------------------- sanitize 상수
CONTRACT_BOILERPLATE_RE = re.compile(
    r"\(무배당[^)]*\)|\(간편\)|\(간편가입\)|\(해약환급금\s*미지급형\)|\(갱신형\)|\(일반형\)"
)
MAIN_WRAP_RE = re.compile(r"^주계약\((.*)\)$", re.S)
GENERIC_TERMS = {"보험", "약관", "보험금", "특약", "보험계약", "계약", "회사", "피보험자",
                 "계약자", "보험료", "지급", "보장", "해당", "경우"}
NAV_RE = re.compile(r"^[-=─━\s]+$|^\d+$")

# 조 번호+제목: "제3조 보험금의 지급사유", "제2-3조(보험금 지급에 관한 세부규정)",
# "제2조의2 한국표준질병·사인분류 적용 기준"
# 제목에는 '|'(목차 표의 페이지 열 구분자)와 괄호를 넣지 않는다.
ARTICLE_RE = re.compile(
    r"^\s*제\s*(\d+(?:\s*-\s*\d+)?)\s*조(?:\s*의\s*(\d+))?\s*[（(]?\s*([^\n|（()）]{0,40})",
    re.M)

# ---------------------------------------------------------------- JSON Schema (structured output)
# LLM에는 5개 슬롯만 맡긴다 — 구조 규칙이 채우는 3개는 스키마에서 아예 뺀다.
ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "subject": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "array", "items": {"type": "string"}},
                    "qualifier": {"type": "array", "items": {"type": "string"}},
                    "reference": {"type": "array", "items": {"type": "string"}},
                    "table": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["i", "subject", "role", "qualifier", "reference", "table"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- 시스템 프롬프트
SYSTEM_PROMPT = """너는 한국 보험약관 색인 전문가다. 약관 엘리먼트 텍스트를 읽고 아래 5개
슬롯을 채운다. 각 슬롯은 문자열 리스트(list[str])다. 해당 사항이 없을 때만 빈 리스트 [].

## 슬롯 정의

1. **subject** (가장 중요) — 이 엘리먼트가 실제로 이름을 대고 다루는 급여금·보장·질병·
   담보 대상 용어. 본문에 **글자 그대로 등장한 표현만** 쓴다. 최대 5개.
   예: ["암진단비", "갑상선암", "제자리암"], ["뇌졸중진단비"], ["수술급여금", "관혈수술"]

2. **role** — 의미역. 아래 11개 영문 식별자 중에서만 고른다. 최대 3개.
   - exclusion_exception : 면책·제외·예외·부지급·지급하지 않는 사유
   - premium_waiver      : 보험료 납입면제
   - payment_trigger     : 보험금 지급사유·지급조건
   - payment_amount      : 지급금액·지급률·산정·계산
   - limit_frequency     : 지급한도·횟수·일수·최초 1회
   - timing_period       : 보장개시·책임개시·대기기간·감액기간·보험기간
   - definition          : 용어의 정의·의미
   - criteria_rule       : 진단확정·판정기준·적용기준
   - contract_lifecycle  : 갱신·해지·소멸·무효·부활·환급
   - claim_procedure     : 보험금 청구·구비서류·절차
   - code_reference      : 질병분류코드·수가코드·부표·분류표

3. **qualifier** — 조건·기간·금액·비율 같은 구체적 한정 표현. 본문 축자. 최대 4개.
   예: ["90일 이내", "가입금액의 100%", "최초 1회에 한하여", "계약일부터 1년 미만"]

4. **reference** — 본문이 가리키는 다른 조항·별표·부표. 최대 5개.
   예: ["제5조", "별표3", "부표2"]

5. **table** — 엘리먼트가 표를 포함하면 표의 헤더/행 키. 표가 없으면 [].

## 절대 규칙

- **축자 추출**: 모든 값은 본문에 실제로 나타난 표현이어야 한다. 새 용어를 지어내거나,
  본문에 없는 상위 범주("질병", "재해", "보장" 등)로 바꿔 쓰지 마라.
- **subject가 비면 안 되는 경우**: 본문에 급여금 명칭·질병명·수술명·진단명·담보명이
  하나라도 이름으로 등장하면 subject를 빈 리스트로 두는 것은 **오류다**. 반드시 채워라.
  조 제목(예: "제3조 보험금의 지급사유")에 담긴 대상어, 표 헤더의 급여금 명칭,
  본문 중 "○○진단비", "○○수술급여금", "○○입원급여금" 같은 표현을 모두 훑어라.
- **일반어 금지**: "보험", "약관", "보험금", "특약", "계약", "회사", "피보험자" 같은
  범용 단어만 단독으로 subject에 넣지 마라. 단, "암보험금"처럼 대상어가 붙은 복합어는 OK.
- role은 반드시 위 11개 식별자 문자열 그대로. 한글로 쓰지 마라.

## 예시

[예시 1] 본문:
  제3조 보험금의 지급사유
  회사는 피보험자가 이 특약의 보험기간 중 암보장개시일 이후에 "암"으로 진단확정
  되었을 때에는 최초 1회에 한하여 암진단보험금(가입금액의 100%)을 지급합니다.
  다만, 갑상선암, 기타피부암, 제자리암 및 경계성종양은 제외합니다.
출력:
  {"subject":["암진단보험금","암","갑상선암","기타피부암","제자리암"],
   "role":["payment_trigger","payment_amount","exclusion_exception"],
   "qualifier":["암보장개시일 이후","최초 1회에 한하여","가입금액의 100%"],
   "reference":[],"table":[]}

[예시 2] 본문:
  제2-5조 보험료 납입면제
  회사는 피보험자가 특약보험기간 중 재해로 장해분류표에서 정한 장해지급률 50% 이상에
  해당하는 장해상태가 되었을 때에는 차회 이후의 특약보험료 납입을 면제합니다.
  장해분류표는 별표1(장해분류표)에서 정한 바에 따릅니다.
출력:
  {"subject":["장해상태","장해지급률","재해","장해분류표"],
   "role":["premium_waiver","criteria_rule"],
   "qualifier":["장해지급률 50% 이상","특약보험기간 중"],
   "reference":["별표1"],"table":[]}

[예시 3] 본문(표):
  | 급여금의 종류 | 지급사유 | 지급금액 |
  | 뇌출혈진단급여금 | 뇌출혈로 진단확정시 | 1,000만원 |
  | 급성심근경색증진단급여금 | 급성심근경색증으로 진단확정시 | 2,000만원 |
출력:
  {"subject":["뇌출혈진단급여금","뇌출혈","급성심근경색증진단급여금","급성심근경색증"],
   "role":["payment_trigger","payment_amount"],
   "qualifier":["1,000만원","2,000만원"],
   "reference":[],
   "table":["급여금의 종류","지급사유","지급금액"]}
"""

BATCH_FORMAT = """
--- 출력 형식 (반드시 지켜라) ---
아래에 엘리먼트 {N}개가 `=== [{{num}}] ===` 로 구분되어 주어진다.
각 엘리먼트마다 위 5개 슬롯을 채우고, 다음 형태의 JSON 객체 하나로 출력해라:

  {{"items": [{{"i":0,"subject":[],"role":[],"qualifier":[],"reference":[],"table":[]}}, ...]}}

- items 원소는 정확히 {N}개, `i`는 0부터 {LAST}까지 순서대로.
- 모든 슬롯은 문자열 리스트. 해당 없으면 빈 리스트 [].
- subject는 대상어가 본문에 있으면 절대 비우지 마라.
"""

# ⚠ 이 정규식은 **실패한 호출의 출력에만** 건다. codex exec는 프롬프트 전문을 stderr로
# 되울리므로, 성공 경로까지 검사하면 약관 본문의 숫자·문자열이 오탐된다.
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


# ---------------------------------------------------------------- 구조 규칙 슬롯
def rule_contract(element: dict) -> list[str]:
    """contract_scope에서 특약명을 뽑고 보일러플레이트를 제거한다."""
    scope = (element.get("contract_scope") or "").strip()
    if not scope:
        return []
    m = MAIN_WRAP_RE.match(scope)          # "주계약(...)" 껍데기를 먼저 벗긴다
    if m:
        scope = m.group(1).strip()
    name = CONTRACT_BOILERPLATE_RE.sub("", scope)
    name = re.sub(r"\s+", " ", name).strip(" ,")
    return [name[:40]] if len(name) >= 2 else []


def rule_article(element: dict) -> list[str]:
    """본문에서 '제N조(제목)' 형태를 뽑는다. 실재하는 조 번호만."""
    out: list[str] = []
    for m in ARTICLE_RE.finditer(element.get("text", "")):
        head = f"제{re.sub(r'\\s+', '', m.group(1))}조"
        if m.group(2):
            head += "의" + m.group(2)
        title = (m.group(3) or "").strip().strip("）)").strip()
        label = head + (f"({title})" if title else "")
        if label not in out:
            out.append(label[:40])
        if len(out) >= 5:
            break
    return out


def rule_schema(element: dict) -> list[str]:
    et = (element.get("element_type") or "").strip()
    return [et] if et in VALID_SCHEMAS else ["paragraph"]


# ---------------------------------------------------------------- navigation 판정
def is_navigation(element: dict) -> bool:
    """구분선·페이지번호 등 LLM 호출 없이 규칙 슬롯만 채우면 되는 엘리먼트."""
    text = element.get("text", "").strip()
    return len(text) < 5 or bool(NAV_RE.fullmatch(text))


def rule_only_row(element: dict) -> dict:
    """LLM 없이 구조 슬롯만 채운 행."""
    return {
        "element_id": element["element_id"], "ok": True,
        "contract": rule_contract(element),
        "subject": [], "role": [],
        "article": rule_article(element),
        "table": [], "qualifier": [], "reference": [],
        "schema": rule_schema(element),
    }


# ---------------------------------------------------------------- sanitize
def sanitize_tags(raw: dict, element: dict) -> dict:
    """LLM 응답 1건(5슬롯)을 정제하고, 구조 규칙 3슬롯을 합친다."""
    def strs(key: str) -> list[str]:
        return [v.strip() for v in (raw.get(key) or [])
                if isinstance(v, str) and v.strip()]

    # --- LLM 슬롯
    subjects = [s[:24] for s in strs("subject")]
    subjects = [s for s in subjects if len(s) >= 2 and s not in GENERIC_TERMS]
    subjects = list(dict.fromkeys(subjects))[:5]

    roles = [r for r in strs("role") if r in VALID_ROLES_SET]
    roles = list(dict.fromkeys(roles))[:3]

    qualifiers = [q[:40] for q in strs("qualifier") if q not in GENERIC_TERMS]
    qualifiers = list(dict.fromkeys(qualifiers))[:4]

    references = list(dict.fromkeys(r[:40] for r in strs("reference")))[:5]
    tables = list(dict.fromkeys(t[:40] for t in strs("table")))[:10]

    return {
        "contract": rule_contract(element),
        "subject": subjects,
        "role": roles,
        "article": rule_article(element),
        "table": tables,
        "qualifier": qualifiers,
        "reference": references,
        "schema": rule_schema(element),
    }


# ---------------------------------------------------------------- codex 호출
def call_codex(prompt: str, work: Path, call_id: int, model: str,
               stats: "Stats") -> tuple[list | None, str | None]:
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

    combined = (r.stdout.decode("utf-8", "replace") + "\n"
                + r.stderr.decode("utf-8", "replace"))
    m = re.search(r"tokens used\s*\n\s*([\d,]+)", combined)
    tok = int(m.group(1).replace(",", "")) if m else 0

    try:
        obj = json.loads(o.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        # 레이트리밋 판정은 실패 경로에서만 — 성공 경로는 정의상 레이트리밋이 아니다.
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


# ---------------------------------------------------------------- 배치 프롬프트
def _element_block(k: int, elem: dict) -> str:
    text = elem.get("text", "")
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS] + "\n... (이하 생략)"
    etype = elem.get("element_type", "unknown")
    scope = elem.get("contract_scope", "")
    return f"\n=== [{k}] ===\n[element_type: {etype}] [특약: {scope}]\n{text}"


def build_batch_prompt(elements: list[dict]) -> list[dict]:
    n = len(elements)
    user = [BATCH_FORMAT.format(N=n, LAST=n - 1)]
    user += [_element_block(k, e) for k, e in enumerate(elements)]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user)}]


def build_batch_prompt_text(elements: list[dict]) -> str:
    msgs = build_batch_prompt(elements)
    return msgs[0]["content"] + "\n" + msgs[1]["content"]


# ---------------------------------------------------------------- 정렬 검증
def align_items(items: list, n: int) -> list[dict] | None:
    if not isinstance(items, list) or len(items) != n:
        return None
    out: list[dict | None] = [None] * n
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        idx = item.get("i")
        if not isinstance(idx, int) or not (0 <= idx < n) or out[idx] is not None:
            idx = pos
        if idx < 0 or idx >= n or out[idx] is not None:
            return None
        out[idx] = item
    return None if any(x is None for x in out) else out  # type: ignore[return-value]


# ---------------------------------------------------------------- 배치 처리
class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = self.fails = self.tokens = 0

    def add(self, tokens: int = 0, fail: bool = False) -> None:
        with self.lock:
            self.calls += 1
            self.fails += int(fail)
            self.tokens += tokens

    def line(self) -> str:
        return f"calls={self.calls} fails={self.fails} tokens={self.tokens:,}"


def process_batch(elements: list[dict], stats: Stats, work: Path, model: str,
                  max_retries: int = 3) -> list[dict]:
    """배치 1개 처리. 실패 시 백오프 재시도 → 반분할 재시도 → ok:false."""
    n = len(elements)
    prompt = build_batch_prompt_text(elements)

    for attempt in range(max_retries):
        items, err = call_codex(prompt, work, next(_call_counter), model, stats)
        if err is None:
            aligned = align_items(items, n)
            if aligned is not None:
                return [{"element_id": e["element_id"], "ok": True,
                         **sanitize_tags(raw, e)}
                        for e, raw in zip(elements, aligned)]
        time.sleep(2 ** attempt)

    if len(elements) > 1:
        mid = len(elements) // 2
        return (process_batch(elements[:mid], stats, work, model, max_retries)
                + process_batch(elements[mid:], stats, work, model, max_retries))

    e = elements[0]
    return [{"element_id": e["element_id"], "ok": False, "error": "all_retries_failed",
             "contract": rule_contract(e), "subject": [], "role": [],
             "article": rule_article(e), "table": [], "qualifier": [],
             "reference": [], "schema": rule_schema(e)}]


# ---------------------------------------------------------------- resume
def load_done_ids(path: Path) -> set[str]:
    """성공(ok:true)한 element_id만 done으로 본다 — 실패 행은 재시도 대상."""
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
            if row.get("ok"):
                done.add(row.get("element_id", ""))
    done.discard("")
    return done


# ---------------------------------------------------------------- 채움률 게이트
def report_fill_rates(rows: list[dict]) -> float:
    """8슬롯 채움률 + 고유값 수 표를 찍고 subject 채움률(0~1)을 반환한다."""
    # element_id 중복(재시도 흔적) 제거 — 마지막 성공 행을 채택
    dedup: dict[str, dict] = {}
    for r in rows:
        eid = r.get("element_id")
        if not eid:
            continue
        if eid not in dedup or (r.get("ok") and not dedup[eid].get("ok")):
            dedup[eid] = r
    rows = list(dedup.values())
    n = max(len(rows), 1)

    print(f"\n{'슬롯':<12}{'채움률':>9}{'채움건수':>10}{'고유값':>9}")
    print("-" * 42)
    subject_fill = 0.0
    for field in FIELDS:
        filled = sum(1 for r in rows if r.get(field))
        distinct = len({v for r in rows for v in (r.get(field) or [])})
        pct = filled / n
        if field == "subject":
            subject_fill = pct
        mark = "  <-- 핵심" if field == "subject" else ""
        print(f"{field:<12}{pct * 100:8.1f}%{filled:10,}{distinct:9,}{mark}")
    print("-" * 42)
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"전체 {len(rows):,}건 / ok:true {n_ok:,} / ok:false {len(rows) - n_ok:,}")

    role_c = collections.Counter(v for r in rows for v in (r.get("role") or []))
    print(f"role 분포: {dict(role_c.most_common())}")
    subj_c = collections.Counter(v for r in rows for v in (r.get("subject") or []))
    print(f"subject 상위 15: {[k for k, _ in subj_c.most_common(15)]}")
    return subject_fill


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="엘리먼트 8-슬롯 태그 생성 v2 (구조 규칙 + codex LLM)")
    ap.add_argument("--input", default="out/elements_psection.jsonl",
                    help="입력 jsonl (기본 out/elements_psection.jsonl)")
    ap.add_argument("--output", default="out/element_tags_v2.jsonl",
                    help="출력 jsonl (기본 out/element_tags_v2.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 엘리먼트 수 (0=전체)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true",
                    help="codex 호출 없이 배치 프롬프트 1개와 규칙 슬롯 샘플만 출력")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = BASE / output_path

    if not input_path.exists():
        print(f"[오류] 입력 파일이 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    elements = [json.loads(ln) for ln in
                input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        elements = elements[:args.limit]

    nav_elems = [e for e in elements if is_navigation(e)]
    non_nav = [e for e in elements if not is_navigation(e)]

    # ------------------------------------------------------------ dry-run
    if args.dry_run:
        sample = non_nav[:args.batch_size]
        msgs = build_batch_prompt(sample)
        print("=== SYSTEM ===")
        print(msgs[0]["content"])
        print("=== USER ===")
        print(msgs[1]["content"])
        print("\n=== 구조 규칙 슬롯 미리보기 (LLM 없이 채워지는 3개) ===")
        for e in sample[:5]:
            print(json.dumps({"element_id": e["element_id"],
                              "contract": rule_contract(e),
                              "article": rule_article(e),
                              "schema": rule_schema(e)}, ensure_ascii=False))
        rc = sum(1 for e in elements if rule_contract(e))
        ra = sum(1 for e in elements if rule_article(e))
        print(f"\n[규칙 채움률/{len(elements):,}건] contract {100*rc/max(len(elements),1):.1f}% "
              f"article {100*ra/max(len(elements),1):.1f}% schema 100.0%")
        print(f"[DRY-RUN] 배치 {len(sample)} / 비-항법 {len(non_nav):,} / "
              f"항법 {len(nav_elems):,} / 전체 {len(elements):,} / model={args.model} "
              f"/ 본문 상한 {MAX_BODY_CHARS}자 (codex 호출 없음)")
        return

    # ------------------------------------------------------------ 실행
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "schema.json").write_text(
        json.dumps(ENVELOPE_SCHEMA, ensure_ascii=False), encoding="utf-8")

    done = load_done_ids(output_path)
    fh = output_path.open("a", encoding="utf-8")
    wlock = threading.Lock()

    def emit(rows: list[dict]) -> None:
        with wlock:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    nav_rows = [rule_only_row(e) for e in nav_elems if e["element_id"] not in done]
    if nav_rows:
        emit(nav_rows)
        print(f"[항법] {len(nav_rows):,}건 규칙 슬롯만 기록 (LLM 호출 없음)")

    todo = [e for e in non_nav if e["element_id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    print(f"[gen_element_tags_v2] 엘리먼트 {len(elements):,} / 캐시 {len(done):,} / "
          f"항법 {len(nav_elems):,} / API 대상 {len(todo):,} -> 배치 {len(batches):,}개 "
          f"(batch={args.batch_size}, workers={args.workers}, model={args.model})",
          flush=True)
    print(f"출력: {output_path}")

    stats = Stats()
    t0 = time.time()
    rate_limited: str | None = None

    if batches:
        n_done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_batch, b, stats, WORK_DIR, args.model)
                    for b in batches]
            try:
                for fut in as_completed(futs):
                    rows = fut.result()
                    emit(rows)
                    n_done += len(rows)
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    if n_done % 100 < args.batch_size or n_done == len(todo):
                        print(f"   {n_done:,}/{len(todo):,}  {rate:.1f}/s  "
                              f"~{(len(todo) - n_done) / max(rate, 1e-9) / 60:.1f}분 남음  "
                              f"{stats.line()}", flush=True)
            except RateLimitDetected as exc:
                rate_limited = str(exc)
                ex.shutdown(wait=False, cancel_futures=True)

    fh.close()

    if rate_limited is not None:
        print(f"\n[중단] codex 레이트리밋. 부분 산출물: {output_path}\n"
              f"감지된 신호: {rate_limited}", file=sys.stderr)
        sys.exit(1)

    all_rows = [json.loads(ln) for ln in
                output_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"\n완료: {stats.calls:,}호출 / 실패 {stats.fails} / {time.time() - t0:.0f}초 / "
          f"토큰 {stats.tokens:,}")

    subject_fill = report_fill_rates(all_rows)

    if subject_fill < SUBJECT_FILL_GATE:
        print(f"\n[게이트 실패] subject 채움률 {subject_fill*100:.1f}% < "
              f"{SUBJECT_FILL_GATE*100:.0f}%. v1의 9.7% 재현 위험 — 프롬프트를 고쳐라.",
              file=sys.stderr)
        sys.exit(2)
    print(f"\n[게이트 통과] subject 채움률 {subject_fill*100:.1f}% >= "
          f"{SUBJECT_FILL_GATE*100:.0f}%")


if __name__ == "__main__":
    main()
