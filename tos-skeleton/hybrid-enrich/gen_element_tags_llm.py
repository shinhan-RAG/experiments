#!/usr/bin/env python3
"""엘리먼트 단위 시멘틱 태그 생성 — codex CLI(gpt-5.4-mini) 배치 호출.

SlotFileSearch(slot_filesearch.py)의 8-슬롯 스키마와 동일한 구조의 태그를 LLM으로
채운다. 입력은 elements_repaired_v1.jsonl, 출력은 element_tags_llm_v1.jsonl.

전송 계층은 로컬 vLLM HTTP API 대신 codex CLI 서브프로세스를 쓴다
(`codex exec --output-schema ... -o out.json -`). 프롬프트는 stdin으로, 결과는
`-o` 로 지정한 파일에서 읽는다. 스키마·시스템 프롬프트·sanitize·배치 정렬 로직은
vLLM판과 동일하게 유지한다 — 바뀐 것은 API 호출 메커니즘뿐이다.

배치 처리(--batch-size, 기본 20)로 프로세스 호출 수를 줄이고, ThreadPoolExecutor로
병렬화한다(--workers, 기본 8). codex는 API 기반이라 로컬 GPU 부하가 없으므로
vLLM판보다 워커·배치 크기를 크게 잡는다.

사용법
    python gen_element_tags_llm.py
    python gen_element_tags_llm.py --limit 50 --dry-run
    python gen_element_tags_llm.py --workers 4 --batch-size 10
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

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"

# ---------------------------------------------------------------- codex CLI 설정
# npm shim(`codex`)은 실제로 존재가 확인된 절대 경로를 쓴다.
CODEX = os.environ.get("CODEX_BIN") or str(
    Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")

WORK_DIR = OUT / "codex_tag_work"
WORKERS_DEFAULT = int(os.environ.get("TAG_WORKERS", "8"))
BATCH_DEFAULT = int(os.environ.get("CODEX_TAG_BATCH", "20"))
CALL_TIMEOUT = int(os.environ.get("CODEX_TAG_TIMEOUT", "600"))
MODEL_DEFAULT = os.environ.get("CODEX_TAG_MODEL", "gpt-5.4-mini")

# ---------------------------------------------------------------- 8-슬롯 스키마
FIELDS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")

VALID_ROLES = {
    "exclusion_exception", "premium_waiver", "payment_trigger", "payment_amount",
    "limit_frequency", "timing_period", "definition", "criteria_rule",
    "contract_lifecycle", "claim_procedure", "code_reference",
}

ROLE_ALIASES = {
    "exclusion_exception": "면책 제외 예외 부지급 지급하지 않는 사유",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}

# ---------------------------------------------------------------- sanitize 상수
CONTRACT_BOILERPLATE_RE = re.compile(
    r"\(무배당[^)]*\)|\(간편\)|\(해약환급금\s*미지급형\)|\(갱신형\)|\(일반형\)"
)
GENERIC_TERMS = {"보험", "약관", "보험금", "특약", "보험계약"}
NAV_RE = re.compile(r"^[-=─━\s]+$|^\d+$")

# ---------------------------------------------------------------- JSON Schema (structured output)
ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "i": {"type": "integer"},
        "contract": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "array", "items": {"type": "string"}},
        "role": {"type": "array", "items": {"type": "string"}},
        "article": {"type": "array", "items": {"type": "string"}},
        "table": {"type": "array", "items": {"type": "string"}},
        "qualifier": {"type": "array", "items": {"type": "string"}},
        "reference": {"type": "array", "items": {"type": "string"}},
        "schema": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["i", "contract", "subject", "role", "article", "table", "qualifier", "reference", "schema"],
}

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": ITEM_SCHEMA,
        }
    },
    "required": ["items"],
}

# codex `--output-schema`는 JSON Schema 파일을 요구한다. gen_meta_codex_v9.py와
# 같은 형태로 봉투(envelope) 스키마를 만든다 — additionalProperties: False로
# 여분 필드를 막는다.
ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "contract": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "array", "items": {"type": "string"}},
                    "article": {"type": "array", "items": {"type": "string"}},
                    "table": {"type": "array", "items": {"type": "string"}},
                    "qualifier": {"type": "array", "items": {"type": "string"}},
                    "reference": {"type": "array", "items": {"type": "string"}},
                    "schema": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["i", "contract", "subject", "role", "article",
                             "table", "qualifier", "reference", "schema"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- 시스템 프롬프트
SYSTEM_PROMPT = """너는 한국 보험약관 도메인 전문가다. 주어진 약관 엘리먼트 텍스트를 읽고,
아래 8개 슬롯을 빠짐없이 채워라. 각 슬롯은 문자열 리스트(list[str])다.
해당 사항이 없으면 빈 리스트 []로 남겨라.

## 슬롯 정의

1. **contract**: 이 엘리먼트가 속한 특약명. 예: ["(무)질병입원특약"]
2. **subject**: 핵심 주제 키워드. 급여금 명칭, 질병명, 보장항목 등. 예: ["암진단비", "갑상선암"]
3. **role**: 의미역. 아래 11가지 중에서만 선택:
   - exclusion_exception (면책 제외 예외 부지급 지급하지 않는 사유)
   - premium_waiver (보험료 납입면제)
   - payment_trigger (보험금 지급사유 지급조건)
   - payment_amount (보험금 지급금액 지급률 산정 계산)
   - limit_frequency (지급한도 횟수 일수 최초 1회)
   - timing_period (보장개시 책임개시 대기기간 감액기간 보험기간)
   - definition (용어 정의 의미)
   - criteria_rule (진단확정 판정기준 적용기준)
   - contract_lifecycle (갱신 해지 소멸 무효 환급)
   - claim_procedure (보험금 청구 구비서류 절차)
   - code_reference (질병분류코드 수가코드 부표 분류표)
4. **article**: 조 번호+제목. 예: ["제3조(보험금의 지급사유)"]
5. **table**: 표가 있으면 표의 헤더/행 키. 표가 아니면 []
6. **qualifier**: 조건/제한/수치 규정. 예: ["90일 이내", "보험가입금액의 100%"]
7. **reference**: 참조하는 다른 조항·별표·부표. 예: ["제5조", "별표3"]
8. **schema**: 엘리먼트 유형 태그. 예: ["paragraph"], ["table"], ["heading"]

## 규칙
- contract: 특약명을 정확히 적되, "(무배당, ...)", "(간편)", "(해약환급금 미지급형)" 같은 보일러플레이트는 제거
- subject: 구체적 키워드만. "보험", "약관", "보험금", "특약", "보험계약" 같은 일반 용어는 넣지 마라
- role: 반드시 위 11가지 영문 식별자 중에서만. 여러 역할이 있으면 최대 3개
- article: "제N조(제목)" 형태. 본문에 실재하는 조 번호만
- qualifier: 구체적 조건·제한만. 최대 4개
- 빈 텍스트나 구분선은 모든 슬롯을 []로

"""

BATCH_FORMAT = """
--- 출력 형식 (반드시 지켜라) ---
아래에 엘리먼트 {N}개가 `=== [{{num}}] ===` 로 구분되어 주어진다.
각 엘리먼트마다 위에서 정의한 8개 슬롯을 채우고, 다음 형태의 JSON 객체 하나로 출력해라:

  {{"items": [{{"i":0,"contract":[],"subject":[],"role":[],"article":[],"table":[],"qualifier":[],"reference":[],"schema":[]}}, ...]}}

- items 원소는 정확히 {N}개, `i`는 0부터 {LAST}까지 순서대로.
- 모든 슬롯은 문자열 리스트. 해당 없으면 빈 리스트 [].
"""

# ⚠ 이 정규식은 **실패한 호출의 출력에만** 건다. 성공 경로에 걸면 오탐한다.
# codex exec는 사용자 프롬프트 전문을 stderr로 되울린다. stdout+stderr를 통째로
# 검사하면 약관 본문에 들어 있는 숫자·문자열이 패턴에 걸릴 수 있다(gen_meta_codex_v9.py
# 참고: 실측 사고로 `\b429\b`가 본문의 표 번호에 매칭됐다). 그래서 숫자 단독 패턴은
# 빼고, 실패한 호출에서만 검사한다.
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
    """레이트리밋으로 판정된 근거 줄만 뽑는다(되울려진 본문 전체를 로그에 남기지 않는다)."""
    hits = [ln.strip() for ln in combined.splitlines() if _RATE_LIMIT_RX.search(ln)]
    return " | ".join(hits[:3])[:300] if hits else combined.strip()[-200:]


# ---------------------------------------------------------------- navigation 판정
def is_navigation(element: dict) -> bool:
    """구분선·페이지번호 등 LLM 호출 없이 빈 태그를 줄 수 있는 엘리먼트."""
    text = element.get("text", "").strip()
    if len(text) < 5:
        return True
    if NAV_RE.fullmatch(text):
        return True
    return False


def empty_tags(element_id: str) -> dict:
    return {
        "element_id": element_id, "ok": True,
        "contract": [], "subject": [], "role": [], "article": [],
        "table": [], "qualifier": [], "reference": [], "schema": [],
    }


# ---------------------------------------------------------------- sanitize
def sanitize_contract(value: str) -> str:
    """특약명에서 보일러플레이트를 제거하고 길이를 자른다."""
    value = CONTRACT_BOILERPLATE_RE.sub("", value).strip()
    return value[:40]


def sanitize_tags(raw: dict, element_type: str = "") -> dict:
    """LLM 응답 1건을 정제한다."""
    result: dict[str, list[str]] = {}

    # contract
    contracts = [sanitize_contract(v) for v in (raw.get("contract") or []) if isinstance(v, str) and v.strip()]
    contracts = [c for c in contracts if c and c not in GENERIC_TERMS]
    result["contract"] = contracts[:3]

    # subject
    subjects = [v.strip()[:20] for v in (raw.get("subject") or []) if isinstance(v, str) and v.strip()]
    subjects = [s for s in subjects if s not in GENERIC_TERMS and len(s) >= 2]
    result["subject"] = subjects[:5]

    # role
    roles = [v.strip() for v in (raw.get("role") or []) if isinstance(v, str) and v.strip() in VALID_ROLES]
    result["role"] = list(dict.fromkeys(roles))[:3]

    # article
    articles = [v.strip()[:40] for v in (raw.get("article") or []) if isinstance(v, str) and v.strip()]
    result["article"] = articles[:5]

    # table
    tables = [v.strip()[:40] for v in (raw.get("table") or []) if isinstance(v, str) and v.strip()]
    result["table"] = tables[:10]

    # qualifier
    qualifiers = [v.strip()[:40] for v in (raw.get("qualifier") or []) if isinstance(v, str) and v.strip()]
    qualifiers = [q for q in qualifiers if q not in GENERIC_TERMS]
    result["qualifier"] = qualifiers[:4]

    # reference
    references = [v.strip()[:40] for v in (raw.get("reference") or []) if isinstance(v, str) and v.strip()]
    result["reference"] = references[:5]

    # schema
    schemas = [v.strip() for v in (raw.get("schema") or []) if isinstance(v, str) and v.strip()]
    if not schemas and element_type:
        schemas = [element_type]
    result["schema"] = schemas[:3]

    return result


# ---------------------------------------------------------------- codex 호출
def call_codex(prompt: str, work: Path, call_id: int, model: str,
               stats: "Stats") -> tuple[list | None, str | None]:
    """codex exec를 한 번 부른다. 프롬프트는 stdin, 결과는 `-o` 파일로 받는다.

    gen_meta_codex_v9.py의 호출 방식을 그대로 가져왔다 — model_reasoning_effort=low,
    --sandbox read-only --skip-git-repo-check --ephemeral, --output-schema 강제.
    """
    o = work / f"out_{call_id}.json"
    cmd = [CODEX, "exec", "--model", model, "-c", "model_reasoning_effort=low",
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--output-schema", str(work / "schema.json"), "-o", str(o), "-"]
    try:
        r = subprocess.run(cmd, input=prompt.encode("utf-8"),
                           capture_output=True, timeout=CALL_TIMEOUT)
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
        # 여기서만 레이트리밋을 판정한다 — 성공 경로(파싱 OK)는 정의상 레이트리밋이
        # 아니므로 검사조차 하지 않는다(프롬프트 되울림으로 인한 본문 오탐 방지).
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


# ---------------------------------------------------------------- 배치 프롬프트 구성
def build_batch_prompt_text(elements: list[dict]) -> str:
    """elements 리스트로 codex에 stdin으로 넘길 프롬프트 문자열을 구성한다."""
    n = len(elements)
    parts = [SYSTEM_PROMPT, BATCH_FORMAT.format(N=n, LAST=n - 1)]
    for k, elem in enumerate(elements):
        text = elem.get("text", "")
        # 긴 텍스트는 잘라서 토큰 절약
        if len(text) > 2000:
            text = text[:2000] + "\n... (이하 생략)"
        etype = elem.get("element_type", "unknown")
        parts.append(f"\n=== [{k}] ===\n[element_type: {etype}]\n{text}")

    return "\n".join(parts)


def build_batch_prompt(elements: list[dict]) -> list[dict]:
    """dry-run에서 system/user를 나눠 보여주기 위한 호환용 헬퍼."""
    n = len(elements)
    user_parts = [BATCH_FORMAT.format(N=n, LAST=n - 1)]
    for k, elem in enumerate(elements):
        text = elem.get("text", "")
        if len(text) > 2000:
            text = text[:2000] + "\n... (이하 생략)"
        etype = elem.get("element_type", "unknown")
        user_parts.append(f"\n=== [{k}] ===\n[element_type: {etype}]\n{text}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------- 정렬 검증
def align_items(items: list, n: int) -> list[dict] | None:
    """응답 items를 i 인덱스로 정렬한다. 개수/범위가 안 맞으면 None."""
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


def process_batch(elements: list[dict], stats: Stats, work: Path, model: str,
                  max_retries: int = 3) -> list[dict]:
    """배치 1개를 처리한다. 실패 시 지수 백오프로 재시도, 전부 실패하면 ok:false.

    RateLimitDetected는 여기서 잡지 않는다 — 그대로 위(main)까지 전파시켜
    즉시 중단시킨다(재시도 루프에 태우지 않는다).
    """
    n = len(elements)
    prompt = build_batch_prompt_text(elements)

    for attempt in range(max_retries):
        call_id = next(_call_counter)
        items, err = call_codex(prompt, work, call_id, model, stats)
        if err is None:
            aligned = align_items(items, n)
            if aligned is not None:
                results = []
                for elem, raw in zip(elements, aligned):
                    tags = sanitize_tags(raw, elem.get("element_type", ""))
                    results.append({
                        "element_id": elem["element_id"],
                        "ok": True,
                        **tags,
                    })
                return results
            # 정렬 실패 — 개수가 안 맞으면 반으로 쪼개 재시도
        time.sleep(2 ** attempt)

    # 배치가 2개 이상이면 반으로 쪼개 재시도
    if len(elements) > 1:
        mid = len(elements) // 2
        return (process_batch(elements[:mid], stats, work, model, max_retries)
                + process_batch(elements[mid:], stats, work, model, max_retries))

    # 1개짜리도 실패하면 ok:false 기록
    return [{
        "element_id": elements[0]["element_id"],
        "ok": False,
        "error": "all_retries_failed",
        "contract": [], "subject": [], "role": [], "article": [],
        "table": [], "qualifier": [], "reference": [], "schema": [],
    }]


# ---------------------------------------------------------------- resume
def load_done_ids(path: Path) -> set[str]:
    """이미 처리된 element_id를 로드한다."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["element_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="엘리먼트 시멘틱 태그 생성 (codex CLI, gpt-5.4-mini)")
    ap.add_argument("--limit", type=int, default=0,
                    help="처리할 최대 엘리먼트 수 (0=전체)")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT,
                    help=f"ThreadPoolExecutor 워커 수 (기본 {WORKERS_DEFAULT})")
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT,
                    help=f"배치당 엘리먼트 수 (기본 {BATCH_DEFAULT})")
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help=f"codex exec --model (기본 {MODEL_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="codex를 호출하지 않고 배치 프롬프트 1개만 출력하고 종료")
    ap.add_argument("--input", default="elements_repaired_v1.jsonl",
                    help="입력 파일명 (out/ 기준)")
    ap.add_argument("--output", default="element_tags_llm_v1.jsonl",
                    help="출력 파일명 (out/ 기준)")
    args = ap.parse_args()

    input_path = OUT / args.input
    output_path = OUT / args.output

    if not input_path.exists():
        print(f"[오류] 입력 파일이 없습니다: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 엘리먼트 로드
    elements = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        elements = elements[:args.limit]

    # 항법 엘리먼트 분리
    nav_elems = [e for e in elements if is_navigation(e)]
    non_nav = [e for e in elements if not is_navigation(e)]

    # dry-run
    if args.dry_run:
        sample = non_nav[:args.batch_size]
        messages = build_batch_prompt(sample)
        print("=== SYSTEM ===")
        print(messages[0]["content"])
        print("\n=== USER ===")
        print(messages[1]["content"])
        print(f"\n[DRY-RUN] 배치 크기 {len(sample)} / 비-항법 엘리먼트 {len(non_nav):,} / "
              f"항법 엘리먼트 {len(nav_elems):,} / 전체 {len(elements):,} / "
              f"model={args.model} (codex 호출 없음)")
        return

    # 작업 디렉터리 준비 — schema.json은 여기 한 번만 쓰고 모든 호출이 공유한다.
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "schema.json").write_text(
        json.dumps(ENVELOPE_SCHEMA, ensure_ascii=False), encoding="utf-8")

    # resume
    done = load_done_ids(output_path)

    fh = output_path.open("a", encoding="utf-8")
    wlock = threading.Lock()

    def emit(rows: list[dict]) -> None:
        with wlock:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    # 항법 엘리먼트는 LLM 호출 없이 빈 태그
    nav_rows = [empty_tags(e["element_id"]) for e in nav_elems if e["element_id"] not in done]
    if nav_rows:
        emit(nav_rows)
        print(f"[항법] {len(nav_rows):,}개 빈 태그 기록 (LLM 호출 없음)")

    # codex 대상
    todo = [e for e in non_nav if e["element_id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    print(f"[gen_element_tags_llm/codex] 엘리먼트 {len(elements):,} / 캐시 {len(done):,} / "
          f"항법 {len(nav_elems):,} / API 대상 {len(todo):,} -> "
          f"배치 {len(batches):,}개 (batch={args.batch_size}, workers={args.workers}, "
          f"model={args.model})",
          flush=True)
    print(f"출력: {output_path}")

    if not batches:
        print("처리할 엘리먼트가 없습니다.")
        fh.close()
        return

    stats = Stats()
    t0 = time.time()
    n_done = 0
    rate_limited: str | None = None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_batch, batch, stats, WORK_DIR, args.model): len(batch)
                for batch in batches}
        try:
            for fut in as_completed(futs):
                rows = fut.result()
                emit(rows)
                n_done += len(rows)
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                remaining = (len(todo) - n_done) / max(rate, 1e-9)
                if n_done % 100 < args.batch_size or n_done == len(todo):
                    print(f"   {n_done:,}/{len(todo):,}  "
                          f"{rate:.1f}/s  "
                          f"~{remaining / 60:.1f}분 남음  "
                          f"{stats.line()}",
                          flush=True)
        except RateLimitDetected as exc:
            rate_limited = str(exc)
            ex.shutdown(wait=False, cancel_futures=True)

    fh.close()

    if rate_limited is not None:
        print(f"""
[중단] codex 레이트리밋으로 보인다. 재시도로 시간을 태우지 않고 즉시 중단한다.
부분 산출물: {output_path}
감지된 신호: {rate_limited}
""", file=sys.stderr)
        sys.exit(1)

    # 최종 보고
    elapsed = time.time() - t0
    all_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_total = len(all_rows)
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    n_fail = sum(1 for r in all_rows if r.get("ok") is False)

    filled: dict[str, int] = {}
    for field in FIELDS:
        filled[field] = sum(1 for r in all_rows if r.get(field))

    print(f"\n완료: {stats.calls:,}호출 / 실패 {stats.fails} / {elapsed:.0f}초 / "
          f"토큰 {stats.tokens:,}")
    print(f"\n필드 채움률 (전체 {n_total:,}건):")
    for field in FIELDS:
        pct = 100 * filled[field] / max(n_total, 1)
        print(f"   {field:16s} {pct:5.1f}%  ({filled[field]:,}건)")
    print(f"   ok:true  {n_ok:,}건 / ok:false  {n_fail:,}건")


if __name__ == "__main__":
    main()
