#!/usr/bin/env python3
"""v13-C Boundary Evidence Card 메타데이터 생성 — codex CLI(gpt-5.4-mini) 배치 호출판.

스키마 출처: LLM_메타데이터_스키마_제안_20260812.md §7 (v13-C)
모든 의미 값은 LLM이 생성한다. 정규식/사전/규칙 기반 추출은 사용하지 않는다.

2단계 생성
    Stage 1 — 문서 카드: 원본 파일명+문서 앞뒤를 보고 `document_scope`를 만든다.
    Stage 2 — 청크 카드: 각 청크에 대해 document_scope + 현재 청크 + 앞뒤 청크를
              보고 location/fact/applies/excludes/anchors 5필드를 생성한다.

v9/v10과 다른 것
    (a) 2단계 생성 (document_scope는 문서 단위 1회, 나머지는 청크 단위).
    (b) 앞뒤 청크 문맥을 함께 제공 — 청크 하나만 보면 조항명/표 위치를 틀릴 수 있다.
    (c) 질문을 만들지 않고 답변 명제(fact)와 적용 경계(applies/excludes)를 만든다.
    (d) sanitize가 v10과 완전히 다른 필드를 강제한다 — import 하지 않고 자체 정의한다.

v9/v10과 같은 것 (계약)
    codex CLI 전송 계층(subprocess, --output-schema, -o 파일 수신),
    배치 봉투 스키마({items: [{i, ...}]}), 정렬 안전장치(align_items),
    이진 분할 재시도(process_batch), 레이트리밋 감지, 증분 저장.

모델
    gpt-5.4-mini (v9/v10과 동일). v13은 새 스키마이므로 기존 arm과의 1요인 비교는
    V13 arm 간(같은 모델·같은 생성·직렬화만 다름)에서만 성립한다.

사용법
    python gen_meta_codex_v13.py --chunkset fixed600 --dry-run --limit 40 --offset 200
    python gen_meta_codex_v13.py --chunkset fixed600 --workers 8
"""
from __future__ import annotations

import argparse
import collections
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common import OUT, read_jsonl, write_jsonl
from gen_llm_meta import SEP_ONLY, is_navigation  # noqa: F401

CODEX = os.environ.get("CODEX_BIN") or str(
    Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")

WORKERS_DEFAULT = int(os.environ.get("V13_WORKERS", "8"))
BATCH_DEFAULT = int(os.environ.get("CODEX_META_BATCH_V13", "15"))
CALL_TIMEOUT = int(os.environ.get("CODEX_META_TIMEOUT", "600"))
MODEL_DEFAULT = os.environ.get("CODEX_META_MODEL", "gpt-5.4-mini")

# ---------------------------------------------------------------- 봉투 스키마
ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "location": {"type": "string"},
                    "fact": {"type": "string"},
                    "applies": {"type": "string"},
                    "excludes": {"type": "string"},
                    "anchors": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["i", "location", "fact", "applies", "excludes", "anchors"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

DOC_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "document_scope": {"type": "string"},
    },
    "required": ["document_scope"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- 배치 프롬프트 형식
BATCH_FORMAT_V13 = """

--- 출력 형식 (반드시 지켜라) ---
아래에 청크 %(N)d개가 `=== [번호] ===` 로 구분되어 주어진다.
각 청크마다 위에서 정의한 5개 필드를 만들고, 다음 형태의 JSON 객체 하나로 출력해라:

  {"items": [{"i":0,"location":"...","fact":"...","applies":"","excludes":"","anchors":[]}, ...]}

- items 원소는 정확히 %(N)d개, `i`는 0부터 %(LAST)d까지 순서대로.
- 각 필드에 근거가 없으면 location/fact/applies/excludes는 빈 문자열, anchors는 빈 배열.
- 질문("~인가요?")을 만들지 마라. 사실("~한다")만 적어라.
- anchors는 원소마다 0~2개. 범용어("보험금","계약","보장")를 넣지 마라.
"""

# ---------------------------------------------------------------- 레이트리밋 감지
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
    pass


def _rl_snippet(combined: str) -> str:
    hits = [ln.strip() for ln in combined.splitlines() if _RATE_LIMIT_RX.search(ln)]
    return " | ".join(hits[:3])[:300] if hits else combined.strip()[-200:]


# ---------------------------------------------------------------- usage
class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = self.fails = self.tok = 0

    def add(self, tok: int = 0, fail: bool = False) -> None:
        with self.lock:
            self.calls += 1
            self.fails += int(fail)
            self.tok += tok

    def line(self) -> str:
        return f"calls={self.calls} fails={self.fails} tokens={self.tok:,}"


# ---------------------------------------------------------------- 프롬프트
SYSTEM_DOC_CARD = """너는 한국 보험 문서를 색인하는 검색 엔지니어다.
주어진 문서의 첫 부분을 읽고, 이 문서를 다른 문서·다른 버전·다른 문서종류와
구별하기 위한 짧은 검색용 정체성을 만들어라.

출력 필드 (필수):
  document_scope [문자열, 최대 36자]
    이 문서를 식별하는 최단 표현. 다음 상황에서 구별에 필요한 정보만 남겨라:
    - 같은 상품의 판매약관과 사업방법서
    - 일반심사형과 간편심사형
    - 같은 상품의 서로 다른 판매 시점(YYMMDD 형식 날짜)
    - 이름이 비슷한 다른 상품

절대 하지 말 것:
- 파일명을 그대로 복사하지 마라.
- 근거가 불분명한 날짜나 버전을 추측하지 마라.
- "(무배당, ...)", "(갱신형)" 같은 상품 유형 보일러플레이트는 빼라.
"""

SYSTEM_CHUNK_CARD = """너는 한국 보험 약관 청크를 색인하는 검색 엔지니어다.
주어진 약관 청크와 주변 문맥을 읽고, 검색 색인에 붙일 필드 5개를 JSON으로 만든다.

이 작업의 핵심 원칙 — 질문을 예측하지 마라. 이 청크가 근거가 되는 **사실**을 적어라.
고객이 어떻게 물어올지는 예측할 수 없지만, 이 청크가 뒷받침하는 결론은 확정적이다.
질문 말투가 달라도 사실과 질문 사이의 의미 관계는 안정적으로 정렬된다.

이 문서의 정체: %(document_scope)s

출력 필드 (모두 필수, 근거가 없으면 빈 문자열/""/빈 배열[]):
  location  [문자열, 최대 40자]
    이 청크가 속한 가장 구체적인 계층 위치.
    특약명 > 조항/절/표 제목 형태로 적되, 문서에 근거가 있는 수준까지만.
    예: "수술특약 > 보험금 지급 세부규정", "고혈압 약물치료특약 > 특약의 무효"
    확실하지 않으면 빈 문자열.

  fact  [문자열, 최대 70자]
    이 청크가 근거가 되는 답변형 사실. 주어·판단·결과가 드러나야 한다.
    "보험금 지급에 관한 내용" 같은 주제 요약은 쓸모없다.
    좋은 예: "동일한 날 여러 수술을 받아도 동일 수술이면 1회만 지급한다"
    나쁜 예: "수술비는 어떻게 지급되나요?" (질문이다)
            "보험금, 지급, 수술" (키워드 나열이다)

  applies  [문자열, 최대 24자 또는 빈 문자열]
    적용 대상, 발생 시점, 횟수, 기간, 수치 문턱.
    청크에 적용 조건이 명시되어 있을 때만 적는다.
    예: "암보장개시일 이후 치료", "동일 부위 장해가 악화된 경우"

  excludes  [문자열, 최대 24자 또는 빈 문자열]
    면책, 제외 대상, 단서, 비적용 상황.
    청크에 제외/면책이 명시되어 있을 때만 적는다. 상식적 예외를 지어내지 마라.
    예: "최종 지급률이 기존보다 높지 않은 경우"

  anchors  [문자열 0~2개, 각 14자 이내]
    이 청크를 인접 유사 청크와 구별하는 고유 식별자.
    질병분류코드, 급여금 정확한 명칭, 부표/표 식별명, 형제를 가르는 수치.
    범용어("보험금", "계약", "보장")는 쓰지 마라. 없으면 빈 배열.

절대 하지 말 것:
- 본문에 없는 숫자·코드·금액·질병분류코드를 지어내지 마라.
- "보험", "약관", "보험금" 같은 범용어를 fact에 단독으로 쓰지 마라.
- 질문("~인가요?", "~되나요?")을 만들지 마라.
- 키워드 목록이나 주제어 나열을 만들지 마라.
- 청크에 없는 내용을 QA 정답을 아는 것처럼 확장하지 마라.

표·목차 청크에 대하여:
  표 머리글이나 목차도 검색 대상이다. 그 표가 정하는 사실을 적어라.
  (구분선·페이지번호뿐인 청크는 별도 처리되어 이 프롬프트에 오지 않는다.)"""


def build_system_v13(document_scope: str) -> str:
    return SYSTEM_CHUNK_CARD % {"document_scope": document_scope}


def build_prompt_v13(chunk: dict, prev_chunk: dict | None, next_chunk: dict | None) -> str:
    parts = []
    if prev_chunk:
        parts.append(f"--- 앞 청크 (참고용, 위치 판단에만 사용) ---\n{prev_chunk['text'][:600]}")
    parts.append(f"--- 현재 청크 (이것에 대해 답하라) ---\n{chunk['text'][:2400]}")
    if next_chunk:
        parts.append(f"--- 뒤 청크 (참고용, 위치 판단에만 사용) ---\n{next_chunk['text'][:600]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- sanitize
GENERIC_TERMS = frozenset({"보험금", "보험", "계약", "보장", "약관", "특약", "보험료",
                           "계약자", "피보험자", "보험회사", "회사", "지급", "납입"})


def _clip_str(val, maxlen: int) -> str:
    s = str(val or "").strip()
    return s[:maxlen]


def _clip_list(val, maxitems: int, maxlen: int) -> list[str]:
    if not isinstance(val, list):
        return []
    out = []
    for item in val[:maxitems]:
        s = str(item or "").strip()[:maxlen]
        if s and s not in GENERIC_TERMS:
            out.append(s)
    return out


def sanitize_v13(raw: dict, nav: bool, body: str = "") -> dict:
    if nav:
        return {"location": "", "fact": "", "applies": "", "excludes": "",
                "anchors": [], "low_content": True,
                "dropped_body": 0, "dropped_question": 0}

    dropped_body = dropped_question = 0

    location = _clip_str(raw.get("location"), 40)
    if location and location in body:
        location, dropped_body = "", dropped_body + 1

    fact = _clip_str(raw.get("fact"), 70)
    if fact and fact in body:
        fact, dropped_body = "", dropped_body + 1
    elif fact and fact.rstrip().endswith("?"):
        fact, dropped_question = "", dropped_question + 1

    applies = _clip_str(raw.get("applies"), 24)
    if applies and applies in body:
        applies, dropped_body = "", dropped_body + 1
    if applies and applies == fact:
        applies = ""

    excludes = _clip_str(raw.get("excludes"), 24)
    if excludes and excludes in body:
        excludes, dropped_body = "", dropped_body + 1
    if excludes and (excludes == fact or excludes == applies):
        excludes = ""

    anchors = _clip_list(raw.get("anchors"), 2, 14)

    return {
        "location": location,
        "fact": fact,
        "applies": applies,
        "excludes": excludes,
        "anchors": anchors,
        "low_content": False,
        "dropped_body": dropped_body,
        "dropped_question": dropped_question,
    }


# ---------------------------------------------------------------- codex 호출
def call_codex(prompt: str, work: Path, call_id: int, model: str,
               usage: Usage, schema_file: str = "schema.json"
               ) -> tuple[list | dict | None, str | None]:
    o = work / f"out_{call_id}.json"
    cmd = [CODEX, "exec", "--model", model, "-c", "model_reasoning_effort=low",
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--output-schema", str(work / schema_file), "-o", str(o), "-"]
    try:
        r = subprocess.run(cmd, input=prompt.encode("utf-8"),
                           capture_output=True, timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        usage.add(fail=True)
        return None, "timeout"
    except OSError as exc:
        usage.add(fail=True)
        return None, f"spawn:{exc}"

    out_text = r.stdout.decode("utf-8", "replace")
    err_text = r.stderr.decode("utf-8", "replace")
    combined = out_text + "\n" + err_text

    m = re.search(r"tokens used\s*\n\s*([\d,]+)", combined)
    tok = int(m.group(1).replace(",", "")) if m else 0

    try:
        obj = json.loads(o.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        usage.add(fail=True)
        if r.returncode != 0 and _RATE_LIMIT_RX.search(combined):
            raise RateLimitDetected(_rl_snippet(combined))
        return None, f"output_parse:{type(exc).__name__}:{combined[-300:]}"
    finally:
        try:
            o.unlink()
        except OSError:
            pass

    usage.add(tok=tok)
    return obj, None


# ---------------------------------------------------------------- Stage 1: 문서 카드
def generate_doc_card(source_doc: Path, work: Path, model: str,
                      usage: Usage) -> str:
    raw = source_doc.read_text(encoding="utf-8")
    head = raw[:3000]
    tail = raw[-1000:] if len(raw) > 4000 else ""

    prompt_parts = [SYSTEM_DOC_CARD, f"\n--- 파일명 ---\n{source_doc.name}",
                    f"\n--- 문서 첫 부분 ---\n{head}"]
    if tail:
        prompt_parts.append(f"\n--- 문서 끝 부분 ---\n{tail}")
    prompt = "\n".join(prompt_parts)

    (work / "doc_schema.json").write_text(
        json.dumps(DOC_CARD_SCHEMA, ensure_ascii=False), encoding="utf-8")

    for attempt in range(3):
        call_id = next(_call_counter)
        obj, err = call_codex(prompt, work, call_id, model, usage,
                              schema_file="doc_schema.json")
        if err is None and isinstance(obj, dict) and obj.get("document_scope"):
            scope = str(obj["document_scope"]).strip()[:36]
            return scope
        if attempt < 2:
            time.sleep(2)

    raise SystemExit("[오류] 문서 카드 생성 3회 실패. codex 연결을 확인하라.")


# ---------------------------------------------------------------- Stage 2: 청크 카드
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
        if out[idx] is not None:
            return None
        out[idx] = item
    return None if any(x is None for x in out) else out  # type: ignore[return-value]


def build_batch_prompt(system: str, blocks: list[str]) -> str:
    n = len(blocks)
    parts = [system, BATCH_FORMAT_V13 % {"N": n, "LAST": n - 1}]
    for k, b in enumerate(blocks):
        parts.append(f"\n=== [{k}] ===\n{b}")
    return "\n".join(parts)


def process_batch(items: list[tuple[dict, str]], system: str, work: Path, model: str,
                  usage: Usage, attempts: int = 3) -> list[dict]:
    blocks = [b for _, b in items]
    n = len(blocks)
    for _ in range(attempts):
        call_id = next(_call_counter)
        prompt = build_batch_prompt(system, blocks)
        obj, err = call_codex(prompt, work, call_id, model, usage)
        if err is None and isinstance(obj, dict):
            raw_items = obj.get("items")
            if raw_items is not None:
                aligned = align_items(raw_items, n)
                if aligned is not None:
                    return [{"chunk_id": c["chunk_id"], "ok": True,
                             **sanitize_v13(r, False, body=c["text"])}
                            for (c, _), r in zip(items, aligned)]
        time.sleep(2)

    if len(items) > 1:
        mid = len(items) // 2
        return (process_batch(items[:mid], system, work, model, usage, attempts)
                + process_batch(items[mid:], system, work, model, usage, attempts))

    c = items[0][0]
    return [{"chunk_id": c["chunk_id"], "ok": False, "error": "batch_parse_failed",
             **sanitize_v13({}, False, body=c["text"])}]


# ---------------------------------------------------------------- 증분 저장
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
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunkset", default="fixed600")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="스모크 전용. 앞에서 N개를 건너뛴 뒤 --limit개를 쓴다.")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="codex를 호출하지 않고 배치 프롬프트 1개만 출력하고 종료")
    args = ap.parse_args()

    work = Path(__file__).resolve().parent / "codex_v13_work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "schema.json").write_text(json.dumps(ENVELOPE_SCHEMA, ensure_ascii=False),
                                      encoding="utf-8")

    usage = Usage()

    # Stage 1: 문서 카드
    from tos_runtime.config import SOURCE_DOC
    doc_card_path = OUT / "doc_card_v13.json"
    if doc_card_path.exists():
        document_scope = json.loads(doc_card_path.read_text(encoding="utf-8"))["document_scope"]
        print(f"[Stage 1] 캐시된 문서 카드: {document_scope!r}")
    else:
        print(f"[Stage 1] 문서 카드 생성 중... (model={args.model})")
        document_scope = generate_doc_card(SOURCE_DOC, work, args.model, usage)
        doc_card_path.write_text(
            json.dumps({"document_scope": document_scope, "source": str(SOURCE_DOC),
                        "model": args.model}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[Stage 1] 완료: {document_scope!r}")

    system = build_system_v13(document_scope)

    # Stage 2: 청크 카드
    all_chunks = list(read_jsonl(OUT / f"chunks_{args.chunkset}.jsonl"))
    chunks = all_chunks[:]
    if args.offset:
        chunks = chunks[args.offset:]
    if args.limit:
        chunks = chunks[:args.limit]

    nav_ids = {c["chunk_id"] for c in chunks if is_navigation(c)}
    non_nav = [c for c in chunks if c["chunk_id"] not in nav_ids]

    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}
    chunk_order = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}

    def get_prev_next(chunk: dict) -> tuple[dict | None, dict | None]:
        idx = chunk_order.get(chunk["chunk_id"])
        if idx is None:
            return None, None
        prev_c = all_chunks[idx - 1] if idx > 0 else None
        next_c = all_chunks[idx + 1] if idx < len(all_chunks) - 1 else None
        return prev_c, next_c

    if args.dry_run:
        sample = non_nav[:args.batch] if args.batch else non_nav[:15]
        blocks = []
        for c in sample:
            prev_c, next_c = get_prev_next(c)
            blocks.append(build_prompt_v13(c, prev_c, next_c))
        print(build_batch_prompt(system, blocks))
        print(f"\n[DRY-RUN] 배치 크기 {len(blocks)} / 비-항법 청크 {len(non_nav):,} / "
              f"model={args.model} (codex 호출 없음)")
        return

    default_name = f"llm_meta_json_v13_{args.chunkset}.jsonl"
    path = Path(args.out) if args.out else OUT / default_name
    done = load_done_ids(path)

    fh = path.open("a", encoding="utf-8")
    wlock = threading.Lock()

    def emit(rows: list[dict]) -> None:
        with wlock:
            for r in rows:
                r["document_scope"] = document_scope
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    nav_rows = [{"chunk_id": c["chunk_id"], "ok": True,
                 **sanitize_v13({}, True)}
                for c in chunks if c["chunk_id"] in nav_ids and c["chunk_id"] not in done]
    emit(nav_rows)

    todo = []
    for c in non_nav:
        if c["chunk_id"] in done:
            continue
        prev_c, next_c = get_prev_next(c)
        block = build_prompt_v13(c, prev_c, next_c)
        todo.append((c, block))

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    print(f"[v13/codex] 청크 {len(chunks):,} / 캐시 {len(done):,} / 항법 {len(nav_ids):,} / "
          f"API 대상 {len(todo):,} → 배치 {len(batches):,}개 (batch={args.batch}, "
          f"workers={args.workers}, model={args.model})", flush=True)
    print(f"출력: {path}")

    t0 = time.time()
    n_done = 0
    rate_limited: str | None = None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_batch, b, system, work, args.model, usage): len(b)
                for b in batches}
        try:
            for fut in as_completed(futs):
                rows = fut.result()
                emit(rows)
                n_done += len(rows)
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                print(f"   {n_done:,}/{len(todo):,}  {rate:.1f}/s  "
                      f"~{(len(todo) - n_done) / max(rate, 1e-9) / 60:.1f}분 남음  {usage.line()}",
                      flush=True)
        except RateLimitDetected as exc:
            rate_limited = str(exc)
            ex.shutdown(wait=False, cancel_futures=True)

    fh.close()

    if rate_limited is not None:
        print(f"""
[중단] codex 레이트리밋으로 보인다.
부분 산출물: {path}
감지된 신호: {rate_limited}
""", file=sys.stderr)
        sys.exit(1)

    rows = list(read_jsonl(path))
    fails = sum(1 for r in rows if r.get("ok") is False)
    nonlow = [r for r in rows if not r.get("low_content")]
    n_nonlow = max(len(nonlow), 1)

    filled_loc = sum(1 for r in nonlow if r.get("location"))
    filled_fact = sum(1 for r in nonlow if r.get("fact"))
    filled_applies = sum(1 for r in nonlow if r.get("applies"))
    filled_excludes = sum(1 for r in nonlow if r.get("excludes"))
    filled_anchors = sum(1 for r in nonlow if r.get("anchors"))
    dropped_body = sum(int(r.get("dropped_body") or 0) for r in nonlow)
    dropped_question = sum(int(r.get("dropped_question") or 0) for r in nonlow)

    print(f"\n완료 {usage.calls:,}호출 / 실패 {usage.fails} / {time.time() - t0:.0f}초 / "
          f"총토큰 {usage.tok:,}")
    print(f"\n생성 요약 (n={len(rows):,}, 비-항법 n={len(nonlow):,}):")
    print(f"   location 채움률   {100 * filled_loc / n_nonlow:5.1f}%")
    print(f"   fact 채움률       {100 * filled_fact / n_nonlow:5.1f}%")
    print(f"   applies 채움률    {100 * filled_applies / n_nonlow:5.1f}%")
    print(f"   excludes 채움률   {100 * filled_excludes / n_nonlow:5.1f}%")
    print(f"   anchors 채움률    {100 * filled_anchors / n_nonlow:5.1f}%")
    print(f"   폐기: 본문복사 {dropped_body:,} / 질문형태 {dropped_question:,}")
    print(f"   ok:false(실패) {fails:,}건 / 전체 {len(rows):,}건")
    print(f"   document_scope: {document_scope!r}")


if __name__ == "__main__":
    main()
