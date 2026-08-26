#!/usr/bin/env python3
"""8슬롯 전 LLM 태그 생성기 v3.

설계 원칙:
- 8슬롯 전부 LLM 생성. subject_key를 쿼리 프로젝션 허브로 확장 (12개: 문서4+질의4+동의어4).
- contract_key: 폐쇄 어휘(유효 특약 목록)에서만 선택, 목록 외 → 규칙값 fallback.
- locator/schema_tag: LLM 출력 후 정규식 검증, 불일치 시 규칙값 fallback.
- SHA 캐시로 재실행 시 재호출 없음. 배치(15/호출)·병렬(4).

전송 계층:
- --transport claude : claude -p CLI (기본)
- --transport codex  : codex exec + --output-schema (gpt-5.6-luna 등)
"""
import argparse
import concurrent.futures
import hashlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"

CODEX_BIN = os.environ.get("CODEX_BIN") or str(
    Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")
CALL_TIMEOUT = 300
_call_counter = itertools.count()

RATE_LIMIT_RX = re.compile(
    r"rate[ _-]?limit|429\s+too many|quota exceeded|usage limit", re.I)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "contract_key": {"type": "string"},
                    "subject_key": {"type": "array", "items": {"type": "string"}},
                    "role": {"type": "array", "items": {"type": "string"}},
                    "qualifier": {"type": "array", "items": {"type": "string"}},
                    "reference": {"type": "array", "items": {"type": "string"}},
                    "locator": {
                        "type": "object",
                        "properties": {
                            "article": {"type": "string"},
                            "article_title": {"type": "string"},
                            "section": {"type": "string"},
                        },
                        "required": ["article", "article_title", "section"],
                        "additionalProperties": False,
                    },
                    "table_summary": {"type": "string"},
                    "schema_tag": {"type": "string"},
                },
                "required": ["i", "contract_key", "subject_key", "role",
                             "qualifier", "reference", "locator",
                             "table_summary", "schema_tag"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

VALID_ROLES = [
    "payment_trigger", "exclusion_exception", "claim_procedure", "definition",
    "contract_lifecycle", "timing_period", "limit_frequency", "payment_amount",
    "criteria_rule", "premium_waiver", "code_reference",
]
VALID_SCHEMAS = {"paragraph", "table", "formula"}
ARTICLE_RE = re.compile(r"제\d+(?:-\d+)?조(?:의\d+)?")
SECTION_RE = re.compile(r"제\d+(?:관|편|장|절)")

ROLE_DESC = {
    "payment_trigger": "보험금 지급사유·지급조건",
    "exclusion_exception": "면책·부지급·예외",
    "claim_procedure": "청구 절차·구비서류",
    "definition": "용어의 정의",
    "contract_lifecycle": "갱신·해지·소멸·무효·환급금",
    "timing_period": "보장개시일·대기기간·감액기간·보험기간",
    "limit_frequency": "지급한도·횟수·일수",
    "payment_amount": "지급금액·지급률·산정",
    "criteria_rule": "진단확정·판정기준",
    "premium_waiver": "보험료 납입면제",
    "code_reference": "질병분류코드·부표·분류표",
}

PROMPT_TEMPLATE = """보험약관 조각들에 검색용 태그를 붙여라. 각 조각마다 JSON 한 줄씩, 조각 수만큼 정확히 출력하라(설명 금지).

## 출력 형식
{{"i": 조각번호, "contract_key": "특약/주계약명. [유효목록]에서만 선택. 못 고르면 빈 문자열", "subject_key": ["최대12개. 아래 규칙 참고"], "role": ["역할코드 최대3개"], "qualifier": ["한정 조건 최대6개"], "reference": ["참조 조/표 최대8개"], "locator": {{"article": "제N조", "article_title": "조 제목", "section": "제N관/편/장"}}, "table_summary": "표면 요약(표가 아니면 빈 문자열)", "schema_tag": "paragraph|table|formula"}}

## subject_key 규칙 (가장 중요)
- 앞 4개: 원문에 있는 핵심 명사구(질병명·급부명·제도명). 원문 표기 그대로, 완결된 구만.
- 중간 4개: 고객이 이 조각을 찾으려 검색창에 칠 구어체 표현. 질문 형태도 좋다.
- 뒤 4개: 위 대상어의 동의어·별칭·약어·유사 표현.
- 잘리거나 문장 중간인 구절은 뽑지 마라. 빈 내용이면 개수를 줄여라.

## 기타 규칙
- contract_key는 [유효목록]에서 정확히 복사. 목록에 없는 이름 생성 금지.
- reference: "제N조(...)" 형태 + 부표/별표/별첨 등 참조도 포함.
- schema_tag: 표(| 시작 줄)→table, 수식($$)→formula, 나머지→paragraph.

[역할코드]
{roles}

[유효목록]
{contracts}

"""


def build_prompt(contracts_list):
    roles = "\n".join(f"- {r}: {ROLE_DESC[r]}" for r in VALID_ROLES)
    contracts = "\n".join(contracts_list)
    return PROMPT_TEMPLATE.format(roles=roles, contracts=contracts)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _build_user_prompt(items):
    return "".join(
        f"--- 조각 {i} (계약: {e.get('contract_scope', '')[:60]}) ---\n"
        f"{e['text'][:2000]}\n\n"
        for i, e in items
    )


def llm_batch_claude(model, prompt_base, items, timeout):
    body = _build_user_prompt(items)
    full_prompt = prompt_base + body
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=full_prompt, capture_output=True, text=True, timeout=timeout,
    )
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            if isinstance(d.get("i"), int):
                out[d["i"]] = d
        except Exception:
            continue
    return out


def llm_batch_codex(model, prompt_base, items, timeout, work_dir, retries=3):
    body = _build_user_prompt(items)
    full_prompt = prompt_base + "\n\n" + body

    for attempt in range(retries):
        call_id = next(_call_counter)
        o = work_dir / f"out_{call_id}.json"
        cmd = [CODEX_BIN, "exec", "--model", model,
               "-c", "model_reasoning_effort=low",
               "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
               "--output-schema", str(work_dir / "schema.json"),
               "-o", str(o), "-"]
        try:
            r = subprocess.run(cmd, input=full_prompt.encode("utf-8"),
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                continue
            return {}
        except OSError:
            return {}

        combined = (r.stdout.decode("utf-8", "replace") + "\n"
                    + r.stderr.decode("utf-8", "replace"))
        try:
            obj = json.loads(o.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if RATE_LIMIT_RX.search(combined):
                wait = min(120, 15 * (attempt + 1))
                print(f"  [rate-limit] {wait}s ...", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {}
        finally:
            try:
                o.unlink()
            except OSError:
                pass

        out = {}
        raw_items = obj.get("items") or []
        for d in raw_items:
            if isinstance(d, dict) and isinstance(d.get("i"), int):
                out[d["i"]] = d
        if out:
            return out
        if attempt < retries - 1:
            time.sleep(1)
            continue
    return {}


def llm_batch(transport, model, prompt_base, items, timeout, work_dir=None):
    if transport == "codex":
        return llm_batch_codex(model, prompt_base, items, timeout, work_dir)
    return llm_batch_claude(model, prompt_base, items, timeout)


def validate_tag(llm_out, rule_tag, element, valid_contracts_set):
    tag = {}

    ck = str(llm_out.get("contract_key") or "").strip()
    if ck and ck in valid_contracts_set:
        tag["contract_key"] = ck
    else:
        tag["contract_key"] = rule_tag.get("contract_key", "")

    subj = llm_out.get("subject_key") or []
    tag["subject_key"] = [str(s).strip()[:60] for s in subj if str(s).strip()][:12]

    roles = llm_out.get("role") or []
    tag["role"] = [r for r in roles if r in VALID_ROLES][:3]

    quals = llm_out.get("qualifier") or []
    tag["qualifier"] = [str(q).strip()[:60] for q in quals if str(q).strip()][:6]

    refs = llm_out.get("reference") or []
    tag["reference"] = [str(r).strip()[:80] for r in refs if str(r).strip()][:8]

    loc_raw = llm_out.get("locator") or {}
    if not isinstance(loc_raw, dict):
        loc_raw = {}
    rule_loc = rule_tag.get("locator") or {}
    art = str(loc_raw.get("article") or "").strip()
    if art and not ARTICLE_RE.match(art):
        art = rule_loc.get("article", "")
    art_title = str(loc_raw.get("article_title") or "").strip()[:80]
    sec = str(loc_raw.get("section") or "").strip()
    if sec and not SECTION_RE.match(sec):
        sec = rule_loc.get("section", "")
    tag["locator"] = {
        "article": art or rule_loc.get("article", ""),
        "article_title": art_title or rule_loc.get("article_title", ""),
        "table_headers": rule_loc.get("table_headers", []),
        "row_keys": rule_loc.get("row_keys", []),
        "section": sec or rule_loc.get("section", ""),
    }

    ts = str(llm_out.get("table_summary") or "").strip()[:120]
    tag["table_summary"] = ts

    schema = str(llm_out.get("schema_tag") or "").strip()
    if schema not in VALID_SCHEMAS:
        schema = rule_tag.get("schema_tag", element.get("element_type", "paragraph"))
    tag["schema_tag"] = schema

    return tag


def build_search_text(tag):
    parts = []
    if tag.get("contract_key"):
        parts.append(f"[contract] {tag['contract_key']}")
    if tag.get("subject_key"):
        parts.append("[subject] " + " | ".join(tag["subject_key"]))
    if tag.get("role"):
        role_ko = {
            "payment_trigger": "지급사유", "exclusion_exception": "면책 제외",
            "claim_procedure": "청구 절차 서류", "definition": "정의",
            "contract_lifecycle": "갱신 해지 소멸", "timing_period": "기간 시점 보장개시",
            "limit_frequency": "한도 횟수", "payment_amount": "지급금액 산정",
            "criteria_rule": "판정 기준", "premium_waiver": "납입면제",
            "code_reference": "분류코드 부표",
        }
        role_text = " | ".join(
            f"{r} {role_ko.get(r, '')}" for r in tag["role"]
        )
        parts.append(f"[role] {role_text}")
    if tag.get("qualifier"):
        parts.append("[qualifier] " + " | ".join(tag["qualifier"]))
    loc = tag.get("locator") or {}
    loc_parts = [loc.get("article", ""), loc.get("article_title", ""), loc.get("section", "")]
    loc_str = " ".join(p for p in loc_parts if p)
    if loc_str:
        parts.append(f"[article] {loc_str}")
    if tag.get("reference"):
        parts.append("[reference] " + " | ".join(tag["reference"]))
    if tag.get("table_summary"):
        parts.append(f"[table] {tag['table_summary']}")
    parts.append(f"[schema] {tag.get('schema_tag', 'paragraph')}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="8슬롯 전 LLM 태그 생성기 v3")
    ap.add_argument("--elements", default=str(FS / "out/elements_u3.jsonl"))
    ap.add_argument("--base-tags", default=str(FS / "out/tags_u4_fact_rules.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out/tags_u4_llm_v3.jsonl"))
    ap.add_argument("--cache", default=str(HERE / "out/.tagllm_v3_cache.jsonl"))
    ap.add_argument("--transport", default="codex", choices=("claude", "codex"),
                    help="LLM 전송: claude(-p) 또는 codex(exec)")
    ap.add_argument("--model", default="gpt-5.6-luna",
                    help="모델 (codex: gpt-5.6-luna, claude: haiku/sonnet)")
    ap.add_argument("--offset", type=int, default=0, help="시작 인덱스 (분할 실행용)")
    ap.add_argument("--limit", type=int, default=0, help="처리할 개수 (0=전체)")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    work_dir = None
    if a.transport == "codex":
        work_dir = HERE / "codex_v3_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir / "schema.json", "w", encoding="utf-8") as sf:
            sf.write(json.dumps(OUTPUT_SCHEMA, ensure_ascii=False))

    els = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    el_map = {e["element_id"]: e for e in els}
    base_rows = [json.loads(l) for l in open(a.base_tags, encoding="utf-8")]
    base_map = {t["element_id"]: t for t in base_rows}

    valid_contracts = sorted({t["contract_key"] for t in base_rows if t.get("contract_key")} - {""})
    valid_contracts_set = set(valid_contracts)
    prompt_base = build_prompt(valid_contracts)

    all_ids = [e["element_id"] for e in els]
    if a.limit:
        targets = all_ids[a.offset:a.offset + a.limit]
    elif a.offset:
        targets = all_ids[a.offset:]
    else:
        targets = all_ids

    cache = {}
    cache_path = Path(a.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(l)
                cache[r["text_sha"]] = r["tag"]
            except Exception:
                continue

    lock = threading.Lock()
    cache_f = open(a.cache, "a", encoding="utf-8")

    def sha_of(eid):
        return sha_text(el_map[eid]["text"])

    todo = [eid for eid in targets if sha_of(eid) not in cache]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    done = [0]
    errors = [0]

    print(json.dumps({
        "total_elements": len(els), "targets": len(targets),
        "cached": len(targets) - len(todo), "todo": len(todo),
        "batches": len(batches), "transport": a.transport, "model": a.model,
        "contracts": len(valid_contracts),
    }, ensure_ascii=False), flush=True)

    def run_batch(batch):
        items = [(i, el_map[eid]) for i, eid in enumerate(batch)]
        try:
            got = llm_batch(a.transport, a.model, prompt_base, items, a.timeout, work_dir)
        except Exception as e:
            print(f"batch error: {e}", flush=True)
            got = {}
            with lock:
                errors[0] += len(batch)
        with lock:
            for i, eid in enumerate(batch):
                raw = got.get(i, {})
                validated = validate_tag(raw, base_map.get(eid, {}), el_map[eid], valid_contracts_set)
                cache_f.write(json.dumps({
                    "text_sha": sha_of(eid),
                    "tag": validated,
                    "llm_raw": bool(raw),
                }, ensure_ascii=False) + "\n")
                cache[sha_of(eid)] = validated
            cache_f.flush()
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(batches):
                print(f"batch {done[0]}/{len(batches)}", flush=True)

    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run_batch, batches))
    cache_f.close()

    filled = empty = fallback_contract = 0
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out_f:
        for eid in [e["element_id"] for e in els]:
            base = dict(base_map.get(eid, {}))
            llm_tag = cache.get(sha_text(el_map[eid]["text"])) if eid in el_map else None

            if llm_tag and any(llm_tag.get(k) for k in ("subject_key", "role")):
                tag = {
                    "element_id": eid,
                    "schema_version": "semtag-llm-v3-1.0",
                    "contract_key": llm_tag.get("contract_key") or base.get("contract_key", ""),
                    "subject_key": llm_tag.get("subject_key", []),
                    "role": llm_tag.get("role", []),
                    "qualifier": llm_tag.get("qualifier", []),
                    "reference": llm_tag.get("reference", []),
                    "locator": llm_tag.get("locator", base.get("locator", {})),
                    "table_summary": llm_tag.get("table_summary", ""),
                    "schema_tag": llm_tag.get("schema_tag", base.get("schema_tag", "paragraph")),
                }
                tag["search_text"] = build_search_text(tag)

                for k in ("parent_jo", "evidence_anchor", "answer_values",
                           "benefit_aliases", "fact_roles", "document_key",
                           "explicit_table_references", "appendix_key",
                           "reference_source_element_ids", "reference_target_element_id",
                           "fact_tag_version"):
                    if k in base:
                        tag[k] = base[k]

                if llm_tag.get("contract_key") != base.get("contract_key", ""):
                    if llm_tag.get("contract_key") == base.get("contract_key", ""):
                        pass
                    elif not llm_tag.get("contract_key"):
                        fallback_contract += 1

                filled += 1
            else:
                tag = base
                tag["element_id"] = eid
                empty += 1

            out_f.write(json.dumps(tag, ensure_ascii=False) + "\n")

    stats = {
        "version": "semtag-llm-v3-1.0",
        "transport": a.transport,
        "model": a.model,
        "total": len(els),
        "targets": len(targets),
        "filled": filled,
        "empty_fallback_rule": empty,
        "errors": errors[0],
        "fallback_contract": fallback_contract,
        "out": a.out,
    }
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
