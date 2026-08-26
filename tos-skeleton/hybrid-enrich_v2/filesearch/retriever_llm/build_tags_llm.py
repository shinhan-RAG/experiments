#!/usr/bin/env python3
"""LLM 검색기 태그 생성기 — element별 U4 동일 스키마 태그를 LLM으로 생성.

설계 원칙(대장 'LLM 배치 원칙' 준수):
- LLM은 **적재 시 1회**만 호출된다. 산출물(tags_u4_llm.jsonl)은 파일로 동결·sha 검증되며,
  검색 시점은 규칙 검색기와 완전히 동일한 결정론 엔진(retriever_rules의 SlotSearch/BM25F)을 쓴다.
- 출력 스키마는 U4 규칙 태그와 동일(contract_key/subject_key/role/locator/qualifier/reference/
  search_text/fact_roles) — 엔진·arm 수정 없이 태그 파일 교체만으로 전환 가능.
- contract_key 는 LLM에 맡기지 않고 element 의 contract_scope 에서 규칙 복사(환각 방지).
  LLM 담당 축: subject_key(대상어 — 규칙의 절단 조각 문제 개선이 목표), role, qualifier.
- element 텍스트 sha 기반 캐시로 재실행·문서 증분 갱신 시 미변경분은 재호출하지 않는다.

Usage:
    python build_tags_llm.py --elements ../out/elements_u3.jsonl \
        --base-tags ../out/tags_u4_fact_rules.jsonl --out ../out/tags_u4_llm.jsonl \
        [--model claude-sonnet-5] [--limit N] [--dry-run]

상태: 스캐폴드 — 실제 대량 생성은 비용 승인 후 실행(32,046 elements).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parent

PROMPT = """다음 보험약관 조각에 검색용 태그를 붙여라. JSON 한 줄로만 답하라.
{{"subject_key": [대상어(질병명·급부명·개념어) 최대 8개, 원문 표기 그대로],
  "role": [이 조각의 기능: payment_trigger/exclusion/claim_procedure/definition/
           contract_lifecycle/timing_period/limit 중 해당되는 것],
  "qualifier": [한정 조건 문구(다만/최초 1회/연간 N회 등) 최대 4개]}}
조각이 속한 계약: {contract}
조각 원문:
{text}"""


def llm_call(model, prompt):
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=prompt, capture_output=True, text=True, timeout=120)
    return proc.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(FS / "out/elements_u3.jsonl"))
    ap.add_argument("--base-tags", default=str(FS / "out/tags_u4_fact_rules.jsonl"),
                    help="규칙 태그 — locator/reference/구조 축은 여기서 승계")
    ap.add_argument("--out", default=str(FS / "out/tags_u4_llm.jsonl"))
    ap.add_argument("--cache", default=str(FS / "out/.tagllm_v2_cache.jsonl"))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--limit", type=int, default=0, help="선행 파일럿용 상한(0=전체)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    els = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    base = {json.loads(l)["element_id"]: json.loads(l)
            for l in open(a.base_tags, encoding="utf-8")}
    cache = {}
    if Path(a.cache).exists():
        for l in open(a.cache, encoding="utf-8"):
            r = json.loads(l)
            cache[r["text_sha"]] = r["tag"]
    cache_f = open(a.cache, "a", encoding="utf-8")
    out_f = open(a.out, "w", encoding="utf-8")
    n_call = n_hit = 0
    for e in els[: a.limit or None]:
        b = dict(base.get(e["element_id"], {}))
        sha = hashlib.sha256(e["text"].encode()).hexdigest()
        if sha in cache:
            llm = cache[sha]; n_hit += 1
        elif a.dry_run:
            llm = {}
        else:
            raw = llm_call(a.model, PROMPT.format(
                contract=e.get("contract_scope", ""), text=e["text"][:2000]))
            try:
                llm = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            except Exception:
                llm = {}
            cache_f.write(json.dumps({"text_sha": sha, "tag": llm},
                                     ensure_ascii=False) + "\n")
            n_call += 1
        for k in ("subject_key", "role", "qualifier"):
            if llm.get(k):
                b[k] = llm[k]
        b["schema_version"] = "semtag-llm-1.0"
        parts = [f"[schema] {b.get('schema_tag', '')}"]
        if b.get("subject_key"):
            parts.append("[subject] " + " ".join(b["subject_key"]))
        if b.get("role"):
            parts.append("[role] " + " ".join(b["role"]))
        b["search_text"] = " | ".join(parts)
        out_f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(json.dumps({"elements": len(els[: a.limit or None]),
                      "llm_calls": n_call, "cache_hits": n_hit,
                      "out": a.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
