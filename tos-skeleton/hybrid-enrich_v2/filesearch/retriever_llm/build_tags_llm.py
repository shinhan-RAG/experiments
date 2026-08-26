#!/usr/bin/env python3
"""LLM 검색기 태그 생성기 — element별 U4 동일 스키마 태그를 LLM으로 생성.

설계 원칙: LLM은 적재 시 1회, 산출물 동결·sha 검증, 검색 시점 결정론 유지.
스키마는 U4와 동일 — 엔진·arm 수정 없이 태그 파일 교체로 전환.
contract/locator/reference 는 규칙 태그에서 승계(환각 방지), LLM 담당 축 = subject/role/qualifier.
--only-broken: 규칙 subject 가 절단 조각(괄호 불균형·과길이·공백 낀 장문)인 element 만 LLM 재태깅
하고 나머지는 규칙 태그 그대로 통과(혼합 인덱스) — P1 표적 파일럿 모드.
element 텍스트 sha 캐시로 재실행·증분 갱신 시 재호출 없음. 배치(기본 15개/호출)·병렬(기본 4).
"""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parent

PROMPT = """보험약관 조각들에 검색용 태그를 붙여라. 각 조각마다 JSON 한 줄씩, 조각 수만큼 정확히 출력하라(설명 금지).
형식: {"i": 조각번호, "subject_key": [대상어(질병명·급부명·핵심 개념어) 최대 8개, 원문 표기 그대로, 완결된 명사구만], "role": [payment_trigger|exclusion|claim_procedure|definition|contract_lifecycle|timing_period|limit 중 해당], "qualifier": [한정 조건 문구(다만/최초 1회/연간 N회 등) 최대 4개]}
잘리거나 문장 중간인 구절을 대상어로 뽑지 마라.

"""


def broken_subject(keys):
    return any(k.count("(") != k.count(")") or k.count("[") != k.count("]")
               or len(k) > 20 or (" " in k and len(k) > 12) for k in keys or [])


def llm_batch(model, items, timeout):
    body = "".join(f"--- 조각 {i} (계약: {e.get('contract_scope','')[:50]}) ---\n"
                   f"{e['text'][:1500]}\n\n" for i, e in items)
    proc = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text"],
                          input=PROMPT + body, capture_output=True, text=True, timeout=timeout)
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            if isinstance(d.get("i"), int):
                def as_list(v):
                    if isinstance(v, str):
                        v = [v]
                    return [str(x).strip() for x in (v or []) if str(x).strip()]
                out[d["i"]] = {k: as_list(d.get(k)) for k in ("subject_key", "role", "qualifier")}
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(FS / "out/elements_u3.jsonl"))
    ap.add_argument("--base-tags", default=str(FS / "out/tags_u4_fact_rules.jsonl"))
    ap.add_argument("--out", default=str(FS / "out/tags_u4_llm.jsonl"))
    ap.add_argument("--cache", default=str(FS / "out/.tagllm_v2_cache.jsonl"))
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--only-broken", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--merge", action="store_true",
                    help="대체가 아니라 병기: subject/role/qualifier = 규칙 ∪ LLM (어휘 표면 보존)")
    a = ap.parse_args()

    els = {json.loads(l)["element_id"]: json.loads(l) for l in open(a.elements, encoding="utf-8")}
    base_rows = [json.loads(l) for l in open(a.base_tags, encoding="utf-8")]
    targets = [t["element_id"] for t in base_rows
               if (not a.only_broken or broken_subject(t.get("subject_key")))
               and t["element_id"] in els]
    if a.limit:
        targets = targets[: a.limit]
    cache = {}
    if Path(a.cache).exists():
        for l in open(a.cache, encoding="utf-8"):
            r = json.loads(l)
            cache[r["text_sha"]] = r["tag"]
    lock = threading.Lock()
    cache_f = open(a.cache, "a", encoding="utf-8")

    def sha_of(eid):
        return hashlib.sha256(els[eid]["text"].encode()).hexdigest()

    todo = [eid for eid in targets if sha_of(eid) not in cache]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    done = [0]

    def run_batch(batch):
        items = [(i, els[eid]) for i, eid in enumerate(batch)]
        try:
            got = llm_batch(a.model, items, a.timeout)
        except Exception:
            got = {}
        with lock:
            for i, eid in enumerate(batch):
                cache_f.write(json.dumps({"text_sha": sha_of(eid),
                                          "tag": got.get(i, {})}, ensure_ascii=False) + "\n")
                cache[sha_of(eid)] = got.get(i, {})
            cache_f.flush()
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"batch {done[0]}/{len(batches)}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run_batch, batches))

    target_set = set(targets)
    filled = empty = 0
    with open(a.out, "w", encoding="utf-8") as out_f:
        for t in base_rows:
            b = dict(t)
            if t["element_id"] in target_set:
                llm = cache.get(sha_of(t["element_id"])) or {}
                if any(llm.get(k) for k in ("subject_key", "role", "qualifier")):
                    for k in ("subject_key", "role", "qualifier"):
                        v = llm.get(k)
                        if isinstance(v, str):
                            v = [v]
                        v = [str(x).strip() for x in (v or []) if str(x).strip()]
                        if v:
                            if a.merge:
                                b[k] = list(dict.fromkeys((b.get(k) or []) + v))
                            else:
                                b[k] = v
                    b["schema_version"] = "semtag-llm-1.0"
                    parts = [f"[schema] {b.get('schema_tag', '')}"]
                    if b.get("subject_key"):
                        parts.append("[subject] " + " ".join(map(str, b["subject_key"])))
                    if b.get("role"):
                        parts.append("[role] " + " ".join(map(str, b["role"])))
                    b["search_text"] = " | ".join(parts)
                    filled += 1
                else:
                    empty += 1
            out_f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(json.dumps({"targets": len(targets), "llm_batches": len(batches),
                      "filled": filled, "llm_empty_fallback_rule": empty,
                      "out": a.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
