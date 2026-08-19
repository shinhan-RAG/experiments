#!/usr/bin/env python3
"""LLM 질의 태그(qtags) 생성 — v2ds 의 "query_slots(규칙) + LLM qtags" 중 LLM 부분.

입력은 **질문 텍스트와 특약명 사전(태그의 contract_key 집합)만**. 문서·gold·정답은 주지 않는다(순환 방지).
출력: out/qtags_<model>.jsonl  {qid, contract[], role[], subject[], qualifier[], schema[], raw}
claude CLI(-p, --output-format json) 사용, 결과 캐시(같은 qid 는 재호출하지 않음). 병렬 --workers.
"""
import argparse, json, re, subprocess, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLES = ["exclusion_exception", "premium_waiver", "payment_trigger", "payment_amount", "limit_frequency",
         "timing_period", "definition", "criteria_rule", "contract_lifecycle", "claim_procedure", "code_reference"]
ROLE_DESC = {
    "exclusion_exception": "보험금을 지급하지 않는 사유·면책·예외", "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유·지급조건", "payment_amount": "지급금액·지급률·산정",
    "limit_frequency": "지급한도·횟수·일수", "timing_period": "보장개시일·대기기간·감액기간·보험기간·나이",
    "definition": "용어의 정의", "criteria_rule": "진단확정·판정기준", "contract_lifecycle": "갱신·해지·소멸·무효·환급금",
    "claim_procedure": "청구 절차·구비서류", "code_reference": "질병분류코드·부표·분류표",
}

PROMPT = """당신은 보험 약관 검색용 질의 분석기입니다. 아래 사용자 질문을 읽고 JSON 하나만 출력하세요(설명 금지).

필드:
- "contract": 질문이 가리키는 특약/주계약 이름. 반드시 아래 [특약 목록]에 있는 문자열을 그대로 복사. 특정할 수 없으면 []. 여러 개면 모두. 질문이 상품 전체·주계약이면 "(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)".
- "role": 질문이 묻는 조항의 역할 코드. 아래 [역할 코드] 중에서만, 최대 3개.
- "subject": 질문의 핵심 대상어(급여금명·질병명·용어 등)를 약관 문서 표기로 2~5개. 예: "암보장개시일", "허혈심장질환진단급여금".
- "qualifier": 질문에 있는 조건·수치(횟수, 기간, %, 금액 등) 그대로, 없으면 [].
- "schema": 질문이 표(분류표·부표)나 산식을 묻으면 ["table"] 또는 ["formula"], 아니면 [].

[역할 코드]
{roles}

[특약 목록]
{contracts}

[질문]
{question}

JSON:"""


def call(model, prompt):
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json"], input=prompt,
                       capture_output=True, text=True, timeout=180)
    try:
        d = json.loads(r.stdout)
        txt = d.get("result", "")
    except Exception:
        txt = r.stdout
    m = re.search(r"\{.*\}", txt, re.S)
    return (json.loads(m.group(0)) if m else {}), txt[:500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "out/gold_spans_train.jsonl"))
    ap.add_argument("--tags", default=str(HERE / "out/tags_u2_rules.jsonl"))
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=-1)
    a = ap.parse_args()
    contracts = sorted({json.loads(l)["contract_key"] for l in open(a.tags, encoding="utf-8")} - {""})
    roles = "\n".join(f"- {r}: {ROLE_DESC[r]}" for r in ROLES)
    out_path = HERE / "out" / f"qtags_{a.model}.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            d = json.loads(l); done[d["qid"]] = d
    Q = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    if a.n > 0:
        Q = Q[: a.n]
    todo = [q for q in Q if q["qid"] not in done]
    print(f"contracts={len(contracts)} total={len(Q)} cached={len(Q)-len(todo)} todo={len(todo)}", flush=True)

    def work(q):
        p = PROMPT.format(roles=roles, contracts="\n".join(contracts), question=q["q"])
        js, raw = call(a.model, p)
        rec = {"qid": q["qid"], "q_sha": hashlib.sha256(q["q"].encode()).hexdigest()[:12],
               "contract": [c for c in (js.get("contract") or []) if c in contracts],
               "role": [r for r in (js.get("role") or []) if r in ROLES],
               "subject": [str(s)[:60] for s in (js.get("subject") or [])][:6],
               "qualifier": [str(s)[:60] for s in (js.get("qualifier") or [])][:6],
               "schema": [s for s in (js.get("schema") or []) if s in ("table", "formula")],
               "model": a.model, "parse_ok": bool(js)}
        return rec

    with ThreadPoolExecutor(a.workers) as ex, open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(work, todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 20 == 0:
                print(i, rec["qid"], rec["contract"][:1], rec["role"], flush=True)
    print("done")


if __name__ == "__main__":
    main()
