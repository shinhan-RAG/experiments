#!/usr/bin/env python3
"""엘리먼트 텍스트에서 8-슬롯 태그를 정규식/규칙으로 기계 추출. LLM 호출 없음."""
from __future__ import annotations
import json, re, sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "out"

VALID_ROLES = {
    "exclusion_exception", "premium_waiver", "payment_trigger", "payment_amount",
    "limit_frequency", "timing_period", "definition", "criteria_rule",
    "contract_lifecycle", "claim_procedure", "code_reference",
}
ROLE_KEYWORDS = {
    "exclusion_exception": re.compile(r"면책|부지급|지급하지\s*않|지급제한|제외"),
    "premium_waiver": re.compile(r"납입면제|보험료.*면제"),
    "payment_trigger": re.compile(r"지급사유|지급.*조건|보험금.*지급"),
    "payment_amount": re.compile(r"지급금액|지급률|보험가입금액|산정|계산"),
    "limit_frequency": re.compile(r"지급한도|횟수|최초\s*1회|일수제한|한도"),
    "timing_period": re.compile(r"보장개시|책임개시|대기기간|감액기간|보험기간"),
    "definition": re.compile(r"정의|용어.*뜻|말합니다|의미합니다"),
    "criteria_rule": re.compile(r"진단확정|판정기준|적용기준"),
    "contract_lifecycle": re.compile(r"갱신|해지|소멸|무효|환급|부활|취소"),
    "claim_procedure": re.compile(r"청구.*서류|구비서류|청구절차|제출"),
    "code_reference": re.compile(r"질병분류|분류코드|수가코드|부표|분류표"),
}
RE_ARTICLE = re.compile(r"제\s*(\d+)\s*조\s*[\(（]([^)）]+)[\)）]")
RE_ARTICLE_NUM = re.compile(r"제\s*(\d+)\s*조")
RE_REFERENCE = re.compile(r"(?:제\s*\d+\s*조|별표\s*\d+|부표\s*\d+|【[^】]+】)")
RE_QUALIFIER = re.compile(
    r"\d+[%％]|\d+일|\d+년|\d+원|\d+세|\d+개월"
    r"|보험가입금액의\s*\d+[%％]"
    r"|이상|이하|초과|미만|이내"
)
RE_CONTRACT = re.compile(
    r"(?:\(무\)|(?:무배당[,\s]*)?)"
    r"[가-힣]+(?:특약|보험|보장)"
    r"(?:\([^)]*\))?"
)
BOILERPLATE = re.compile(r"\(무배당[^)]*\)|\(간편\)|\(해약환급금\s*미지급형\)|\(갱신형\)|\(일반형\)")
GENERIC = {"보험", "약관", "보험금", "특약", "보험계약"}
RE_SUBJECT = re.compile(r"[가-힣]{2,8}(?:진단비|수술비|입원비|사망보험금|장해급여금|생활자금|간병비|연금|환급금)")


def extract(elem: dict) -> dict:
    text = elem.get("text", "")
    etype = elem.get("element_type", "")
    scope = elem.get("contract_scope", "")

    contract = []
    if scope and scope not in GENERIC:
        c = BOILERPLATE.sub("", scope).strip()
        if c and len(c) >= 2:
            contract.append(c[:40])

    subjects = list({m.group() for m in RE_SUBJECT.finditer(text)})[:5]

    roles = []
    for role, rx in ROLE_KEYWORDS.items():
        if rx.search(text):
            roles.append(role)
            if len(roles) >= 3:
                break

    articles = []
    for m in RE_ARTICLE.finditer(text):
        articles.append(f"제{m.group(1)}조({m.group(2)})")
    if not articles:
        for m in RE_ARTICLE_NUM.finditer(text):
            articles.append(f"제{m.group(1)}조")
    articles = list(dict.fromkeys(articles))[:5]

    table = []
    if etype in ("table", "table_row", "table_header"):
        headers = [c.strip() for c in text.split("|") if c.strip()][:6]
        table = [h[:30] for h in headers if len(h) >= 2]

    qualifiers = list({m.group() for m in RE_QUALIFIER.finditer(text)})[:4]
    references = list({m.group().strip() for m in RE_REFERENCE.finditer(text)} - set(articles))[:5]
    schema = [etype] if etype else []

    return {
        "element_id": elem["element_id"],
        "ok": True,
        "contract": contract,
        "subject": subjects,
        "role": roles,
        "article": articles,
        "table": table,
        "qualifier": qualifiers,
        "reference": references,
        "schema": schema,
    }


def main():
    inp = OUT / "elements_repaired_v1.jsonl"
    out = OUT / "element_tags_llm_v1.jsonl"

    elements = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]

    with out.open("w", encoding="utf-8") as f:
        for elem in elements:
            f.write(json.dumps(extract(elem), ensure_ascii=False) + "\n")

    filled = {k: 0 for k in ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")}
    for elem in elements:
        tags = extract(elem)
        for k in filled:
            if tags[k]:
                filled[k] += 1

    print(f"완료: {len(elements):,}건")
    for k, v in filled.items():
        print(f"  {k:16s} {100*v/len(elements):5.1f}% ({v:,}건)")


if __name__ == "__main__":
    main()
