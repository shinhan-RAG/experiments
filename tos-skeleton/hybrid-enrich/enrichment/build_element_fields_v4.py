#!/usr/bin/env python3
"""Refill schema-aware semantic fields for every immutable element.

Inputs are document-derived elements and v3 structural context only. QA and
gold files are intentionally not opened by this generator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
VERSION = "element-semantic-fields-v4.0"

QUOTED_RE = re.compile(r'["“「『<](.{2,60}?)["”」』>]')
SUBJECT_RE = re.compile(
    r"[가-힣A-Za-z0-9·()\- ]{2,60}?(?:보험금|급여금|지원비|치료비|진단비|수술비|"
    r"입원비|검사비|환급금|보험료|보험나이|가입연령|납입면제|질병|질환|신생물|"
    r"장해|재해|암|수술|치료|검사|입원|통원)"
)
DEFINITION_RE = re.compile(r"([가-힣A-Za-z0-9·()\- ]{2,50}?)(?:이라 함은|라 함은|이란)")
CONDITION_RE = re.compile(r"([^\n.]{2,100}(?:경우|때에는|한하여|이상|미만|초과|이내|이후|이전)[^\n.]{0,80})")
VALUE_RE = re.compile(
    r"(?:보험가입금액|가입금액)의?\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%|"
    r"최초\s*1회(?:한)?|\d+회(?:한)?|\d+일(?:\s*한도|분)?|\d+개월|\d+년|"
    r"\d+(?:,\d{3})*(?:만)?원|\d+세|[A-Z]\d{2}(?:\.\d+)?(?:~[A-Z]?\d{2})?"
)
GENERIC = {
    "보험금", "급여금", "질병", "질환", "치료", "수술", "검사", "입원", "통원",
    "특약", "주계약", "피보험자", "보험수익자", "계약자", "회사", "해당", "경우",
    "보험금 지급사유", "보험금 지급에 관한 세부규정", "용어의 정의",
}
BAD_KEY_RE = re.compile(
    r"제\d+(?:-\d+)?\s*(?:조|관)|부표\s?\d|별표\s?\d|한국표준질병|보험금 지급|"
    r"주계약 약관|세부규정|간편심사형|직접적인 치료|지급절차|제\s?\d+항|주\d+\)|"
    r"계약자는|경우|동안|에도 불구하고|에 따라|^이라 함은|^은\s|^의\s"
)
BAD_SUBJECT_FRAGMENT_RE = re.compile(
    r"^(?:및|의|을|를|은|는|이|가|으로|로|에서|에|등)\s|(?:받는 방법의 변경|지급하지 않는 사유|"
    r"및 진단확정|에 관한 세부규정|보험금 등의 지급절차)$|^\d+(?:\.\d+)?$|"
    r"^(?:/?tr|/?td|/?th|일반형|제\d+편\s+일반사항|\(간편\)암)$"
)
DOMAIN_SUBJECT_RE = re.compile(
    r"보험금|급여금|지원비|치료비|진단비|수술비|입원비|검사비|환급금|보험료|"
    r"보험나이|가입연령|납입면제|계약|질병|질환|신생물|장해|재해|암|수술|"
    r"치료|검사|입원|통원|분류|코드|대장점막|방광|피부|갑상선|뇌혈관|심장|"
    r"화상|부식|중환자실|일상생활|골절|치매|파킨슨|통풍|대상포진|요양"
)
ROLE_KO = {
    "exclusion_exception": "면책 제외 예외 부지급",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정",
    "limit_frequency": "지급한도 횟수 일수",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}
ROLE_RULES = [
    (r"지급하지 않|보상하지 않|면책|제외", "exclusion_exception"),
    (r"납입.?면제", "premium_waiver"),
    (r"지급사유|지급 조건", "payment_trigger"),
    (r"지급금액|지급률|산정|계산", "payment_amount"),
    (r"한도|횟수|최초\s*1회", "limit_frequency"),
    (r"보장개시|책임개시|감액|대기기간|보험기간|보험나이|가입연령", "timing_period"),
    (r"정의|이라 함은|라 함은", "definition"),
    (r"진단확정|판정기준|적용 기준|등록 및 결정", "criteria_rule"),
    (r"갱신|해지|소멸|무효|환급", "contract_lifecycle"),
    (r"청구|구비서류", "claim_procedure"),
    (r"분류표|분류코드|질병코드|수가코드", "code_reference"),
]


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def unique(values, limit=30):
    seen, output = set(), []
    for value in values:
        value = re.sub(r"\s+", " ", str(value)).strip(" |,.;:[]#")
        if (not value or value in GENERIC or len(value) < 2 or len(value) > 100
                or value in seen or BAD_KEY_RE.search(value)):
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def clean_subjects(values, limit=30, require_domain=False):
    output = []
    for value in values:
        value = re.sub(r"^[\s\"'“”「」『』]+|[\s\"'“”「」『』]+$", "", str(value))
        value = re.sub(r"^(?:및|또는)\s+", "", value)
        value = re.sub(r"\s+", " ", value).strip(" |,.;:[]#")
        if (not value or len(value) < 2 or len(value) > 60 or BAD_SUBJECT_FRAGMENT_RE.search(value)
                or BAD_KEY_RE.search(value) or not re.search(r"[가-힣A-Za-z]{2}", value)):
            continue
        if require_domain and not DOMAIN_SUBJECT_RE.search(value):
            continue
        output.append(value)
    return unique(output, limit)


def table_structure(text):
    rows = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [re.sub(r"<[^>]+>", " ", cell).strip() for cell in line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r"[-: ]+", cell)]
        if cells:
            rows.append(cells)
    headers = unique(rows[0], 12) if rows else []
    row_keys = unique([cells[0] for cells in rows[1:] if cells], 40)
    # Tables produced from OCR sometimes put the logical row key in column 2.
    if len(set(row_keys)) <= 1:
        row_keys = unique(row_keys + [cells[1] for cells in rows[1:] if len(cells) > 1], 40)
    return headers, row_keys


def article_parts(locator):
    article = locator[0] if locator else ""
    match = re.match(r"(제\d+(?:-\d+)?조(?:의\d+)?)\s*(.*)", article)
    return (match.group(1), match.group(2).strip("() ")) if match else (article, "")


def main():
    elements_path = Path(os.environ.get("ELEMENTS_INPUT", OUT / "elements.jsonl"))
    old_path = Path(os.environ.get("ELEMENT_TAGS_V3_INPUT", OUT / "element_tags_v3.jsonl"))
    output_path = Path(os.environ.get("ELEMENT_FIELDS_V4_OUTPUT", OUT / "element_semantic_fields_v4.jsonl"))
    grep_path = Path(os.environ.get("GREP_TAG_V4_OUTPUT", OUT / "grep_tag_v4.jsonl"))
    manifest_path = Path(os.environ.get("ELEMENT_FIELDS_V4_MANIFEST", OUT / "element_semantic_fields_v4_manifest.json"))
    elements = [json.loads(line) for line in elements_path.read_text(encoding="utf-8").splitlines()]
    old = {row["element_id"]: row for row in (json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines())}
    output = []
    running_section = ""
    previous_contract = None
    for index, element in enumerate(elements):
        if element.get("contract_scope") != previous_contract:
            running_section = ""
            previous_contract = element.get("contract_scope")
        previous = "\n".join(item["text"] for item in elements[max(0, index - 3):index] if len(item["text"]) <= 160)
        text = element["text"]
        stripped = re.sub(r"^#+\s*", "", text.strip())
        if (element["element_type"] == "heading" or re.match(r"^\[?(?:부표|별첨)\s?\d", stripped)) and len(stripped) <= 120:
            running_section = stripped
        base = old[element["element_id"]]
        article_id, article_title = article_parts(base.get("locator_tag", []))
        headers, row_keys = table_structure(text) if element["element_type"] == "table" else ([], [])

        explicit_quotes = QUOTED_RE.findall(text[:5000])
        explicit_definitions = DEFINITION_RE.findall(text[:3000])
        explicit_domain_terms = SUBJECT_RE.findall(text[:5000])
        subjects = (
            clean_subjects(explicit_quotes + explicit_definitions, 20, require_domain=True)
            + clean_subjects(explicit_domain_terms, 25, require_domain=True)
            + clean_subjects(row_keys, 30, require_domain=True)
        )
        if article_title and re.search(r"보험금|급여금|지원비|치료비|진단비|수술비|질병|질환|암|수술|치료|검사|입원|통원", article_title):
            cleaned_title = re.sub(r"보험금의?|지급(?:사유|금액|에 관한 세부규정)|정의|세부규정|기준", " ", article_title)
            subjects.append(cleaned_title)
        if element["element_type"] == "formula":
            subjects += QUOTED_RE.findall(previous) + SUBJECT_RE.findall(previous)
        subjects = clean_subjects(subjects, 30)

        inferred_roles = [role for pattern, role in ROLE_RULES if re.search(pattern, article_title + "\n" + running_section + "\n" + text[:4000])]
        roles = unique((base.get("role_tag") or []) + inferred_roles, 8)
        condition_keys = CONDITION_RE.findall(text[:6000])
        value_keys = VALUE_RE.findall(text[:6000])
        qualifiers = condition_keys + value_keys
        qualifiers = unique(qualifiers, 20)
        references = unique([value for value in (base.get("reference_tag") or []) if value.replace(" ", "") != article_id.replace(" ", "")], 12)

        locator = {
            "article": article_id,
            "article_title": article_title,
            "table_headers": headers,
            "row_keys": row_keys,
            "formula_context": unique(SUBJECT_RE.findall(previous), 8) if element["element_type"] == "formula" else [],
            "section": running_section or ("document_header" if index < 10 else ""),
        }
        contract = base.get("contract_key_tag", "")
        role_ko = [ROLE_KO.get(role, role) for role in roles]
        serial = [
            f"[schema] {element['element_type']}",
            f"[contract] {contract}" if contract else "",
            f"[subject] {' | '.join(subjects)}" if subjects else "",
            f"[role] {' | '.join(roles)} | {' | '.join(role_ko)}" if roles else "",
            f"[article] {article_id} {article_title}" if article_id or article_title else "",
            f"[columns] {' | '.join(headers)}" if headers else "",
            f"[rows] {' | '.join(row_keys)}" if row_keys else "",
            f"[section] {locator['section']}" if locator["section"] else "",
            f"[qualifier] {' | '.join(qualifiers)}" if qualifiers else "",
            f"[reference] {' | '.join(references)}" if references else "",
        ]
        row = {
            "element_id": element["element_id"], "schema_version": VERSION,
            "schema_tag": element["element_type"], "contract_key": contract,
            "subject_key": subjects, "role": roles, "locator": locator,
            "qualifier": qualifiers, "reference": references,
            "field_sources": {
                "subject": {
                    "explicit_quote": unique(explicit_quotes, 20),
                    "explicit_definition": unique(explicit_definitions, 20),
                    "domain_term": unique(explicit_domain_terms, 20),
                    "table_row": row_keys,
                },
                "qualifier": {"condition": unique(condition_keys, 20), "value": unique(value_keys, 20)},
            },
            "search_text": " ".join(value for value in serial if value),
        }
        output.append(row)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    element_map = {row["element_id"]: row["text"] for row in elements}
    with grep_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps({"element_id": row["element_id"], "g": row["search_text"] + " ||| " + element_map[row["element_id"]].replace("\n", " ")}, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": VERSION, "rows": len(output), "qa_accessed": False,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filled": {
            "contract": sum(bool(row["contract_key"]) for row in output),
            "subject": sum(bool(row["subject_key"]) for row in output),
            "role": sum(bool(row["role"]) for row in output),
            "qualifier": sum(bool(row["qualifier"]) for row in output),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
