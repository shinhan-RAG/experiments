#!/usr/bin/env python3
"""Build QA-blind retrieval fields for one policy document.

This is the experiment-side reference implementation.  It deliberately does
not depend on a Docurator schema and never opens QA artifacts.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
LEGACY_V30 = os.environ.get("VECTOR_METADATA_VARIANT") == "v3_0"
SCHEMA_VERSION = "retrieval-fields-v3.0" if LEGACY_V30 else "retrieval-fields-v3.1-vector-separated"

ARTICLE_RE = re.compile(
    r"^#{0,6}\s*(제\s?\d+(?:-\d+)?조(?:의\s?\d+)?)\s*[\(（]?([^)）\n]{0,60})[\)）]?\s*$"
)
APPENDIX_RE = re.compile(r"^#{0,6}\s*\[?(부표\s?[\d-]+(?:-\d+)?)\]?\s*(.{0,60})$")
NUMBER_RE = re.compile(
    r"(?:보험가입금액|가입금액)의?\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%|"
    r"최초\s*1회(?:한)?|\d+회(?:한)?|\d+일(?:\s*한도|분)?|\d+개월|\d+년|"
    r"\d+(?:,\d{3})*(?:만)?원|\d+세"
)
REFERENCE_RE = re.compile(r"(?:제\s?\d+(?:-\d+)?조(?:의\s?\d+)?|부표\s?[\d-]+(?:-\d+)?|[A-Z]\d{2}(?:\.\d+)?)")
ENTITY_RE = re.compile(
    r"[가-힣A-Za-z0-9·()\-]{2,45}(?:특약|보험금|급여금|진단비|수술비|치료비|"
    r"질환|질병|진단|치료|수술|입원|통원|검사|장해|재해|암)"
)

ROLE_RULES = [
    (r"지급하지 않|보상하지 않|면책|제외", "exclusion_exception"),
    (r"납입면제|납입 면제", "premium_waiver"),
    (r"지급사유|보험금의 지급", "payment_trigger"),
    (r"지급금액|지급액|지급률|산식|계산", "payment_amount"),
    (r"한도|횟수|최초\s*1회", "limit_frequency"),
    (r"보장개시|책임개시|감액|대기기간|보험기간", "timing_period"),
    (r"정의|용어", "definition"),
    (r"진단확정|판정기준|기준", "criteria_rule"),
    (r"갱신|해지|소멸|무효|환급", "contract_lifecycle"),
    (r"청구|구비서류", "claim_procedure"),
    (r"분류표|질병코드|수가코드", "code_reference"),
]
ROLE_SEARCH_TERMS = {
    "exclusion_exception": "면책 제외 예외 지급하지 않는 경우",
    "premium_waiver": "보험료 납입면제 납입 면제",
    "payment_trigger": "보험금 지급사유 지급조건 보장조건",
    "payment_amount": "보험금 지급금액 지급률 산정금액 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 청구절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}
SCHEMA_SEARCH_TERMS = {
    "table": "표 테이블 행 열 지급기준표",
    "formula": "산식 수식 계산식",
    "paragraph": "조항 본문 문단",
    "heading": "제목 목차",
}
EVENTS = ["진단확정", "진단", "입원", "통원", "수술", "치료", "검사", "투약", "사망", "장해", "해지", "갱신", "납입"]
TOPICS = {
    "암": "cancer_tumor", "뇌": "brain_heart_circulatory", "심장": "brain_heart_circulatory",
    "입원": "hospital_care", "통원": "outpatient_care", "수술": "surgery",
    "약물": "drug_treatment", "치료": "treatment", "장해": "disability",
    "재해": "accident", "보험료": "premium_contract", "환급": "refund_contract",
    "청구": "claim_process", "분류표": "code_reference",
}
ALIAS_GROUPS = {
    "보험금": ["보험금", "보장금액", "받는 금액"],
    "급여금": ["급여금", "보험금", "보장금액"],
    "진단비": ["진단비", "진단보험금", "진단 급여금"],
    "납입면제": ["납입면제", "보험료 면제", "보험료를 내지 않는 경우"],
    "보장개시": ["보장개시", "책임개시", "보장이 시작되는 때"],
    "감액": ["감액", "일부 지급", "감액기간"],
    "면책": ["면책", "보장 제외", "지급하지 않는 경우"],
    "해약환급금": ["해약환급금", "해지환급금"],
    "입원": ["입원", "병원에 입원"],
    "통원": ["통원", "외래 진료"],
    "수술": ["수술", "수술 치료"],
}


def compact(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" |,.;:")
        if value and value not in seen:
            seen.add(value)
            out.append(value)
        if len(out) >= limit:
            break
    return out


def sentences(text: str) -> list[str]:
    return compact(re.split(r"(?<=[.다요)])\s+|\n+", text), 80)


def roles(text: str) -> list[str]:
    return compact([label for pattern, label in ROLE_RULES if re.search(pattern, text)], 4)


def entities(text: str) -> list[str]:
    candidates = [m.group(0) for m in ENTITY_RE.finditer(text)]
    candidates.sort(key=lambda value: (len(value), value))
    return compact(candidates, 10)


def fields(text: str, scope: str, article: str, section_path: list[str]) -> dict:
    head = " ".join(section_path[-3:] + [article])
    context = (head + "\n" + text[:1600]).strip()
    role_values = roles(context)
    entity_values = entities(context)
    topic_values = compact([label for token, label in TOPICS.items() if token in context], 8)
    event_values = compact([event for event in EVENTS if event in context], 8)
    condition_values = compact(
        [s for s in sentences(text[:2400]) if re.search(r"경우|때에는|한하여|이상|미만|초과|직접|최초|의료기관", s)], 5
    )
    exception_values = compact(
        [s for s in sentences(text[:2400]) if re.search(r"제외|지급하지 않|보상하지 않|다만|예외|면책", s)], 4
    )
    temporal_numeric = compact(NUMBER_RE.findall(text[:3000]), 10)
    references = compact(REFERENCE_RE.findall(context), 8)
    relation_values = [f"{event}:{entity_values[0]}" for event in event_values[:4] if entity_values]
    alias_values = [] if LEGACY_V30 else compact(
        [alias for anchor, aliases in ALIAS_GROUPS.items() if anchor in context for alias in aliases]
        + [value.replace(" ", "") for value in entity_values if " " in value],
        12,
    )

    summary_parts = compact(
        [scope, article, " · ".join(role_values), " · ".join(entity_values[:4]),
         " · ".join(event_values[:4]), " · ".join(condition_values[:2])], 6
    )
    retrieval_summary = ". ".join(summary_parts)
    embedding_parts = [
        f"검색설명: {retrieval_summary}" if retrieval_summary else "",
        f"범위: {scope}" if scope else "",
        f"주제: {', '.join(topic_values)}" if topic_values else "",
        f"대상: {', '.join(entity_values)}" if entity_values else "",
        f"관련표현: {', '.join(alias_values)}" if alias_values else "",
        f"관계: {', '.join(relation_values)}" if relation_values else "",
        f"조건: {' / '.join(condition_values)}" if condition_values else "",
        f"예외: {' / '.join(exception_values)}" if exception_values else "",
        f"기간값: {', '.join(temporal_numeric)}" if temporal_numeric else "",
        f"참조: {', '.join(references)}" if references else "",
    ]
    result = {
        "retrieval_summary": retrieval_summary,
        "scope_context": scope,
        "topic_context": topic_values,
        "canonical_entities": entity_values,
        "event_relations": relation_values,
        "condition_context": condition_values,
        "exception_context": exception_values,
        "temporal_numeric_context": temporal_numeric,
        "reference_context": references,
        "aggregation_context": "cross_element" if len(references) > 1 else "local",
        "embedding_text": "\n".join(part for part in embedding_parts if part),
    }
    if LEGACY_V30:
        result["role_context"] = role_values
    else:
        result["aliases"] = alias_values
    return result


def main() -> None:
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    lines = raw.splitlines()
    article_points: list[tuple[int, str]] = []
    heading_points: list[tuple[int, int, str]] = []
    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            heading_points.append((line_no, len(heading.group(1)), heading.group(2).strip()))
        match = ARTICLE_RE.match(stripped) or APPENDIX_RE.match(stripped)
        if match and "|" not in stripped:
            article_points.append((line_no, f"{match.group(1).replace(' ', '')} {match.group(2).strip()}".strip()))
    article_lines = [item[0] for item in article_points]
    contract_head = re.compile(
        r"^(?:[^|\n]{2,180}특약[^()\n]{0,40}\(무배당[^)\n]*\)|"
        r"신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^\n]*\))\s*$"
    )
    contract_lines = [line_no for line_no, line in enumerate(lines, 1)
                      if line_no > 200 and "|" not in line and contract_head.match(line.strip())]

    def contract_start_at(line_no: int) -> int:
        idx = bisect.bisect_right(contract_lines, line_no) - 1
        return contract_lines[idx] if idx >= 0 else 0

    def article_at(line_no: int) -> str:
        idx = bisect.bisect_right(article_lines, line_no) - 1
        return article_points[idx][1] if idx >= 0 and article_points[idx][0] >= contract_start_at(line_no) else ""

    def path_at(line_no: int) -> list[str]:
        stack: list[tuple[int, str]] = []
        contract_start = contract_start_at(line_no)
        for point, level, title in heading_points:
            if point > line_no:
                break
            if point < contract_start:
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        return [title for _, title in stack[-4:]]

    chunks = [json.loads(line) for line in (OUT / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    elements_path = Path(os.environ.get("ELEMENTS_INPUT", OUT / "elements.jsonl"))
    tags_path = Path(os.environ.get("ELEMENT_TAGS_V3_OUTPUT", OUT / "element_tags_v3.jsonl"))
    elements = [json.loads(line) for line in elements_path.read_text(encoding="utf-8").splitlines()]

    metadata_rows = []
    for chunk in chunks:
        row = {"chunk_id": chunk["chunk_id"], "schema_version": SCHEMA_VERSION}
        context_line = chunk.get("core_line_start", chunk["line_start"])
        row.update(fields(chunk["text"], chunk.get("contract_scope", ""), article_at(context_line), chunk.get("section_path") or path_at(context_line)))
        metadata_rows.append(row)

    tag_rows = []
    for element in ([] if os.environ.get("CHUNK_METADATA_ONLY") == "1" else elements):
        article = article_at(element["line_start"])
        path = path_at(element["line_start"])
        extracted = fields(element["text"], element.get("contract_scope", ""), article, path)
        table_headers: list[str] = []
        row_keys: list[str] = []
        if element["element_type"] == "table":
            table_lines = [line for line in element["text"].splitlines() if line.lstrip().startswith("|")]
            for index, line in enumerate(table_lines[:30]):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                cells = [cell for cell in cells if cell and not re.fullmatch(r"[-: ]+", cell)]
                if not cells:
                    continue
                if index == 0:
                    table_headers = compact(cells, 8)
                elif cells:
                    row_keys.extend(cells[:1])
        locator = compact([article] + table_headers + row_keys, 14)
        tag = {
            "element_id": element["element_id"],
            "schema_version": SCHEMA_VERSION,
            "schema_tag": element["element_type"],
            "contract_key_tag": element.get("contract_scope", ""),
            "subject_key_tag": extracted["canonical_entities"],
            "role_tag": extracted["role_context"],
            "locator_tag": locator,
            "qualifier_tag": compact(extracted["temporal_numeric_context"] + extracted["condition_context"], 10),
            "reference_tag": extracted["reference_context"],
            "aggregation_member_tag": extracted["aggregation_context"],
        }
        lexical_roles = [ROLE_SEARCH_TERMS.get(role, role) for role in tag["role_tag"]]
        tag["search_text"] = " ".join(
            compact(
                [tag["schema_tag"], SCHEMA_SEARCH_TERMS.get(tag["schema_tag"], ""), tag["contract_key_tag"]]
                + tag["subject_key_tag"] + lexical_roles + tag["locator_tag"]
                + tag["qualifier_tag"] + tag["reference_tag"],
                50,
            )
        )
        tag_rows.append(tag)

    metadata_output = OUT / ("chunk_metadata_v3_0_reconstructed.jsonl" if LEGACY_V30 else "chunk_metadata_v3.jsonl")
    outputs = [(metadata_output, metadata_rows)]
    if os.environ.get("CHUNK_METADATA_ONLY") != "1":
        outputs.append((tags_path, tag_rows))
    for path, rows in outputs:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "document_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "qa_accessed": False,
        "metadata_rows": len(metadata_rows),
        "tag_rows": len(tag_rows) if os.environ.get("CHUNK_METADATA_ONLY") != "1" else "unchanged",
        "metadata_nonempty": {key: sum(bool(row.get(key)) for row in metadata_rows) for key in metadata_rows[0] if key not in {"chunk_id", "schema_version"}},
        "tag_nonempty": ({key: sum(bool(row.get(key)) for row in tag_rows) for key in tag_rows[0] if key not in {"element_id", "schema_version"}}
                         if tag_rows else "unchanged"),
    }
    (OUT / "retrieval_fields_v3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
