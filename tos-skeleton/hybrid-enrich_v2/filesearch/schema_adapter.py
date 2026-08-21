"""Semantic Tag JSON을 문서 종류에 독립적인 검색 축으로 사상한다.

검색기는 특정 태그 스키마의 key를 직접 소비하지 않는다. 문서 유형별 태거는 value
extractor/plugin으로 교체할 수 있고, 검색기는 아래의 안정된 내부 축만 사용한다.

container 문서·파일·상품처럼 하위 section을 포함하는 상위 식별자
identity  특약·조직·section·엔티티의 직접 소속/식별자
topic     대상·주제·제목
function  정의/조건/절차/지급 등 문단 기능
locator   장·절·조·항·페이지·breadcrumb
table     표 제목·열·행·셀
constraint 값·조건·날짜·금액·범위·코드
relation  참조·인용·링크·관련 항목
structure 문서/요소/스키마 유형
extra     처음 보는 필드의 scalar 값(저가중 폴백; 유실 방지)
evidence  문장·표 행에서 추출한 답변 근거/값/범용 표면형
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


AXES = ("container", "identity", "topic", "function", "locator", "table",
        "constraint", "relation", "structure", "evidence", "extra")

# terminal key 별칭. 새 태거는 adapt_tag(key_aliases=...)로 별칭을 추가할 수 있다.
KEY_AXIS = {
    "document_key": "container", "document_scope": "container",
    "container_key": "container", "container_identity": "container",
    "contract_key": "identity", "contract_scope": "identity", "scope": "identity",
    "section_key": "identity", "section_identity": "identity",
    "product": "identity", "product_name": "identity", "entity": "identity",
    "entity_id": "identity", "document_id": "identity", "document_name": "identity",
    "owner": "identity", "organization": "identity", "category": "identity",
    "subject_key": "topic", "subject": "topic", "topic": "topic",
    "title": "topic", "name": "topic", "aliases": "topic", "keywords": "topic",
    "canonical_entities": "topic", "formula_subject": "topic", "table_title": "topic",
    "list_title": "topic", "item_topics": "topic",
    "role": "function", "semantic_role": "function", "function": "function",
    "intent": "function", "predicate": "function", "action": "function",
    "locator": "locator", "path": "locator", "breadcrumb": "locator",
    "chapter": "locator", "section": "locator", "subsection": "locator",
    "article": "locator", "article_title": "locator", "paragraph": "locator",
    "clause": "locator", "item": "locator", "page": "locator", "page_number": "locator",
    "heading": "locator", "article_label": "locator",
    "table": "table", "table_headers": "table", "column_headers": "table",
    "appendix_key": "evidence", "table_key": "evidence",
    "columns": "table", "row_keys": "table", "row_labels": "table",
    "cells": "table", "formula_context": "table", "variables": "table",
    "qualifier": "constraint", "qualifiers": "constraint", "values": "constraint",
    "value": "constraint", "conditions": "constraint", "condition": "constraint",
    "amount": "constraint", "range": "constraint", "date": "constraint",
    "period": "constraint", "code": "constraint", "codes": "constraint",
    "exact_keys": "constraint", "definition_terms": "constraint",
    "reference": "relation", "references": "relation", "ref": "relation",
    "citation": "relation", "citations": "relation", "link": "relation",
    "links": "relation", "related": "relation", "parent": "relation",
    "schema_tag": "structure", "schema_version": "structure",
    "element_type": "structure", "document_type": "structure", "doc_type": "structure",
    "mime_type": "structure", "type": "structure",
    "evidence_anchor": "evidence", "answer_kernel": "evidence",
    "answer_value": "evidence", "answer_values": "evidence",
    "benefit_alias": "evidence", "benefit_aliases": "evidence",
    "enumerated_item": "evidence", "enumerated_items": "evidence",
    "linked_identity": "evidence", "reference_identity": "evidence",
    "fact_role": "evidence", "fact_roles": "evidence",
    "explicit_table_reference": "evidence", "explicit_table_references": "evidence",
    "parent_jo": "evidence", "parent_unit": "evidence",
}

IGNORED_KEYS = {
    "element_id", "id", "field_sources", "search_text", "text", "body", "content",
    "char_start", "char_end", "line_start", "line_end", "members", "is_toc",
    "fact_tag_version",
    "reference_source_element_ids", "reference_target_element_id",
}


def _scalars(value: Any):
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)):
        s = str(value).strip()
        if s:
            yield s
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalars(child)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _scalars(child)


def adapt_tag(tag: dict, element: dict | None = None,
              key_aliases: dict[str, str] | None = None) -> dict[str, list[str]]:
    """임의 JSON tag를 범용 축으로 변환하고 미지 scalar를 extra에 보존한다."""
    aliases = {**KEY_AXIS, **(key_aliases or {})}
    out = defaultdict(list)
    recognized_paths = []
    unknown_paths = []

    def visit(key: str, value: Any, inherited: str | None = None, path: str = ""):
        low = str(key).lower()
        here = f"{path}.{low}" if path else low
        if low in IGNORED_KEYS:
            return
        axis = aliases.get(low) or inherited
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child_key, child, axis, here)
            return
        if axis:
            vals = list(_scalars(value))
            if vals:
                out[axis].extend(vals)
                recognized_paths.append(here)
            return
        vals = list(_scalars(value))
        if vals:
            out["extra"].extend(vals)
            unknown_paths.append(here)

    for key, value in (tag or {}).items():
        visit(key, value)

    element = element or {}
    if not out["identity"] and element.get("contract_scope"):
        out["identity"].append(str(element["contract_scope"]))
    if element.get("title"):
        out["locator"].append(str(element["title"]))
    if element.get("element_type") and not out["structure"]:
        out["structure"].append(str(element["element_type"]))

    result = {axis: list(dict.fromkeys(out[axis])) for axis in AXES}
    result["_recognized_paths"] = recognized_paths
    result["_unknown_paths"] = unknown_paths
    return result
