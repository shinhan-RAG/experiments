"""Contracts and serialization for QA-aligned chunk metadata.

This module deliberately has no dependency on QA answers or qrels.  The only
QA-derived artifact it accepts is a set of question strings used after model
generation to remove accidental verbatim aliases.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


QUESTION_INTENTS = (
    "보장 여부·가입 가능",
    "지급 조건·진단 요건",
    "금액·비율·계산",
    "횟수·한도",
    "개시일·대기기간·면책기간",
    "제외·면책·보장 제한",
    "갱신·소멸·부활·해지",
    "보험료 납입·납입면제",
    "청구 절차·필요 서류",
    "정의·질병분류·수가코드",
    "대상·범위·포함 여부",
)

FIELD_LIMITS = {
    "질문의도": 3,
    "시맨틱태그": 3,
    "핵심대상": 5,
    "수치조건": 6,
    "제외제한": 4,
    "결과급부": 4,
    "질의별칭": 3,
    "검색키워드": 8,
}

ALL_FIELDS = (
    "핵심주제",
    "질문의도",
    "시맨틱태그",
    "핵심대상",
    "수치조건",
    "제외제한",
    "결과급부",
    "질의별칭",
    "검색키워드",
)

QA_FACET_FIELDS = ("시맨틱태그", "핵심대상", "수치조건", "제외제한", "검색키워드")
QA_QUESTION_FIELDS = ("질의별칭",)
QA_COMBINED_FIELDS = (
    "시맨틱태그", "수치조건", "제외제한", "질의별칭", "검색키워드"
)

GENERIC_STANDALONE_TERMS = {"보험", "계약", "피보험자"}
TAG_RE = re.compile(r"^[^>\s][^>]*>[^>\s][^>]*>[^>\s][^>]*$")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "핵심주제": {"type": "string", "minLength": 1, "maxLength": 60},
        "질문의도": {
            "type": "array",
            "items": {"type": "string", "enum": list(QUESTION_INTENTS)},
            "minItems": 1,
            "maxItems": 3,
        },
        "시맨틱태그": {
            "type": "array",
            "items": {"type": "string", "pattern": TAG_RE.pattern},
            "minItems": 1,
            "maxItems": 3,
        },
        "핵심대상": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "수치조건": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "제외제한": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "결과급부": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "질의별칭": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 3,
        },
        "검색키워드": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": list(ALL_FIELDS),
    "additionalProperties": False,
}


def normalized_question(text: str) -> str:
    """Normalize only superficial differences for exact-copy detection."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip("?？.!。 ")


def _clean_list(value, limit: int, *, max_chars: int = 80) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        item = re.sub(r"\s+", " ", str(item)).strip()[:max_chars]
        if not item or item in GENERIC_STANDALONE_TERMS or item in cleaned:
            continue
        cleaned.append(item)
    return cleaned[:limit]


def normalize_metadata(metadata: dict, qa_questions: set[str] | None = None) -> tuple[dict, int]:
    """Normalize one model result and remove exact QA question aliases.

    Returns ``(metadata, removed_alias_count)``.  QA strings are never used to
    add or rewrite metadata; they can only remove leaked aliases.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    result = {"핵심주제": re.sub(r"\s+", " ", str(metadata.get("핵심주제", ""))).strip()[:60]}
    for field, limit in FIELD_LIMITS.items():
        max_chars = 120 if field == "질의별칭" else 80
        result[field] = _clean_list(metadata.get(field), limit, max_chars=max_chars)

    result["질문의도"] = [x for x in result["질문의도"] if x in QUESTION_INTENTS]
    result["시맨틱태그"] = [x for x in result["시맨틱태그"] if TAG_RE.fullmatch(x)]

    removed = 0
    if qa_questions:
        normalized_qa = {normalized_question(q) for q in qa_questions}
        kept = []
        for alias in result["질의별칭"]:
            if normalized_question(alias) in normalized_qa:
                removed += 1
            else:
                kept.append(alias)
        result["질의별칭"] = kept
    return result, removed


def validate_metadata(metadata: dict) -> list[str]:
    """Application-side validation mirroring the API JSON Schema contract."""
    errors = []
    if not isinstance(metadata, dict):
        return ["metadata must be an object"]
    missing = [field for field in ALL_FIELDS if field not in metadata]
    extra = sorted(set(metadata) - set(ALL_FIELDS))
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"extra fields: {extra}")
    if not isinstance(metadata.get("핵심주제"), str) or not metadata.get("핵심주제", "").strip():
        errors.append("핵심주제 must be a non-empty string")
    if len(str(metadata.get("핵심주제", ""))) > 60:
        errors.append("핵심주제 exceeds 60 characters")
    for field, limit in FIELD_LIMITS.items():
        values = metadata.get(field)
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            errors.append(f"{field} must be a string array")
            continue
        minimum = 1 if field in {"질문의도", "시맨틱태그"} else 2 if field == "질의별칭" else 0
        if not minimum <= len(values) <= limit:
            errors.append(f"{field} item count must be {minimum}..{limit}")
        if len(values) != len(set(values)):
            errors.append(f"{field} contains duplicates")
    intents = metadata.get("질문의도", [])
    if isinstance(intents, list) and any(x not in QUESTION_INTENTS for x in intents):
        errors.append("질문의도 contains a value outside the controlled vocabulary")
    tags = metadata.get("시맨틱태그", [])
    if isinstance(tags, list) and any(not TAG_RE.fullmatch(x) for x in tags if isinstance(x, str)):
        errors.append("시맨틱태그 must have exactly three non-empty levels")
    for field in FIELD_LIMITS:
        values = metadata.get(field, [])
        if isinstance(values, list) and any(x in GENERIC_STANDALONE_TERMS for x in values):
            errors.append(f"{field} contains a forbidden standalone generic term")
    return errors


def serialize_metadata_fields(metadata: dict, fields: list[str] | tuple[str, ...] | None = None,
                              max_chars: int = 240) -> str:
    """Render selected retrieval fields without exceeding ``max_chars``."""
    if max_chars <= 0:
        return ""
    selected = tuple(fields) if fields is not None else tuple(metadata)
    parts = []
    for field in selected:
        if field not in metadata:
            continue
        value = metadata[field]
        if isinstance(value, list):
            rendered_value = " ".join(str(x).strip() for x in value if str(x).strip())
        elif isinstance(value, dict):
            rendered_value = " ".join(f"{k}:{v}" for k, v in value.items() if str(v).strip())
        else:
            rendered_value = str(value).strip()
        if not rendered_value:
            continue
        candidate = f"{field}:{rendered_value}"
        separator = " | " if parts else ""
        remaining = max_chars - len("".join(parts)) - len(separator)
        if remaining <= 0:
            break
        parts.append(separator + candidate[:remaining].rstrip())
    return "".join(parts)


def metadata_quality_report(corpus: list[dict], metadata_by_id: dict[str, dict],
                            qa_questions: set[str]) -> dict:
    """Compute the non-leakage and diversity gates for a completed corpus."""
    normalized_qa = {normalized_question(q) for q in qa_questions}
    missing = [row["_id"] for row in corpus if row["_id"] not in metadata_by_id]
    invalid = {}
    copied = []
    fingerprints_by_doc = defaultdict(list)
    for chunk in corpus:
        chunk_id = chunk["_id"]
        metadata = metadata_by_id.get(chunk_id)
        if metadata is None:
            continue
        errors = validate_metadata(metadata)
        if errors:
            invalid[chunk_id] = errors
        for alias in metadata.get("질의별칭", []):
            if normalized_question(alias) in normalized_qa:
                copied.append({"chunk_id": chunk_id, "alias": alias})
        fingerprint = serialize_metadata_fields(
            metadata, ("핵심주제", "시맨틱태그", "핵심대상"), max_chars=1000
        )
        fingerprints_by_doc[str(chunk.get("doc", "unknown"))].append(fingerprint)

    doc_ratios = {
        doc: len(set(values)) / len(values) if values else 0.0
        for doc, values in fingerprints_by_doc.items()
    }
    covered = len(corpus) - len(missing)
    coverage = covered / len(corpus) if corpus else 0.0
    weighted_unique = (
        sum(len(set(values)) for values in fingerprints_by_doc.values()) / covered
        if covered else 0.0
    )
    minimum_unique = min(doc_ratios.values(), default=0.0)
    return {
        "chunk_count": len(corpus),
        "covered_chunk_count": covered,
        "coverage": round(coverage, 6),
        "missing_chunk_ids": missing,
        "invalid_chunk_count": len(invalid),
        "invalid_chunks": invalid,
        "exact_qa_copy_count": len(copied),
        "exact_qa_copies": copied,
        "within_document_unique_ratio": round(weighted_unique, 6),
        "minimum_within_document_unique_ratio": round(minimum_unique, 6),
        "within_document_unique_ratio_by_doc": {
            key: round(value, 6) for key, value in sorted(doc_ratios.items())
        },
        "gates": {
            "coverage_100pct": coverage == 1.0,
            "valid_schema_100pct": not invalid,
            "within_document_unique_gte_95pct": minimum_unique >= 0.95,
            "exact_qa_copy_zero": not copied,
        },
    }
