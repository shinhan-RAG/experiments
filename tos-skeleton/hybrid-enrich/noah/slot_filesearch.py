#!/usr/bin/env python3
"""Schema-aware lexical FileSearch using AND slots, without relevance ranking."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"
FIELDS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")
ROLE_ALIASES = {
    "exclusion_exception": "면책 제외 예외 부지급 지급하지 않는 사유",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def compact(value):
    return re.sub(r"[^가-힣a-z0-9]", "", str(value).lower())


def contract_core(value):
    value = re.sub(r"\(무배당[^)]*\)|\(간편\)|해약환급금\s*미지급형|갱신형|일반형", "", str(value))
    value = compact(value).replace("특약", "")
    return value.replace("허혈성심장질환", "허혈심장질환").replace("대상포진진단", "대상포진통풍진단")


def trigrams(value):
    value = compact(value)
    return {value[index:index + 3] for index in range(max(0, len(value) - 2))}


def fuzzy_contains(query, target, field):
    q = contract_core(query) if field == "contract" else compact(query)
    t = contract_core(target) if field == "contract" else compact(target)
    if not q or not t:
        return False
    if q in t or t in q:
        return True
    if field != "contract" or min(len(q), len(t)) < 5:
        return False
    qg, tg = trigrams(q), trigrams(t)
    # Partial contract names may omit a bundled coverage term, but must retain
    # both a leading and trailing anchor. This prevents similarly named riders
    # (e.g. CAR-T vs targeted anticancer drug) from sharing a namespace.
    anchored = q[:3] in t and q[-4:] in t
    return anchored and len(qg & tg) / max(1, len(qg)) >= 0.72


def canonical_new(row):
    locator = row.get("locator") or {}
    roles = row.get("role") or []
    return {
        "element_id": row["element_id"],
        "contract": [row.get("contract_key", "")],
        "subject": row.get("subject_key") or [],
        "role": roles + [ROLE_ALIASES.get(role, role) for role in roles],
        "article": [locator.get("article", ""), locator.get("article_title", ""), locator.get("section", "")],
        "table": (locator.get("table_headers") or []) + (locator.get("row_keys") or []) + (locator.get("formula_context") or []),
        "qualifier": row.get("qualifier") or [],
        "reference": row.get("reference") or [],
        "schema": [row.get("schema_tag", "")],
    }


def canonical_old(row):
    return {
        "element_id": row["element_id"],
        "contract": [row.get("contract_scope", ""), row.get("topic", "")],
        "subject": (row.get("aliases") or []) + [row.get("table_title", ""), row.get("formula_subject", "")],
        "role": [row.get("semantic_role", "")],
        "article": [row.get("article", "")],
        "table": row.get("table_headers") or [],
        "qualifier": (row.get("values") or []) + (row.get("conditions") or []),
        "reference": [],
        "schema": [row.get("element_type", "")],
    }


def canonical_llm(row):
    return {
        "element_id": row["element_id"],
        "contract": row.get("contract") or [],
        "subject": row.get("subject") or [],
        "role": row.get("role") or [],
        "article": row.get("article") or [],
        "table": row.get("table") or [],
        "qualifier": row.get("qualifier") or [],
        "reference": row.get("reference") or [],
        "schema": row.get("schema") or [],
    }


class SlotSearchUnavailable(RuntimeError):
    """Raised with a clear message when a variant's tag file is missing on disk."""


class SlotFileSearch:
    _VARIANTS = {
        "new": ("element_tags_new_repaired_full_v1.jsonl", canonical_new),
        "old": ("element_tags_old_repaired_full_v1.jsonl", canonical_old),
        "llm": ("element_tags_llm_v1.jsonl", canonical_llm),
        "v2": ("element_tags_v2.jsonl", canonical_llm),
    }

    def __init__(self, variant="v2"):
        source, adapter = self._VARIANTS.get(variant, self._VARIANTS["v2"])
        tag_path = OUT / source
        if not tag_path.exists():
            raise SlotSearchUnavailable(
                f"slot_filesearch: tag file missing for variant={variant!r}: {tag_path}"
            )
        elements_path = OUT / "elements_psection.jsonl"
        if not elements_path.exists():
            raise SlotSearchUnavailable(
                f"slot_filesearch: elements file missing: {elements_path}"
            )
        self.rows = [adapter(row) for row in load(source) if row.get("ok", True)]
        self.elements = {row["element_id"]: row for row in load("elements_psection.jsonl")}

    @staticmethod
    def _queries(value):
        return [str(item) for item in (value if isinstance(value, list) else [value]) if str(item).strip()]

    def field_match(self, row, field, requested):
        queries = self._queries(requested)
        values = [str(value) for value in row.get(field) or [] if str(value).strip()]
        # A list inside one field is OR; independent fields are AND.
        return any(fuzzy_contains(query, value, field) for query in queries for value in values)

    def search(self, filters, raw_regex="", limit=20):
        filters = {key: value for key, value in filters.items() if key in FIELDS and self._queries(value)}
        regex = None
        if raw_regex:
            try:
                regex = re.compile(raw_regex, re.I)
            except re.error:
                regex = re.compile(re.escape(raw_regex), re.I)
        candidates = []
        separate_counts = collections.Counter()
        for row in self.rows:
            individual = {field: self.field_match(row, field, requested) for field, requested in filters.items()}
            for field, matched in individual.items():
                separate_counts[field] += int(matched)
            if filters and not all(individual.values()):
                continue
            raw_match = not regex or bool(regex.search(self.elements[row["element_id"]]["text"]))
            if not raw_match:
                continue
            candidates.append(row)

        result_rows = []
        for row in candidates[:limit]:
            eid = row["element_id"]
            result_rows.append({
                "element_id": eid,
                "matched_slots": {field: filters[field] for field in filters},
                "address": {field: row[field] for field in FIELDS if row[field]},
                "preview": re.sub(r"\s+", " ", self.elements[eid]["text"])[:240],
            })
        suggestions = {}
        if len(candidates) > limit:
            for field in FIELDS:
                if field in filters:
                    continue
                values = collections.Counter(value for row in candidates for value in row[field] if value)
                suggestions[field] = [{"value": value, "count": count} for value, count in values.most_common(8)]
        return {
            "filters": filters, "raw_regex": raw_regex, "total_matches": len(candidates),
            "truncated": len(candidates) > limit, "results": result_rows,
            "single_slot_match_counts": dict(separate_counts),
            "refine_with": suggestions,
            "semantics": "AND across fields; OR within a field; document order; no relevance score",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(SlotFileSearch._VARIANTS), default="v2")
    parser.add_argument("--filters", default="{}")
    parser.add_argument("--raw-regex", default="")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    try:
        engine = SlotFileSearch(args.variant)
    except SlotSearchUnavailable as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return
    print(json.dumps(engine.search(json.loads(args.filters), args.raw_regex, args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
