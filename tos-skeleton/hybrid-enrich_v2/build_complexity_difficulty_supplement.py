#!/usr/bin/env python3
"""Generate a separate evidence-first supplement for complexity and expression difficulty."""

import bisect
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from eda_question_type_coverage import DOC_PATH, OUT, DOC_RULES, classify, document_blocks
from build_type_coverage_supplement import EVIDENCE_PATTERNS, clean_evidence, compact_scope, evidence_window


COMPLEXITY_TARGET = {
    "adjacent_multi": 25,
    "distributed_multi": 35,
    "global_aggregation": 15,
}
DIFFICULTY_SEQUENCE = (
    ["paraphrase"] * 30
    + ["alias"] * 15
    + ["implicit"] * 18
    + ["adversarial"] * 12
)

ROLE_LABEL = {
    "definition": "정의",
    "payment_trigger": "보험금 지급조건",
    "amount_rate": "지급금액",
    "limit_frequency": "횟수·한도",
    "timing_period": "적용 시점·기간",
    "exclusion_exception": "보장 제외·예외",
    "renewal_termination": "갱신·소멸",
    "contract_refund": "계약·환급",
    "claim_procedure": "청구절차",
    "case_calculation": "산정기준",
}
ROLES = tuple(ROLE_LABEL)

TITLE_REQUIREMENTS = {
    "definition": r"정의|용어",
    "payment_trigger": r"지급사유",
    "amount_rate": r"지급기준|부표|지급금액",
    "limit_frequency": r"세부규정|지급사유|지급기준",
    "timing_period": r"보장개시|보험기간|세부규정",
    "exclusion_exception": r"지급하지|특칙|제외|면책",
    "renewal_termination": r"갱신|소멸|효력",
    "contract_refund": r"해지|환급|철회|무효|취소|부활|계약내용",
    "claim_procedure": r"청구|지급절차|서류",
    "case_calculation": r"세부규정|계산|산정",
}


def short_title(title):
    return re.sub(r"\s+", " ", title).strip(" #-[]()")[:70]


def alias_phrase(role):
    return {
        "premium_waiver": "납면",
        "contract_refund": "해약금",
        "claim_procedure": "보험금 서류",
        "payment_trigger": "보험금 나오는 조건",
        "amount_rate": "받는 돈",
        "limit_frequency": "몇 번까지",
        "timing_period": "언제부터",
        "exclusion_exception": "돈이 안 나오는 경우",
        "renewal_termination": "다시 갱신되는지",
        "definition": "무슨 뜻인지",
        "case_calculation": "얼마로 계산되는지",
    }.get(role, ROLE_LABEL.get(role, role))


def pair_question(scope, a, b, difficulty):
    la, lb = ROLE_LABEL[a["role"]], ROLE_LABEL[b["role"]]
    if difficulty == "alias":
        return f"{scope}에서 {a['title']}의 {alias_phrase(a['role'])}와 {b['title']}의 {alias_phrase(b['role'])}를 같이 알려줘."
    if difficulty == "implicit":
        return f"{scope}에서 {a['title']}의 {la}와 {b['title']}의 {lb}를 함께 보면, 실제로 받을 수 있는 조건이 어떻게 달라지나요?"
    if difficulty == "adversarial":
        return f"{scope}은 {a['title']}만 충족하면 무조건 보장되는 게 맞나요? {b['title']}까지 함께 확인해 주세요."
    return f"{scope}에서 {a['title']}에 따른 {la}, {b['title']}에 따른 {lb}를 함께 설명해 주세요."


def main():
    raw = unicodedata.normalize("NFC", DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n"))
    lines = raw.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    elements = [json.loads(line) for line in (OUT / "elements.jsonl").open(encoding="utf-8")]
    scope_points = [(e["line_start"], compact_scope(e["contract_scope"])) for e in elements if e.get("contract_scope")]
    scope_lines = [x[0] for x in scope_points]

    def scope_at(line_no):
        idx = bisect.bisect_right(scope_lines, line_no) - 1
        return scope_points[idx][1] if idx >= 0 else "주계약"

    atoms = []
    seen = set()
    for block in document_blocks(raw):
        if block["line_start"] < 3500:
            continue
        start = offsets[block["line_start"] - 1]
        end = offsets[block["line_end"] - 1] if block["line_end"] - 1 < len(offsets) else len(raw)
        text = raw[start:end]
        roles = [role for role in classify(block["title"] + "\n" + text, DOC_RULES) if role in ROLES]
        for role in roles:
            if not re.search(TITLE_REQUIREMENTS[role], block["title"], re.I):
                continue
            evidence = evidence_window(text, EVIDENCE_PATTERNS[role], max_chars=520)
            if not evidence or not clean_evidence(evidence, role):
                continue
            local = text.find(evidence)
            char_start = start + local
            char_end = char_start + len(evidence)
            scope = scope_at(block["line_start"])
            key = (scope, role, short_title(block["title"]))
            if key in seen:
                continue
            seen.add(key)
            atoms.append({
                "scope": scope, "role": role, "title": short_title(block["title"]),
                "line": block["line_start"], "quote": evidence,
                "char_start": char_start, "char_end": char_end,
                "line_start": raw.count("\n", 0, char_start) + 1,
                "line_end": raw.count("\n", 0, char_end) + 1,
            })

    by_scope = collections.defaultdict(list)
    for atom in atoms:
        by_scope[atom["scope"]].append(atom)
    for values in by_scope.values():
        values.sort(key=lambda item: item["line"])

    candidates = []
    used_pairs = set()

    def add_pair(complexity, a, b):
        if a["char_start"] == b["char_start"] or not (
            a["char_end"] <= b["char_start"] or b["char_end"] <= a["char_start"]
        ):
            return False
        key = (complexity, a["scope"], a["role"], b["role"], a["line"], b["line"])
        if key in used_pairs:
            return False
        used_pairs.add(key)
        difficulty = DIFFICULTY_SEQUENCE[len(candidates) % len(DIFFICULTY_SEQUENCE)]
        candidates.append({
            "qid": f"supp-complex-{len(candidates) + 1:03d}",
            "question": pair_question(a["scope"], a, b, difficulty),
            "answer": a["quote"] + "\n\n" + b["quote"],
            "question_types": list(dict.fromkeys([a["role"], b["role"]])),
            "primary_type": a["role"],
            "evidence_complexity": complexity,
            "expression_difficulty": difficulty,
            "reasoning_steps": 2,
            "sources": [
                {k: a[k] for k in ("quote", "char_start", "char_end", "line_start", "line_end")},
                {k: b[k] for k in ("quote", "char_start", "char_end", "line_start", "line_end")},
            ],
            "contract_scope": a["scope"],
            "article_titles": [a["title"], b["title"]],
            "source": "complexity_difficulty_supplement",
            "review_status": "draft_needs_human_review",
        })
        return True

    # Adjacent multi: distinct evidence roles in the same rider within 180 source lines.
    for scope, values in sorted(by_scope.items()):
        if sum(c["evidence_complexity"] == "adjacent_multi" for c in candidates) >= COMPLEXITY_TARGET["adjacent_multi"]:
            break
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if b["role"] == a["role"] or b["line"] - a["line"] > 180:
                    continue
                if add_pair("adjacent_multi", a, b):
                    break
            if sum(c["evidence_complexity"] == "adjacent_multi" for c in candidates) >= COMPLEXITY_TARGET["adjacent_multi"]:
                break

    # Distributed multi: distinct roles in the same rider, separated by at least 300 lines.
    for scope, values in sorted(by_scope.items()):
        if sum(c["evidence_complexity"] == "distributed_multi" for c in candidates) >= COMPLEXITY_TARGET["distributed_multi"]:
            break
        for i, a in enumerate(values):
            for b in reversed(values[i + 1:]):
                if b["role"] == a["role"] or b["line"] - a["line"] < 300:
                    continue
                if add_pair("distributed_multi", a, b):
                    break
            if sum(c["evidence_complexity"] == "distributed_multi" for c in candidates) >= COMPLEXITY_TARGET["distributed_multi"]:
                break

    # Global aggregation: enumerate all rider titles containing a controlled keyword.
    scope_keywords = [
        "암진단", "입원", "수술", "납입면제", "항암", "뇌혈관", "허혈심장",
        "간병인", "치매", "재해장해", "질병장해", "검사", "통원", "골절", "치료비",
        "종합병원", "약물치료", "방사선치료", "진단특약", "순환계질환",
    ]
    unique_scopes = list(dict.fromkeys(scope for _, scope in scope_points))
    for keyword in scope_keywords:
        matched = [scope for scope in unique_scopes if keyword in scope]
        if len(matched) < 2:
            continue
        sources = []
        exact_scopes = []
        for scope in matched:
            char_start = raw.find(scope)
            if char_start < 0:
                continue
            quote = raw[char_start:char_start + len(scope)]
            sources.append({
                "quote": quote, "char_start": char_start, "char_end": char_start + len(quote),
                "line_start": raw.count("\n", 0, char_start) + 1,
                "line_end": raw.count("\n", 0, char_start) + 1,
            })
            exact_scopes.append(scope)
        if len(sources) < 2:
            continue
        difficulty = DIFFICULTY_SEQUENCE[len(candidates) % len(DIFFICULTY_SEQUENCE)]
        if difficulty == "alias":
            question = f"이 약관에서 이름에 '{keyword}'가 들어간 담보들을 전부 뽑아줘."
        elif difficulty == "implicit":
            question = f"이 상품에서 '{keyword}' 관련 선택 항목이 무엇인지 빠짐없이 확인해 주세요."
        elif difficulty == "adversarial":
            question = f"'{keyword}' 관련 특약은 하나뿐인 것으로 아는데 맞나요? 약관 전체에서 확인해 주세요."
        else:
            question = f"이 판매약관에서 명칭에 '{keyword}'가 포함된 특약을 모두 알려주세요."
        candidates.append({
            "qid": f"supp-complex-{len(candidates) + 1:03d}",
            "question": question,
            "answer": "\n".join(f"- {scope}" for scope in exact_scopes),
            "question_types": ["enumeration_summary", "product_composition"],
            "primary_type": "enumeration_summary",
            "evidence_complexity": "global_aggregation",
            "expression_difficulty": difficulty,
            "reasoning_steps": len(sources),
            "sources": sources,
            "contract_scope": "document_global",
            "article_titles": ["특약 구성"],
            "source": "complexity_difficulty_supplement",
            "review_status": "draft_needs_human_review",
        })
        if sum(c["evidence_complexity"] == "global_aggregation" for c in candidates) >= COMPLEXITY_TARGET["global_aggregation"]:
            break

    target = OUT / "qa_complexity_difficulty_supplement_draft.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    prior = [json.loads(line) for line in (OUT / "qa432_coverage_balanced_draft.jsonl").open(encoding="utf-8")]
    combined = prior + candidates
    with (OUT / f"qa{len(combined)}_expanded_draft.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "status": "draft_needs_human_review",
        "preserved_type_supplement": 129,
        "new_supplement": len(candidates),
        "combined_total": len(combined),
        "complexity": dict(collections.Counter(c["evidence_complexity"] for c in candidates)),
        "difficulty": dict(collections.Counter(c["expression_difficulty"] for c in candidates)),
        "targets": COMPLEXITY_TARGET,
        "document_sha256_prefix": hashlib.sha256(raw.encode()).hexdigest()[:16],
    }
    (OUT / "qa_complexity_difficulty_supplement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
