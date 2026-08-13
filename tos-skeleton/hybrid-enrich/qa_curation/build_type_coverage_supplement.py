#!/usr/bin/env python3
"""Build evidence-first draft QA candidates for underrepresented question types."""

import bisect
import json
import re
import unicodedata
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from eda_question_type_coverage import DOC_PATH, OUT, QA_PATH, DOC_RULES, TYPE_LABELS, classify, document_blocks


TARGETS = {
    "definition": 30,
    "payment_trigger": 30,
    "amount_rate": 30,
    "limit_frequency": 30,
    "timing_period": 30,
    "exclusion_exception": 30,
    "renewal_termination": 30,
    "contract_refund": 30,
    "claim_procedure": 30,
    "case_calculation": 30,
    "document_audit": 20,
    "product_composition": 20,
}


TEMPLATES = {
    "definition": "{scope} 약관의 {title}에서 해당 용어를 어떻게 정의하나요?",
    "payment_trigger": "{scope}의 {title}에서 정한 보험금 지급사유는 무엇인가요?",
    "amount_rate": "{scope}의 {title}에서 지급금액 또는 지급비율은 어떻게 정해지나요?",
    "limit_frequency": "{scope}의 {title}에서 지급 횟수 또는 보장 한도는 어떻게 정해지나요?",
    "timing_period": "{scope}의 {title}에서 적용되는 시점 또는 기간은 어떻게 되나요?",
    "exclusion_exception": "{scope}의 {title}에서 보장 제외사유나 예외는 무엇인가요?",
    "renewal_termination": "{scope}의 {title}에서 갱신·유지 또는 소멸 조건은 어떻게 정해지나요?",
    "contract_refund": "{scope}의 {title}에서 계약의 해지·철회·환급 관련 기준은 무엇인가요?",
    "claim_procedure": "{scope}의 {title}에서 보험금 청구에 필요한 절차나 서류는 무엇인가요?",
    "case_calculation": "{scope}의 {title} 규정을 적용할 때 지급액이나 지급률은 어떻게 산정하나요?",
    "document_audit": "{scope}의 {title} 조항에 적용 기준과 예외가 함께 명시되어 있는지 확인해 주세요.",
}


EVIDENCE_PATTERNS = {
    "definition": r"(?:이라|라) 함은|정의",
    "payment_trigger": r"지급사유|지급합니다|지급할 때",
    "amount_rate": r"지급금액|보험가입금액의\s*\d|\d+(?:\.\d+)?%|\d+(?:,\d{3})+원",
    "limit_frequency": r"최초\s*1회|\d+회|한도|최대",
    "timing_period": r"보장개시일|보험기간|납입기간|\d+일|\d+년|기간 중",
    "exclusion_exception": r"지급하지|제외|면책|다만",
    "renewal_termination": r"갱신|재가입|소멸|효력이 없습니다|종료",
    "contract_refund": r"해약환급금|해지|철회|무효|취소|부활|계약내용의 변경",
    "claim_procedure": r"청구|서류|지급절차|제출|신청",
    "case_calculation": r"계산|합산|더하여|산식|지급률|경과기간",
    "document_audit": r"다만|제외|우선하여 적용|서로 다른|개정|정정",
}


def compact_scope(scope):
    scope = re.sub(r"\(무배당[^)]*\)", "", scope).strip()
    return scope or "해당 계약"


def clean_title(title):
    title = re.sub(r"\s+", " ", title).strip(" #-[]()")
    return title[:90] or "해당 조항"


def evidence_window(text, pattern, max_chars=700):
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    start = max(0, match.start() - 180)
    end = min(len(text), match.end() + 420)
    # Expand to nearby line boundaries while keeping the evidence compact.
    left = text.rfind("\n", max(0, start - 150), start)
    right = text.find("\n", end, min(len(text), end + 150))
    start = left + 1 if left >= 0 else start
    end = right if right >= 0 else end
    evidence = text[start:end].strip()
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars].rstrip()
    return evidence


def clean_evidence(evidence, kind):
    noise = (
        "약 관 목 차", "<!-- pages", "SHINHAN LIFE", "The main ",
        "To extract the text", "The following is", "white bear",
    )
    if any(token in evidence for token in noise):
        return False
    if re.search(r"\(\d{2,4},\d{2,4}\),\(\d{2,4},\d{2,4}\)", evidence):
        return False
    if kind == "definition" and not re.search(r"정의|(?:이라|라) 함은|의미합니다|말합니다", evidence):
        return False
    if len(re.sub(r"\s+", "", evidence)) < 60:
        return False
    return True


def main():
    raw = unicodedata.normalize("NFC", DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n"))
    lines = raw.splitlines(keepends=True)
    line_offsets, pos = [], 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line)

    blocks = document_blocks(raw)
    elements = [json.loads(line) for line in (OUT / "elements.jsonl").open(encoding="utf-8")]
    scoped = [(element["line_start"], element["contract_scope"]) for element in elements if element.get("contract_scope")]
    scope_lines = [line for line, _ in scoped]

    def scope_at(line_no):
        idx = bisect.bisect_right(scope_lines, line_no) - 1
        return compact_scope(scoped[idx][1]) if idx >= 0 else "주계약"

    existing = [json.loads(line) for line in (OUT / "qa359_question_types.jsonl").open(encoding="utf-8")]
    current = {kind: sum(kind in row["question_types"] for row in existing) for kind in TARGETS}
    needed = {kind: max(0, target - current[kind]) for kind, target in TARGETS.items()}

    candidates = []
    used = set()
    for kind, need in needed.items():
        if kind == "product_composition" or need <= 0:
            continue
        pattern = EVIDENCE_PATTERNS[kind]
        eligible = []
        for block in blocks:
            if block["line_start"] < 3500:
                continue
            if kind not in classify(block["title"] + "\n" + raw[line_offsets[block["line_start"] - 1]:line_offsets[min(block["line_end"] - 1, len(line_offsets) - 1)]], DOC_RULES):
                continue
            block_start = line_offsets[block["line_start"] - 1]
            block_end = line_offsets[block["line_end"] - 1] if block["line_end"] - 1 < len(line_offsets) else len(raw)
            block_text = raw[block_start:block_end]
            evidence = evidence_window(block_text, pattern)
            if not evidence or not clean_evidence(evidence, kind):
                continue
            scope = scope_at(block["line_start"])
            key = (kind, scope, clean_title(block["title"]))
            if key in used:
                continue
            eligible.append((scope, block, block_start, block_text, evidence, key))
        # Spread selection across the document rather than taking only the first riders.
        if eligible:
            step = len(eligible) / max(1, need)
            picks = [eligible[min(len(eligible) - 1, int(i * step))] for i in range(min(need, len(eligible)))]
        else:
            picks = []
        for scope, block, block_start, block_text, evidence, key in picks:
            used.add(key)
            local = block_text.find(evidence)
            char_start = block_start + local
            char_end = char_start + len(evidence)
            line_start = raw.count("\n", 0, char_start) + 1
            line_end = raw.count("\n", 0, char_end) + 1
            question = TEMPLATES[kind].format(scope=scope, title=clean_title(block["title"]))
            candidates.append({
                "qid": f"supp-{kind}-{len([c for c in candidates if c['primary_type'] == kind]) + 1:03d}",
                "question": question,
                "answer": evidence,
                "question_types": [kind],
                "primary_type": kind,
                "evidence_complexity": "single",
                "sources": [{"quote": evidence, "char_start": char_start, "char_end": char_end,
                             "line_start": line_start, "line_end": line_end}],
                "contract_scope": scope,
                "article_title": clean_title(block["title"]),
                "source": "coverage_supplement",
                "review_status": "draft_needs_human_review",
            })

    # Product composition candidates use exact rider-title evidence.
    need = needed["product_composition"]
    scopes = ["신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)"]
    for line_no, scope in scoped:
        scope = compact_scope(scope)
        if scope not in scopes:
            scopes.append(scope)
    if scopes and need:
        step = len(scopes) / need
        for i in range(min(need, len(scopes))):
            scope = scopes[min(len(scopes) - 1, int(i * step))]
            char_start = raw.find(scope)
            if char_start < 0:
                continue
            else:
                evidence = raw[char_start:char_start + len(scope)]
                char_end = char_start + len(evidence)
                line_start = raw.count("\n", 0, char_start) + 1
                line_end = line_start
            candidates.append({
                "qid": f"supp-product_composition-{i + 1:03d}",
                "question": f"이 판매약관의 특약 구성에 {scope}가 포함되어 있나요?",
                "answer": f"네. 판매약관에 {scope}가 포함되어 있습니다.",
                "question_types": ["product_composition", "coverage_scope"],
                "primary_type": "product_composition",
                "evidence_complexity": "single",
                "sources": [{"quote": evidence, "char_start": char_start, "char_end": char_end,
                             "line_start": line_start, "line_end": line_end}],
                "contract_scope": scope,
                "article_title": "특약 구성",
                "source": "coverage_supplement",
                "review_status": "draft_needs_human_review",
            })

    target = OUT / "qa_type_coverage_supplement_draft.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    original_by_qid = {
        str(row["no"]): row
        for row in (json.loads(line) for line in QA_PATH.open(encoding="utf-8") if line.strip())
        if row.get("신뢰도") == "확정"
    }
    typed_by_qid = {row["qid"]: row for row in existing}
    combined = []
    for qid, original in original_by_qid.items():
        combined.append({
            "qid": qid,
            "question": original["질문"],
            "answer": original["정답"],
            "question_types": typed_by_qid[qid]["question_types"],
            "business_type": original["사업구분"],
            "sources": original.get("출처", []),
            "source": "field_qa359_confirmed",
            "review_status": "field_confirmed",
        })
    combined.extend(candidates)
    with (OUT / "qa432_coverage_balanced_draft.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    produced = {kind: sum(c["primary_type"] == kind for c in candidates) for kind in TARGETS}
    manifest = {
        "status": "draft_needs_human_review",
        "existing_confirmed": len(existing),
        "targets": TARGETS,
        "current_before_supplement": current,
        "needed": needed,
        "produced": produced,
        "supplement_total": len(candidates),
        "combined_total_if_approved": len(existing) + len(candidates),
        "document_sha256_prefix": __import__("hashlib").sha256(raw.encode()).hexdigest()[:16],
    }
    (OUT / "qa_type_coverage_supplement_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
