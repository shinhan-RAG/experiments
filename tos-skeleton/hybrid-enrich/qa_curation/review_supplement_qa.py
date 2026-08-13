#!/usr/bin/env python3
"""Strict audit of the 204 generated supplement drafts."""

import collections
import csv
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")

FILES = (
    OUT / "qa_type_coverage_supplement_draft.jsonl",
    OUT / "qa_complexity_difficulty_supplement_draft.jsonl",
)

TITLE_EXPECTATION = {
    "definition": r"정의|용어",
    "payment_trigger": r"지급사유|지급기준",
    "amount_rate": r"지급기준|부표|지급금액",
    "limit_frequency": r"지급사유|세부규정|지급기준",
    "timing_period": r"보장개시|보험기간|세부규정|지급기준",
    "exclusion_exception": r"지급하지|면책|제외|특칙",
    "renewal_termination": r"갱신|소멸|효력",
    "contract_refund": r"해지|환급|철회|무효|취소|부활|계약내용",
    "claim_procedure": r"청구|지급절차|서류",
    "case_calculation": r"세부규정|계산|산정|지급기준",
    "document_audit": r"해석|개정|정정|특칙|우선|적용",
    "product_composition": r"특약 구성",
}

NOISE = re.compile(
    r"SHINHAN LIFE|The main |To extract the text|The following is|white bear|"
    r"\(\d{2,4},\d{2,4}\),\(\d{2,4},\d{2,4}\)|<!-- pages|약 관 목 차|"
    r"\$\$|\\frac|---\s*(?:\n|$)",
    re.I,
)

COHERENT_PAIRS = {
    frozenset(("timing_period", "payment_trigger")),
    frozenset(("payment_trigger", "amount_rate")),
    frozenset(("payment_trigger", "limit_frequency")),
    frozenset(("payment_trigger", "exclusion_exception")),
    frozenset(("amount_rate", "limit_frequency")),
    frozenset(("amount_rate", "timing_period")),
    frozenset(("renewal_termination", "contract_refund")),
    frozenset(("definition", "payment_trigger")),
    frozenset(("claim_procedure", "timing_period")),
    frozenset(("case_calculation", "amount_rate")),
    frozenset(("case_calculation", "limit_frequency")),
    frozenset(("timing_period", "exclusion_exception")),
    frozenset(("definition", "timing_period")),
    frozenset(("definition", "exclusion_exception")),
}


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def review(row, raw):
    reasons = []
    decision = "accept"

    # Mechanical integrity.
    if not row.get("question") or not row.get("answer") or not row.get("sources"):
        return "reject", ["필수 필드 누락"]
    for source in row["sources"]:
        start, end = source.get("char_start"), source.get("char_end")
        if start is None or end is None or raw[start:end] != source.get("quote"):
            return "reject", ["원문 quote/offset 불일치"]
    if NOISE.search(row["answer"]):
        decision = "revise"
        reasons.append("답변 근거에 페이지·수식·OCR 노이즈 포함")

    complexity = row.get("evidence_complexity", "single")
    if complexity == "single":
        primary = row.get("primary_type")
        title = row.get("article_title", "")
        expected = TITLE_EXPECTATION.get(primary)
        if expected and not re.search(expected, title, re.I):
            return "reject", [f"질문 유형({primary})과 조항 제목 불일치"]
        if primary == "definition" and "해당 용어" in row["question"] and not re.search(r"정의|용어", title):
            return "reject", ["정의 대상이 특정되지 않음"]
        if len(row["answer"]) > 650:
            decision = "revise"
            reasons.append("답변이 길어 핵심 문장 축약 필요")
        if re.search(r"에서 해당 용어를", row["question"]):
            decision = "revise" if decision == "accept" else decision
            reasons.append("질문에서 정의 대상 명시 필요")

    elif complexity in ("adjacent_multi", "distributed_multi"):
        if len(row["sources"]) != 2:
            return "reject", ["다중 근거 라벨과 source 수 불일치"]
        a, b = row["sources"]
        if not (a["char_end"] <= b["char_start"] or b["char_end"] <= a["char_start"]):
            return "reject", ["두 근거가 중복됨"]
        roles = frozenset(row.get("question_types", []))
        if roles not in COHERENT_PAIRS:
            return "reject", ["두 근거 역할의 결합이 하나의 자연스러운 질문을 구성하지 않음"]
        difficulty = row.get("expression_difficulty")
        if difficulty == "adversarial" and roles not in {
            frozenset(("payment_trigger", "exclusion_exception")),
            frozenset(("amount_rate", "limit_frequency")),
            frozenset(("timing_period", "exclusion_exception")),
        }:
            return "reject", ["adversarial 전제와 근거 조합 불일치"]
        if NOISE.search(row["answer"]) or len(row["answer"]) > 1000:
            decision = "revise"
            reasons.append("다중 근거 답변 정리·축약 필요")
        if re.search(r"(?:기간|조건)이 어떻게 달라지나요", row["question"]) and roles not in {
            frozenset(("amount_rate", "timing_period")),
            frozenset(("payment_trigger", "exclusion_exception")),
            frozenset(("timing_period", "exclusion_exception")),
        }:
            decision = "revise"
            reasons.append("질문 표현이 두 근거 관계를 정확히 설명하지 않음")

    elif complexity == "global_aggregation":
        if len(row["sources"]) < 2:
            return "reject", ["전역 집계인데 근거가 2개 미만"]
        # Exact-title collection is useful, but OCR variants and completeness must be reviewed manually.
        decision = "revise"
        reasons.append("특약명 OCR 변형 통합 및 전체 목록 완전성 수동 확인 필요")
        if len(set(source["quote"] for source in row["sources"])) != len(row["sources"]):
            reasons.append("중복 특약 근거 제거 필요")
    else:
        return "reject", ["알 수 없는 근거 복잡도"]

    if not reasons:
        reasons.append("기계·의미 검수 통과")
    return decision, reasons


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    rows = []
    for path in FILES:
        for row in load_jsonl(path):
            row["draft_file"] = path.name
            rows.append(row)

    reviewed = []
    for row in rows:
        decision, reasons = review(row, raw)
        reviewed.append(row | {"review_decision": decision, "review_reasons": reasons})

    paths = {
        "all": OUT / "qa_supplement_reviewed.jsonl",
        "accept": OUT / "qa_supplement_accepted.jsonl",
        "revise": OUT / "qa_supplement_needs_revision.jsonl",
        "reject": OUT / "qa_supplement_rejected.jsonl",
    }
    for label, path in paths.items():
        selected = reviewed if label == "all" else [row for row in reviewed if row["review_decision"] == label]
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    original = [
        row for row in load_jsonl(OUT / "qa432_coverage_balanced_draft.jsonl")
        if row.get("source") == "field_qa359_confirmed"
    ]
    accepted = [
        row | {"review_status": "screening_accepted"}
        for row in reviewed if row["review_decision"] == "accept"
    ]
    with (OUT / f"qa{len(original) + len(accepted)}_screened.jsonl").open("w", encoding="utf-8") as handle:
        for row in original + accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (OUT / "qa_supplement_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "qid", "draft_file", "review_decision", "review_reasons", "primary_type",
            "evidence_complexity", "expression_difficulty", "question", "answer_preview",
        ])
        writer.writeheader()
        for row in reviewed:
            writer.writerow({
                "qid": row["qid"], "draft_file": row["draft_file"],
                "review_decision": row["review_decision"],
                "review_reasons": " | ".join(row["review_reasons"]),
                "primary_type": row.get("primary_type"),
                "evidence_complexity": row.get("evidence_complexity"),
                "expression_difficulty": row.get("expression_difficulty", "exact"),
                "question": row["question"],
                "answer_preview": re.sub(r"\s+", " ", row["answer"])[:300],
            })

    summary = {
        "total": len(reviewed),
        "decision": dict(collections.Counter(row["review_decision"] for row in reviewed)),
        "by_draft": {
            name: dict(collections.Counter(row["review_decision"] for row in reviewed if row["draft_file"] == name))
            for name in sorted(set(row["draft_file"] for row in reviewed))
        },
        "accepted_by_complexity": dict(collections.Counter(
            row["evidence_complexity"] for row in reviewed if row["review_decision"] == "accept"
        )),
        "note": "Strict screening review. revise items require rewritten question/answer and manual evidence-completeness confirmation before acceptance.",
    }
    (OUT / "qa_supplement_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    detail = [
        "# QA Supplement 비통과 항목별 사유",
        "",
        "`revise`는 질문 또는 답변을 고치면 재사용 가능한 항목이다. `reject`는 현재 질문–근거 조합을 폐기하고 근거부터 다시 선택해야 하는 항목이다. 질문 유형 자체가 불가능하다는 뜻은 아니다.",
        "",
    ]
    for decision, title in (("revise", "수정 필요 54건"), ("reject", "현재 조합 제외 109건")):
        selected = [row for row in reviewed if row["review_decision"] == decision]
        detail.extend([f"## {title}", ""])
        for index, row in enumerate(selected, 1):
            detail.extend([
                f"### {index}. {row['qid']}",
                "",
                f"- 질문: {row['question'].replace(chr(10), ' ')}",
                f"- 유형: {', '.join(row.get('question_types', []))}",
                f"- 근거 복잡도: {row.get('evidence_complexity', 'single')}",
                f"- 표현 난이도: {row.get('expression_difficulty', 'exact')}",
                f"- 판정: `{decision}`",
                f"- 사유: {'; '.join(row['review_reasons'])}",
                f"- 근거 조항: {', '.join(row.get('article_titles', [row.get('article_title', '')]))}",
                "",
            ])
    (ROOT / "QA_SUPPLEMENT_NONPASS_DETAILS.md").write_text("\n".join(detail), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
