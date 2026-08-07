#!/usr/bin/env python3
"""Build a user-natural, document-grounded coverage supplement.

The old supplements are used only as evidence locators. Their generated questions
and review decisions are not reused. Every new question below is explicitly
authored before taxonomy labels are attached.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
OLD_REVIEWED = OUT / "qa_supplement_reviewed.jsonl"
FIELD_QA = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl")
TARGET = OUT / "qa_document_coverage_v2.jsonl"
COMBINED = OUT / "qa333_field_plus_document_coverage_v2.jsonl"
REPORT = OUT / "qa_document_coverage_v2_report.json"
FIELD_TYPES = OUT / "qa359_question_types.jsonl"


TYPE_LABELS = {
    "definition": "정의·의미",
    "coverage_scope": "보장 여부·대상 범위",
    "payment_trigger": "지급사유·발생조건",
    "amount_rate": "금액·비율",
    "limit_frequency": "횟수·한도",
    "timing_period": "시점·기간",
    "exclusion_exception": "면책·제외·예외",
    "premium_waiver": "보험료·납입면제",
    "renewal_termination": "갱신·유지·소멸",
    "contract_refund": "계약·해지·환급·철회",
    "claim_procedure": "청구·서류·절차",
    "enumeration_summary": "목록·전체 요약·집계",
    "comparison": "비교·차이",
    "case_calculation": "사례판정·계산",
    "criteria_rule": "기준·요건·규정 확인",
    "code_classification_lookup": "분류·코드 조회",
    "document_audit": "약관 정합성·누락·오류 검증",
    "interpretation_explanation": "해석·설명·상담지원",
    "product_composition": "상품·특약 구성 확인",
}


def main_cases():
    """Cases grounded in the main contract; evidence is found by exact anchors."""
    return [
        {
            "question": "질병 코드가 개정됐다면 가입 당시 코드와 진단받은 날의 코드 중 어느 기준을 따르나요?",
            "answer": "질병의 진단이나 재해가 발생한 당시 시행 중인 한국표준질병·사인분류를 기준으로 판단합니다.",
            "anchor": "이후 한국표준질병·사인분류가 개정되는 경우에는 질병의 진단 및 재해 발생 당시에 시행되고 있는 한국표준질병·사인분류를 기준으로 판단합니다.",
            "types": ["timing_period", "criteria_rule", "code_classification_lookup"],
            "complexity": "single", "difficulty": "paraphrase",
        },
        {
            "question": "보험금을 거절당한 뒤 질병 코드 기준이 바뀌어 새 기준으로는 보장 대상이 되었습니다. 기존 거절 결과도 다시 판단하나요?",
            "answer": "아닙니다. 진단 당시의 분류 기준으로 판단하며, 이후 코드가 변경되어도 기존 지급 여부를 다시 판단하지 않습니다.",
            "anchor": "이후 한국표준질병·사인분류 개정으로 분류코드가 변경되더라도 이 약관(특약 포함)에서 보장하는 분류코드의 해당 여부를 다시 판단하지 않습니다.",
            "types": ["coverage_scope", "document_audit", "case_calculation"],
            "complexity": "adjacent_multi", "difficulty": "implicit",
        },
        {
            "question": "이 상품의 주계약은 어떤 상황에서 보험금이 나오나요?",
            "answer": "보험기간 중 피보험자가 사망하면 보험수익자에게 약정한 사망보험금을 지급합니다.",
            "anchor": "회사는 보험기간 중 피보험자가 사망하였을 때에는 보험수익자에게 약정한 사망보험금(<부표1> \"보험금 지급기준표\" 참조)을 지급합니다.",
            "types": ["payment_trigger"], "complexity": "single", "difficulty": "direct",
        },
        {
            "question": "간편심사형으로 가입한 뒤 2년이 되기 전에 질병으로 사망하면 사망보험금을 전액 받나요?",
            "answer": "재해 이외의 원인으로 계약일부터 2년 미만에 사망하면 보험가입금액의 50%를 지급합니다.",
            "anchor": "보험가입금액의 100% (다만, \"간편심사형\"에 한하여 계약일부터 2년 미만에 재해 이외의 원인으로 지급사유가 발생한 경우 \"보험가입금액의 50%\"를 지급함)",
            "types": ["amount_rate", "timing_period", "comparison", "case_calculation"],
            "complexity": "distributed_multi", "difficulty": "implicit",
        },
        {
            "question": "주계약 보험료는 암이나 질병을 진단받으면 자동으로 납입이 면제되나요?",
            "answer": "주계약 자체에는 납입면제가 없습니다. 주계약 보험료를 면제할 수 있는 특약을 가입하고 그 특약의 조건을 충족해야 차회 이후 보험료가 면제됩니다.",
            "anchor": "주계약은 납입면제가 없습니다. 다만, 이 주계약의 보험료를 납입면제 할 수 있는 특약을 가입한 경우, 보험료 납입기간 중 피보험자가 해당 특약에서 정하는 기준에 따라 납입면제 사유에 해당되는 경우에는 차 회 이후 주계약의 보험료 납입을 면제합니다.",
            "types": ["premium_waiver", "coverage_scope", "criteria_rule"],
            "complexity": "single", "difficulty": "adversarial",
        },
        {
            "question": "보험금을 청구할 때 기본적으로 어떤 서류를 준비해야 하나요?",
            "answer": "회사 양식의 청구서, 사고증명서, 신분증을 제출해야 하며 필요하면 추가 서류를 요청받을 수 있습니다.",
            "anchor": "1. 청구서 (회사양식)",
            "types": ["claim_procedure", "enumeration_summary"],
            "complexity": "adjacent_multi", "difficulty": "direct",
        },
        {
            "question": "보험금을 늦게 청구해도 되나요? 청구할 수 있는 기간이 정해져 있나요?",
            "answer": "보험금 청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.",
            "anchor": "보험금 청구권, 보험료 반환청구권, 해약환급금 청구권 및 계약자적립액 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.",
            "types": ["claim_procedure", "timing_period", "contract_refund"],
            "complexity": "single", "difficulty": "paraphrase",
        },
        {
            "question": "약관 문구의 뜻이 애매하면 보험사에게 유리하게 풀어야 하나요?",
            "answer": "아닙니다. 약관의 뜻이 명백하지 않으면 계약자에게 유리하게 해석합니다.",
            "anchor": "회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다.",
            "types": ["interpretation_explanation", "document_audit", "criteria_rule"],
            "complexity": "single", "difficulty": "adversarial",
        },
        {
            "question": "설계사가 준 안내자료와 약관 내용이 다르면 어느 내용으로 계약된 것으로 보나요?",
            "answer": "회사가 제작한 보험안내자료가 약관과 다르면 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.",
            "anchor": "보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료(계약의 청약을 권유하기 위해 만든 자료 등을 말합니다) 내용이 이 약관의 내용과 다른 경우에는 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.",
            "types": ["comparison", "document_audit", "interpretation_explanation"],
            "complexity": "single", "difficulty": "implicit",
        },
        {
            "question": "청약하면 승낙이 나기 전까지는 보장을 못 받나요?",
            "answer": "청약과 함께 제1회 보험료를 받은 후 회사가 승낙하면, 제1회 보험료를 받은 날부터 보장이 시작됩니다.",
            "anchor": "회사가 청약과 함께 제1회 보험료를 받은 후 승낙한 경우에도 제1회 보험료를 받은 때부터 보장이 개시됩니다.",
            "types": ["timing_period", "coverage_scope"],
            "complexity": "single", "difficulty": "adversarial",
        },
        {
            "question": "보험료를 연체해 계약이 끊겼는데, 언제까지 다시 살릴 수 있나요?",
            "answer": "해약된 날부터 3년 이내에 회사가 정한 절차에 따라 부활을 청약할 수 있습니다. 다만 해약환급금을 받지 않은 경우여야 합니다.",
            "anchor": "계약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수 있습니다.",
            "types": ["contract_refund", "renewal_termination", "timing_period"],
            "complexity": "single", "difficulty": "paraphrase",
        },
        {
            "question": "보험금을 받을 사람과 보험사가 지급 여부에 합의하지 못하면 어떻게 하나요?",
            "answer": "보험수익자와 회사가 함께 제3자를 정해 그 의견에 따를 수 있습니다. 제3자는 종합병원 소속 전문의 중에서 정하며 의료비용은 회사가 부담합니다.",
            "anchor": "보험수익자와 회사가 보험금 지급사유에 대해 합의하지 못할 때에는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.",
            "types": ["claim_procedure", "criteria_rule", "interpretation_explanation"],
            "complexity": "single", "difficulty": "paraphrase",
        },
        {
            "question": "보험금을 한 번 받은 뒤에도 주계약이 계속 유지되나요?",
            "answer": "피보험자가 사망하거나 약관에서 정한 보험금 발생 가능성이 더 이상 없어지면 계약은 그때부터 효력이 없습니다. 단순히 보험금을 받았다는 사실만으로는 판단할 수 없습니다.",
            "anchor": "보험기간 중 피보험자가 사망하거나 이 약관에서 정하는 보험금 지급사유가 더 이상 발생할 수 없는 경우에는 이 계약은 그때부터 효력이 없습니다.",
            "types": ["renewal_termination", "coverage_scope"],
            "complexity": "single", "difficulty": "implicit",
        },
        {
            "question": "보험료를 연체했는데 별도 안내 없이 바로 계약이 해지되나요?",
            "answer": "바로 해지되지 않습니다. 회사는 14일 이상(보험기간 1년 미만이면 7일 이상)의 납입최고 기간을 정해 안내해야 합니다.",
            "anchor": "회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고(독촉)기간",
            "types": ["contract_refund", "timing_period", "exclusion_exception"],
            "complexity": "single", "difficulty": "adversarial",
        },
        {
            "question": "보험료를 자동대출로 납입하는 상태를 계속 유지할 수 있나요?",
            "answer": "자동대출납입 기간은 최초 시작일로부터 1년이 한도이며, 그 이후에는 다시 신청해야 합니다.",
            "anchor": "자동대출납입기간은 최초 자동대출납입일부터",
            "types": ["limit_frequency", "timing_period", "contract_refund"],
            "complexity": "single", "difficulty": "paraphrase",
        },
    ]


SOURCE_REWRITES = {
    "supp-definition-002": ("암진단특약에서 기타피부암이나 갑상선암도 일반암으로 보나요?", "해당 특약의 암 범위에서 기타피부암, 갑상선암, 대장점막내암, 비침습방광암 및 전암 상태는 제외됩니다.", ["definition", "coverage_scope", "exclusion_exception"], "single", "paraphrase"),
    "supp-definition-003": ("제자리암이라는 결과를 받았는데 어떤 검사를 기준으로 진단이 확정되나요?", "병리과 전문의가 조직검사, 미세바늘흡인검사 또는 혈액검사의 현미경 소견을 기초로 진단해야 하며, 결과보고 시점을 진단확정 시점으로 봅니다.", ["definition", "criteria_rule", "timing_period"], "single", "implicit"),
    "supp-definition-004": ("CT나 MRI 없이도 급성뇌경색증 진단으로 인정받을 수 있나요?", "원칙적으로 특이적 증상과 CT·MRI 등을 기초로 진단합니다. 다만 피보험자가 사망해 검사를 진단 기초로 삼을 수 없는 경우에만 기존 진단·치료 기록을 사용할 수 있습니다.", ["definition", "criteria_rule", "exclusion_exception"], "single", "adversarial"),
    "supp-definition-006": ("대상포진과 통풍 중 하나만 골라서 가입할 수 있나요?", "아닙니다. 대상포진 보장계약과 통풍 보장계약을 동시에 체결해야 하고, 두 계약의 보험가입금액도 같아야 합니다.", ["product_composition", "coverage_scope", "criteria_rule"], "single", "adversarial"),
    "supp-definition-007": ("암세포가 없는 상태에서 면역력을 높이려고 약을 맞은 것도 항암약물치료에 포함되나요?", "포함되지 않습니다. 암의 직접적인 치료를 목적으로 항암화학요법 또는 항암면역요법으로 항암약물을 투여한 경우를 말하며, 암세포가 없을 때 면역력만 높이는 약물치료는 제외됩니다.", ["definition", "coverage_scope", "exclusion_exception"], "single", "implicit"),
    "supp-payment_trigger-003": ("고액암으로 두 번 진단받으면 진단급여금도 두 번 받을 수 있나요?", "아닙니다. 고액암보장개시일 이후 고액암으로 진단확정된 경우 최초 1회에 한해 진단급여금을 지급합니다.", ["payment_trigger", "limit_frequency", "timing_period"], "single", "adversarial"),
    "supp-amount_rate-008": ("NGS 유전자패널 검사를 가입 후 1년이 되기 전에 받으면 얼마를 지급하나요?", "최초계약의 계약일부터 1년 미만에 지급 조건이 발생하면 특약 보험가입금액의 0.5%를 지급합니다. 1년 이후에는 1%입니다.", ["amount_rate", "timing_period", "comparison"], "single", "implicit"),
    "supp-amount_rate-009": ("뇌혈관질환 진단을 가입 후 1년 안에 받으면 진단급여금이 절반으로 줄어드나요?", "네. 최초계약일부터 1년 미만에 발생하면 특약 보험가입금액의 50%를 지급하고, 그 이후에는 100%를 지급합니다.", ["amount_rate", "timing_period", "comparison"], "single", "implicit"),
    "supp-limit_frequency-002": ("같은 질병으로 여러 번 입원해 간병인을 사용하면 입원할 때마다 185일씩 보장되나요?", "같은 질병·재해로 2회 이상 입원하면 원칙적으로 1회 입원으로 보고 사용일수를 합산해 최대 185일을 적용합니다. 다만 최종 사용일 다음 날부터 180일이 지난 뒤 사용한 경우는 새로운 입원으로 봅니다.", ["limit_frequency", "case_calculation", "timing_period"], "single", "adversarial"),
    "supp-renewal_termination-007": ("암주요치료비특약 가입자가 사망하면 특약과 적립금은 어떻게 되나요?", "피보험자가 보험기간 중 사망하면 특약은 효력을 잃습니다. 회사는 산출방법서에 따라 사망 당시의 계약자적립액을 계약자에게 지급합니다.", ["renewal_termination", "amount_rate"], "single", "implicit"),
    "supp-contract_refund-003": ("납입면제 조건이 이미 발생한 뒤에도 납입면제특약만 따로 해지할 수 있나요?", "원칙적으로 특약은 소멸 전에 해지할 수 있지만, 납입면제 사유가 발생한 후에는 주계약을 해지하는 경우에만 해지할 수 있습니다.", ["contract_refund", "premium_waiver", "exclusion_exception"], "single", "adversarial"),
    "supp-contract_refund-004": ("해약환급금 미지급형 인공관절치환수술특약을 보험료 납입 중에 해지하면 환급금이 있나요?", "없습니다. 해약환급금 미지급형은 보험료 납입기간 중 해지하면 해약환급금이 없습니다.", ["contract_refund", "timing_period"], "single", "direct"),
    "supp-claim_procedure-013": ("보험금을 대신 청구할 사람으로 누구를 지정할 수 있나요?", "피보험자의 가족관계등록부상 또는 주민등록상 배우자, 피보험자의 3촌 이내 친족 중에서 최대 2명까지 지정할 수 있습니다.", ["claim_procedure", "criteria_rule", "limit_frequency"], "single", "paraphrase"),
    "supp-case_calculation-008": ("같은 질병으로 1인실에 20일 입원한 뒤 180일이 지나기 전에 15일을 다시 입원했습니다. 35일 전체가 보장되나요?", "아닙니다. 같은 질병으로 180일 이내에 재입원하면 1회 입원으로 합산하고, 1회 입원당 30일 한도이므로 30일만 보장되고 5일은 제외됩니다.", ["case_calculation", "limit_frequency", "timing_period"], "adjacent_multi", "implicit"),
    "supp-document_audit-001": ("진단받은 뒤에 질병 분류체계가 바뀌면 보험금 판단도 다시 하는지 약관에 명확하게 나와 있나요?", "네. 진단·재해 발생 당시의 분류를 기준으로 판단하고, 이후 분류코드가 변경되어도 기존 보장 해당 여부를 다시 판단하지 않는다고 명시되어 있습니다.", ["document_audit", "code_classification_lookup", "timing_period"], "single", "paraphrase"),
}

# Narrow spans used only for gold target mapping. Full source quotes remain available
# for answer verification, but page artifacts and neighbouring elements must not
# become alternative gold hits.
SOURCE_TARGET_ANCHORS = {
    "supp-definition-002": "다만, 아래에 해당하는 질병은 제외합니다.",
    "supp-definition-003": "\"제자리암\"의 진단확정은 병리과 전문의 자격증을 가진 자에 의하여 내려져야 하며",
    "supp-definition-004": "\"급성뇌경색증\"의 진단확정은 의료기관의 의사(치과의사 제외)에 의하여 내려져야 하며",
    "supp-definition-006": "계약자는 각 보장계약을 동시에 체결하여야 하며, 각각의 보장계약의 보험가입금액은 동일하여야 합니다.",
    "supp-definition-007": "암세포가 없는 상태에서 면역력을 증가시키는 약물치료는 제외됩니다.",
    "supp-payment_trigger-003": "(다만, 최초 1회의 진단확정에 한함)",
    "supp-amount_rate-008": "특약보험가입금액의 1% (다만, 최초계약의 계약일부터 1년 미만에 지급사유가 발생한 경우 \"특약보험가입금액의 0.5%\"를 지급함)",
    "supp-amount_rate-009": "특약보험가입금액의 100% (다만, 최초계약의 계약일부터 1년 미만에 지급사유가 발생한 경우 \"특약보험가입금액의 50%\"를 지급함)",
    "supp-limit_frequency-002": "사용일수는 각각 1회 입원당 185일을 최고 한도로 합니다.",
    "supp-renewal_termination-007": "이 특약의 피보험자가 특약보험기간 중 사망한 경우에는 이 특약의 산출방법서에서 정하는 바에 따라",
    "supp-contract_refund-003": "보험료납입면제사유가 발생한 이후에는 주계약을 해지하는 경우에 한하여 해지할 수 있습니다.",
    "supp-contract_refund-004": "이 특약이 보험료 납입기간 중 해지될 경우 해약환급금이 없습니다.",
    "supp-claim_procedure-013": "피보험자의 3촌 이내의 친족",
    "supp-case_calculation-008": "지급일수는 1회 입원당 30일을 최고 한도로 합니다.",
    "supp-document_audit-001": "이후 한국표준질병·사인분류 개정으로 분류코드가 변경되더라도",
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n")


def clean_scope(scope: str) -> str:
    scope = scope.replace("수출", "수술")
    return re.sub(r"\(무배당[^)]*\)", "", scope).strip()


def locate_anchor(raw: str, anchor: str) -> tuple[int, int]:
    start = raw.find(anchor)
    if start < 0:
        raise ValueError(f"anchor not found: {anchor[:80]}")
    return start, start + len(anchor)


def evidence_window(raw: str, start: int, end: int, radius: int = 500) -> tuple[int, int]:
    """Keep exact offsets while including enough surrounding text to support the answer."""
    left = max(0, start - radius)
    right = min(len(raw), end + radius)
    line_left = raw.find("\n", left, start)
    if line_left >= 0:
        left = line_left + 1
    line_right = raw.rfind("\n", end, right)
    if line_right >= end:
        right = line_right
    return left, right


def line_range(raw: str, start: int, end: int) -> tuple[int, int]:
    return raw.count("\n", 0, start) + 1, raw.count("\n", 0, end) + 1


def main():
    raw = normalize(DOC.read_text(encoding="utf-8"))
    old = {row["qid"]: row for row in (json.loads(x) for x in OLD_REVIEWED.open(encoding="utf-8"))}
    chunks = [json.loads(x) for x in (OUT / "chunks.jsonl").open(encoding="utf-8")]
    elements = [json.loads(x) for x in (OUT / "elements.jsonl").open(encoding="utf-8")]
    cases = []

    for spec in main_cases():
        target_start, target_end = locate_anchor(raw, spec["anchor"])
        start, end = evidence_window(raw, target_start, target_end)
        ls, le = line_range(raw, start, end)
        cases.append({
            "question": spec["question"], "answer": spec["answer"],
            "question_types": spec["types"],
            "evidence_complexity": spec["complexity"],
            "expression_difficulty": spec["difficulty"],
            "contract_scope": "주계약(신한(간편가입)통합건강보험 원(ONE))",
            "sources": [{"quote": raw[start:end], "char_start": start, "char_end": end,
                         "line_start": ls, "line_end": le}],
            "provenance": "direct_document_curation",
            "_target_ranges": [(target_start, target_end)],
        })

    for source_qid, (question, answer, types, complexity, difficulty) in SOURCE_REWRITES.items():
        source = old[source_qid]
        sources = source["sources"]
        for evidence in sources:
            quote = evidence["quote"]
            if raw[evidence["char_start"]:evidence["char_end"]] != quote:
                raise ValueError(f"source offset mismatch: {source_qid}")
        anchor = SOURCE_TARGET_ANCHORS[source_qid]
        search_start = max(0, sources[0]["char_start"] - 2500)
        search_end = min(len(raw), sources[-1]["char_end"] + 2500)
        target_start = raw.find(anchor, search_start, search_end)
        if target_start < 0:
            raise ValueError(f"narrow target anchor not found: {source_qid}")
        target_end = target_start + len(anchor)
        cases.append({
            "question": question, "answer": answer, "question_types": types,
            "evidence_complexity": complexity, "expression_difficulty": difficulty,
            "contract_scope": clean_scope(source.get("contract_scope", "")),
            "sources": sources, "provenance": "document_evidence_recurated",
            "legacy_evidence_id": source_qid,
            "_target_ranges": [(target_start, target_end)],
        })

    banned = [r"해당 용어", r"어떻게 정의", r"지급사유는 무엇", r"제\d+(?:-\d+)?조"]
    for i, case in enumerate(cases, 1):
        case["qid"] = f"doccov-v2-{i:03d}"
        case["source"] = "document_coverage_v2"
        case["review_status"] = "curated_needs_domain_signoff"
        case["qa_sha256"] = hashlib.sha256(
            (case["question"] + "\n" + case["answer"]).encode("utf-8")
        ).hexdigest()[:16]
        if any(re.search(pattern, case["question"]) for pattern in banned):
            raise ValueError(f"unnatural/internal wording: {case['qid']} {case['question']}")
        if not case["sources"]:
            raise ValueError(f"no evidence: {case['qid']}")
        ranges = case.pop("_target_ranges")
        case["gold_chunk_ids"] = sorted({
            c["chunk_id"] for c in chunks
            if any(c["char_start"] < end and c["char_end"] > start for start, end in ranges)
        })
        case["gold_element_ids"] = sorted({
            e["element_id"] for e in elements
            if any(e["char_start"] < end and e["char_end"] > start for start, end in ranges)
        })
        if not case["gold_chunk_ids"] or not case["gold_element_ids"]:
            raise ValueError(f"gold target mapping failed: {case['qid']}")

    with TARGET.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    field_rows = [
        row for row in (json.loads(x) for x in FIELD_QA.open(encoding="utf-8") if x.strip())
        if row.get("신뢰도") == "확정"
    ]
    with COMBINED.open("w", encoding="utf-8") as handle:
        for row in field_rows:
            handle.write(json.dumps({
                "qid": f"field-{row['no']}", "question": row["질문"], "answer": row["정답"],
                "business_type": row.get("사업구분"), "sources": row.get("출처", []),
                "source": "field_qa359_confirmed", "review_status": "field_confirmed",
            }, ensure_ascii=False) + "\n")
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_type = collections.Counter(t for case in cases for t in case["question_types"])
    field_type_rows = [json.loads(x) for x in FIELD_TYPES.open(encoding="utf-8")]
    field_by_type = collections.Counter(t for row in field_type_rows for t in row["question_types"])
    combined_by_type = field_by_type + by_type
    by_complexity = collections.Counter(case["evidence_complexity"] for case in cases)
    by_difficulty = collections.Counter(case["expression_difficulty"] for case in cases)
    duplicate_questions = len(cases) - len({case["question"] for case in cases})
    scope_counts = collections.Counter(case["contract_scope"] for case in cases)
    covered = sorted(by_type)
    report = {
        "n_cases": len(cases),
        "n_field_confirmed_preserved": len(field_rows),
        "n_combined": len(field_rows) + len(cases),
        "taxonomy_in_scope": sorted(TYPE_LABELS),
        "taxonomy_covered": covered,
        "taxonomy_missing": sorted(set(TYPE_LABELS) - set(covered)),
        "by_type": dict(sorted(by_type.items())),
        "field_303_by_type": dict(sorted(field_by_type.items())),
        "combined_333_by_type": dict(sorted(combined_by_type.items())),
        "by_complexity": dict(sorted(by_complexity.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "duplicate_questions": duplicate_questions,
        "n_contract_scopes": len(scope_counts),
        "by_contract_scope": dict(sorted(scope_counts.items())),
        "exact_source_offsets_validated": True,
        "gold_chunk_and_element_ids_mapped": True,
        "questions_are_methodology_blind": True,
        "notes": [
            "This is the core document-coverage supplement, not a method-specific challenge suite.",
            "curated_needs_domain_signoff means retrieval-grounded and language-reviewed, but not approved by an insurance domain owner.",
            "The historical 129+75 generated drafts are not included.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
