#!/usr/bin/env python3
"""Compare question-type coverage of the field QA set with affordances in one policy document."""

import collections
import json
import math
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
QA_PATH = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl")
DOC_PATH = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")


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
    "underwriting_eligibility": "가입자격·인수 가능성",
    "document_version_lookup": "가입시점·문서 버전 선택",
}


QA_RULES = {
    "definition": [r"(?:이란|뜻|정의|의미|무엇인가요?$|무엇이야|어떤 치료|뭔가요?$|뭐야$)"],
    "coverage_scope": [r"보장|포함|해당(?:되|하)|가입해야|적용되|가능한가|가능해|가능한지|받을\s*수|지급되|나오나요"],
    "payment_trigger": [r"지급사유|지급조건|지급요건|받는 조건|어떤 경우.*(?:지급|보험금)|언제.*(?:지급|받)|사유는|무조건\s*지급"],
    "amount_rate": [r"얼마|금액|몇\s?%|비율|지급률|지급액|보험가입금액|환급금액"],
    "limit_frequency": [r"몇\s*(?:번|회|일)|횟수|한도|최대|최초\s*1회|반복|여러 번|중복|또\s*받|다시\s*(?:보장|받)|연간\s*1회"],
    "timing_period": [r"언제|기간|며칠|몇\s*년|몇\s*개월|개시일|대기기간|면책기간|삭감기간|시점|이후|이내"],
    "exclusion_exception": [r"지급하지|보장하지|제외|면책|예외|안\s*(?:되|나오|받)|없는 경우|거절"],
    "premium_waiver": [r"보험료|납입면제|납면|납입 면제"],
    "renewal_termination": [r"갱신|재가입|유지|소멸|종료|효력.*없|실효"],
    "contract_refund": [r"해지|해약|환급|철회|취소|무효|계약.*변경|부활|효력회복"],
    "claim_procedure": [r"청구|서류|절차|신청|제출|발급|준비해야"],
    "enumeration_summary": [r"모두|전체|목록|리스트|정리|요약|총\s*.*개|몇\s*개|전부|한번에|특약들"],
    "comparison": [r"비교|차이|다른|달라|보다|A와 B|각각"],
    "case_calculation": [r"경우.*(?:받|지급|보장)|계산|합산|예시|가정|진단받았|치료.*받|가입.*했|라면|때.*얼마"],
    "criteria_rule": [r"기준|요건|규정|판정|충족|어떻게 산정|어떻게 적용"],
    "code_classification_lookup": [r"분류코드|질병코드|수가코드|코드|분류표|구성요소|항목"],
    "document_audit": [r"오타|오탈자|누락|빠져|들어가 있는지|있는지 확인|충돌|애매|잘못|검증|문구.*확인|판매되고 있지"],
    "interpretation_explanation": [r"설명|쉽게|해석|고객에게|추가질문|리스크|유의|불리|관례|알려줘"],
    "product_composition": [r"특약.*(?:구성|개수|갯수|종류)|상품.*구성|주계약과 특약|특약명|상품명|담보.*종류"],
    "underwriting_eligibility": [r"가입\s*(?:이|은|을|도)?\s*(?:가능|할 수)|가입가능|가입연령|가입나이|가입을 하려고|인수.*가능|유병자만 가입|심사.*가입 변경"],
    "document_version_lookup": [r"\d{2}년\s*\d{1,2}월\s*\d{1,2}일\s*가입.*약관|가입.*약관\s*찾"],
}


DOC_RULES = {
    "definition": [r"(?:이라|라) 함은|용어의 정의|의 정의|정의 및 진단확정"],
    "coverage_scope": [r"보장대상|분류표|해당하는|보장합니다|급여금을 지급"],
    "payment_trigger": [r"보험금의 지급사유|지급사유가 발생|지급합니다|지급할 때"],
    "amount_rate": [r"지급금액|보험가입금액의\s*\d|\d+(?:\.\d+)?%|\d+(?:,\d{3})+원|금액을 지급"],
    "limit_frequency": [r"최초\s*1회|\d+회(?:에|를|까지| 한)|최고\s*한도|\d+일\s*한도|연간\s*\d+회|최대\s*\d+"],
    "timing_period": [r"보장개시일|보험기간|납입기간|\d+일\s*(?:이내|이후|이상)|\d+년\s*(?:이내|이후|동안)|기간 중|지난 날"],
    "exclusion_exception": [r"지급하지 않는|보장하지|제외합니다|제외한다|면책|다만,?\s*.*제외|지급하지 않습니다"],
    "premium_waiver": [r"보험료.*납입|납입.*면제|납입면제|보험료납입면제"],
    "renewal_termination": [r"갱신|재가입|계약의 소멸|효력이 없습니다|보험기간의 종료|실효"],
    "contract_refund": [r"해약환급금|계약의 해지|청약의 철회|계약의 무효|계약.*취소|부활\(효력회복\)|계약내용의 변경"],
    "claim_procedure": [r"보험금.*청구|청구서|구비서류|지급절차|서류를 제출|신청방법"],
    "enumeration_summary": [r"다음 각 호|다음 중|각각|분류표|지급기준표|특약.*목록"],
    "comparison": [r"서로 다른|보다 .*경우|각각.*(?:경우|지급)|A형|B형|종과.*종|유형별"],
    "case_calculation": [r"계산|합산|더하여|산식|예시|예제|경과기간|지급률.*적용|해당하는 경우"],
    "criteria_rule": [r"기준|요건|규정|판정|충족|적용 기준"],
    "code_classification_lookup": [r"분류코드|질병·사인분류|수가코드|코드표|분류표|대상이 되는 질병"],
    "document_audit": [r"약관의 해석|내용이 서로 다른|우선하여 적용|오류|누락|정정|개정"],
    "interpretation_explanation": [r"해석|설명|유의사항|안내|예시|쉽게 찾기|주\)"],
    "product_composition": [r"특약약관|특약의 구성|주계약|특약.*종류|목\s*차|상품의 구성"],
    "underwriting_eligibility": [r"가입나이|가입연령|가입할 수|피보험자의 범위|계약인수|가입 조건"],
    # A single fixed document cannot itself afford selecting the correct historical version.
    "document_version_lookup": [],
}


MANUAL_OVERRIDES = {
    "231": ["coverage_scope", "code_classification_lookup"],
    "250": ["coverage_scope", "criteria_rule"],
    "253": ["coverage_scope", "exclusion_exception", "criteria_rule"],
    "254": ["coverage_scope"],
    "256": ["coverage_scope", "case_calculation", "criteria_rule"],
    "257": ["coverage_scope"],
    "258": ["coverage_scope"],
    "263": ["coverage_scope"],
    "265": ["coverage_scope", "criteria_rule"],
    "269": ["limit_frequency"],
    "276": ["coverage_scope", "criteria_rule"],
    "284": ["coverage_scope", "payment_trigger"],
    "293": ["coverage_scope", "enumeration_summary", "product_composition"],
    "297": ["underwriting_eligibility", "case_calculation"],
    "317": ["underwriting_eligibility", "case_calculation"],
    "318": ["coverage_scope", "enumeration_summary", "product_composition"],
    "319": ["underwriting_eligibility", "timing_period"],
    "320": ["definition"],
    "331": ["document_version_lookup"],
    "336": ["coverage_scope"],
    "340": ["document_version_lookup"],
    "343": ["coverage_scope"],
    "356": ["coverage_scope", "exclusion_exception", "case_calculation"],
}


def matches(text, rules):
    return any(re.search(pattern, text, re.I) for pattern in rules)


def classify(text, ruleset):
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    return [kind for kind, rules in ruleset.items() if matches(normalized, rules)]


ARTICLE = re.compile(r"^#{0,6}\s*(제\s*\d+(?:-\d+)?조(?:의\s*\d+)?)\s*[\[(（]?([^\])）\n]{0,100})[\])）]?\s*$")
APPENDIX = re.compile(r"^#{0,6}\s*[<\[]?(부표|별표)\s*([\d-]+)[>\]]?\s*(.{0,100})$")


def document_blocks(raw):
    lines = raw.splitlines()
    starts = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) > 140 or stripped.startswith("|"):
            continue
        article = ARTICLE.match(stripped)
        appendix = APPENDIX.match(stripped)
        if article:
            starts.append((index, article.group(1).replace(" ", "") + " " + article.group(2).strip()))
        elif appendix:
            starts.append((index, appendix.group(1) + appendix.group(2) + " " + appendix.group(3).strip()))
    blocks = []
    for pos, (start, title) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else min(len(lines), start + 500)
        text = "\n".join(lines[start:end])
        # TOC-like matches generally have no substantive body and page-number rows dominate.
        if len(text.strip()) < 80:
            continue
        blocks.append({"block_id": f"b{len(blocks):05d}", "title": title, "line_start": start + 1,
                       "line_end": end, "text": text})
    return blocks


def pct(n, d):
    return round(n / d, 4) if d else 0.0


def main():
    qa_all = [json.loads(line) for line in QA_PATH.open(encoding="utf-8") if line.strip()]
    qa = [row for row in qa_all if row.get("신뢰도") == "확정"]
    raw = unicodedata.normalize("NFC", DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n"))
    blocks = document_blocks(raw)

    qa_rows = []
    for row in qa:
        types = classify(row["질문"], QA_RULES)
        if str(row["no"]) in MANUAL_OVERRIDES:
            types = list(dict.fromkeys(types + MANUAL_OVERRIDES[str(row["no"])]))
        qa_rows.append({"qid": str(row["no"]), "question": row["질문"], "business_type": row["사업구분"],
                        "question_types": types or ["unclassified"]})

    block_rows = []
    for block in blocks:
        types = classify(block["title"] + "\n" + block["text"], DOC_RULES)
        block_rows.append({k: block[k] for k in ("block_id", "title", "line_start", "line_end")} |
                          {"question_affordances": types})

    qa_counts = collections.Counter(t for row in qa_rows for t in row["question_types"] if t != "unclassified")
    doc_counts = collections.Counter(t for row in block_rows for t in row["question_affordances"])
    examples = collections.defaultdict(list)
    for row in qa_rows:
        for kind in row["question_types"]:
            if len(examples[kind]) < 5:
                examples[kind].append({"qid": row["qid"], "question": row["question"]})

    coverage = []
    for kind, label in TYPE_LABELS.items():
        qn, dn = qa_counts[kind], doc_counts[kind]
        coverage.append({
            "type": kind,
            "label": label,
            "qa_count": qn,
            "qa_share": pct(qn, len(qa)),
            "document_block_count": dn,
            "document_block_share": pct(dn, len(blocks)),
            "present_in_qa": qn > 0,
            "present_in_document": dn > 0,
            "examples": examples[kind],
        })

    qa_type_sets = collections.Counter(len(row["question_types"]) for row in qa_rows)
    unclassified = [row for row in qa_rows if row["question_types"] == ["unclassified"]]
    report = {
        "method": {
            "focus": "question/answer form and reasoning type, not insurance content topic",
            "classification": "deterministic multi-label regex v1",
            "unit_for_document": "article/appendix block",
            "warning": "Counts are screening estimates and require sample review; one item may carry multiple labels.",
        },
        "qa": {"all": len(qa_all), "confirmed": len(qa), "unclassified": len(unclassified),
               "labels_per_question": dict(sorted(qa_type_sets.items()))},
        "document": {"article_appendix_blocks": len(blocks)},
        "coverage": coverage,
        "qa_unclassified_examples": unclassified[:20],
    }
    OUT.mkdir(exist_ok=True)
    with (OUT / "qa359_question_types.jsonl").open("w", encoding="utf-8") as handle:
        for row in qa_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "document_question_affordances.jsonl").open("w", encoding="utf-8") as handle:
        for row in block_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "qa359_type_coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
