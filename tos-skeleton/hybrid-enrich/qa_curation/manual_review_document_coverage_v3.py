#!/usr/bin/env python3
"""Apply the human review of the 170 document-coverage questions.

Question text in this file is an explicit per-item editorial decision.  This is
not a question generator.  Code is only used to apply the decisions and check
the resulting dataset mechanically.
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SOURCE = OUT / "qa_document_coverage_v3_expansion.jsonl"
CORE = OUT / "qa_document_coverage_v2.jsonl"
FIELD = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl")


# Each entry below was written after reading the individual question, answer,
# and evidence row.  Questions absent from this mapping were explicitly kept.
QUESTION_OVERRIDES = {
    "doccov-v3-033": "[50%이상장해형] 6대질병장해특약에서 장해급여금을 받으려면 어떤 질병으로 인한 장해여야 하고 장해지급률은 몇 % 이상이어야 하나요?",
    "doccov-v3-047": "1-5종단일금액보장수술특약의 수술급여금 지급 요건과 연간 지급 한도는 무엇인가요?",
    "doccov-v3-049": "1-5종수술특약에서 3종 수술급여금의 지급 대상이 되는 수술은 무엇인가요?",
    "doccov-v3-053": "간병인사용입원특약에서 요양병원을 제외한 의료기관에 입원한 경우, 간병인사용 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-057": "급여뇌심장질환특정검사지원특약의 검사급여금은 어떤 목적으로 검사를 받아야 지급되며, 연간 몇 회까지 보장되나요?",
    "doccov-v3-059": "급여 갑상선 바늘생검 조직병리진단급여금의 지급 요건과 연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-061": "급여암특정재활치료특약에서 기본물리치료 급여금을 받을 수 있는 암의 범위와 치료 요건, 지급 횟수 한도는 무엇인가요?",
    "doccov-v3-063": "급여주요뇌심장질환특정재활치료특약의 기본물리치료 급여금은 어떤 경우에 지급되며, 하루와 연간 지급 횟수는 어떻게 제한되나요?",
    "doccov-v3-065": "급여 특정NGS 유전자패널 검사 지원급여금의 지급 대상 질병과 검사 목적, 연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-069": "뇌혈관질환으로 상급종합병원에 입원한 경우, 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-070": "뇌혈관질환으로 상급종합병원에 입원하면 입원급여금은 하루에 얼마씩 지급되나요?",
    "doccov-v3-073": "뇌혈관질환 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-079": "비급여 암주요치료비특약은 어떤 암의 치료를 보장하며, 최초 진단 후 몇 년 동안 몇 회까지 지급하나요?",
    "doccov-v3-081": "비급여 항암약물치료급여금의 지급 대상 암과 지급기간·횟수 한도는 어떻게 되나요?",
    "doccov-v3-083": "상급종합병원 암주요치료비특약에서 기타피부암이나 갑상선암 치료는 어떤 의료기관에서 받아야 하며, 최대 몇 회까지 보장되나요?",
    "doccov-v3-085": "상급종합병원 2인실 또는 3인실 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-087": "뇌혈관질환 또는 허혈심장질환으로 상급종합병원에서 주요치료를 받은 경우, 치료비는 몇 년 동안 몇 회까지 지급되나요?",
    "doccov-v3-089": "상급종합병원 1인실 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-091": "상급종합병원질병수술특약의 수술급여금 지급 요건과 동일 질병당 지급 횟수는 어떻게 되나요?",
    "doccov-v3-093": "암다빈치로봇수술특약에서 갑상선암과 전립선암을 제외한 다빈치로봇수술급여금의 지급 요건은 무엇인가요?",
    "doccov-v3-095": "암수술특약에서 어떤 암의 직접 치료를 위한 수술이 암수술급여금 지급 대상인가요?",
    "doccov-v3-097": "암주요치료비특약에서 암·대장점막내암·비침습방광암 주요치료는 몇 년 동안 몇 회까지 보장되나요?",
    "doccov-v3-099": "암특정검사지원특약에서 CT 검사는 어떤 암의 치료 또는 진행 여부 확인을 위한 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-100": "암 CT검사 지원비는 계약일부터 1년 미만일 때와 1년 이후에 각각 얼마가 지급되나요?",
    "doccov-v3-101": "사고로 치아 이외의 부위가 골절된 경우 재해골절 치료급여금을 받을 수 있나요?",
    "doccov-v3-103": "재해수술특약의 수술급여금은 어떤 경우에 지급되나요?",
    "doccov-v3-107": "중증질환자[암] 산정특례대상 보장급여금은 진단만 받으면 지급되나요, 아니면 산정특례 신규 등록까지 해야 하나요?",
    "doccov-v3-109": "진심을품은일반암진단Plus특약의 올페이Plus 급여금은 어떤 암으로 진단받았을 때 지급되나요?",
    "doccov-v3-111": "진심을품은일반암진단특약의 올페이 급여금은 어떤 암으로 진단받았을 때 지급되나요?",
    "doccov-v3-113": "질병수술(8대특정수술제외)특약의 수술급여금 지급 요건과 동일 질병당 지급 횟수는 어떻게 되나요?",
    "doccov-v3-115": "질병수술(백내장및대장용종제외)특약은 어떤 수술을 보장하며, 동일 질병으로 여러 번 수술하면 매번 지급되나요?",
    "doccov-v3-117": "질병수술특약의 수술급여금 지급 요건과 동일 질병당 지급 횟수는 어떻게 되나요?",
    "doccov-v3-119": "6대질병으로 중환자실에 입원한 경우 입원급여금을 받을 수 있는 질병 범위와 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-125": "표적항암약물허가치료급여금은 어떤 암의 직접 치료를 위해 해당 치료를 받은 경우 지급되며, 몇 회까지 받을 수 있나요?",
    "doccov-v3-127": "항암세기조절방사선치료급여금의 지급 대상 암과 지급 횟수 한도는 어떻게 되나요?",
    "doccov-v3-129": "항암양성자방사선치료급여금의 지급 대상 암과 지급 횟수 한도는 어떻게 되나요?",
    "doccov-v3-137": "1-5종수술특약에서 4종 수술급여금의 지급 대상이 되는 수술은 무엇인가요?",
    "doccov-v3-139": "특정 급성심근경색증으로 혈전용해치료를 받으면 어떤 조건에서 급여금이 지급되며, 몇 회까지 받을 수 있나요?",
    "doccov-v3-141": "급여 혈관조영술 검사급여금은 어떤 목적으로 검사를 받아야 지급되며, 연간 몇 회까지 보장되나요?",
    "doccov-v3-143": "급여 유방 바늘생검 조직병리진단급여금의 지급 요건과 연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-145": "급여암특정재활치료특약에서 단순재활치료 급여금을 받을 수 있는 암의 범위와 하루·연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-147": "급여주요뇌심장질환특정재활치료특약의 단순재활치료 급여금은 어떤 경우에 지급되며, 하루와 연간 지급 횟수는 어떻게 제한되나요?",
    "doccov-v3-149": "허혈심장질환으로 상급종합병원에 입원한 경우, 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-150": "허혈심장질환으로 상급종합병원에 입원하면 입원급여금은 하루에 얼마씩 지급되나요?",
    "doccov-v3-151": "허혈심장질환 치료를 위해 상급종합병원에 통원하면 하루와 연간 몇 회까지 통원급여금을 받을 수 있나요?",
    "doccov-v3-153": "허혈심장질환 입원급여금의 지급 요건과 1회 입원당 보장일수는 어떻게 되나요?",
    "doccov-v3-155": "암·대장점막내암·비침습방광암의 항암호르몬약물치료급여금은 어떤 경우에 지급되며, 연간 몇 회까지 받을 수 있나요?",
    "doccov-v3-157": "암특정검사지원특약에서 PET 검사는 어떤 암의 치료 또는 진행 여부 확인을 위한 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-158": "암 PET검사 지원비는 계약일부터 1년 미만일 때와 1년 이후에 각각 얼마가 지급되나요?",
    "doccov-v3-161": "특정순환계질환주요치료비특약의 일반심사형에서 급여 부분체외순환치료는 어떤 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-162": "특정순환계질환주요치료비특약의 일반심사형에서 급여 부분체외순환치료 급여금은 얼마인가요?",
    "doccov-v3-165": "1-5종수술특약에서 5종 수술급여금의 지급 대상이 되는 수술은 무엇인가요?",
    "doccov-v3-167": "급여 전립선 바늘생검 조직병리진단급여금의 지급 요건과 연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-169": "급여암특정재활치료특약에서 전문재활치료 급여금을 받을 수 있는 암의 범위와 하루·연간 지급 횟수는 어떻게 되나요?",
    "doccov-v3-171": "급여주요뇌심장질환특정재활치료특약의 전문재활치료 급여금은 어떤 경우에 지급되며, 하루와 연간 지급 횟수는 어떻게 제한되나요?",
    "doccov-v3-173": "기타피부암·갑상선암·제자리암·경계성종양의 항암호르몬약물치료급여금은 어떤 경우에 지급되며, 연간 몇 회까지 받을 수 있나요?",
    "doccov-v3-175": "암특정검사지원특약에서 MRI 검사는 어떤 암의 치료 또는 진행 여부 확인을 위한 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-176": "암 MRI검사 지원비는 계약일부터 1년 미만일 때와 1년 이후에 각각 얼마가 지급되나요?",
    "doccov-v3-177": "특정순환계질환주요치료비특약의 일반심사형에서 지속적 신대체요법(CRRT)은 어떤 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-178": "특정순환계질환주요치료비특약의 일반심사형에서 지속적 신대체요법(CRRT) 치료급여금은 얼마인가요?",
    "doccov-v3-179": "암특정검사지원특약에서 초음파검사는 어떤 암의 치료 또는 진행 여부 확인을 위한 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-180": "암초음파검사 지원비는 계약일부터 1년 미만일 때와 1년 이후에 각각 얼마가 지급되나요?",
    "doccov-v3-181": "특정순환계질환주요치료비특약의 일반심사형에서 12시간을 초과한 인공호흡기치료는 어떤 경우에 보장되나요?",
    "doccov-v3-182": "특정순환계질환주요치료비특약의 일반심사형에서 12시간 초과 인공호흡기치료 급여금은 얼마인가요?",
    "doccov-v3-183": "특정순환계질환주요치료비특약의 일반심사형에서 급여 저체온요법치료는 어떤 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-184": "특정순환계질환주요치료비특약의 일반심사형에서 급여 저체온요법치료 급여금은 얼마인가요?",
    "doccov-v3-185": "특정순환계질환주요치료비특약에서 급여 혈전제거술과 혈전용해치료를 제외한 수술은 어떤 경우에 보장되며, 연간 몇 회까지 지급되나요?",
    "doccov-v3-186": "특정순환계질환주요치료비특약의 특정순환계질환 수술급여금은 얼마인가요?",
    "doccov-v3-187": "6대질병으로 장해지급률 50% 이상의 장해가 생긴 경우, 장해급여금은 최초 한 번만 지급되나요?",
    "doccov-v3-188": "질병으로 장해지급률 50% 이상의 장해가 생긴 경우, 질병장해급여금은 최초 한 번만 지급되나요?",
    "doccov-v3-194": "1-5종단일금액보장수술특약의 수술급여금은 연간 몇 회까지 받을 수 있나요?",
    "doccov-v3-199": "급여뇌심장질환 검사급여금은 연간 몇 회까지 받을 수 있나요?",
    "doccov-v3-200": "급여 갑상선 바늘생검 조직병리진단급여금은 연간 몇 회까지 받을 수 있나요?",
}


FIELD_OVERRIDES = {
    qid: {"contract_scope": "(간편)특정순환계질환주요치료비특약(4대중증치료)(치료별 연간1회)"}
    for qid in ("doccov-v3-161", "doccov-v3-162", "doccov-v3-177", "doccov-v3-178",
                "doccov-v3-181", "doccov-v3-182", "doccov-v3-183", "doccov-v3-184")
}
FIELD_OVERRIDES.update({
    "doccov-v3-185": {"contract_scope": "(간편)특정순환계질환주요치료비특약(치료별 연간1회)"},
    "doccov-v3-186": {"contract_scope": "(간편)특정순환계질환주요치료비특약(치료별 연간1회)"},
})


ANSWER_OVERRIDES = {
    "doccov-v3-083": "특약보험기간 중 기타피부암 또는 갑상선암으로 최초 진단확정되고, 보험금 지급기간 안에 해당 암의 직접 치료를 목적으로 상급종합병원(국립암센터 및 원자력병원 포함)에서 주요치료를 받은 경우 지급합니다. 최초 진단확정일부터 최대 10년간 연간 1회, 최대 10회 지급합니다.",
    "doccov-v3-089": "질병 또는 재해의 직접 치료를 목적으로 상급종합병원 1인실에 1일 이상 계속 입원한 경우 지급하며, 1회 입원당 30일까지 보장합니다.",
    "doccov-v3-090": "입원일수 1일당 특약보험가입금액의 1%를 지급합니다. 간편심사형에서 계약일부터 1년 미만에 재해 이외의 원인으로 지급사유가 발생하면 0.5%를 지급합니다.",
    "doccov-v3-113": "질병(특정다빈도 7대질병 제외)의 직접 치료를 목적으로 특약보험기간 중 수술(제왕절개 제외)을 받은 경우 동일 질병당 1회 지급합니다.",
    "doccov-v3-114": "특약보험가입금액의 1%를 지급합니다. 계약일부터 1년 미만에 지급사유가 발생하면 0.5%를 지급합니다.",
    "doccov-v3-115": "백내장과 대장용종을 제외한 질병의 직접 치료를 목적으로 수술을 받은 경우 동일 질병당 1회 지급합니다.",
    "doccov-v3-116": "특약보험가입금액의 1%를 지급합니다. 계약일부터 1년 미만에 지급사유가 발생하면 0.5%를 지급합니다.",
    "doccov-v3-185": "특정순환계질환으로 진단확정되고 그 직접 치료를 목적으로 수술(급여 혈전제거술과 혈전용해치료 제외)을 받은 경우 연간 1회 지급합니다.",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def dump_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = load_jsonl(SOURCE)
    assert len(rows) == 170
    seen = {row["qid"] for row in rows}
    assert set(QUESTION_OVERRIDES) <= seen
    assert set(FIELD_OVERRIDES) <= seen
    assert set(ANSWER_OVERRIDES) <= seen

    for row in rows:
        qid = row["qid"]
        if qid in QUESTION_OVERRIDES:
            row["question"] = QUESTION_OVERRIDES[qid]
            row["review_status"] = "manual_reviewed_revised"
        else:
            row["review_status"] = "manual_reviewed_keep"
        if qid in FIELD_OVERRIDES:
            row.update(FIELD_OVERRIDES[qid])
        if qid in ANSWER_OVERRIDES:
            row["answer"] = ANSWER_OVERRIDES[qid]
        row["qa_sha256"] = hashlib.sha256(
            (row["question"] + "\n" + row["answer"]).encode()
        ).hexdigest()

    reviewed_path = OUT / "qa_document_coverage_v3_expansion_reviewed.jsonl"
    dump_jsonl(reviewed_path, rows)

    core = load_jsonl(CORE)
    coverage = core + rows
    dump_jsonl(OUT / "qa_document_coverage_v3_reviewed.jsonl", coverage)

    field_original = [row for row in load_jsonl(FIELD) if row.get("신뢰도") == "확정"]
    assert len(field_original) == 303
    field = [{
        "qid": f"field-{row['no']}",
        "question": row["질문"],
        "answer": row["정답"],
        "business_type": row.get("사업구분"),
        "sources": row.get("출처", []),
        "source": "field_qa359_confirmed",
        "review_status": "field_confirmed",
    } for row in field_original]
    combined = field + coverage
    dump_jsonl(OUT / "qa503_field_plus_document_coverage_v3_reviewed.jsonl", combined)

    original_fields = ["no", "사업구분", "질문", "정답", "신뢰도", "출처", "방법"]
    original_rows = []
    for row in field_original:
        original_rows.append({key: row.get(key, "") for key in original_fields})
    for number, row in enumerate(coverage, 360):
        source_text = "\n".join(s["quote"] for s in row.get("sources", []))
        original_rows.append({
            "no": number,
            "사업구분": "문서커버리지",
            "질문": row["question"],
            "정답": row["answer"],
            "신뢰도": "검수필요",
            "출처": source_text,
            "방법": "약관 원문 직접 검수",
        })

    csv_path = OUT / "정답셋_503_현업303_문서보강200_v2_직접검수.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=original_fields)
        writer.writeheader()
        writer.writerows(original_rows)
    dump_jsonl(OUT / "정답셋_503_현업303_문서보강200_v2_직접검수.jsonl", original_rows)

    questions = [row["question"] for row in rows]
    report = {
        "total_reviewed": len(rows),
        "kept": sum(row["review_status"] == "manual_reviewed_keep" for row in rows),
        "revised": sum(row["review_status"] == "manual_reviewed_revised" for row in rows),
        "duplicate_questions": len(questions) - len(set(questions)),
        "combined_total": len(original_rows),
        "question_generation": "human_editorial_only",
        "code_usage": "apply_edits_and_mechanical_validation_only",
        "domain_signoff_required": True,
    }
    (OUT / "qa_document_coverage_v3_manual_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
