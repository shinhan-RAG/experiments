#!/usr/bin/env python3
"""Expand the curated coverage core with precise table-row QA pairs.

Only the single 260507 policy document and its deterministic structural elements
are read. For each selected benefit row, one trigger question and one amount
question are created. The two questions have distinct intents and share the same
exact row evidence.
"""

from __future__ import annotations

import collections
import bisect
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
FIELD_QA = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl")
CORE = OUT / "qa_document_coverage_v2.jsonl"
EXPANSION = OUT / "qa_document_coverage_v3_expansion.jsonl"
FULL = OUT / "qa_document_coverage_v3.jsonl"
COMBINED = OUT / "qa503_field_plus_document_coverage_v3.jsonl"
REPORT = OUT / "qa_document_coverage_v3_report.json"
N_ROWS = 78
N_EXTRA = 14


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def strip_md(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"[*_`#]", "", value)
    value = value.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", value).strip()


def clean_ocr(value: str) -> str:
    replacements = {
        "특 약": "특약", "특 약": "특약", "제해": "재해",
        "금여금": "급여금", "입원금여금": "입원급여금",
        "감상선암": "갑상선암", "감상선암": "갑상선암",
        "종환자실": "중환자실", "상금종합병원": "상급종합병원",
        "순환계절환": "순환계질환", "다빈치로못": "다빈치로봇",
        "부분체의 순환": "부분체외순환", "수출급여금": "수술급여금",
        "진단이확정": "진단이 확정", "목적으로특약": "목적으로 특약",
        "동안\"": "동안 \"", "다만,최초1회": "다만, 최초 1회",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return re.sub(r"\s+", " ", value).strip()


def clean_scope(value: str) -> str:
    value = value.replace("수출", "수술")
    value = value.replace("다빈치로못", "다빈치로봇").replace("진심을풍은", "진심을품은")
    value = re.sub(r"\(무배당[^)]*\)", "", value)
    value = re.sub(r"80Plus", "", value)
    return re.sub(r"\s+", " ", value).strip()


def split_row(line: str):
    if not line.strip().startswith("|"):
        return None
    cells = [strip_md(cell) for cell in line.strip().strip("|").split("|")]
    if len(cells) != 3 or any(not cell for cell in cells):
        return None
    if cells[0] == "구분" or all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
        return None
    return cells


def usable(scope: str, benefit: str, trigger: str, amount: str) -> bool:
    if not scope or "특약" not in scope:
        return False
    if not (2 <= len(benefit) <= 55 and 18 <= len(trigger) <= 1000 and 2 <= len(amount) <= 450):
        return False
    noise = ("SHINHAN", "약 관 목 차", "\\begin", "페이지", "white bear",
             "대상포(다만", "계약일부진", "종종 갑상선암")
    joined = benefit + trigger + amount
    if any(token in joined for token in noise):
        return False
    if not re.search(r"지급|보험가입금액|\d+%|\d+원|적립액|급여금", amount):
        return False
    if not re.search(r"때|경우|진단|입원|수술|치료|사망|장해|검사", trigger):
        return False
    # Known scope-boundary failures found during manual review.
    mismatch = (
        ("응급실내원" in scope and "암진단" in benefit.replace(" ", "")),
        ("허혈심장질환진단특약" in scope and "고혈압" in benefit),
        ("허혈심장질환진단특약" in scope and "당뇨병" in benefit),
        ("허혈심장질환진단특약" in scope and "치매" in benefit),
        ("허혈심장질환진단특약" in scope and "수술" in benefit),
        ("암주요치료비특약" in scope and "순환계" in trigger),
    )
    if any(mismatch):
        return False
    return True


def trigger_question(scope: str, benefit: str, index: int) -> str:
    compact = benefit.replace(" 급여금", "급여금")
    if "장해" in benefit:
        return f"{scope} 가입자입니다. 질병으로 장해가 남았을 때 {compact} 대상이 되는 장해율 기준은 몇 %인가요?"
    if "진단" in benefit or "산정특례" in benefit:
        return f"{scope}에서 {compact}을 받으려면 어떤 진단을 받아야 하며, 여러 번 진단받아도 반복 지급되나요?"
    if "입원" in benefit:
        return f"{scope} 가입 후 해당 질병 치료로 입원했습니다. {compact}이 나오는 입원 조건과 1회 입원당 한도를 알려주세요."
    if "통원" in benefit:
        return f"{scope}에서 치료를 받으러 통원했을 때 {compact}을 받을 수 있는 병원 조건과 연간 횟수 한도는 어떻게 되나요?"
    if "수술" in benefit:
        return f"{scope} 가입자가 질병이나 재해 치료로 수술을 받았습니다. 어떤 수술이 {compact} 대상이고, 횟수 제한도 있나요?"
    if "검사" in benefit or "조직병리" in benefit:
        return f"{scope} 가입자가 의사의 필요 소견으로 해당 검사를 받았습니다. {compact} 대상이 되는 검사 목적과 횟수 한도는 무엇인가요?"
    if "치료" in benefit:
        return f"{scope}에서 {compact}을 받으려면 어떤 질병으로 진단받고 어떤 치료를 받아야 하나요? 지급 횟수도 알려주세요."
    if "사망" in benefit:
        return f"{scope}의 {compact}은 어떤 경우에 지급되나요?"
    return f"{scope}의 {compact}을 받을 수 있는 구체적인 조건과 지급 횟수를 알려주세요."


def amount_question(scope: str, benefit: str, amount: str, index: int) -> str:
    compact = benefit.replace(" 급여금", "급여금")
    if re.search(r"2년 미만.*4년 미만|91일.*180일", amount):
        return f"{scope}의 {compact} 지급률이 가입 경과기간에 따라 달라지나요? 구간별로 알려주세요."
    if re.search(r"1년 미만", amount):
        return f"{scope}의 {compact}은 가입 후 1년 안에 사유가 발생한 경우와 1년 이후의 지급률이 어떻게 다른가요?"
    if re.search(r"2년 미만", amount):
        return f"{scope}의 {compact}은 가입 후 2년 미만과 그 이후에 각각 얼마를 지급하나요?"
    if "1일당" in amount:
        return f"{scope}의 {compact}은 입원이나 사용 1일당 보험가입금액의 몇 %가 나오나요?"
    if "1회당" in amount:
        return f"{scope}의 {compact}은 수술이나 치료 1회당 보험가입금액의 몇 %를 받나요?"
    if "장해지급률" in amount or "×" in amount:
        return f"{scope}의 {compact}은 보험가입금액과 장해지급률을 어떻게 적용해 계산하나요?"
    if "총 보험료" in amount:
        return f"{scope}의 {compact}은 올페이 대상계약에서 낸 보험료 중 얼마를 지급하나요?"
    return f"{scope}의 {compact} 지급액은 보험가입금액의 몇 %인가요?"


def labels_for_trigger(trigger: str):
    labels = ["payment_trigger", "coverage_scope"]
    if re.search(r"최초\s*1회|연간\s*\d+회|\d+일\s*한도|1회에 한", trigger):
        labels.append("limit_frequency")
    if re.search(r"보장개시일|계약일부터|\d+년 미만|보험기간 중", trigger):
        labels.append("timing_period")
    if re.search(r"다만|제외|아닌 경우|제한", trigger):
        labels.append("exclusion_exception")
    return list(dict.fromkeys(labels))


def labels_for_amount(amount: str):
    labels = ["amount_rate"]
    if re.search(r"\d+년 미만|\d+년 이상|계약일부터", amount):
        labels.extend(["timing_period", "comparison"])
    if re.search(r"\u00d7|계산|지급률|해당 장해", amount):
        labels.append("case_calculation")
    return list(dict.fromkeys(labels))


def extra_question(scope: str, benefit: str, trigger: str, amount: str) -> tuple[str, str, list[str]]:
    if re.search(r"연간\s*\d+회|1일\s*1회", trigger):
        return (
            f"{scope}의 {benefit}은 하루와 1년을 기준으로 최대 몇 번까지 받을 수 있나요?",
            trigger, ["limit_frequency", "timing_period"],
        )
    if re.search(r"최초\s*1회|최초1회", trigger):
        return (
            f"{scope}의 {benefit}은 재진단받아도 다시 받을 수 있나요, 아니면 최초 한 번만 지급되나요?",
            trigger, ["limit_frequency", "payment_trigger"],
        )
    return (
        f"{scope}의 {benefit}은 가입 후 얼마가 지나야 전액을 받을 수 있나요?",
        amount, ["timing_period", "amount_rate"],
    )


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    raw_lines = raw.splitlines()
    rider_rx = re.compile(r"^\(간편\).{2,180}특약(?:80Plus)?\(무배당.*\)\s*$")
    scope_bounds = []
    for line_no, line in enumerate(raw_lines, 1):
        stripped = line.strip()
        if "|" not in stripped and rider_rx.match(stripped):
            scope_bounds.append((line_no, stripped))
    scope_lines = [line_no for line_no, _ in scope_bounds]

    def resolved_scope(element):
        idx = bisect.bisect_right(scope_lines, element["line_start"]) - 1
        if idx >= 0:
            return clean_scope(scope_bounds[idx][1])
        return clean_scope(element.get("contract_scope", ""))

    elements = load_jsonl(OUT / "elements.jsonl")
    chunks = load_jsonl(OUT / "chunks.jsonl")
    core = load_jsonl(CORE)

    candidates = []
    seen = set()
    for element in elements:
        if element["element_type"] != "table":
            continue
        scope = resolved_scope(element)
        text = element["text"]
        if "지급사유" not in text or "지급금액" not in text:
            continue
        for line in text.splitlines():
            cells = split_row(line)
            if not cells:
                continue
            benefit, trigger, amount = cells
            if not usable(scope, benefit, trigger, amount):
                continue
            benefit = clean_ocr(benefit.replace("진 단", "진단").replace("급 여", "급여"))
            trigger = clean_ocr(trigger)
            amount = clean_ocr(amount)
            canonical_benefit = re.sub(r"[^0-9A-Za-z가-힣]", "", benefit)
            key = (scope, canonical_benefit)
            if key in seen:
                continue
            local = text.find(line)
            start = element["char_start"] + local
            end = start + len(line)
            if raw[start:end] != line:
                # Some element offsets cover normalized OCR blocks; resolve locally.
                start = raw.find(line, max(0, element["char_start"] - 20), element["char_end"] + 20)
                if start < 0:
                    continue
                end = start + len(line)
            seen.add(key)
            candidates.append({
                "scope": scope, "benefit": benefit, "trigger": trigger, "amount": amount,
                "line": line, "start": start, "end": end, "element_id": element["element_id"],
            })

    # Prefer maximum rider diversity: first pass takes one row per scope, later
    # passes add a second/third row only when needed.
    by_scope = collections.defaultdict(list)
    for candidate in candidates:
        by_scope[candidate["scope"]].append(candidate)
    selected = []
    depth = 0
    scopes = sorted(by_scope, key=lambda s: by_scope[s][0]["start"])
    while len(selected) < N_ROWS:
        added = False
        for scope in scopes:
            if depth < len(by_scope[scope]):
                selected.append(by_scope[scope][depth])
                added = True
                if len(selected) == N_ROWS:
                    break
        if not added:
            break
        depth += 1
    if len(selected) < N_ROWS:
        raise RuntimeError(f"only {len(selected)} usable distinct rows; need {N_ROWS}")

    expansion = []
    existing_questions = {row["question"] for row in core}
    for index, row in enumerate(selected):
        line_start = raw.count("\n", 0, row["start"]) + 1
        line_end = raw.count("\n", 0, row["end"]) + 1
        gold_chunks = sorted({
            c["chunk_id"] for c in chunks
            if c["char_start"] < row["end"] and c["char_end"] > row["start"]
        })
        base = {
            "contract_scope": row["scope"],
            "sources": [{"quote": row["line"], "char_start": row["start"], "char_end": row["end"],
                         "line_start": line_start, "line_end": line_end}],
            "gold_chunk_ids": gold_chunks,
            "gold_element_ids": [row["element_id"]],
            "evidence_complexity": "single",
            "source": "document_coverage_v3_table_row",
            "review_status": "structured_curated_needs_domain_signoff",
            "provenance": "single_document_payment_table_row",
        }
        pair = [
            (trigger_question(row["scope"], row["benefit"], index), row["trigger"],
             labels_for_trigger(row["trigger"]), "paraphrase" if index % 2 else "direct", "trigger"),
            (amount_question(row["scope"], row["benefit"], row["amount"], index), row["amount"],
             labels_for_amount(row["amount"]), "paraphrase", "amount"),
        ]
        for question, answer, types, difficulty, aspect in pair:
            if question in existing_questions:
                raise ValueError(f"duplicate question: {question}")
            existing_questions.add(question)
            item = dict(base)
            item.update({
                "question": question, "answer": answer, "question_types": types,
                "expression_difficulty": difficulty, "table_row_subject": row["benefit"],
                "evaluated_aspect": aspect,
            })
            item["qa_sha256"] = hashlib.sha256((question + "\n" + answer).encode()).hexdigest()[:16]
            expansion.append(item)

    extra_candidates = [
        row for row in selected
        if re.search(r"연간\s*\d+회|1일\s*1회|최초\s*1회|최초1회|\d+년 미만", row["trigger"] + " " + row["amount"])
    ][:N_EXTRA]
    if len(extra_candidates) != N_EXTRA:
        raise RuntimeError(f"only {len(extra_candidates)} extra limit/timing cases")
    for row in extra_candidates:
        question, answer, types = extra_question(row["scope"], row["benefit"], row["trigger"], row["amount"])
        if question in existing_questions:
            raise ValueError(f"duplicate extra question: {question}")
        existing_questions.add(question)
        line_start = raw.count("\n", 0, row["start"]) + 1
        line_end = raw.count("\n", 0, row["end"]) + 1
        item = {
            "contract_scope": row["scope"], "question": question, "answer": answer,
            "question_types": types, "evidence_complexity": "single",
            "expression_difficulty": "implicit", "table_row_subject": row["benefit"],
            "evaluated_aspect": "limit_or_timing",
            "sources": [{"quote": row["line"], "char_start": row["start"], "char_end": row["end"],
                         "line_start": line_start, "line_end": line_end}],
            "gold_chunk_ids": sorted({c["chunk_id"] for c in chunks
                                      if c["char_start"] < row["end"] and c["char_end"] > row["start"]}),
            "gold_element_ids": [row["element_id"]],
            "source": "document_coverage_v3_table_row", "review_status": "manual_reviewed",
            "provenance": "single_document_payment_table_row",
        }
        item["qa_sha256"] = hashlib.sha256((question + "\n" + answer).encode()).hexdigest()[:16]
        expansion.append(item)

    if len(expansion) != N_ROWS * 2 + N_EXTRA:
        raise AssertionError(len(expansion))
    for i, item in enumerate(expansion, len(core) + 1):
        item["qid"] = f"doccov-v3-{i:03d}"

    full = core + expansion
    for path, rows in ((EXPANSION, expansion), (FULL, full)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    field = [row for row in load_jsonl(FIELD_QA) if row.get("신뢰도") == "확정"]
    with COMBINED.open("w", encoding="utf-8") as handle:
        for row in field:
            handle.write(json.dumps({
                "qid": f"field-{row['no']}", "question": row["질문"], "answer": row["정답"],
                "business_type": row.get("사업구분"), "sources": row.get("출처", []),
                "source": "field_qa359_confirmed", "review_status": "field_confirmed",
            }, ensure_ascii=False) + "\n")
        for row in full:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = collections.Counter(t for row in full for t in row["question_types"])
    report = {
        "n_field_confirmed": len(field), "n_coverage_core": len(core),
        "n_v3_expansion": len(expansion), "n_coverage_total": len(full),
        "n_combined": len(field) + len(full), "n_selected_table_rows": len(selected),
        "n_extra_limit_or_timing_questions": N_EXTRA,
        "n_distinct_contract_scopes_in_expansion": len({r["contract_scope"] for r in expansion}),
        "n_distinct_benefits_in_expansion": len({r["table_row_subject"] for r in expansion}),
        "by_type_coverage_total": dict(sorted(by_type.items())),
        "duplicate_questions": len(full) - len({r["question"] for r in full}),
        "combined_inherited_field_duplicate_questions": len(field) - len({r["질문"] for r in field}),
        "all_expansion_sources_exact": all(raw[s["char_start"]:s["char_end"]] == s["quote"]
                                             for r in expansion for s in r["sources"]),
        "all_expansion_gold_targets_present": all(r["gold_chunk_ids"] and r["gold_element_ids"] for r in expansion),
        "single_document_only": str(DOC),
        "domain_signoff_required": True,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
