#!/usr/bin/env python3
"""Split the reviewed QA CSV by independent rows with balanced label/topic shares."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
INPUT_CSV = OUT / "정답셋_503_현업303_문서보강200_v2_직접검수.csv"
INPUT_JSON = OUT / "qa503_field_plus_document_coverage_v3_reviewed.jsonl"
FIELD_TYPES = OUT / "qa359_question_types.jsonl"
TRAIN_CSV = OUT / "정답셋_350_train_유형내용균형_v1.csv"
TEST_CSV = OUT / "정답셋_150_test_유형내용균형_v1.csv"
EXCLUDED_CSV = OUT / "정답셋_3_분할제외_완전중복.csv"
MANIFEST = OUT / "qa500_train_test_split_manifest.jsonl"
REPORT = OUT / "qa500_train_test_split_report.json"

FIELDS = ["no", "사업구분", "질문", "정답", "신뢰도", "출처", "방법"]
EXCLUDED_QIDS = {"field-339", "field-340", "field-341"}


CONTENT_RULES = {
    "cancer_tumor": r"암|종양|항암|다빈치|NGS",
    "brain_heart_circulatory": r"뇌|심장|심근|혈관|순환계|부정맥|혈전",
    "surgery": r"수술|수술술",
    "hospital_outpatient_care": r"입원|통원|간병|중환자실|병실|병원|의료기관",
    "diagnosis_test_code": r"진단|검사|생검|병리|MRI|CT|PET|초음파|분류코드|질병코드|수가코드",
    "treatment_drug_rehab": r"치료|약물|방사선|재활|호르몬|혈전용해|인공호흡기|신대체요법",
    "disability_accident_death": r"장해|재해|골절|사망|상해",
    "contract_premium_refund": r"계약|보험료|납입|해지|해약|환급|갱신|무효|철회|취소|소멸|연금",
    "claim_documents_process": r"청구|구비서류|제출|지급절차|신청",
    "product_underwriting": r"상품|특약리스트|특약 목록|가입가능|가입 가능|가입연령|가입나이|인수|심사형",
    "other_specific_disease": r"당뇨|고혈압|통풍|신부전|간질환|폐질환|치매|대상포진|갑상선|신장질환",
    "rules_definition_interpretation": r"정의|기준|약관|해석|분류|예외|제외|면책|보장개시일",
}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def content_labels(row: dict, rich: dict) -> list[str]:
    text = " ".join([
        row["질문"], row["정답"],
        str(rich.get("contract_scope", "")),
        str(rich.get("table_row_subject", "")),
    ])
    labels = [name for name, pattern in CONTENT_RULES.items() if re.search(pattern, text, re.I)]
    return labels or ["general_other"]


def score(test_indices: set[int], row_labels: list[set[str]], targets: dict[str, int],
          duplicate_groups: list[set[int]]) -> float:
    counts = Counter(label for i in test_indices for label in row_labels[i])
    value = 0.0
    for label, target in targets.items():
        # Rare labels matter, but one-item rounding differences must not dominate.
        value += ((counts[label] - target) ** 2) / max(2.0, target)
    # Exact duplicate questions must not appear on both sides of the split.
    for group in duplicate_groups:
        in_test = len(group & test_indices)
        if 0 < in_test < len(group):
            value += 1000.0
    return value


def balanced_test_indices(row_labels: list[set[str]], duplicate_groups: list[set[int]],
                          seed: int = 260507) -> set[int]:
    n = len(row_labels)
    test_n = 150
    totals = Counter(label for labels in row_labels for label in labels)
    targets = {label: round(count * test_n / n) for label, count in totals.items()}
    rng = random.Random(seed)
    best_set = None
    best_score = math.inf

    # Row-level split: swaps always exchange exactly one row, so 350/150 is fixed.
    for restart in range(24):
        current = set(rng.sample(range(n), test_n))
        current_score = score(current, row_labels, targets, duplicate_groups)
        for _ in range(18000):
            a = rng.choice(tuple(current))
            b = rng.choice(tuple(set(range(n)) - current))
            candidate = current.copy()
            candidate.remove(a)
            candidate.add(b)
            candidate_score = score(candidate, row_labels, targets, duplicate_groups)
            if candidate_score < current_score:
                current, current_score = candidate, candidate_score
                if current_score == 0:
                    break
        if current_score < best_score:
            best_set, best_score = current, current_score
        if best_score == 0:
            break
    assert best_set is not None and len(best_set) == test_n
    return best_set


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def distribution(rows_meta: list[dict], key: str) -> dict:
    counts = Counter(v for row in rows_meta for v in row[key])
    return dict(sorted(counts.items()))


def main():
    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    rich_rows = load_jsonl(INPUT_JSON)
    assert len(csv_rows) == len(rich_rows) == 503
    assert all(a["질문"] == b["question"] for a, b in zip(csv_rows, rich_rows))

    type_by_field_qid = {
        f"field-{row['qid']}": row["question_types"] for row in load_jsonl(FIELD_TYPES)
    }
    records = []
    excluded = []
    for csv_row, rich in zip(csv_rows, rich_rows):
        qid = rich["qid"]
        if qid in EXCLUDED_QIDS:
            excluded.append(csv_row)
            continue
        qtypes = rich.get("question_types") or type_by_field_qid[qid]
        contents = content_labels(csv_row, rich)
        origin = "field" if qid.startswith("field-") else "document_coverage"
        business = rich.get("business_type") or "document_coverage"
        labels = (
            {f"type:{x}" for x in qtypes}
            | {f"content:{x}" for x in contents}
            | {f"origin:{origin}", f"business:{business}"}
        )
        records.append({
            "csv": csv_row, "qid": qid, "question_types": qtypes,
            "content_labels": contents, "origin": origin,
            "business_type": business, "labels": labels,
        })

    assert len(records) == 500 and len(excluded) == 3
    question_groups = {}
    for i, row in enumerate(records):
        question_groups.setdefault(row["csv"]["질문"].strip(), set()).add(i)
    duplicate_groups = [indices for indices in question_groups.values() if len(indices) > 1]
    test_indices = balanced_test_indices([row["labels"] for row in records], duplicate_groups)
    train = [row for i, row in enumerate(records) if i not in test_indices]
    test = [row for i, row in enumerate(records) if i in test_indices]
    assert len(train) == 350 and len(test) == 150

    write_csv(TRAIN_CSV, [row["csv"] for row in train])
    write_csv(TEST_CSV, [row["csv"] for row in test])
    write_csv(EXCLUDED_CSV, excluded)

    with MANIFEST.open("w") as f:
        for split, rows in (("train", train), ("test", test)):
            for row in rows:
                f.write(json.dumps({
                    "qid": row["qid"], "no": row["csv"]["no"], "split": split,
                    "question_types": row["question_types"],
                    "content_labels": row["content_labels"],
                    "origin": row["origin"], "business_type": row["business_type"],
                }, ensure_ascii=False) + "\n")

    all_meta = train + test
    report = {
        "unit": "independent_csv_row",
        "input": 503,
        "excluded_exact_duplicates": 3,
        "train": len(train),
        "test": len(test),
        "csv_schema": FIELDS,
        "excluded_qids": sorted(EXCLUDED_QIDS),
        "type_distribution": {
            "all": distribution(all_meta, "question_types"),
            "train": distribution(train, "question_types"),
            "test": distribution(test, "question_types"),
        },
        "content_distribution": {
            "all": distribution(all_meta, "content_labels"),
            "train": distribution(train, "content_labels"),
            "test": distribution(test, "content_labels"),
        },
        "origin_distribution": {
            "all": dict(Counter(r["origin"] for r in all_meta)),
            "train": dict(Counter(r["origin"] for r in train)),
            "test": dict(Counter(r["origin"] for r in test)),
        },
        "business_distribution": {
            "all": dict(Counter(r["business_type"] for r in all_meta)),
            "train": dict(Counter(r["business_type"] for r in train)),
            "test": dict(Counter(r["business_type"] for r in test)),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
