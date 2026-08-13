#!/usr/bin/env python3
"""EDA for the 350/150 split and coverage comparison against the source document."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qa_curation"))
from eda_question_type_coverage import TYPE_LABELS
from split_qa503_train_test import CONTENT_RULES


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TRAIN = OUT / "정답셋_350_train_유형내용균형_v1.csv"
TEST = OUT / "정답셋_150_test_유형내용균형_v1.csv"
MANIFEST = OUT / "qa500_train_test_split_manifest.jsonl"
DOC_BLOCKS = OUT / "document_question_affordances.jsonl"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
REPORT_JSON = OUT / "qa500_train_test_document_eda.json"
REPORT_MD = OUT / "QA500_TRAIN_TEST_DOCUMENT_EDA.md"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 1)


def length_stats(rows, key):
    values = [len(row[key]) for row in rows]
    return {
        "mean": round(sum(values) / len(values), 1),
        "p25": percentile(values, .25), "median": percentile(values, .5),
        "p75": percentile(values, .75), "p95": percentile(values, .95),
        "min": min(values), "max": max(values),
    }


def prevalence(count, denominator):
    return count / denominator if denominator else 0.0


def vector_similarity(a_counts, a_n, b_counts, b_n, labels):
    a = [prevalence(a_counts[x], a_n) for x in labels]
    b = [prevalence(b_counts[x], b_n) for x in labels]
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
    diffs = [abs(x-y) for x, y in zip(a, b)]
    return {
        "cosine_similarity": round(dot / norm, 4) if norm else 0,
        "mean_absolute_prevalence_diff_pp": round(sum(diffs) / len(diffs) * 100, 2),
        "max_absolute_prevalence_diff_pp": round(max(diffs) * 100, 2),
        "max_diff_label": labels[diffs.index(max(diffs))],
    }


def duplicate_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["질문"].strip()].append(row["no"])
    duplicates = {q: nos for q, nos in groups.items() if len(nos) > 1}
    return {"duplicate_question_groups": len(duplicates),
            "duplicate_excess_rows": sum(len(x)-1 for x in duplicates.values()),
            "groups": duplicates}


def pct(v):
    return f"{v*100:.1f}%"


def main():
    train, test = read_csv(TRAIN), read_csv(TEST)
    manifest = read_jsonl(MANIFEST)
    doc_blocks = read_jsonl(DOC_BLOCKS)
    assert len(train) == 350 and len(test) == 150 and len(manifest) == 500

    split_meta = defaultdict(list)
    for row in manifest:
        split_meta[row["split"]].append(row)

    type_counts = {
        split: Counter(t for row in rows for t in row["question_types"])
        for split, rows in split_meta.items()
    }
    content_counts = {
        split: Counter(t for row in rows for t in row["content_labels"])
        for split, rows in split_meta.items()
    }
    doc_type_counts = Counter(t for row in doc_blocks for t in row["question_affordances"])

    raw = DOC.read_text()
    lines = raw.splitlines()
    doc_content_counts = Counter()
    for block in doc_blocks:
        text = "\n".join(lines[block["line_start"]-1:block["line_end"]])
        for label, pattern in CONTENT_RULES.items():
            if re.search(pattern, text, re.I):
                doc_content_counts[label] += 1

    type_labels = sorted(set(doc_type_counts) | set(type_counts["train"]) | set(type_counts["test"]))
    content_labels = sorted(set(doc_content_counts) | set(content_counts["train"]) | set(content_counts["test"]))
    train_test_type = vector_similarity(type_counts["train"], 350, type_counts["test"], 150, type_labels)
    train_test_content = vector_similarity(content_counts["train"], 350, content_counts["test"], 150, content_labels)
    train_doc_type = vector_similarity(type_counts["train"], 350, doc_type_counts, len(doc_blocks), type_labels)
    train_doc_content = vector_similarity(content_counts["train"], 350, doc_content_counts, len(doc_blocks), content_labels)

    train_questions = {r["질문"].strip() for r in train}
    test_questions = {r["질문"].strip() for r in test}
    type_doc_present = {x for x, n in doc_type_counts.items() if n}
    type_train_present = {x for x, n in type_counts["train"].items() if n}
    type_test_present = {x for x, n in type_counts["test"].items() if n}
    content_doc_present = {x for x, n in doc_content_counts.items() if n}

    report = {
        "summary": {
            "document_count": 1, "document_blocks": len(doc_blocks),
            "train_rows": len(train), "test_rows": len(test),
            "train_test_row_id_overlap": len({r['no'] for r in train} & {r['no'] for r in test}),
            "train_test_exact_question_overlap": len(train_questions & test_questions),
        },
        "similarity": {
            "train_vs_test_question_type": train_test_type,
            "train_vs_test_content": train_test_content,
            "train_vs_document_affordance_type_screening": train_doc_type,
            "train_vs_document_content_screening": train_doc_content,
        },
        "coverage": {
            "document_affordance_types": sorted(type_doc_present),
            "train_types": sorted(type_train_present),
            "test_types": sorted(type_test_present),
            "document_types_missing_in_train": sorted(type_doc_present-type_train_present),
            "document_types_missing_in_test": sorted(type_doc_present-type_test_present),
            "train_types_not_afforded_by_single_document": sorted(type_train_present-type_doc_present),
            "document_content_topics": sorted(content_doc_present),
            "document_content_missing_in_train": sorted(content_doc_present-set(content_counts['train'])),
            "document_content_missing_in_test": sorted(content_doc_present-set(content_counts['test'])),
        },
        "lengths": {
            "train": {k: length_stats(train, k) for k in ("질문", "정답", "출처")},
            "test": {k: length_stats(test, k) for k in ("질문", "정답", "출처")},
        },
        "origin": {
            split: dict(Counter(row["origin"] for row in rows)) for split, rows in split_meta.items()
        },
        "business_type": {
            split: dict(Counter(row["business_type"] for row in rows)) for split, rows in split_meta.items()
        },
        "duplicates": {
            "train": duplicate_summary(train), "test": duplicate_summary(test),
        },
        "type_counts": {"train": dict(type_counts["train"]), "test": dict(type_counts["test"]),
                        "document_blocks": dict(doc_type_counts)},
        "content_counts": {"train": dict(content_counts["train"]), "test": dict(content_counts["test"]),
                           "document_blocks": dict(doc_content_counts)},
        "method_note": "Document ratios count article/appendix blocks with an affordance/topic. Repeated policy clauses inflate block counts, so train-vs-document proportions are screening signals, not target ratios.",
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    md = []
    md.append("# QA 500 Train/Test 및 원문 문서 EDA\n")
    md.append("## 결론\n")
    md.append(f"- Train/Test 유형 분포 cosine similarity: `{train_test_type['cosine_similarity']}`; 평균 절대 차이 `{train_test_type['mean_absolute_prevalence_diff_pp']}%p`.")
    md.append(f"- Train/Test 내용 분포 cosine similarity: `{train_test_content['cosine_similarity']}`; 평균 절대 차이 `{train_test_content['mean_absolute_prevalence_diff_pp']}%p`.")
    md.append(f"- Train/Test 동일 질문 교집합: `{len(train_questions & test_questions)}`건.")
    md.append(f"- 문서가 제공하는 질문 유형 중 Train 미포함: `{', '.join(sorted(type_doc_present-type_train_present)) or '없음'}`.")
    md.append(f"- 문서 내용 주제 중 Train 미포함: `{', '.join(sorted(content_doc_present-set(content_counts['train']))) or '없음'}`.\n")

    md.append("## 질문 유형 분포\n")
    md.append("| 유형 | Train | Test | 차이(%p) | 문서 블록 |")
    md.append("|---|---:|---:|---:|---:|")
    for label in type_labels:
        tr, te, dc = type_counts['train'][label], type_counts['test'][label], doc_type_counts[label]
        diff = (tr/350-te/150)*100
        md.append(f"| {TYPE_LABELS.get(label,label)} (`{label}`) | {tr} ({pct(tr/350)}) | {te} ({pct(te/150)}) | {diff:+.2f} | {dc} ({pct(dc/len(doc_blocks))}) |")

    md.append("\n## 내용 주제 분포\n")
    md.append("| 내용 주제 | Train | Test | 차이(%p) | 문서 블록 |")
    md.append("|---|---:|---:|---:|---:|")
    for label in content_labels:
        tr, te, dc = content_counts['train'][label], content_counts['test'][label], doc_content_counts[label]
        diff = (tr/350-te/150)*100
        md.append(f"| `{label}` | {tr} ({pct(tr/350)}) | {te} ({pct(te/150)}) | {diff:+.2f} | {dc} ({pct(dc/len(doc_blocks))}) |")

    md.append("\n## 길이 분포\n")
    md.append("| 필드 | Split | 평균 | 중앙값 | P95 | 최소 | 최대 |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for key in ("질문", "정답", "출처"):
        for split in ("train", "test"):
            s = report['lengths'][split][key]
            md.append(f"| {key} | {split} | {s['mean']} | {s['median']} | {s['p95']} | {s['min']} | {s['max']} |")

    md.append("\n## 해석 주의\n")
    md.append("문서 블록 비율은 약관에서 같은 조항·지급표가 반복되는 횟수의 영향을 받습니다. 따라서 QA 분포가 문서 블록 비율과 정확히 같아야 하는 것은 아닙니다. 이 비교는 누락된 유형과 내용 주제를 찾는 용도로 사용합니다.")
    REPORT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({"summary": report["summary"], "similarity": report["similarity"],
                      "coverage": report["coverage"], "lengths": report["lengths"],
                      "duplicates": report["duplicates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
