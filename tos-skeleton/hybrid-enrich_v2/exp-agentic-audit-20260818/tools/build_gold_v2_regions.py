#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""noah 통합 gold(0814, coord_space=raw) → v2: region 병합 + 핵심/보조 등급.

입력  : out/gold_train.jsonl (173) · out/gold_test.jsonl (67)  [noah_test_0814]
원문  : DOC_PATH 환경변수 (sha256 앞 16자리 = 40a471d54b2ec8c8 이어야 함)
출력  : out/gold_train_v2_regions.jsonl · out/gold_test_v2_regions.jsonl
        out/gold_v2_build_report.json (통계 + 검수 대기열)

규칙 (LLM 0회, 전부 결정론)
  [병합] span을 정렬 후 gap ≤ 200자면 하나의 evidence region으로 병합.
         근거: 전체 인접 gap 598개 중 287개가 ≤200 (목록 항목·같은 조항 블록).
  [등급] region 단위로 required / supporting:
    a) fragment  : 텍스트 <45자이고 문장 종결(다.)이 아님 → supporting
                   (예: "[일반심사형]", "1회당, 1일 1회에 한함", 조항 제목 단독)
    b) noise     : 한글 비율 <50%이고 30자 이상 → supporting + 검수 플래그
                   (파서가 삽입한 영어 필러 텍스트 등)
    c) outlier   : region이 3개 이상이고, 최근접 region과 30,000자 이상
                   떨어진 비최장 region → supporting + 검수 플래그
                   (예: 본문 조항 근거 + 문서 반대편 용어해설 조합)
    d) 나머지     → required
    e) required가 0개면 최장 region을 required로 승격 (문항당 최소 1개 보장)

주의: c)는 자동 강등이므로 검수 대기열(review_queue)에서 사람이 재승격 여부를
확인해야 한다. supporting은 버려지는 게 아니라 부지표용으로 보존된다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"

MERGE_GAP = 200
FRAGMENT_LEN = 45
OUTLIER_DIST = 30_000
NOISE_HANGUL_RATIO = 0.5
EXPECTED_DOC_SHA = "40a471d54b2ec8c8"

DOC_PATH = os.environ.get("DOC_PATH", "")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge_spans(spans: list[list[int]]) -> list[list[list[int]]]:
    """정렬 → gap ≤ MERGE_GAP 이면 같은 region. region = 원본 span 목록."""
    ordered = sorted((tuple(s) for s in spans))
    regions: list[list[list[int]]] = []
    for st, en in ordered:
        if regions and st - regions[-1][-1][1] <= MERGE_GAP:
            regions[-1].append([st, en])
        else:
            regions.append([[st, en]])
    return regions


def hangul_ratio(text: str) -> float:
    if not text:
        return 1.0
    body = re.sub(r"\s", "", text)
    if not body:
        return 1.0
    return sum(1 for c in body if "가" <= c <= "힣") / len(body)


def grade_regions(regions: list[dict]) -> list[dict]:
    longest = max(regions, key=lambda r: r["char_end"] - r["char_start"])
    for r in regions:
        text = r["text"]
        stripped = text.strip()
        if len(stripped) < FRAGMENT_LEN and not re.search(r"(다|니다)\s*\.?\s*$", stripped):
            r["required"], r["grade_reason"] = False, "fragment"
        elif len(stripped) >= 30 and hangul_ratio(stripped) < NOISE_HANGUL_RATIO:
            r["required"], r["grade_reason"] = False, "parser_noise"
            r["needs_review"] = True
        else:
            r["required"], r["grade_reason"] = True, "default"
    if len(regions) >= 3:
        for r in regions:
            if not r["required"] or r is longest:
                continue
            dist = min(
                abs(r["char_start"] - o["char_end"]) if r["char_start"] > o["char_end"]
                else abs(o["char_start"] - r["char_end"])
                for o in regions if o is not r
            )
            if dist > OUTLIER_DIST:
                r["required"], r["grade_reason"] = False, "far_outlier"
                r["needs_review"] = True
    if not any(r["required"] for r in regions):
        longest["required"], longest["grade_reason"] = True, "promoted_longest"
    return regions


def build(rows: list[dict], doc: str, split: str) -> tuple[list[dict], list[dict]]:
    out_rows, review = [], []
    for g in rows:
        regions = []
        for i, span_group in enumerate(merge_spans(g["gold_spans"])):
            st, en = span_group[0][0], span_group[-1][1]
            regions.append({
                "group_id": i,
                "spans": span_group,
                "char_start": st,
                "char_end": en,
                "text": doc[st:en],
            })
        regions = grade_regions(regions)
        row = {
            "qid": g["qid"],
            "question": g["question"],
            "task_type": g.get("task_type"),
            "core_retrieval": g.get("core_retrieval"),
            "사업구분": g.get("사업구분"),
            "coord_space": "raw",
            "evidence_groups": [
                {k: r[k] for k in ("group_id", "spans", "char_start", "char_end",
                                    "required", "grade_reason")}
                | {"preview": r["text"][:80].replace("\n", " ")}
                for r in regions
            ],
            "n_groups": len(regions),
            "n_required": sum(1 for r in regions if r["required"]),
            "n_spans_v1": g["n_spans"],
            "source": f"noah gold_{split}.jsonl (noah_test_0814)",
            "builder": f"v2 regions: merge_gap<={MERGE_GAP}, grades a-e (build_gold_v2_regions.py)",
        }
        flags = [r for r in regions if r.get("needs_review")]
        if flags:
            row["needs_review"] = True
            review.append({
                "qid": g["qid"], "split": split, "question": g["question"][:60],
                "flagged": [{"group_id": r["group_id"], "reason": r["grade_reason"],
                             "preview": r["text"][:70].replace("\n", " ")} for r in flags],
            })
        out_rows.append(row)
    return out_rows, review


def main() -> None:
    doc = open(DOC_PATH, encoding="utf-8").read()
    sha = hashlib.sha256(doc.encode()).hexdigest()[:16]
    if sha != EXPECTED_DOC_SHA:
        sys.exit(f"DOC_PATH sha 불일치: {sha} (기대 {EXPECTED_DOC_SHA})")

    report = {"doc_sha256": sha, "merge_gap": MERGE_GAP, "splits": {}, "review_queue": []}
    for split in ("train", "test"):
        rows = load_jsonl(OUT / f"gold_{split}.jsonl")
        v2, review = build(rows, doc, split)
        out_path = OUT / f"gold_{split}_v2_regions.jsonl"
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in v2) + "\n",
                            encoding="utf-8")
        n_spans = sum(r["n_spans_v1"] for r in v2)
        n_groups = sum(r["n_groups"] for r in v2)
        n_req = sum(r["n_required"] for r in v2)
        reasons: dict[str, int] = {}
        for r in v2:
            for grp in r["evidence_groups"]:
                reasons[grp["grade_reason"]] = reasons.get(grp["grade_reason"], 0) + 1
        report["splits"][split] = {
            "n_questions": len(v2), "n_spans_v1": n_spans, "n_regions_v2": n_groups,
            "n_required": n_req, "n_supporting": n_groups - n_req,
            "grade_reasons": reasons,
            "n_flagged_questions": sum(1 for r in v2 if r.get("needs_review")),
        }
        report["review_queue"].extend(review)
        print(f"[{split}] 문항 {len(v2)} · span {n_spans} → region {n_groups} "
              f"(required {n_req} / supporting {n_groups - n_req}) → {out_path.name}")

    (OUT / "gold_v2_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report -> out/gold_v2_build_report.json")


if __name__ == "__main__":
    main()
