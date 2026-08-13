#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""noah 제공 v3 QA셋(497문항, train348+test149)을 elements_semtag.jsonl 기준으로
gold 매핑. 출처 필드가 줄번호 JSON이 아니라 원문 인용 텍스트라, 원문 .md에서
정규화된 텍스트로 위치(줄번호)를 먼저 찾은 뒤, 기존 line-overlap 매칭 로직 재사용."""
import json, re, csv
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)
OUT = f"{BASE}/out"
NOAH = f"{BASE}/noah_qaset"


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def ns(s):
    return re.sub(r"\s+", "", s)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_doc_index(doc_path):
    """줄 단위로 원문을 읽어, 정규화(공백제거)된 전체 문자열과
    '정규화 문자열 상 위치 -> 줄번호' 매핑을 만든다."""
    lines = open(doc_path, encoding="utf-8").read().split("\n")
    norm_full = []
    offset_to_line = []  # offset_to_line[i] = 그 줄의 (start_offset, end_offset, line_no)
    cur = 0
    for i, line in enumerate(lines, start=1):
        nline = ns(line)
        if nline:
            norm_full.append(nline)
            offset_to_line.append((cur, cur + len(nline), i))
            cur += len(nline)
    return "".join(norm_full), offset_to_line


def offset_to_lineno(offset, offset_to_line):
    lo, hi = 0, len(offset_to_line) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, ln = offset_to_line[mid]
        if offset < s:
            hi = mid - 1
        elif offset >= e:
            lo = mid + 1
        else:
            return ln
    # 범위 밖이면 가장 가까운 줄 반환
    if offset_to_line:
        return offset_to_line[min(max(lo, 0), len(offset_to_line) - 1)][2]
    return None


def find_span(quote, norm_doc, offset_to_line):
    nq = ns(quote)
    if not nq:
        return None
    idx = norm_doc.find(nq)
    if idx == -1:
        # 정확매칭 실패 -> 앞부분 60%로 재시도
        shorter = nq[: max(20, int(len(nq) * 0.6))]
        idx = norm_doc.find(shorter)
        if idx == -1:
            return None
        end = idx + len(shorter)
    else:
        end = idx + len(nq)
    ls = offset_to_lineno(idx, offset_to_line)
    le = offset_to_lineno(end - 1, offset_to_line)
    return ls, le


def map_gold(ls, le, elements_sorted):
    mid = (ls + le) / 2
    best, best_overlap = None, -1
    for e in elements_sorted:
        els, ele = e.get("line_start"), e.get("line_end")
        if els is None or ele is None:
            continue
        overlap = min(le, ele) - max(ls, els)
        if overlap > best_overlap:
            best_overlap = overlap
            best = e
        elif overlap == best_overlap and best is not None:
            if els <= mid <= ele:
                best = e
    return best


def main():
    elements = load_jsonl(f"{OUT}/elements_semtag.jsonl")
    elements_sorted = sorted(elements, key=lambda e: e.get("line_start", 0))

    doc_path = f"{NOAH}/판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
    norm_doc, offset_to_line = build_doc_index(doc_path)
    print(f"원문 정규화 완료: {len(norm_doc)}자, {len(offset_to_line)}줄", flush=True)

    rows = []
    for fn in ["정답셋_149_test_v3_retrieval_250212.csv", "정답셋_348_train_v3_retrieval_250212.csv"]:
        rows.extend(load_csv(f"{NOAH}/{fn}"))
    print(f"QA 전체 {len(rows)}건 로드", flush=True)

    out = []
    n_matched, n_failed = 0, 0
    for r in rows:
        quote = r.get("출처", "")
        span = find_span(quote, norm_doc, offset_to_line)
        gold_ids = []
        if span:
            ls, le = span
            best = map_gold(ls, le, elements_sorted)
            if best:
                gold_ids.append(best["element_id"])
        if gold_ids:
            n_matched += 1
        else:
            n_failed += 1
        out.append({
            "qid": r["qid"],
            "q": r["질문"],
            "gold": gold_ids,
            "task_type": r.get("task_type", ""),
            "core_retrieval": r.get("core_retrieval", ""),
            "사업구분": r.get("사업구분", ""),
        })

    with open(f"{OUT}/gold_mapped_noah_v3.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"매칭 성공 {n_matched}건, 실패 {n_failed}건", flush=True)
    print("저장: gold_mapped_noah_v3.jsonl", flush=True)


if __name__ == "__main__":
    main()
