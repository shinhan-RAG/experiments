#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""156건(gold매칭 실패)에 대해 출처를 \n\n 구간별로 쪼개서 각각 위치를 찾아
multi-gold를 구성. not_answerable(15건, 출처 0자)은 그대로 gold=[] 유지."""
import json, re, csv

BASE = "/home/work/source/embed_exp/dr-dci-lsh/tos-skeleton/hybrid-enrich"
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
    lines = open(doc_path, encoding="utf-8").read().split("\n")
    norm_full = []
    offset_to_line = []
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
    if offset_to_line:
        return offset_to_line[min(max(lo, 0), len(offset_to_line) - 1)][2]
    return None


def find_span(quote, norm_doc, offset_to_line):
    nq = ns(quote)
    if len(nq) < 8:  # 너무 짧은 조각은 오탐 위험 커서 스킵
        return None
    idx = norm_doc.find(nq)
    if idx == -1:
        shorter = nq[: max(20, int(len(nq) * 0.6))]
        if len(shorter) < 8:
            return None
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
    print(f"원문 정규화 완료: {len(norm_doc)}자", flush=True)

    rows = []
    for fn in ["정답셋_149_test_v3_retrieval_250212.csv", "정답셋_348_train_v3_retrieval_250212.csv"]:
        rows.extend(load_csv(f"{NOAH}/{fn}"))
    by_qid = {r["qid"]: r for r in rows}

    # 기존 341건 매칭성공 gold 불러오기(그대로 유지)
    existing = {r["qid"]: r for r in load_jsonl(f"{OUT}/gold_mapped_noah_v3.jsonl")}
    failed_qids = [qid for qid, r in existing.items() if not r["gold"]]
    print(f"multi-gold 시도 대상: {len(failed_qids)}건", flush=True)

    out = []
    n_multi_built = 0
    for qid, r in existing.items():
        if qid not in failed_qids:
            out.append(r)
            continue
        row = by_qid[qid]
        quote = row.get("출처", "")
        segs = [s for s in quote.split("\n\n") if s.strip()]
        gold_ids = []
        seen = set()
        for seg in segs:
            span = find_span(seg, norm_doc, offset_to_line)
            if span:
                ls, le = span
                best = map_gold(ls, le, elements_sorted)
                if best and best["element_id"] not in seen:
                    seen.add(best["element_id"])
                    gold_ids.append(best["element_id"])
        if gold_ids:
            n_multi_built += 1
        out.append({
            "qid": qid, "q": row["질문"], "gold": gold_ids,
            "task_type": row.get("task_type", ""), "core_retrieval": row.get("core_retrieval", ""),
            "사업구분": row.get("사업구분", ""),
        })

    with open(f"{OUT}/gold_mapped_noah_v3_multigold.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"기존 실패 156건 중 multi-gold 구성 성공: {n_multi_built}건", flush=True)
    still_empty = 156 - n_multi_built
    print(f"여전히 gold 없음: {still_empty}건 (not_answerable 등)", flush=True)
    print("저장: gold_mapped_noah_v3_multigold.jsonl", flush=True)


if __name__ == "__main__":
    main()
