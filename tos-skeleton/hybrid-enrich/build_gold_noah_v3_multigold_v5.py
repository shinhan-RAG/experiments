#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3: v2에 이어, '순수 조항 제목만' 또는 '문서 전역에 5개+ 서로 다른 특약에서
토씨 하나 안 틀리고 반복되는 보일러플레이트' 세그먼트는 gold 후보에서 아예 제외.
이런 세그먼트는 특정 특약을 가리키는 정보가 없어 어느 occurrence를 골라도
검색 관점에서 의미가 없다 -- 추측하지 않고 통째로 버린다."""
import json, re, csv

BASE = "/home/work/source/embed_exp/dr-dci-lsh/tos-skeleton/hybrid-enrich"
OUT = f"{BASE}/out"
NOAH = f"{BASE}/noah_qaset"

BARE_TITLE_RE = re.compile(r"^제\s?\d+(?:\s?-\s?\d+)?\s?조(?:\s?의\s?\d+)?\s*[\(\)\wㄱ-힣·,\s]{0,25}$")
MANY_SCOPE_THRESHOLD = 5  # 이 이상 서로 다른 특약에서 나오면 보일러플레이트로 간주


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def ns(s):
    return re.sub(r"\s+", "", s)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


PAGEBREAK_HEADER_RE = re.compile(r"^#?\s*SHINHAN LIFE\s*$", re.I)
PAGE_NUM_RE = re.compile(r"^\d+$")
SEPARATOR_RE = re.compile(r"^-{3,}$")


def find_pagebreak_skip_lines(lines):
    """문장 중간에 끼어드는 'SHINHAN LIFE / 페이지번호 / ---' 3줄 블록을 찾아
    (원본 line index, 1-based) 집합으로 반환 -- 이 줄들은 정규화에서 제외한다."""
    skip = set()
    n = len(lines)
    for i in range(n):
        if not PAGEBREAK_HEADER_RE.match(lines[i].strip()):
            continue
        j = i + 1
        found_num = None
        while j < n and j < i + 4:
            s = lines[j].strip()
            if s == "":
                j += 1
                continue
            if PAGE_NUM_RE.match(s):
                found_num = j
            break
        if found_num is None:
            continue
        k = found_num + 1
        found_sep = None
        while k < n and k < found_num + 4:
            s = lines[k].strip()
            if s == "":
                k += 1
                continue
            if SEPARATOR_RE.match(s):
                found_sep = k
            break
        if found_sep is None:
            continue
        for idx in range(i, found_sep + 1):
            skip.add(idx)
    return skip


def build_doc_index(doc_path):
    lines = open(doc_path, encoding="utf-8").read().split("\n")
    skip_idx = find_pagebreak_skip_lines(lines)
    norm_full = []
    offset_to_line = []
    cur = 0
    for i, line in enumerate(lines, start=1):
        if (i - 1) in skip_idx:
            continue  # 페이지 넘김 보일러플레이트 -> 정규화에서 제외(문장 이어붙이기)
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


def find_all_spans(quote, norm_doc, offset_to_line, max_hits=30):
    nq = ns(quote)
    if len(nq) < 8:
        return []
    spans = []
    start = 0
    n_hits = 0
    while n_hits < max_hits:
        idx = norm_doc.find(nq, start)
        if idx == -1:
            break
        end = idx + len(nq)
        ls = offset_to_lineno(idx, offset_to_line)
        le = offset_to_lineno(end - 1, offset_to_line)
        spans.append((ls, le))
        start = idx + 1
        n_hits += 1
    if spans:
        return spans
    shorter = nq[: max(20, int(len(nq) * 0.6))]
    if len(shorter) < 8:
        return []
    idx = norm_doc.find(shorter)
    if idx == -1:
        return []
    end = idx + len(shorter)
    ls = offset_to_lineno(idx, offset_to_line)
    le = offset_to_lineno(end - 1, offset_to_line)
    return [(ls, le)]


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


def scope_core(s):
    s = re.sub(r"\(무배당[^)]*\)", "", s or "")
    s = re.sub(r"^\(간편\)", "", s)
    return s.strip()


def build_scope_index(elements):
    cores = {}
    for e in elements:
        s = e.get("contract_scope", "")
        if s and s != "주계약":
            c = scope_core(s)
            if len(c) >= 4:
                cores.setdefault(c, s)
    return sorted(cores.items(), key=lambda x: -len(x[0]))


def question_scope_hint(question, scope_cores):
    for core, full in scope_cores:
        if core in question:
            return full
    # 대괄호 수식어([기본]/[삭감없음용] 등)를 뗀 형태로 재시도 -- 질문은 보통 이걸 생략한다
    stripped_cores = []
    for core, full in scope_cores:
        stripped = re.sub(r"^\[[^\]]+\]", "", core)
        if stripped != core and len(stripped) >= 4:
            stripped_cores.append((stripped, full))
    stripped_cores.sort(key=lambda x: -len(x[0]))
    for stripped, full in stripped_cores:
        if stripped in question:
            return full
    return None


def main():
    elements = load_jsonl(f"{OUT}/elements_semtag.jsonl")
    elements_sorted = sorted(elements, key=lambda e: e.get("line_start", 0))
    scope_cores = build_scope_index(elements)

    doc_path = f"{NOAH}/판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
    norm_doc, offset_to_line = build_doc_index(doc_path)
    print(f"원문 정규화 완료: {len(norm_doc)}자", flush=True)

    rows = []
    for fn in ["정답셋_149_test_v3_retrieval_250212.csv", "정답셋_348_train_v3_retrieval_250212.csv"]:
        rows.extend(load_csv(f"{NOAH}/{fn}"))
    by_qid = {r["qid"]: r for r in rows}

    existing = {r["qid"]: r for r in load_jsonl(f"{OUT}/gold_mapped_noah_v3.jsonl")}
    failed_qids = [qid for qid, r in existing.items() if not r["gold"]]
    print(f"multi-gold 시도 대상: {len(failed_qids)}건", flush=True)

    out = []
    n_multi_built = 0
    n_dropped_boilerplate = 0
    n_dropped_ambiguous = 0
    for qid, r in existing.items():
        if qid not in failed_qids:
            out.append(r)
            continue
        row = by_qid[qid]
        quote = row.get("출처", "")
        segs = [s for s in quote.split("\n\n") if s.strip()]

        seg_candidates = []
        for seg in segs:
            nseg = ns(seg)
            if BARE_TITLE_RE.match(nseg):
                n_dropped_boilerplate += 1
                continue  # 순수 조항 제목만 -> 정보 없음, 아예 스킵
            spans = find_all_spans(seg, norm_doc, offset_to_line)
            cands = []
            seen_eid = set()
            for ls, le in spans:
                best = map_gold(ls, le, elements_sorted)
                if best and best["element_id"] not in seen_eid:
                    seen_eid.add(best["element_id"])
                    cands.append(best)
            seg_candidates.append((seg, cands))

        confirmed = []
        ambiguous = []
        for seg, cands in seg_candidates:
            if not cands:
                continue
            scopes = set(c.get("contract_scope", "") for c in cands)
            if len(scopes) == 1:
                confirmed.append(cands[0])
            else:
                # 여러 특약(5개+ 포함, 보일러플레이트 후보)도 곧바로 버리지 않고
                # 질문의 특약명 힌트/이 질문의 다른 확정 세그먼트로 우선 해소를 시도한다.
                ambiguous.append((seg, cands))

        confirmed_scopes = set(c.get("contract_scope", "") for c in confirmed)
        q_hint = question_scope_hint(row["질문"], scope_cores)

        for seg, cands in ambiguous:
            pick = None
            if confirmed_scopes:
                for c in cands:
                    if c.get("contract_scope", "") in confirmed_scopes:
                        pick = c
                        break
            if pick is None and q_hint:
                for c in cands:
                    if c.get("contract_scope", "") == q_hint:
                        pick = c
                        break
            if pick is not None:
                confirmed.append(pick)
                confirmed_scopes.add(pick.get("contract_scope", ""))
            else:
                n_dropped_ambiguous += 1
                if len(set(c.get("contract_scope", "") for c in cands)) >= MANY_SCOPE_THRESHOLD:
                    n_dropped_boilerplate += 1

        gold_ids = []
        seen = set()
        for e in confirmed:
            if e["element_id"] not in seen:
                seen.add(e["element_id"])
                gold_ids.append(e["element_id"])

        if gold_ids:
            n_multi_built += 1
        out.append({
            "qid": qid, "q": row["질문"], "gold": gold_ids,
            "task_type": row.get("task_type", ""), "core_retrieval": row.get("core_retrieval", ""),
            "사업구분": row.get("사업구분", ""),
        })

    with open(f"{OUT}/gold_mapped_noah_v3_multigold_v5.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"기존 실패 156건 중 multi-gold 구성 성공: {n_multi_built}건", flush=True)
    print(f"진짜 보일러플레이트로(질문힌트도 실패) 드롭: {n_dropped_boilerplate}", flush=True)
    print(f"모호해서 드롭한 세그먼트(전체): {n_dropped_ambiguous}", flush=True)
    still_empty = 156 - n_multi_built
    print(f"여전히 gold 없음: {still_empty}건", flush=True)
    print("저장: gold_mapped_noah_v3_multigold_v5.jsonl", flush=True)


if __name__ == "__main__":
    main()
