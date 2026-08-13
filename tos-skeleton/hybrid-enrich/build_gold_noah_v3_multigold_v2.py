#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""156건(gold매칭 실패)에 대해 출처를 \n\n 구간별로 쪼개서 각각 위치를 찾아
multi-gold를 구성 -- v2: 짧은/보일러플레이트 문구가 여러 특약에 반복되어
'문서상 첫 occurrence'로 잘못 매칭되던 버그 수정.

수정 로직:
  1패스: 각 세그먼트의 모든 occurrence를 찾는다. 전부 같은 특약(scope)으로
         귀결되면(진짜 유일해서든, 반복되도 같은 특약 안에서든) 확정 채택.
         서로 다른 특약에 걸쳐 있으면 '모호'로 보류.
  확정된 세그먼트들의 scope 집합 = 이 질문이 다루는 특약(들)로 간주.
  2패스: 모호했던 세그먼트는, 그 occurrence들 중 확정 scope 집합에 속하는
         것을 우선 채택. 그래도 안 되면(확정 scope가 아직 없거나 안 겹치면)
         질문 텍스트에 언급된 특약명으로 재시도. 그래도 안 되면 스킵(추측성
         gold를 넣지 않는다 -- 틀린 gold보다 빈 gold가 낫다)."""
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


def find_all_spans(quote, norm_doc, offset_to_line, max_hits=30):
    """quote가 문서 안에 등장하는 모든 위치를 (ls, le)로 반환."""
    nq = ns(quote)
    if len(nq) < 8:
        return []
    spans = []
    start = 0
    used = nq
    n_hits = 0
    while n_hits < max_hits:
        idx = norm_doc.find(used, start)
        if idx == -1:
            break
        end = idx + len(used)
        ls = offset_to_lineno(idx, offset_to_line)
        le = offset_to_lineno(end - 1, offset_to_line)
        spans.append((ls, le))
        start = idx + 1
        n_hits += 1
    if spans:
        return spans
    # 정확 매칭 실패 -> 축약 매칭 1회만
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
    n_dropped_ambiguous = 0
    for qid, r in existing.items():
        if qid not in failed_qids:
            out.append(r)
            continue
        row = by_qid[qid]
        quote = row.get("출처", "")
        segs = [s for s in quote.split("\n\n") if s.strip()]

        # 세그먼트별 후보(occurrence -> element) 계산
        seg_candidates = []  # [(seg, [element, ...]), ...]
        for seg in segs:
            spans = find_all_spans(seg, norm_doc, offset_to_line)
            cands = []
            seen_eid = set()
            for ls, le in spans:
                best = map_gold(ls, le, elements_sorted)
                if best and best["element_id"] not in seen_eid:
                    seen_eid.add(best["element_id"])
                    cands.append(best)
            seg_candidates.append((seg, cands))

        # 1패스: 확정(전 occurrence가 같은 scope) 세그먼트만 채택
        confirmed = []  # element dict list
        ambiguous = []  # (seg, cands) list
        for seg, cands in seg_candidates:
            if not cands:
                continue
            scopes = set(c.get("contract_scope", "") for c in cands)
            if len(scopes) == 1:
                confirmed.append(cands[0])
            else:
                ambiguous.append((seg, cands))

        confirmed_scopes = set(c.get("contract_scope", "") for c in confirmed)

        # 질문에서 특약명 힌트 (확정 scope가 없을 때 보조용)
        q_hint = question_scope_hint(row["질문"], scope_cores)

        # 2패스: 모호한 세그먼트 처리
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
                n_dropped_ambiguous += 1  # 못 정하면 버림(추측 안 함)

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

    with open(f"{OUT}/gold_mapped_noah_v3_multigold_v2.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"기존 실패 156건 중 multi-gold 구성 성공: {n_multi_built}건", flush=True)
    print(f"모호해서 드롭한 세그먼트 수: {n_dropped_ambiguous}", flush=True)
    still_empty = 156 - n_multi_built
    print(f"여전히 gold 없음: {still_empty}건 (not_answerable 등)", flush=True)
    print("저장: gold_mapped_noah_v3_multigold_v2.jsonl", flush=True)


if __name__ == "__main__":
    main()
