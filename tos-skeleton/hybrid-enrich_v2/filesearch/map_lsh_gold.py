#!/usr/bin/env python3
"""lsh gold(element id) → 원문 char span → 채점용 gold(out/gold_spans_lsh_train.jsonl).
입력: ../out/gold_mapped_noah_v3_348_train.jsonl (lsh 공식 업로드) + span 텍스트 사전(cowork 원문 span 매핑 파일).
span 사전에 없는 gold id 가 있는 문항은 unmapped 로 표기하고 채점 모집단에서 제외된다(수량은 출력에 기록).
gold element 텍스트를 map_gold_spans.map_group 으로 앵커링(줄 단위, 최소매치 줄 기준). group = gold element 1개."""
import json, sys, unicodedata, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from map_gold_spans import norm_map, map_group
CO = Path("/Users/ralph/Desktop/신한라이프/cowork")
DOC = "/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
nd, idx = norm_map(raw)
S = {json.loads(l)["element_id"]: json.loads(l) for l in open(CO / "원문 span 매핑 파일.jsonl", encoding="utf-8")}
R = [json.loads(l) for l in open(HERE.parent / "out" / "gold_mapped_noah_v3_348_train.jsonl", encoding="utf-8")]
Q = {json.loads(l)["qid"]: json.loads(l) for l in open(HERE / "out/gold_spans_train.jsonl", encoding="utf-8")}
# ── 빠진 id 복원: lsh id 는 문서 순서와 단조(검증: 유일 앵커 300개, 순서 위반 0) ──
# 사전에 없는 gold id 는 (1) 이웃 id 들의 span 으로 구간을 협공하고 (2) 그 문항의 QA 출처 인용문 앵커(임시 gold)가
# 구간 안에 떨어지면 그 span 을 쓴다. 협공·인용문이 안 맞으면 미복원으로 남긴다.
span_cache = {}
def span_of(eid):
    if eid in span_cache:
        return span_cache[eid]
    t = S[eid]["text"]
    lines = [x.strip() for x in t.split("\n") if x.strip() and not re.fullmatch(r"-{3,}|\d{1,4}|SHINHAN LIFE|_{3,}|\d{10,}", x.strip())]
    g, un = map_group(nd, idx, lines)
    span_cache[eid] = g
    return g

# ── 채점 규칙 ─────────────────────────────────────────────────────────────────
# 1) 인접 span 병합: gap<=GAP (같은 근거 블록의 조각 정리용으로만 짧게 둔다. group 수는 원 gold id 수를 원칙으로 유지)
# 2) 같은 텍스트(앞 KEY자, 공백 제거) group 은 OR 로 묶는다 (a/a' 동치)
# 3) 동일문구 출현 확장(lenient 만): 문서 안 정확 일치 위치를 OR 멤버로 추가. 40자 미만은 확장하지 않는다.
#    확장 범위는 gold element 가 실제로 속한 특약(scope) 안으로 제한한다 — 질문 텍스트나 검색기 라우터는 쓰지 않는다.
#    strict 파일은 확장을 넣지 않는다(주지표용). lenient 파일은 부지표용.
# 4) 남은 group = AND (비율 채점의 분모)
GAP = 30
KEY = 160
MAX_OCC = 5
E = [json.loads(l) for l in open(HERE / "out/elements_u2.jsonl", encoding="utf-8")]
import bisect
_starts = [e["char_start"] for e in E]
def scope_at(c):
    i = bisect.bisect_right(_starts, c) - 1
    return E[i]["contract_scope"] if i >= 0 else ""
def key_of(c0, c1):
    return re.sub(r"\s", "", raw[c0:c1])[:KEY]
def occurrences(k):
    out, p = [], nd.find(k)
    while p >= 0 and len(out) < 200:
        out.append((idx[p], idx[p + len(k) - 1] + 1)); p = nd.find(k, p + 1)
    return out
def build_groups(spans, expand, expand_named=True):
    spans = sorted(spans)
    merged = []
    for c0, c1, eid in spans:
        if merged and c0 - merged[-1]["c1"] <= GAP:
            merged[-1]["c1"] = max(merged[-1]["c1"], c1); merged[-1]["lsh_eids"].append(eid)
        else:
            merged.append({"c0": c0, "c1": c1, "lsh_eids": [eid]})
    groups = []
    for m in merged:
        k = key_of(m["c0"], m["c1"])
        for g in groups:
            if k and g["key"] == k:
                g["members"].append({"c0": m["c0"], "c1": m["c1"], "src": "gold"}); g["lsh_eids"] += m["lsh_eids"]; break
        else:
            groups.append({"key": k, "members": [{"c0": m["c0"], "c1": m["c1"], "src": "gold"}], "lsh_eids": list(m["lsh_eids"])})
    for g in groups:
        if expand and len(g["key"]) >= 40:
            # 정답기준 명세 v1 §3: 특약 지정 질문 → 정답 특약 scope 안만 / 일반 질문 → 문서 전체 출현 인정
            # 특약 지정 판정 = 질문 문자열에 특약 핵심명이 문자 그대로 존재하는가(검색기 라우터 미사용 — 순환 방지)
            scopes = {scope_at(m["c0"]) for m in g["members"]} if expand_named else None
            added = 0
            for c0, c1 in occurrences(g["key"]):
                if added >= MAX_OCC:
                    break
                if any(c0 < m["c1"] and c1 > m["c0"] for m in g["members"]):
                    continue
                if scopes is not None and scope_at(c0) not in scopes:
                    continue
                g["members"].append({"c0": c0, "c1": c1, "src": "occurrence"}); added += 1
        g["c0"] = min(m["c0"] for m in g["members"]); g["c1"] = max(m["c1"] for m in g["members"])
    return groups

import csv as _csv
QA_SRC = {}
_csvp = Path("/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/정답셋_348_train_v3_retrieval_250212.csv")
for _row in _csv.DictReader(open(_csvp, encoding="utf-8-sig")):
    QA_SRC[_row["qid"]] = [x.strip() for x in re.split(r"\n+", _row["출처"] or "") if x.strip() and not re.fullmatch(r"-{3,}|\d{1,4}|\d{10,}", x.strip())]

def find_in_bracket(qid, lo, hi):
    """출처 인용문 각 줄을 [lo,hi) 구간 안에서만 재앵커링 → 가장 긴 매치 span 반환"""
    best = None
    for ln in QA_SRC.get(qid, []):
        q = re.sub(r"\s", "", unicodedata.normalize("NFC", ln))
        for cand in (q, q[: max(20, int(len(q) * 0.6))]):
            if len(cand) < 12:
                continue
            p = nd.find(cand)
            while p >= 0:
                c0, c1 = idx[p], idx[p + len(cand) - 1] + 1
                if lo <= c0 and c1 <= hi and (best is None or (c1 - c0) > (best[1] - best[0])):
                    best = (c0, c1)
                p = nd.find(cand, p + 1)
            if best:
                break
    return best

# 질문이 특약명을 문자 그대로 담고 있는지(공백 제거·핵심명 4자 이상). 사전은 element 의 contract_scope 에서만 도출(QA 미참조)
_E_scopes = sorted({e["contract_scope"] for e in E if e["contract_scope"]})
def _core_name(sc):
    sc = re.sub(r"^주계약\((.*)\)$", r"\1", sc)
    sc = re.sub(r"\(무배당[^)]*\)|\(간편\)|\[.*?\]", "", sc)
    return re.sub(r"\s", "", sc).replace("특약", "")
_CORES = sorted({c for c in (_core_name(s_) for s_ in _E_scopes) if len(c) >= 4}, key=len, reverse=True)
def question_names_contract(qtext):
    cq = re.sub(r"\s", "", qtext)
    return any(c in cq for c in _CORES)

for eid in S:
    span_of(eid)
for expand, name in ((False, "gold_spans_lsh_train.jsonl"), (True, "gold_spans_lsh_train_lenient.jsonl")):
    n_fail = 0; out = []
    for r in R:
        q = Q.get(r["qid"], {})
        spans = []; fail = []
        for eid in r["gold"]:
            g = span_of(eid) if eid in S else None
            if g:
                spans.append((g["c0"], g["c1"], eid))
            else:
                fail.append(eid)
        if fail:
            # 협공 복원 시도 — 앵커는 (id, 위치) 단조 증가 부분수열(LIS)만 신뢰(오앵커 배제)
            global _lis_anchors
            if "_lis_anchors" not in globals():
                pts = sorted((int(k[1:]), v) for k, v in span_cache.items() if v)
                import bisect as _b
                tails, tidx, prev = [], [], [None] * len(pts)
                for i, (n, v) in enumerate(pts):
                    j = _b.bisect_right(tails, v["c0"])
                    if j == len(tails):
                        tails.append(v["c0"]); tidx.append(i)
                    else:
                        tails[j] = v["c0"]; tidx[j] = i
                    prev[i] = tidx[j - 1] if j else None
                sel, i = [], tidx[-1] if tidx else None
                while i is not None:
                    sel.append(pts[i]); i = prev[i]
                _lis_anchors = sel[::-1]
            known = _lis_anchors
            recovered = []
            for eid in list(fail):
                x = int(eid[1:])
                lo = max((v["c1"] for n, v in known if n < x), default=0)
                hi = min((v["c0"] for n, v in known if n > x), default=len(raw))
                cand = [gr for gr in Q.get(r["qid"], {}).get("groups", []) if lo <= gr["c0"] and gr["c1"] <= hi]
                if cand:
                    g = min(cand, key=lambda gr: gr["c0"])
                    spans.append((g["c0"], g["c1"], eid)); recovered.append(eid); fail.remove(eid)
                else:
                    hit = find_in_bracket(r["qid"], lo, hi)
                    if hit:
                        spans.append((hit[0], hit[1], eid)); recovered.append(eid); fail.remove(eid)
            if recovered:
                r["_recovered"] = recovered
        named = question_names_contract(q.get("q", ""))
        groups = build_groups(spans, expand, expand_named=named) if spans else []
        n_fail += bool(fail)
        out.append({"qid": r["qid"], "q": q.get("q", ""), "task_type": q.get("task_type", ""), "core_retrieval": q.get("core_retrieval", ""),
                    "groups": groups, "lsh_gold": r["gold"], "unmapped": fail, "recovered": r.get("_recovered", []),
                    "status": "ok" if not fail else "partial",
                    "rule": f"gap<={GAP};same_text_or;{'occ_expand(key160,min40,max5,gold_scope)' if expand else 'no_expand'}"})
    path = HERE / "out" / name
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ng = [len(o["groups"]) for o in out if o["groups"]]
    import hashlib
    print(name, {"n": len(out), "with_gold": len(ng), "unmapped_any": n_fail, "recovered_q": sum(1 for o in out if o["recovered"]), "groups_total": sum(ng), "q_multi_group": sum(1 for x in ng if x > 1),
                 "or_members_total": sum(len(g["members"]) for o in out for g in o["groups"]),
                 "groups_with_or": sum(1 for o in out for g in o["groups"] if len(g["members"]) > 1),
                 "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]})
