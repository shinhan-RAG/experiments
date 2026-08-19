#!/usr/bin/env python3
"""lsh v4 gold(element id + text) → 원문 char span → 우리 우주 독립 gold(out/gold_spans_lsh_train.jsonl).
입력: cowork/문항별 순위 jsonl.jsonl (qid, gold ids, ranked ids) + cowork/원문 span 매핑 파일.jsonl (element_id, text).
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
R = [json.loads(l) for l in open(CO / "문항별 순위 jsonl.jsonl", encoding="utf-8")]
Q = {json.loads(l)["qid"]: json.loads(l) for l in open(HERE / "out/gold_spans_train.jsonl", encoding="utf-8")}
span_cache = {}
def span_of(eid):
    if eid in span_cache:
        return span_cache[eid]
    t = S[eid]["text"]
    lines = [x.strip() for x in t.split("\n") if x.strip() and not re.fullmatch(r"-{3,}|\d{1,4}|SHINHAN LIFE|_{3,}|\d{10,}", x.strip())]
    g, un = map_group(nd, idx, lines)
    span_cache[eid] = g
    return g

# ── 채점 규칙(회의 2026-08-19 + 정답기준 명세 v1 준용) ─────────────────────────
GAP = 200          # 인접 span 병합 간격(자) — 같은 근거 블록
KEY = 160          # 동일문구 출현 확장 키 길이(공백 제거)
E = [json.loads(l) for l in open(HERE / "out/elements_u2.jsonl", encoding="utf-8")]
import bisect
_starts = [e["char_start"] for e in E]
def scope_at(c):
    i = bisect.bisect_right(_starts, c) - 1
    return E[i]["contract_scope"] if i >= 0 else ""
sys.path.insert(0, str(HERE))
from clm_search import Router
_router = Router(sorted({e["contract_scope"] for e in E}))
_router.partial = False
def key_of(c0, c1):
    return re.sub(r"\s", "", raw[c0:c1])[:KEY]
def occurrences(k):
    """정규화 문서 nd 에서 k 의 모든 출현 → raw 구간 목록(정확 일치, 최대 200)"""
    out, p = [], nd.find(k)
    while p >= 0 and len(out) < 200:
        out.append((idx[p], idx[p + len(k) - 1] + 1)); p = nd.find(k, p + 1)
    return out
def build_groups(spans, question):
    """spans[(c0,c1,eid)] → AND groups, 각 group 은 OR members. 규칙:
    1) 정렬 후 gap≤GAP 병합(같은 근거 블록)  2) 병합 group 텍스트(앞 160자)가 동일한 group 끼리 OR 병합(a/a')
    3) 동일문구 출현 확장: 문서 내 정확 일치 위치를 OR 멤버로 추가 — 단 질문이 특약을 지정하면 그 특약 scope 안 출현만
    4) 남은 group = AND (fractional 채점의 분모)"""
    spans = sorted(spans)
    merged = []
    for c0, c1, eid in spans:
        if merged and c0 - merged[-1]["c1"] <= GAP:
            merged[-1]["c1"] = max(merged[-1]["c1"], c1); merged[-1]["lsh_eids"].append(eid)
        else:
            merged.append({"c0": c0, "c1": c1, "lsh_eids": [eid]})
    slots, _ = _router.route(question)
    q_scopes = set(slots.get("contract", []))
    q_scopes = {s for s in q_scopes if s} if q_scopes else set()
    groups = []
    for m in merged:
        k = key_of(m["c0"], m["c1"])
        # 2) 같은 텍스트 group 과 OR 병합
        for g in groups:
            if k and g["key"] == k:
                g["members"].append({"c0": m["c0"], "c1": m["c1"], "src": "gold"}); g["lsh_eids"] += m["lsh_eids"]; break
        else:
            groups.append({"key": k, "members": [{"c0": m["c0"], "c1": m["c1"], "src": "gold"}], "lsh_eids": list(m["lsh_eids"])})
    for g in groups:
        k = g["key"]
        if len(k) >= 40:  # 짧은 조각은 확장하지 않음(상용구 오인정 방지)
            for c0, c1 in occurrences(k):
                if any(c0 < m["c1"] and c1 > m["c0"] for m in g["members"]):
                    continue
                if q_scopes and not any(re.sub(r"^주계약\((.*)\)$", r"\1", scope_at(c0)) in q_scopes or scope_at(c0) in q_scopes for _ in [0]):
                    continue
                g["members"].append({"c0": c0, "c1": c1, "src": "occurrence"})
        g["c0"] = min(m["c0"] for m in g["members"]); g["c1"] = max(m["c1"] for m in g["members"])
        g["scope_restricted"] = bool(q_scopes)
    return groups

n_ok = n_fail = 0; out = []
for r in R:
    q = Q.get(r["qid"], {})
    spans = []; fail = []
    for eid in r["gold"]:
        g = span_of(eid) if eid in S else None
        if g:
            spans.append((g["c0"], g["c1"], eid))
        else:
            fail.append(eid)
    groups = build_groups(spans, q.get("q", "")) if spans else []
    n_ok += bool(groups) or not r["gold"]; n_fail += bool(fail)
    out.append({"qid": r["qid"], "q": q.get("q", ""), "task_type": q.get("task_type", ""), "core_retrieval": q.get("core_retrieval", ""),
                "groups": groups, "lsh_gold": r["gold"], "unmapped": fail, "status": "ok" if not fail else "partial",
                "rule": f"merge_gap<={GAP};same_text_or;occurrence_expand(key{KEY},min40,scope_if_named)"})
with open(HERE / "out/gold_spans_lsh_train.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
ng = [len(o["groups"]) for o in out if o["groups"]]
print({"n": len(out), "with_gold": len(ng), "unmapped_any": n_fail, "groups_total": sum(ng),
       "q_multi_group": sum(1 for x in ng if x > 1), "or_members_total": sum(len(g["members"]) for o in out for g in o["groups"]),
       "groups_with_or": sum(1 for o in out for g in o["groups"] if len(g["members"]) > 1),
       "scope_restricted_q": sum(1 for o in out if o["groups"] and o["groups"][0].get("scope_restricted"))})
