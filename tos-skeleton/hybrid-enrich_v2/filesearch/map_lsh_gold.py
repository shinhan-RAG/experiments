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
def build_groups(spans, expand):
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
            scopes = {scope_at(m["c0"]) for m in g["members"]}
            added = 0
            for c0, c1 in occurrences(g["key"]):
                if added >= MAX_OCC:
                    break
                if any(c0 < m["c1"] and c1 > m["c0"] for m in g["members"]):
                    continue
                if scope_at(c0) not in scopes:
                    continue
                g["members"].append({"c0": c0, "c1": c1, "src": "occurrence"}); added += 1
        g["c0"] = min(m["c0"] for m in g["members"]); g["c1"] = max(m["c1"] for m in g["members"])
    return groups

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
        groups = build_groups(spans, expand) if spans else []
        n_fail += bool(fail)
        out.append({"qid": r["qid"], "q": q.get("q", ""), "task_type": q.get("task_type", ""), "core_retrieval": q.get("core_retrieval", ""),
                    "groups": groups, "lsh_gold": r["gold"], "unmapped": fail, "status": "ok" if not fail else "partial",
                    "rule": f"gap<={GAP};same_text_or;{'occ_expand(key160,min40,max5,gold_scope)' if expand else 'no_expand'}"})
    path = HERE / "out" / name
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ng = [len(o["groups"]) for o in out if o["groups"]]
    import hashlib
    print(name, {"n": len(out), "with_gold": len(ng), "unmapped_any": n_fail, "groups_total": sum(ng), "q_multi_group": sum(1 for x in ng if x > 1),
                 "or_members_total": sum(len(g["members"]) for o in out for g in o["groups"]),
                 "groups_with_or": sum(1 for o in out for g in o["groups"] if len(g["members"]) > 1),
                 "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]})
