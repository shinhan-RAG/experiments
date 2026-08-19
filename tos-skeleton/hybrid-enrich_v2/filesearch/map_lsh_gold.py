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
n_ok = n_fail = 0; out = []
for r in R:
    q = Q.get(r["qid"], {})
    groups = []; fail = []
    for eid in r["gold"]:
        g = span_of(eid) if eid in S else None
        if g:
            g = dict(g); g["lsh_eid"] = eid; groups.append(g)
        else:
            fail.append(eid)
    n_ok += bool(groups) or not r["gold"]; n_fail += bool(fail)
    out.append({"qid": r["qid"], "q": q.get("q", ""), "task_type": q.get("task_type", ""), "core_retrieval": q.get("core_retrieval", ""),
                "groups": groups, "lsh_gold": r["gold"], "unmapped": fail, "status": "ok" if not fail else "partial"})
with open(HERE / "out/gold_spans_lsh_train.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print({"n": len(out), "with_gold": sum(1 for o in out if o["groups"]), "unmapped_any": n_fail,
       "ambiguous_anchor_groups": sum(1 for o in out for g in o["groups"] if g["anchor_multi"] > 1), "groups": sum(len(o["groups"]) for o in out)})
# 우리 임시 gold 와의 겹침
ov = same = 0
for o in out:
    q = Q.get(o["qid"])
    if not q or not o["groups"] or not q["groups"]:
        continue
    ov += 1
    hit = all(any(g["c0"] < h["c1"] and g["c1"] > h["c0"] for h in q["groups"]) for g in o["groups"])
    same += hit
print("lsh gold groups all overlap our temp gold:", same, "/", ov)
