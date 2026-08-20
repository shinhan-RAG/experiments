#!/usr/bin/env python3
"""v4 교정 gold 최종 평가 — 개별 arm / 앙상블(RRF) / 분모별 병기.

분모(0817 교훈: 분모 병기 의무):
- n=60: train60 전체, v4 교정 gold (제외 없음)
- n=58: C1 제외 (v4 삭제 + v3 gold 오매핑 확인 2문항)
- n=52: C1+C3 제외 (C3 = v4 근거 인용 일부가 채점 코퍼스(250212판 약관)에 부재 6문항)

앙상블 = 에이전틱 런 제출(가중 2) ⊕ 결정론 태그 변형 10종 ⊕ 메타 hybrid/bm25/dense (RRF k=60).
"""
import json, sys, glob, os, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "filesearch"))
from scoring import score
from units import Units

U = Units(HERE.parent.parent.parent / "filesearch/out/elements_u2jo.jsonl")
gold = {g["qid"]: g for g in (json.loads(l) for l in open(HERE / "out/gold_v4_train60.jsonl", encoding="utf-8"))}
audit = json.load(open(HERE / "out/regold_audit.json", encoding="utf-8"))
C1 = {"v3-offline-0413", "v3-offline-0508"}
C3 = {a["qid"] for a in audit if a.get("status") == "regold" and a.get("missed")}

def jos_of(ids):
    seen, out = set(), []
    for u in U.resolve(ids):
        if u["element_id"] not in seen:
            seen.add(u["element_id"]); out.append(u["element_id"])
    return out

def rrf(lists, k=60, w=None):
    sc, first = collections.defaultdict(float), {}
    for lst, wt in zip(lists, w or [1] * len(lists)):
        for i, j in enumerate(lst):
            sc[j] += wt / (k + i + 1); first.setdefault(j, i)
    return sorted(sc, key=lambda j: (-sc[j], first[j]))

arms = {}
for d in sorted(glob.glob(str(HERE / "out/agent/*60"))):
    rf = os.path.join(d, "results.jsonl")
    if os.path.exists(rf):
        rows = [json.loads(l) for l in open(rf, encoding="utf-8")]
        if len(rows) >= 55:
            arms[os.path.basename(d)] = {r["qid"]: r for r in rows}
dets = {}
for f in glob.glob(str(HERE / "out/det_ranks_*.jsonl")):
    dets[f] = {json.loads(l)["qid"]: json.loads(l).get("jo_top", []) for l in open(f, encoding="utf-8")}
meta = {json.loads(l)["qid"]: json.loads(l)["jo_top"] for l in open(HERE / "out/meta_ranks_train60.jsonl", encoding="utf-8")}
mbd = json.load(open(HERE / "out/meta_bm25_dense_train60.json", encoding="utf-8"))

def ensemble(q):
    subs = [jos_of(arms[a][q]["submitted"]) for a in arms if q in arms[a]]
    det = [d.get(q, []) for d in dets.values()]
    m = [meta.get(q, []), mbd["mb"].get(q, []), mbd["md"].get(q, [])]
    return rrf(subs + det + m, w=[2] * len(subs) + [1] * (len(det) + len(m)))

def report(fn, label):
    for EX, lab in ((set(), "n=60"), (C1, "n=58"), (C1 | C3, "n=52")):
        per, bytt = [], collections.defaultdict(list)
        for q, g in gold.items():
            if q in EX:
                continue
            p = score(U.resolve(fn(q)), g["groups"])
            per.append(p); bytt[g["task_type"]].append(p["R@5"])
        n = len(per)
        m = {k: sum(p[k] for p in per) / n for k in ("R@5", "R@10", "suff@5", "suff@10")}
        tt = " ".join("%s=%.3f(%d)" % (t, sum(v) / len(v), len(v)) for t, v in sorted(bytt.items()))
        print("%-26s %-5s R@5=%.4f R@10=%.4f suff@10=%.4f | %s" % (label, lab, m["R@5"], m["R@10"], m["suff@10"], tt))

print("에이전틱 런:", list(arms))
for a in arms:
    report(lambda q, a=a: jos_of(arms[a][q]["submitted"]) if q in arms[a] else [], a)
report(ensemble, "** 앙상블(전 런+det+meta) **")
