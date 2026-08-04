#!/usr/bin/env python3
"""게이트 2 개선 실험: 하드 필터 vs 소프트 부스트, R 확대.

A      : 전체 BM25 (기준선)
B_hard : 라우팅 top-R 특약 + 주계약으로 후보 제한 (기존)
B_soft : 전체 검색하되 라우팅 특약·주계약 청크 점수 ×boost (정답 잘림 없음)
"""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import ns, BM25, BASE, QA, VERSION

import statistics


def load_all():
    units = json.loads((BASE / "data" / "segmented" / f"{VERSION}.json").read_text())
    chunks = []
    for u in units:
        merged, order = {}, []
        for a in u["articles"]:
            k = (a["no"], a["title"])
            if k not in merged:
                merged[k] = []
                order.append(k)
            merged[k].extend(a["text"])
        for no, title in order:
            txt = " ".join(merged[(no, title)])
            if len(ns(txt)) > 30:
                chunks.append({"unit": u["unit"], "no": no, "title": title, "text": txt})
    ns_chunks = [ns(c["text"]) for c in chunks]
    bm = BM25([c["title"] + " " + c["text"] for c in chunks])
    fm = json.loads((BASE / "frontmatter" / "out" / f"{VERSION}.frontmatter.json").read_text())
    cov = {c["rider"]: c for c in fm["coverage"]}
    profiles = {}
    for c in fm["composition"]:
        parts = [c["rider"]]
        for b in cov.get(c["rider"], {}).get("benefits", []):
            parts += [b.get("name", ""), b.get("trigger", "")] + b.get("key_terms", [])
        profiles[c["rider"]] = " ".join(parts)
    rider_names = list(profiles)
    rider_bm = BM25([profiles[r] for r in rider_names])
    ALWAYS = {u["unit"] for u in units if "특약" not in u["unit"]}
    qa = [json.loads(l) for l in open(QA)]
    mapped = []
    for q in qa:
        gold = set()
        for src in q.get("출처", []):
            quote = ns(src.get("quote", ""))
            if len(quote) < 15:
                continue
            hit = [i for i, t in enumerate(ns_chunks) if quote in t]
            if not hit:
                wins = [quote[:20], quote[len(quote)//2:len(quote)//2+20], quote[-20:]]
                sc = Counter()
                for w in wins:
                    for i, t in enumerate(ns_chunks):
                        if w in t:
                            sc[i] += 1
                hit = [i for i, c in sc.items() if c >= 2]
            gold |= set(hit)
        if gold:
            mapped.append({"q": q["질문"], "gold": gold})
    return units, chunks, bm, rider_names, rider_bm, ALWAYS, mapped


def evaluate(label, ranked_per_q, mapped, KS=(1, 3, 5, 10)):
    n = len(mapped)
    hits = Counter()
    ranks = []
    for item, ranked in zip(mapped, ranked_per_q):
        r1 = next((r for r, i in enumerate(ranked, 1) if i in item["gold"]), None)
        ranks.append(r1)
        for k in KS:
            if item["gold"] & set(ranked[:k]):
                hits[k] += 1
    found = [r for r in ranks if r]
    mrr = sum(1 / r for r in found) / n
    med = statistics.median(found) if found else float("inf")
    print(f"{label:14s} " + " ".join(f"@{k}={hits[k]/n:.3f}" for k in KS)
          + f"  MRR={mrr:.3f} 중앙={med:.0f}등 미발견={ranks.count(None)}")


def main():
    units, chunks, bm, rider_names, rider_bm, ALWAYS, mapped = load_all()
    print(f"문항 {len(mapped)} | 청크 {len(chunks)}")
    all_idx = list(range(len(chunks)))

    resA, res_hard5, res_soft = [], [], {1.3: [], 1.6: [], 2.0: []}
    res_hard10 = []
    for item in mapped:
        q = item["q"]
        base_scores = [bm.score(q, i) for i in all_idx]
        resA.append(sorted(all_idx, key=lambda i: -base_scores[i])[:20])
        routed = rider_bm.rank(q, range(len(rider_names)))
        for R, out in ((5, res_hard5), (10, res_hard10)):
            allowed = ALWAYS | {rider_names[i] for i in routed[:R]}
            cand = [i for i in all_idx if chunks[i]["unit"] in allowed]
            out.append(sorted(cand, key=lambda i: -base_scores[i])[:20])
        top5 = {rider_names[i] for i in routed[:5]}
        for boost, out in res_soft.items():
            sc = [base_scores[i] * (boost if chunks[i]["unit"] in (ALWAYS | top5) else 1.0)
                  for i in all_idx]
            out.append(sorted(all_idx, key=lambda i: -sc[i])[:20])

    evaluate("A 전체검색", resA, mapped)
    evaluate("B hard R=5", res_hard5, mapped)
    evaluate("B hard R=10", res_hard10, mapped)
    for boost, out in res_soft.items():
        evaluate(f"B soft x{boost}", out, mapped)


if __name__ == "__main__":
    main()
