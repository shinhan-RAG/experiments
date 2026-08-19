#!/usr/bin/env python3
"""우주 비교 — char-bigram BM25(k1=1.2, b=0.75; exp23 기준선과 동일 공식) 결정론 측정.

채점 단위 3종을 병기한다(Dense X Retrieval, EMNLP 2024 의 map-back·예산 고정 비교 방식):
  (a) 원 단위 recall@k          — 우주 element 그대로
  (b) 조 map-back recall@k      — u2 순위를 소속 조(u2jo) 로 사상, 중복 제거 후 top-k 조
  (c) 글자 예산 고정 recall@B   — 상위부터 누적 글자수가 B 에 도달할 때까지 반환한 집합
gold = out/gold_spans_train.jsonl (우주 독립 char span). hit = element 구간이 gold group 구간과 겹침.
Recall@k = fractional evidence-group (Top-k 가 덮은 group 수 / group 수). Success@k, MRR@10 병기.
"""
import argparse, collections, json, math, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import bigrams, overlaps, score, budget_recall  # noqa: E402




class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = []
        df = collections.Counter()
        self.dl = []
        for d in docs:
            c = collections.Counter(bigrams(d))
            self.tf.append(c); self.dl.append(sum(c.values()))
            df.update(c.keys())
        N = len(docs); self.avg = sum(self.dl) / max(1, N)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post = collections.defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.post[t].append((i, f))

    def search(self, q, k=200):
        sc = collections.defaultdict(float)
        for t in set(bigrams(q)):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, f in self.post[t]:
                den = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg)
                sc[i] += idf * f * (self.k1 + 1) / den
        return sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:k]








def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", default="out/elements_u2.jsonl")
    ap.add_argument("--coarse", default="out/elements_u2jo.jsonl")
    ap.add_argument("--gold", default="out/gold_spans_train.jsonl")
    ap.add_argument("--budgets", default="1000,3000")
    ap.add_argument("--shuffle-null", type=int, default=0, help="query_shuffle 널 시드 수")
    a = ap.parse_args()
    fine = [json.loads(l) for l in open(a.fine)]
    coarse = [json.loads(l) for l in open(a.coarse)]
    member2jo = {m: j for j, u in enumerate(coarse) for m in u["members"]}
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"]]
    print(f"population n={len(G)} (gold 매핑 성공분)")
    bf = BM25([e["text"] for e in fine]); bc = BM25([e["text"] for e in coarse])
    budgets = [int(x) for x in a.budgets.split(",")]

    def run(queries):
        agg = collections.defaultdict(list)
        for g, q in zip(G, queries):
            rf = [fine[i] for i, _ in bf.search(q, 400)]
            rc = [coarse[i] for i, _ in bc.search(q, 200)]
            # map-back: fine 순위 → 조 dedupe
            seen, rm = set(), []
            for e in rf:
                j = member2jo[e["element_id"]]
                if j not in seen:
                    seen.add(j); rm.append(coarse[j])
            for name, ranked in (("fine(u2)", rf), ("coarse(u2jo)", rc), ("u2→jo map-back", rm)):
                for k, v in score(ranked, g["groups"]).items():
                    agg[(name, k)].append(v)
                for B in budgets:
                    agg[(name, f"R@{B}chars")].append(budget_recall(ranked, g["groups"], B))
                agg[(name, "_core")].append(g["core_retrieval"] == "True")
        return agg

    agg = run([g["q"] for g in G])
    names = ["fine(u2)", "coarse(u2jo)", "u2→jo map-back"]
    keys = ["R@1", "R@5", "R@10", "R@20", "S@5", "RR@10"] + [f"R@{B}chars" for B in budgets]
    print("\n| 채점 | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for nm in names:
        print(f"| {nm} | " + " | ".join(f"{sum(agg[(nm,k)])/len(agg[(nm,k)]):.3f}" for k in keys) + " |")
    # 층화 core / 비core
    for nm in names:
        core = agg[(nm, "_core")]
        row = []
        for k in ("R@5", "R@10"):
            v = agg[(nm, k)]
            c = [x for x, f in zip(v, core) if f]; nc = [x for x, f in zip(v, core) if not f]
            row.append(f"{k} core {sum(c)/len(c):.3f} / 비core {sum(nc)/len(nc):.3f}")
        print(f"  {nm}: " + " · ".join(row))
    if a.shuffle_null:
        vals = collections.defaultdict(list)
        for s in range(a.shuffle_null):
            qs = [g["q"] for g in G]; random.Random(s).shuffle(qs)
            ag = run(qs)
            for nm in names:
                vals[nm].append(sum(ag[(nm, "R@5")]) / len(G))
        for nm in names:
            v = vals[nm]; print(f"  null query_shuffle R@5 {nm}: mean {sum(v)/len(v):.3f} min {min(v):.3f} max {max(v):.3f} (seeds {a.shuffle_null})")


if __name__ == "__main__":
    main()
