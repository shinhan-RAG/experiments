"""shinhan-mixed 중복 청크 gold 보정 재채점 (재실험 불필요).

gold와 텍스트가 동일한 다른 id 청크가 검색돼도 회수로 인정해
part5 결과 JSON의 ranked_top20을 재채점한다 (원채점→보정 병기 출력).
nDCG@10은 gold 그룹당 첫 등장만 이득 1로 인정.
"""
import glob, hashlib, json, math
from collections import defaultdict

norm = lambda s: " ".join((s or "").split())
groups = defaultdict(set)
id2h = {}
for line in open("data/raw/shinhan-mixed/corpus.jsonl", encoding="utf-8"):
    d = json.loads(line)
    h = hashlib.md5(norm(d.get("text")).encode()).hexdigest()
    groups[h].add(str(d["_id"]))
    id2h[str(d["_id"])] = h
nd = sum(len(g) for g in groups.values() if len(g) > 1)
print(f"corpus ids={len(id2h)} dup-group ids={nd} ({nd/len(id2h):.1%})")

gold_by_q = defaultdict(set)
for line in open("data/raw/shinhan-mixed/qrels.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("score", 0) >= 1:
        gold_by_q[str(e["query-id"])].add(str(e["corpus-id"]))

def rid(x):
    if isinstance(x, (list, tuple)):
        return str(x[0])
    if isinstance(x, dict):
        return str(x.get("doc_id") or x.get("_id") or x.get("id"))
    return str(x)

shown = False
for p in sorted(glob.glob("results/part5_pull_backend/2026080*.json")):
    r = json.load(open(p))
    m = r.get("manifest", {})
    if m.get("dataset") != "shinhan-mixed":
        continue
    print("--", p.split("/")[-1], "subset=", m.get("subset_size"))
    for backend, d in r["full_results"].items():
        rows = d["probe_rows"]
        a = defaultdict(float)
        for row in rows:
            if not shown:
                print("  sample:", repr(row["ranked_top20"][:1]))
                shown = True
            ranked = [rid(x) for x in row["ranked_top20"]]
            gold = gold_by_q[str(row["query_id"])]
            ng = len(gold) or 1
            def hits(k):
                topk = ranked[:k]
                return sum(1 for g in gold
                           if any(t in groups.get(id2h.get(g), {g}) for t in topk))
            a["r5"] += hits(5) / ng
            a["r20"] += hits(20) / ng
            cred = set()
            dcg = 0.0
            for i, t in enumerate(ranked[:10]):
                th = id2h.get(t)
                for g in gold:
                    if g not in cred and (t == g or (th and th == id2h.get(g))):
                        dcg += 1 / math.log2(i + 2)
                        cred.add(g)
                        break
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
            a["nd"] += dcg / idcg if idcg else 0.0
            a["o5"] += row["recall_at_5"]
            a["o20"] += row["recall_at_20"]
            a["ond"] += row["ndcg_at_10"]
        n = len(rows)
        print("  %-20s R@5 %.2f->%.2f  R@20 %.2f->%.2f  nDCG@10 %.3f->%.3f" % (
            backend, a["o5"]/n, a["r5"]/n, a["o20"]/n, a["r20"]/n,
            a["ond"]/n, a["nd"]/n))
