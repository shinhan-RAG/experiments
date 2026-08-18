#!/usr/bin/env python3
"""결정론 평가(LLM 0회): 라우터+SlotSearch 를 gold span 에 대해 채점. arm 은 --arms 로 지정.

arm 문법: <mode>[:lex=<binary|count>][:tags=<0|1>][:w=<field>=<num>,...]
  예) clm:lex=binary  |  clm:lex=count  |  and  |  clm:tags=0(어휘채널만=A0)  |  clm:w=contract=3
채점: 원 단위(u2) 와 조 map-back(u2jo) 둘 다. Recall@K = fractional evidence-group, Success@K, MRR@10.
문항별 순위를 out/det_ranks_<arm>.jsonl 로 남긴다(paired 검정용).
"""
import argparse, collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clm_search import SlotSearch
from bm25_compare import score, overlaps


def parse_arm(s):
    parts = s.split(":"); mode = parts[0]; opt = {"lex": "binary", "tags": "1", "w": {}}
    for p in parts[1:]:
        k, v = p.split("=", 1)
        if k == "w":
            for kv in v.split(","):
                f, n = kv.split("="); opt["w"][f] = float(n)
        else:
            opt[k] = v
    return mode, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(HERE / "out/elements_u2.jsonl"))
    ap.add_argument("--tags", default=str(HERE / "out/tags_u2_rules.jsonl"))
    ap.add_argument("--jo", default=str(HERE / "out/elements_u2jo.jsonl"))
    ap.add_argument("--gold", default=str(HERE / "out/gold_spans_train.jsonl"))
    ap.add_argument("--arms", default="clm:lex=binary,clm:lex=count,and,clm:tags=0")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    S = SlotSearch(a.elements, a.tags)
    J = [json.loads(l) for l in open(a.jo)]
    m2j = {m: j for j, u in enumerate(J) for m in u["members"]}
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"]]
    print(f"n={len(G)} tags={Path(a.tags).name}")
    hdr = ["R@1", "R@5", "R@10", "R@20", "S@5", "RR@10"]
    print("| arm | 단위 | " + " | ".join(hdr) + " | core R@5 | 비core R@5 | 후보0 |\n|---|---|" + "---|" * (len(hdr) + 3))
    for arm in a.arms.split(","):
        mode, opt = parse_arm(arm)
        agg = collections.defaultdict(list); zero = 0
        with open(HERE / "out" / f"det_ranks_{arm.replace(':','_').replace('=','-').replace(',','+')}.jsonl", "w") as f:
            for g in G:
                slots, toks = S.router.route(g["q"])
                if opt["tags"] == "0":
                    slots = {}
                res = S.search(slots, toks, mode=mode, lex=opt["lex"], weights=opt["w"], limit=a.limit)
                if not res:
                    zero += 1
                ru = [e for e, _ in res]
                seen, rj = set(), []
                for e in ru:
                    j = m2j[e["element_id"]]
                    if j not in seen:
                        seen.add(j); rj.append(J[j])
                for unit, ranked in (("u2", ru), ("jo", rj)):
                    sc = score(ranked, g["groups"])
                    for k, v in sc.items():
                        agg[(unit, k)].append(v)
                    agg[(unit, "_core")].append(g["core_retrieval"] == "True")
                f.write(json.dumps({"qid": g["qid"], "slots": slots, "n_tokens": len(toks), "n_cand": len(res),
                                    "u2_top": [e["element_id"] for e in ru[:20]], "jo_top": [u["element_id"] for u in rj[:20]]}, ensure_ascii=False) + "\n")
        for unit in ("u2", "jo"):
            row = [f"{sum(agg[(unit,k)])/len(G):.3f}" for k in hdr]
            core = agg[(unit, "_core")]; r5 = agg[(unit, "R@5")]
            c = [x for x, f_ in zip(r5, core) if f_]; nc = [x for x, f_ in zip(r5, core) if not f_]
            print(f"| {arm} | {unit} | " + " | ".join(row) + f" | {sum(c)/len(c):.3f} | {sum(nc)/len(nc):.3f} | {zero} |")


if __name__ == "__main__":
    main()
