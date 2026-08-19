#!/usr/bin/env python3
"""트랙 C 결정론 선별 — 우리 우주(u2/u2jo)에서 태그 채널(CLM) × 임베딩 채널(BGE-m3-ko) 결합 방식 비교.

arm:
  tag            = CLM(LLM qtags 라우터 + contract w2)                                  [C0a]
  vec:<view>     = 임베딩 코사인 top-K (view=base|ctx)                                    [C0b]
  rrf:<view>     = 각각 검색 → RRF(k=60, 1:1)                                             [C1]
  scoped:<view>  = 태그 라우터가 특약을 지목한 문항은 그 특약 scope 안 임베딩 순위(소프트: scope 밖 −δ), 아니면 vec  [C2]
채점: 조 map-back, fractional evidence-group R@K + Success@K + MRR@10, core/비core. 문항별 순위 저장.
쿼리 임베딩은 질문 원문(질의 측 변환 없음). LLM 0회(qtags 는 동결 캐시).
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bm25_compare import score, overlaps
from clm_search import SlotSearch


def rrf(rank_lists, k=60):
    sc = collections.defaultdict(float)
    for rl in rank_lists:
        for r, eid in enumerate(rl):
            sc[eid] += 1.0 / (k + r + 1)
    return [e for e, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--qtags", default=str(HERE / "out/qtags_haiku.jsonl"))
    ap.add_argument("--arms", default="tag;vec:base;vec:ctx;rrf:base;rrf:ctx;scoped:ctx")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--w", default='{"contract":2}')
    a = ap.parse_args()
    import embedder
    S = SlotSearch(str(HERE / "out/elements_u2.jsonl"), str(HERE / "out/tags_u2_rules.jsonl"))
    J = [json.loads(l) for l in open(HERE / "out/elements_u2jo.jsonl")]
    m2j = {m: j for j, u in enumerate(J) for m in u["members"]}
    eidx = {e["element_id"]: i for i, e in enumerate(S.E)}
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"]]
    QT = {json.loads(l)["qid"]: json.loads(l) for l in open(a.qtags)}
    arms = [x for x in a.arms.split(";") if x]
    views = sorted({x.split(":")[1] for x in arms if ":" in x})
    V = {v: np.load(HERE / "out/emb" / f"u2_{v}.npy") for v in views}
    Vids = {v: json.load(open(HERE / "out/emb" / f"u2_{v}_ids.json")) for v in views}
    qv = embedder.encode([g["q"] for g in G], batch=32) if views else None
    W = json.loads(a.w)
    hdr = ["R@1", "R@5", "R@10", "R@20", "R@40", "S@5", "RR@10"]
    agg = {arm: collections.defaultdict(list) for arm in arms}
    files = {arm: open(HERE / "out" / f"hyb_ranks_{arm.replace(':','_')}.jsonl", "w") for arm in arms}
    for qi, g in enumerate(G):
        lt = QT.get(g["qid"], {})
        slots = {k: lt.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
        slots = {k: v for k, v in slots.items() if v}
        _, toks = S.router.route(g["q"])
        M, L = S.match_table(slots, toks)
        tag_rank = [e["element_id"] for e, _ in S.rank(M, L, mode="clm", lex="count", weights=W, limit=a.topk)]
        vec_rank = {}
        for v in views:
            sims = V[v] @ qv[qi]
            top = np.argsort(-sims)[: a.topk]
            vec_rank[v] = [Vids[v][i] for i in top]
        for arm in arms:
            kind, _, view = arm.partition(":")
            if kind == "tag":
                ranked = tag_rank
            elif kind == "vec":
                ranked = vec_rank[view]
            elif kind == "rrf":
                ranked = rrf([tag_rank, vec_rank[view]])
            elif kind == "scoped":
                sc_ = set(slots.get("contract", []))
                if sc_:
                    sims = V[view] @ qv[qi]
                    pen = np.array([0.0 if S.E[eidx[eid]]["contract_scope"].replace("주계약(", "").rstrip(")") in sc_ or S.E[eidx[eid]]["contract_scope"] in sc_ else 0.15 for eid in Vids[view]], dtype="float32")
                    top = np.argsort(-(sims - pen))[: a.topk]
                    ranked = [Vids[view][i] for i in top]
                else:
                    ranked = vec_rank[view]
            seen, rj = set(), []
            for eid in ranked:
                j = m2j.get(eid)
                if j is not None and j not in seen:
                    seen.add(j); rj.append(J[j])
            sc = score(rj, g["groups"], ks=(1, 5, 10, 20, 40))
            for k, v in sc.items():
                agg[arm][k].append(v)
            agg[arm]["_core"].append(g["core_retrieval"] == "True")
            files[arm].write(json.dumps({"qid": g["qid"], "jo_top": [u["element_id"] for u in rj[:40]]}, ensure_ascii=False) + "\n")
    print(f"n={len(G)} gold={Path(a.gold).name}")
    print("| arm | " + " | ".join(hdr) + " | core R@5 | 비core R@5 |\n|---|" + "---|" * (len(hdr) + 2))
    for arm in arms:
        A = agg[arm]; core = A["_core"]; r5 = A["R@5"]
        c = [x for x, f in zip(r5, core) if f]; nc = [x for x, f in zip(r5, core) if not f]
        print(f"| {arm} | " + " | ".join(f"{sum(A[k])/len(G):.3f}" for k in hdr) + f" | {sum(c)/len(c):.3f} | {sum(nc)/len(nc):.3f} |")
    # 상보성 4분할 (S@5): tag vs 첫 vec view
    if views:
        v0 = f"vec:{views[0]}"
        if "tag" in arms and v0 in arms:
            ta, va = agg["tag"]["S@5"], agg[v0]["S@5"]
            print(f"S@5 4분할 tag vs {v0}: tag만 {sum(1 for x,y in zip(ta,va) if x and not y)} · vec만 {sum(1 for x,y in zip(ta,va) if y and not x)} · 둘다 {sum(1 for x,y in zip(ta,va) if x and y)} · 둘다실패 {sum(1 for x,y in zip(ta,va) if not x and not y)} · union {sum(1 for x,y in zip(ta,va) if x or y)/len(G):.3f}")


if __name__ == "__main__":
    main()
