#!/usr/bin/env python3
"""Step 0 — 도달 상한(ceiling) 분석.

질문: 현 검색기(태그 search + 메타 msearch) 인덱스로 R@5=90% 가 물리적으로 가능한가?
방법: 결정론 기준으로 태그 top-40 ∪ 메타 top-40 의 gold 조 커버리지를 그룹 단위로 측정하고
문항을 3분류한다: ①어떤 채널에도 없음(인덱스/코퍼스 한계) ②합집합에는 있으나 top-10 밖(순위
레버) ③top-10 안(제출 정책 레버). 결과가 90% 미만이면 에이전트 정책으로는 90% 불가.

입력: eval_det.py 가 만든 out/det_ranks_*.jsonl (태그 채널 jo_top 40),
      vector_search.ChunkHybridSearch (메타 채널, EMBED_ENDPOINT 필요).
출력: out/ceiling_train.json + 표.
"""
import argparse, collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
sys.path.insert(0, str(FS)); sys.path.insert(0, str(VS))
from scoring import overlaps
from units import Units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(FS / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--ranks", default=str(HERE / "out/det_ranks_clm_lex-count_w-contract-2.jsonl"))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--qids", default="", help="선택: qid 목록 json (train60 등 표본 한정)")
    a = ap.parse_args()

    G = [g for g in (json.loads(l) for l in open(a.gold, encoding="utf-8")) if g["groups"] and g.get("status", "ok") == "ok"]
    if a.qids:
        keep = set(json.load(open(a.qids)))
        G = [g for g in G if g["qid"] in keep]
    tag = {json.loads(l)["qid"]: json.loads(l) for l in open(a.ranks, encoding="utf-8")}
    U = Units()
    jix = {u["element_id"]: u for u in U.J}

    from hybrid_search import ChunkHybridSearch
    hs = ChunkHybridSearch(view="V9")

    def jo_units(ids):
        return [jix[i] for i in ids if i in jix]

    cls = collections.Counter()          # 문항 3분류 (모든 그룹 기준 AND)
    grp = collections.Counter()          # 그룹 단위 커버리지
    rows = []
    for g in G:
        t = tag.get(g["qid"], {})
        tag_jo = jo_units(t.get("jo_top", [])[: a.k])
        res = hs.search(g["q"], strategy="hybrid", top_k=a.k)
        meta_jo, seen = [], set()
        for r in res:
            if r.get("char_start") is None:
                continue
            u = U.jo_of_span(r["char_start"], r["char_end"])
            if u and u["element_id"] not in seen:
                seen.add(u["element_id"]); meta_jo.append(u)
        # 합집합(태그 순위 우선, 이후 메타)
        seen2, union = set(), []
        for u in tag_jo + meta_jo:
            if u["element_id"] not in seen2:
                seen2.add(u["element_id"]); union.append(u)
        gstat = []
        for gr in g["groups"]:
            in_tag = any(overlaps(u, gr) for u in tag_jo)
            in_meta = any(overlaps(u, gr) for u in meta_jo)
            rank_u = next((i + 1 for i, u in enumerate(union) if overlaps(u, gr)), None)
            grp["in_tag40"] += in_tag; grp["in_meta40"] += in_meta
            grp["in_union"] += rank_u is not None
            grp["in_union_top10"] += (rank_u is not None and rank_u <= 10)
            grp["total"] += 1
            gstat.append({"in_tag": in_tag, "in_meta": in_meta, "union_rank": rank_u})
        if any(s["union_rank"] is None for s in gstat):
            c = "①미포착(인덱스한계)"
        elif all(s["union_rank"] <= 10 for s in gstat):
            c = "③top10내(제출정책레버)"
        else:
            c = "②순위밖(순위레버)"
        cls[c] += 1
        rows.append({"qid": g["qid"], "n_groups": len(g["groups"]), "core": g["core_retrieval"] == "True",
                     "task_type": g["task_type"], "class": c, "groups": gstat})
        if len(rows) % 40 == 0:
            print(f"  ...{len(rows)}/{len(G)}", flush=True)

    n = len(G)
    print(f"\n=== ceiling (n={n}, k={a.k}) ===")
    for c, v in sorted(cls.items()):
        print(f"  {c}: {v} ({100*v/n:.1f}%)")
    t = grp["total"]
    print(f"  그룹 단위: tag40 {grp['in_tag40']}/{t} ({100*grp['in_tag40']/t:.1f}%) | meta40 {grp['in_meta40']}/{t} ({100*grp['in_meta40']/t:.1f}%) | union {grp['in_union']}/{t} ({100*grp['in_union']/t:.1f}%)")
    print(f"  ** R@∞(union) 상한 = 그룹 커버리지 {100*grp['in_union']/t:.1f}% — 완벽한 에이전트라도 이 값이 R@5/suff 상한 근사 **")
    suffable = sum(1 for r in rows if r["class"] != "①미포착(인덱스한계)")
    print(f"  suff 상한(모든 그룹이 union 에 존재): {suffable}/{n} ({100*suffable/n:.1f}%)")
    json.dump({"n": n, "k": a.k, "class": dict(cls), "groups": dict(grp), "rows": rows},
              open(HERE / "out/ceiling_train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> out/ceiling_train.json")


if __name__ == "__main__":
    main()
