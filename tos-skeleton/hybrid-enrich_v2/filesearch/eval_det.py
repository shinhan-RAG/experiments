#!/usr/bin/env python3
"""결정론 평가(LLM 0회): 라우터+SlotSearch 를 gold span 에 대해 채점. arm 은 --arms 로 지정.

arm 문법(세미콜론 구분): <mode>[:lex=<binary|count>][:tags=<0|1>][:w=<field>=<num>,...][:oracle=scope]
  구조화 단일단계: clm:ranker=bm25f[:profile=full|core|no_identity_split|no_variant|no_locator|no_extra]
  예) clm:lex=binary  |  clm:lex=count  |  and  |  clm:tags=0(어휘채널만=A0)  |  clm:w=contract=3
채점: 원 단위(u2) 와 조 map-back(u2jo) 둘 다. Recall@K = fractional evidence-group, Success@K, MRR@10.
문항별 순위를 out/det_ranks_<arm>.jsonl 로 남긴다(paired 검정용).
"""
import argparse, collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clm_search import SlotSearch
from scoring import score, overlaps


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
    ap.add_argument("--gold", default=str(HERE / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--arms", default="clm:lex=count;clm:lex=count:w=contract=3;and;clm:tags=0;clm:lex=count:oracle=scope;clm:lex=count:w=contract=3:oracle=scope")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--partial", default="0", help="라우터 부분 특약명 일치 허용(1)")
    ap.add_argument("--qtags", default="", help="LLM qtags jsonl (qtag_llm.py 출력)")
    ap.add_argument("--qmode", default="union", choices=("rule", "llm", "union"), help="라우터 슬롯 결합 방식")
    ap.add_argument("--qids", default="", help="선택 qid JSON 배열(스크리닝용)")
    a = ap.parse_args()
    S = SlotSearch(a.elements, a.tags)
    S.router.partial = a.partial == "1"
    J = [json.loads(l) for l in open(a.jo)]
    m2j = {m: j for j, u in enumerate(J) for m in u["members"]}
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"]
    if a.qids:
        keep_qids = set(json.load(open(a.qids, encoding="utf-8")))
        G = [g for g in G if g["qid"] in keep_qids]
    QT = {}
    if a.qtags:
        for l in open(a.qtags, encoding="utf-8"):
            d = json.loads(l); QT[d["qid"]] = d
    print(f"n={len(G)} tags={Path(a.tags).name}")
    hdr = ["R@1", "R@5", "R@10", "R@20", "R@40", "R@100", "R@200", "R@400", "S@5", "suff@10", "RR@10"]
    arms = [x for x in a.arms.split(";") if x]
    parsed = [parse_arm(x) for x in arms]
    need_match_table = any(opt.get("ranker") != "bm25f" for _, opt in parsed)
    print("| arm | 단위 | " + " | ".join(hdr) + " | core R@5 | 비core R@5 | 후보0 |\n|---|---|" + "---|" * (len(hdr) + 3))
    agg = {arm: collections.defaultdict(list) for arm in arms}; zero = collections.Counter()
    files = {arm: open(HERE / "out" / f"det_ranks_{arm.replace(':','_').replace('=','-').replace(',','+')}.jsonl", "w") for arm in arms}
    for g in G:
        slots, toks = S.router.route(g["q"])
        conf = slots.pop("_conf", {})
        if QT:
            lt = QT.get(g["qid"], {})
            llm = {k: lt.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
            llm = {k: v for k, v in llm.items() if v}
            if a.qmode == "llm":
                slots = llm; conf = {}
            elif a.qmode == "union":
                for k, v in llm.items():
                    slots[k] = list(dict.fromkeys(list(slots.get(k, [])) + v))
                if llm.get("contract"):
                    conf.pop("contract", None)  # LLM 이 특약을 지목하면 정신뢰도
        if need_match_table:
            M, L = S.match_table(slots, toks)
        else:
            M, L = [], S.lexical_counts(toks)
        M0 = [{} for _ in M]  # tags=0 용
        gscopes = {e["contract_scope"] for e in S.E if any(overlaps(e, gr) for gr in g["groups"])}
        for arm, (mode, opt) in zip(arms, parsed):
            w = {f: v * conf.get(f, 1.0) for f, v in opt["w"].items()} if opt.get("pconf") == "1" else opt["w"]
            S.variant_rr = opt.get("vrr") == "1"
            Muse = M
            if opt.get("slots"):  # 라우터 슬롯 제한(예: slots=contract+role) — AND 검색기 공정 대조용
                keep = set(opt["slots"].split("+"))
                Muse = [{f: v for f, v in m.items() if f in keep} for m in M]
            if opt.get("ranker") == "bm25f":
                res = S.rank_structured(slots, toks, g["q"], weights=w,
                                        profile=opt.get("profile", "full"), limit=a.limit,
                                        lexical_counts=L)
            elif opt.get("decomp") == "role" and len(slots.get("role", [])) >= 2:
                # A4: 역할별 하위질의 → 라운드로빈 interleave (multi-evidence 겨냥)
                subs = []
                for r_ in slots["role"]:
                    ss = dict(slots); ss["role"] = [r_]
                    Mr, Lr = S.match_table(ss, toks)
                    subs.append(S.rank(Mr, Lr, mode=mode, lex=opt["lex"], weights=w, limit=a.limit, rare=opt.get("rare") == "1", n_tokens=len(toks), rare_cap=float(opt["cap"]) if opt.get("cap") else None))
                seen_e, res = set(), []
                for k_ in range(a.limit):
                    for sub in subs:
                        if k_ < len(sub) and sub[k_][0]["element_id"] not in seen_e:
                            seen_e.add(sub[k_][0]["element_id"]); res.append(sub[k_])
                res = res[: a.limit]
            else:
                res = S.rank(M0 if opt["tags"] == "0" else Muse, L, mode=mode, lex=opt["lex"], weights=w, limit=a.limit,
                             scope_filter=gscopes if opt.get("oracle") == "scope" else None, rare=opt.get("rare") == "1", n_tokens=len(toks), rare_cap=float(opt["cap"]) if opt.get("cap") else None)
            if not res:
                zero[arm] += 1
            ru = [e for e, _ in res]
            seen, rj = set(), []
            for e in ru:
                j = m2j[e["element_id"]]
                if j not in seen:
                    seen.add(j); rj.append(J[j])
            for unit, ranked in (("u2", ru), ("jo", rj)):
                sc = score(ranked, g["groups"], ks=(1, 5, 10, 20, 40, 100, 200, 400))
                for k, v in sc.items():
                    agg[arm][(unit, k)].append(v)
                agg[arm][(unit, "_core")].append(g["core_retrieval"] == "True")
            files[arm].write(json.dumps({"qid": g["qid"], **sc, "slots": slots, "n_tokens": len(toks), "n_cand": len(res),
                                         "u2_top": [e["element_id"] for e in ru[:a.limit]], "jo_top": [u["element_id"] for u in rj[:a.limit]]}, ensure_ascii=False) + "\n")
    for arm in arms:
        for unit in ("u2", "jo"):
            A = agg[arm]
            row = [f"{sum(A[(unit,k)])/len(G):.3f}" for k in hdr]
            core = A[(unit, "_core")]; r5 = A[(unit, "R@5")]
            c = [x for x, f_ in zip(r5, core) if f_]; nc = [x for x, f_ in zip(r5, core) if not f_]
            print(f"| {arm} | {unit} | " + " | ".join(row) + f" | {sum(c)/len(c):.3f} | {sum(nc)/len(nc):.3f} | {zero[arm]} |")


if __name__ == "__main__":
    main()
