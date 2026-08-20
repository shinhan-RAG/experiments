#!/usr/bin/env python3
"""결정론 평가(LLM 0회) — 0819판. filesearch/eval_det.py 사본에 alias 확장 arm 옵션과
폴백 τ 튜닝용 통계를 추가했다. 데이터·채점은 filesearch/ 를 참조한다.

arm 문법(세미콜론 구분): <mode>[:lex=<binary|count>][:tags=<0|1>][:w=<field>=<num>,...][:alias=1]
  예) clm:lex=count:w=contract=2  |  clm:lex=count:w=contract=2:alias=1
추가 출력: 문항별 top1 CLM score·n_cand·gold 적중 여부(det_ranks) — tune_tau.py 가 τ 를 고른다.
"""
import argparse, collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
from clm_search import SlotSearch
from scoring import score, overlaps
import enhance


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
    ap.add_argument("--elements", default=str(FS / "out/elements_u2.jsonl"))
    ap.add_argument("--tags", default=str(FS / "out/tags_u2_rules.jsonl"))
    ap.add_argument("--jo", default=str(FS / "out/elements_u2jo.jsonl"))
    ap.add_argument("--gold", default=str(FS / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--arms", default="clm:lex=count:w=contract=2;clm:lex=count:w=contract=2:alias=1")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--partial", default="0", help="라우터 부분 특약명 일치 허용(1)")
    ap.add_argument("--qtags", default=str(FS / "out/qtags_haiku.jsonl"), help="LLM qtags jsonl (없으면 빈 문자열)")
    ap.add_argument("--qmode", default="union", choices=("rule", "llm", "union"))
    a = ap.parse_args()
    S = SlotSearch(a.elements, a.tags)
    S.router.partial = a.partial == "1"
    J = [json.loads(l) for l in open(a.jo)]
    m2j = {m: j for j, u in enumerate(J) for m in u["members"]}
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"]
    QT = {}
    if a.qtags:
        for l in open(a.qtags, encoding="utf-8"):
            d = json.loads(l); QT[d["qid"]] = d
    print(f"n={len(G)} tags={Path(a.tags).name}")
    hdr = ["R@1", "R@5", "R@10", "R@20", "R@40", "R@100", "S@5", "suff@10", "RR@10"]
    arms = [x for x in a.arms.split(";") if x]
    parsed = [parse_arm(x) for x in arms]
    any_alias = any(opt.get("alias") == "1" for _, opt in parsed)
    print("| arm | 단위 | " + " | ".join(hdr) + " | core R@5 | 비core R@5 | 후보0 |\n|---|---|" + "---|" * (len(hdr) + 3))
    agg = {arm: collections.defaultdict(list) for arm in arms}; zero = collections.Counter()
    (HERE / "out").mkdir(exist_ok=True)
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
                    conf.pop("contract", None)
        M, L = S.match_table(slots, toks)
        toks_a, alias_log = toks, {}
        Ma, La = M, L
        if any_alias:
            extra, alias_log = enhance.expand_query(g["q"], toks)
            if extra:
                toks_a = toks + extra
                Ma, La = S.match_table(slots, toks_a)
        for arm, (mode, opt) in zip(arms, parsed):
            use_alias = opt.get("alias") == "1"
            Mu, Lu, tk = (Ma, La, toks_a) if use_alias else (M, L, toks)
            w = opt["w"]
            res = S.rank(Mu, Lu, mode=mode, lex=opt["lex"], weights=w, limit=a.limit, rare=opt.get("rare") == "1", n_tokens=len(tk))
            if not res:
                zero[arm] += 1
            ru = [e for e, _ in res]
            seen, rj = set(), []
            for e in ru:
                j = m2j[e["element_id"]]
                if j not in seen:
                    seen.add(j); rj.append(J[j])
            for unit, ranked in (("u2", ru), ("jo", rj)):
                sc = score(ranked, g["groups"], ks=(1, 5, 10, 20, 40, 100))
                for k, v in sc.items():
                    agg[arm][(unit, k)].append(v)
                agg[arm][(unit, "_core")].append(g["core_retrieval"] == "True")
            # 폴백 τ 튜닝용: top1 점수·후보 수·gold 가 태그 결과(전체 limit)에 존재하는지
            gold_in = any(any(overlaps(e, gr) for gr in g["groups"]) for e in ru)
            files[arm].write(json.dumps({"qid": g["qid"], "slots": slots, "n_tokens": len(tk), "n_cand": len(res),
                                         "top_score": res[0][1] if res else 0.0, "gold_in_cand": gold_in,
                                         **({"alias_expanded": alias_log} if use_alias and alias_log else {}),
                                         "u2_top": [e["element_id"] for e in ru[:40]], "jo_top": [u["element_id"] for u in rj[:40]]}, ensure_ascii=False) + "\n")
    for arm in arms:
        for unit in ("u2", "jo"):
            A = agg[arm]
            row = [f"{sum(A[(unit,k)])/len(G):.3f}" for k in hdr]
            core = A[(unit, "_core")]; r5 = A[(unit, "R@5")]
            c = [x for x, f_ in zip(r5, core) if f_]; nc = [x for x, f_ in zip(r5, core) if not f_]
            print(f"| {arm} | {unit} | " + " | ".join(row) + f" | {sum(c)/len(c):.3f} | {sum(nc)/len(nc):.3f} | {zero[arm]} |")


if __name__ == "__main__":
    main()
