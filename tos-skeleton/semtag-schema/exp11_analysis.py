#!/usr/bin/env python3
"""실험 11 웨이브 3 판독 — 주표 · 층화 3종 · 반증 3종 (PREREG_EXP11.md 판독 계획 준수)."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
import stats
from context_v2 import annotate_v2

HERE = Path(__file__).parent
OUT = HERE / "out/exp11"
DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
       "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
ORACLE = {"R1": 0.5208, "R2": 0.6771, "R3": 0.7917, "R4": 0.6771}


def kendall_tau(a, b):
    """두 순열(동일 원소 집합)의 Kendall τ."""
    pos_b = {x: i for i, x in enumerate(b)}
    seq = [pos_b[x] for x in a if x in pos_b]
    n = len(seq)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=HERE).stdout.strip()
    lines = SE.nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    byid = {e["eid"]: e for e in els}
    mapped = {str(m["qid"]): m for m in
              (json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8"))}
    ga = json.loads((OUT / "gold_ambiguity_dev.json").read_text())
    amb_q = {r["qid"] for r in ga["judgements"] if r["gold_ambiguous"]}
    dupA_q = {r["qid"] for r in ga["judgements"]}
    ns = lambda s: re.sub(r"\s+", "", s)
    sigA = Counter(ns(e["text"])[:300] for e in els if len(ns(e["text"])) >= 300)
    runs = {r: json.loads((OUT / f"run_{r}.json").read_text()) for r in ("R1", "R2", "R3", "R4")}
    qids = sorted(runs["R1"]["results"], key=int)

    def hit5(order, qid):
        g = set(mapped[qid]["gold"])
        return int(any(e in g for e in order[:5]))

    def rr10(order, qid):
        g = set(mapped[qid]["gold"])
        r = next((i + 1 for i, e in enumerate(order[:10]) if e in g), None)
        return (1 / r) if r else 0.0

    base5 = {q: hit5(runs["R1"]["results"][q]["input"], q) for q in qids}
    base_rr = {q: rr10(runs["R1"]["results"][q]["input"], q) for q in qids}

    # ── 주표
    main_rows, pvals = {}, []
    for r in ("R1", "R2", "R3", "R4"):
        res = runs[r]["results"]
        h = {q: hit5(res[q]["output"], q) for q in qids}
        rr = {q: rr10(res[q]["output"], q) for q in qids}
        d5 = [h[q] - base5[q] for q in qids]
        imp, reg = sum(1 for x in d5 if x > 0), sum(1 for x in d5 if x < 0)
        p = stats.mcnemar_exact(imp, reg)
        dm = [rr[q] - base_rr[q] for q in qids]
        main_rows[r] = {"r5": round(sum(h.values()) / len(qids), 4),
                        "mrr10": round(sum(rr.values()) / len(qids), 4),
                        "oracle": ORACLE[r],
                        "delta_r5": round(sum(d5) / len(d5), 4),
                        "improve": imp, "regress": reg, "mcnemar_p": round(p, 4),
                        "delta_mrr": round(sum(dm) / len(dm), 4),
                        "mrr_bca": [round(x, 4) for x in stats.bca_ci(dm)],
                        "parse_fail_windows": runs[r]["parse_fail_windows"],
                        "usage": runs[r]["usage"]}
        if r in ("R1", "R2", "R3"):
            pvals.append(p)
    holm_p = stats.holm(pvals)
    for i, r in enumerate(("R1", "R2", "R3")):
        main_rows[r]["holm_p"] = round(holm_p[i], 4)

    # ── 층화 3종 (R@5, 층별)
    def strat(pred, label):
        rows = {}
        for name, sel in pred.items():
            sub = [q for q in qids if sel(q)]
            if not sub:
                continue
            row = {"n": len(sub), "A0": round(sum(base5[q] for q in sub) / len(sub), 4)}
            for r in ("R1", "R2", "R3"):
                res = runs[r]["results"]
                row[r] = round(sum(hit5(res[q]["output"], q) for q in sub) / len(sub), 4)
            rows[name] = row
        return {"strata": label, "rows": rows}

    def gtype(q):
        g0 = mapped[q]["gold"][0]
        return byid[g0]["type"] if g0 in byid else "?"

    strata = [
        strat({"모호": lambda q: q in amb_q, "명확": lambda q: q not in amb_q},
              "gold_ambiguous (트랙 A)"),
        strat({"DUP_A 중복군": lambda q: q in dupA_q, "고유 본문": lambda q: q not in dupA_q},
              "DUP_A 중복/고유"),
        strat({"table": lambda q: gtype(q) == "table", "formula": lambda q: gtype(q) == "formula",
               "text": lambda q: gtype(q) == "text"}, "gold 유형(첫 gold 기준, 서술만·검정 금지)"),
    ]

    # ── 반증 ①: R2 vs R4 (동일 후보·제시 순서만 반전)
    taus, agree5 = [], 0
    for q in qids:
        o2, o4 = runs["R2"]["results"][q]["output"], runs["R4"]["results"][q]["output"]
        taus.append(kendall_tau(o2, o4))
        agree5 += len(set(o2[:5]) & set(o4[:5])) / 5
    d5_24 = [hit5(runs["R2"]["results"][q]["output"], q) -
             hit5(runs["R4"]["results"][q]["output"], q) for q in qids]
    i24, r24 = sum(1 for x in d5_24 if x > 0), sum(1 for x in d5_24 if x < 0)
    ref1 = {"mean_tau_R2_R4": round(sum(taus) / len(taus), 3),
            "top5_overlap": round(agree5 / len(qids), 3),
            "delta_r5_R2_minus_R4": round(sum(d5_24) / len(d5_24), 4),
            "transitions": f"{i24}/{r24}", "mcnemar_p": round(stats.mcnemar_exact(i24, r24), 4)}

    # ── 반증 ②: BM25 에코 (출력 vs 입력 τ)
    ref2 = {}
    for r in ("R1", "R2", "R3", "R4"):
        ts = [kendall_tau(runs[r]["results"][q]["output"], runs[r]["results"][q]["input"])
              for q in qids]
        ref2[r] = round(sum(ts) / len(ts), 3)

    # ── 반증 ③: 실패 전수 분류 (R2 기준 — gold가 후보 50 내인데 top-5 실패)
    fails = []
    for q in qids:
        res = runs["R2"]["results"][q]
        g = set(mapped[q]["gold"])
        in_cand = any(e in g for e in res["input"])
        if not in_cand or hit5(res["output"], q):
            continue
        rank = next((i + 1 for i, e in enumerate(res["output"]) if e in g), None)
        g0 = mapped[q]["gold"][0]
        twin = (g0 in byid and len(ns(byid[g0]["text"])) >= 300
                and sigA.get(ns(byid[g0]["text"])[:300], 0) > 1)
        fails.append({"qid": q, "gold_rank_after": rank,
                      "bucket": ("6-10" if rank and rank <= 10 else
                                 "11-20" if rank and rank <= 20 else ">20"),
                      "gold_type": gtype(q), "dupA_gold": q in dupA_q,
                      "gold_ambiguous": q in amb_q, "twin_exists": twin,
                      "parse_fail": bool(res["parse_fails"])})
    ref3 = {"population": len(fails),
            "by_bucket": dict(Counter(f["bucket"] for f in fails)),
            "by_type": dict(Counter(f["gold_type"] for f in fails)),
            "dupA": sum(1 for f in fails if f["dupA_gold"]),
            "ambiguous": sum(1 for f in fails if f["gold_ambiguous"]),
            "twin": sum(1 for f in fails if f["twin_exists"]),
            "parse_fail_q": sum(1 for f in fails if f["parse_fail"]),
            "detail": fails}

    out = {"code_commit": commit,
           "A0": {"r5": round(sum(base5.values()) / len(qids), 4),
                  "mrr10": round(sum(base_rr.values()) / len(qids), 4)},
           "main": main_rows, "strata": strata,
           "refute_position_bias": ref1, "refute_bm25_echo": ref2,
           "refute_failure_census": ref3}
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "refute_failure_census"},
                     ensure_ascii=False, indent=1))
    print("failure census:", json.dumps({k: v for k, v in ref3.items() if k != "detail"},
                                        ensure_ascii=False))


if __name__ == "__main__":
    main()
