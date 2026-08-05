#!/usr/bin/env python3
"""실험 5c — 조(條) 그룹 재순위화 (BM25 후처리, 전 과정 결정적).

BM25 점수 산출 후, 상위 K 후보를 (scope, jo) 키로 그룹핑하고 그룹 점수로
그룹을 정렬(그룹 내부는 element 점수 순)해 top-10을 재구성한다.
jo가 없는 element는 자기 자신을 단독 그룹으로 취급.

그리드: K ∈ {20, 50} × 그룹점수 ∈ {max, sum, max+0.3×2위}
평가: dev 96만. R@5 / R@10 / MRR@10 + paired bootstrap CI + R@5 전환 +
근접실패(원 gold 6~10위) 이동 분해.

usage:
  python3 exp5c_jorerank.py --doc <260507.md> --out out/exp5c
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).resolve().parent
GOLD = HERE / "out" / "gold_mapped.jsonl"
DEVQ = HERE.parent / "hybrid-enrich" / "out" / "qa100_gold.jsonl"


def full_scores(bm, query):
    """BM25 클래스 내부 로직 재사용 — 전체 element 점수 배열."""
    q = list(Counter(SE.bigrams(query)))
    scores = []
    for i in range(bm.N):
        s = 0.0
        tfi = bm.tf[i]
        for t in q:
            f = tfi.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (bm.N - bm.df[t] + 0.5) / (bm.df[t] + 0.5))
            B = 1 - bm.b + bm.b * (bm.dl[i] / bm.avgdl)
            s += idf * f * (bm.k1 + 1) / (f + bm.k1 * B)
        scores.append(s)
    return scores


def group_score(scores_desc, mode):
    if mode == "max":
        return scores_desc[0]
    if mode == "sum":
        return sum(scores_desc)
    if mode == "max2":  # max + 0.3×2위
        return scores_desc[0] + 0.3 * (scores_desc[1] if len(scores_desc) > 1 else 0.0)
    raise ValueError(mode)


def rerank(order, scores, keys, K, mode):
    """order: 전체 내림차순 인덱스. 상위 K를 (scope,jo) 그룹 재순위 → top-10."""
    head = order[:K]
    rank_of = {i: r for r, i in enumerate(head)}
    groups = {}  # key -> [element index...]
    for i in head:
        groups.setdefault(keys[i], []).append(i)
    ordered_groups = []
    for key, members in groups.items():
        members.sort(key=lambda i: (-scores[i], rank_of[i]))
        gs = group_score([scores[i] for i in members], mode)
        best_rank = min(rank_of[i] for i in members)
        ordered_groups.append((-gs, best_rank, members))
    ordered_groups.sort(key=lambda t: (t[0], t[1]))
    flat = [i for _, _, members in ordered_groups for i in members]
    return flat[:10]


def eval_rows(dev, top10_by_qid, gold_by_qid):
    rows = []
    for it in dev:
        g = gold_by_qid[it["qid"]]
        r = next((k + 1 for k, eid in enumerate(top10_by_qid[it["qid"]]) if eid in g), None)
        rows.append({"qid": it["qid"], "rank": r,
                     "r5": int(bool(r and r <= 5)), "r10": int(bool(r)),
                     "mrr": (1 / r) if r else 0.0})
    return rows


def agg(rows):
    n = len(rows)
    return {"r5": round(sum(r["r5"] for r in rows) / n, 4),
            "r10": round(sum(r["r10"] for r in rows) / n, 4),
            "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}


def run_once(doc_path):
    raw = Path(doc_path).read_text(encoding="utf-8", errors="ignore")
    doc_sha = hashlib.sha256(raw.encode()).hexdigest()
    lines = SE.nfc(raw).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    n_nojo = sum(1 for e in els if not e["jo"])

    # 그룹 키: (scope, jo); jo 없으면 자기 자신 단독 그룹
    keys = [(e["scope"], e["jo"]) if e["jo"] else ("__self__", e["eid"]) for e in els]

    mapped = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    dev_qids = {json.loads(l)["qid"] for l in DEVQ.read_text(encoding="utf-8").splitlines() if l.strip()}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    gold_by_qid = {m["qid"]: set(m["gold"]) for m in dev}

    bm = SE.BM25([e["text"] for e in els])
    eid = [e["eid"] for e in els]

    # 질의별 전체 점수·전체 순위 (한 번만 산출, 전 조합 공유)
    per_q = {}
    for it in dev:
        scores = full_scores(bm, it["q"])
        order = sorted(range(bm.N), key=lambda i: -scores[i])
        per_q[it["qid"]] = (scores, order)

    # ── A0 기준선 (rank10과 동일 로직) ──
    a0_top10 = {q: [eid[i] for i in per_q[q][1][:10]] for q in per_q}
    a0_rows = eval_rows(dev, a0_top10, gold_by_qid)
    a0 = agg(a0_rows)
    a0_rank = {r["qid"]: r["rank"] for r in a0_rows}
    near_miss = sorted((q for q, r in a0_rank.items() if r and 6 <= r <= 10), key=str)
    a0_top5 = sorted((q for q, r in a0_rank.items() if r and r <= 5), key=str)

    variants = {}
    for K in (20, 50):
        for mode in ("max", "sum", "max2"):
            name = f"K{K}_{mode}"
            top10 = {}
            for it in dev:
                scores, order = per_q[it["qid"]]
                top10[it["qid"]] = [eid[i] for i in rerank(order, scores, keys, K, mode)]
            rows = eval_rows(dev, top10, gold_by_qid)
            m = agg(rows)
            am = {r["qid"]: r for r in a0_rows}
            d_mrr = [r["mrr"] - am[r["qid"]]["mrr"] for r in rows]
            d_r5 = [r["r5"] - am[r["qid"]]["r5"] for r in rows]
            rk = {r["qid"]: r["rank"] for r in rows}
            gains = sorted(q for q in near_miss if rk[q] and rk[q] <= 5)
            regress = sorted(q for q in a0_top5 if not (rk[q] and rk[q] <= 5))
            improved = sorted(r["qid"] for r in rows if r["r5"] and not am[r["qid"]]["r5"])
            m.update({
                "d_mrr_vs_A0": round(sum(d_mrr) / len(d_mrr), 4),
                "d_mrr_ci95": [round(x, 4) for x in SE.boot_ci(d_mrr)],
                "d_r5_vs_A0": round(sum(d_r5) / len(d_r5), 4),
                "d_r5_ci95": [round(x, 4) for x in SE.boot_ci(d_r5)],
                "r5_flips": {"improved": improved, "regressed": regress,
                             "n_improved": len(improved), "n_regressed": len(regress)},
                "near_miss_moved_into_top5": {"n": len(gains), "qids": gains},
                "near_miss_ranks_after": {q: rk[q] for q in near_miss},
            })
            variants[name] = m

    return {
        "doc_sha256": doc_sha,
        "n_elements": len(els),
        "n_elements_without_jo": n_nojo,
        "n_dev": len(dev),
        "A0": {**a0,
               "near_miss_6_10": {"n": len(near_miss), "qids": near_miss,
                                  "ranks": {q: a0_rank[q] for q in near_miss}},
               "n_top5": len(a0_top5)},
        "variants": variants,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", default=str(HERE / "out" / "exp5c"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res1 = run_once(args.doc)

    # 게이트: A0 기준선 재현
    if not (res1["A0"]["r5"] == 0.3125 and res1["A0"]["mrr10"] == 0.2647):
        print(f"GATE FAIL: A0 r5={res1['A0']['r5']} mrr10={res1['A0']['mrr10']} "
              f"(기대 .3125 / .2647) — 중단")
        (out / "results.json").write_text(
            json.dumps({"gate": "FAIL", "A0": res1["A0"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise SystemExit(1)
    print(f"gate OK: A0 dev{res1['n_dev']} R@5={res1['A0']['r5']} MRR@10={res1['A0']['mrr10']}")

    # 결정성: 동일 실행 2회 수치 일치
    res2 = run_once(args.doc)
    h1 = hashlib.sha256(json.dumps(res1, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(res2, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    res1["determinism"] = {"run1_sha256": h1, "run2_sha256": h2, "identical": h1 == h2}
    print(f"determinism: {'OK' if h1 == h2 else 'MISMATCH'} ({h1[:12]} / {h2[:12]})")

    (out / "results.json").write_text(
        json.dumps(res1, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, m in res1["variants"].items():
        print(f"{name:10s} R@5={m['r5']:.4f} R@10={m['r10']:.4f} MRR={m['mrr10']:.4f} "
              f"dMRR={m['d_mrr_vs_A0']:+.4f} CI={m['d_mrr_ci95']} "
              f"+{m['r5_flips']['n_improved']}/-{m['r5_flips']['n_regressed']} "
              f"nearmiss→top5={m['near_miss_moved_into_top5']['n']}")


if __name__ == "__main__":
    main()
