#!/usr/bin/env python3
"""실험 10 평가 — 합성 고객 구어 질의셋(cq)에서 어휘 간극 축 재검정. 결정적.

사전 등록:
- 평가는 동결본(out/exp10/cq_frozen.jsonl)만 사용. gold·분할은 기존 그대로 재사용.
- dev arms(각 1변수, 기준 A0): ① S2Q 문서측 별칭(태그 concat) ② Q1 질의측 확장
  (고객 표현→문서 용어 부가). 조합은 개별 판정 후에만.
- 판정: dev paired CI가 0을 제외하는 arm만 test(193) 1회 개봉 — 실행은 상위 세션.
- 참조: 동일 뷰에서 원 질문(orig) 수치 병기 — 분포 이동 자체의 효과 확인용.
- 한계: 결론은 합성 고객 구어 분포에 한정(실제 고객 발화 일반화는 미검증).

usage: python3 exp10_eval.py dev|test
"""
import json
import sys
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2
from semtag_h3 import Q_BRIDGE, build_tag_v2

HERE = Path(__file__).parent
DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
       "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
DEVQ = HERE.parent / "hybrid-enrich/out/qa100_gold.jsonl"

BRIDGE_REV = []
for k, v in Q_BRIDGE.items():
    for p in v.split(";"):
        p = p.strip()
        if p:
            BRIDGE_REV.append((p, k))


def expand(q):
    add = []
    for phrase, term in BRIDGE_REV:
        if phrase in q and term not in q and term not in add:
            add.append(term)
    return (q + " " + " ".join(add)) if add else q, bool(add)


def evaluate(bm, els, items, qkey):
    rows = []
    for it in items:
        top = bm.rank10(it[qkey])
        g = it["gold"]
        r = next((i + 1 for i, ix in enumerate(top) if els[ix]["eid"] in g), None)
        rows.append({"qid": it["qid"], "rank": r,
                     "r5": int(bool(r and r <= 5)), "r10": int(bool(r)),
                     "mrr": (1 / r) if r else 0.0})
    return rows


def agg(rows):
    n = len(rows)
    return {"r5": round(sum(r["r5"] for r in rows) / n, 4),
            "r10": round(sum(r["r10"] for r in rows) / n, 4),
            "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}


def paired(base, var):
    d = [v["mrr"] - b["mrr"] for b, v in zip(base, var)]
    d5 = [v["r5"] - b["r5"] for b, v in zip(base, var)]
    return {"delta_mrr": round(sum(d) / len(d), 4),
            "mrr_ci95": [round(x, 4) for x in SE.boot_ci(d)],
            "delta_r5": round(sum(d5) / len(d5), 4),
            "r5_ci95": [round(x, 4) for x in SE.boot_ci(d5)],
            "r5_transitions": {"improve": sum(1 for x in d5 if x > 0),
                               "regress": sum(1 for x in d5 if x < 0)}}


def main():
    split = sys.argv[1]
    assert split in ("dev", "test")
    lines = SE.nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)

    mapped = {str(m["qid"]): m for m in
              (json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8"))}
    dev_qids = {str(json.loads(l)["qid"]) for l in open(DEVQ)}
    frozen = [json.loads(l) for l in open(HERE / "out/exp10/cq_frozen.jsonl", encoding="utf-8")]
    items = []
    for d in frozen:
        qid = str(d["qid"])
        in_dev = qid in dev_qids
        if (split == "dev") != in_dev:
            continue
        cq_q1, fired = expand(d["cq"])
        items.append({"qid": qid, "orig": d["q_orig"], "cq": d["cq"], "cq_q1": cq_q1,
                      "fired": fired, "fallback": d["fallback"],
                      "gold": set(mapped[qid]["gold"])})
    n = len(items)

    bm_a0 = SE.BM25([e["text"] for e in els])
    views = []
    for e in els:
        t = build_tag_v2(e, "s2q")
        views.append((t + " ||| " + e["text"]) if t else e["text"])
    bm_s2q = SE.BM25(views)

    a0_cq = evaluate(bm_a0, els, items, "cq")
    res = {
        "split": split, "n": n,
        "set_properties": {
            "fallback": sum(1 for it in items if it["fallback"]),
            "docterm_rate_orig": round(sum(1 for it in items if any(k in it["orig"] for k in Q_BRIDGE)) / n, 3),
            "docterm_rate_cq": round(sum(1 for it in items if any(k in it["cq"] for k in Q_BRIDGE)) / n, 3),
            "custphrase_rate_orig": round(sum(1 for it in items if any(p in it["orig"] for p, _ in BRIDGE_REV)) / n, 3),
            "custphrase_rate_cq": round(sum(1 for it in items if any(p in it["cq"] for p, _ in BRIDGE_REV)) / n, 3),
            "q1_fired": sum(1 for it in items if it["fired"]),
        },
        "arms": {
            "A0_orig(참조)": agg(evaluate(bm_a0, els, items, "orig")),
            "A0_cq(기준)": agg(a0_cq),
            "S2Q_cq(문서측 별칭)": agg(evaluate(bm_s2q, els, items, "cq")),
            "Q1_cq(질의측 확장)": agg(evaluate(bm_a0, els, items, "cq_q1")),
        },
        "paired_vs_A0cq": {
            "S2Q": paired(a0_cq, evaluate(bm_s2q, els, items, "cq")),
            "Q1": paired(a0_cq, evaluate(bm_a0, els, items, "cq_q1")),
        },
    }
    out = HERE / f"out/exp10/results_{split}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
