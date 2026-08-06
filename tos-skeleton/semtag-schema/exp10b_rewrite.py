#!/usr/bin/env python3
"""실험 10 수정 이력(10b) — Q2 arm: LLM 질의 정규화(고객 구어→약관 문체 재작성).

등록 사유: dev에서 사전 기반 처방 2종이 실패한 기제가 "정확 문자열 매칭은 표현
변이(조사·어미)를 못 따라감"으로 특정됨 → 열거(사전)가 아니라 일반화(LLM 재작성)를
검증할 필요. 서빙 시 질의당 LLM 1회 비용이 드는 전략이므로 인프라 비용 병기.

누수 통제: 재작성기는 cq(합성 구어)만 봄 — 원 질문(orig)·gold·문서 미제공.
절차: dev 96 재작성(캐시)→동결→A0 인덱스에서 채점, paired vs A0_cq.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).parent
OUT = HERE / "out/exp10"
CACHE = OUT / "rw"
DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
       "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
DEVQ = HERE.parent / "hybrid-enrich/out/qa100_gold.jsonl"

PROMPT = """아래는 보험 고객이 전화로 물은 질문이다. 이 질문을 보험 약관 문서에서 쓰는
공식 용어와 문어체 표현으로 다시 써라. 규칙:
1. 일상 표현을 약관 용어로 바꿔라 (예: "보험료 안 내도 되는" → "보험료 납입면제").
2. 특약 이름·상품 이름은 그대로 유지하라.
3. 묻는 내용(의미)은 바꾸지 말고, 새 정보를 추가하지 마라.
4. 한 문장.

질문: {q}

한 줄 JSON만 출력: {{"q": "다시 쓴 질문"}}"""

TK = re.compile(r"[가-힣A-Za-z0-9()\[\]·]+특약")


def valid(src, rw):
    if not rw or not (8 <= len(rw) <= 160) or not re.search(r"[가-힣]", rw):
        return False
    return all(tk in rw for tk in TK.findall(src))


def rewrite_one(d):
    qid = str(d["qid"])
    f = CACHE / f"{qid}.json"
    if f.exists():
        c = json.loads(f.read_text())
        if c.get("rw"):
            return qid, c
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    rw, fb = "", False
    for _ in range(2):
        try:
            r = subprocess.run(["claude", "-p", "--model", "haiku"],
                               input=PROMPT.format(q=d["cq"]),
                               capture_output=True, text=True, timeout=90, env=env)
            m = re.findall(r"\{[^{}]*\"q\"[\s\S]*?\}", r.stdout)
            cand = str(json.loads(m[-1]).get("q", "")).strip() if m else ""
            if valid(d["cq"], cand):
                rw = cand
                break
        except Exception:
            pass
    if not rw:
        rw, fb = d["cq"], True
    c = {"qid": qid, "cq": d["cq"], "rw": rw, "fallback": fb}
    f.write_text(json.dumps(c, ensure_ascii=False))
    return qid, c


def agg(rows):
    n = len(rows)
    return {"r5": round(sum(r["r5"] for r in rows) / n, 4),
            "r10": round(sum(r["r10"] for r in rows) / n, 4),
            "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    CACHE.mkdir(parents=True, exist_ok=True)
    mapped = {str(m["qid"]): m for m in
              (json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8"))}
    dev_qids = {str(json.loads(l)["qid"]) for l in open(DEVQ)}
    frozen = [json.loads(l) for l in open(OUT / "cq_frozen.jsonl", encoding="utf-8")]
    items = [d for d in frozen if (str(d["qid"]) in dev_qids) == (split == "dev")]
    print(f"split={split} n={len(items)}", flush=True)

    rewritten = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (qid, c) in enumerate(ex.map(rewrite_one, items), 1):
            rewritten[qid] = c
            if i % 20 == 0:
                print(f"rewrite {i}/{len(items)}", flush=True)
    frozen_rw = OUT / f"rw_frozen_{split}.jsonl"
    ordered = [rewritten[str(d["qid"])] for d in items]
    frozen_rw.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in ordered) + "\n")
    sha = hashlib.sha256(frozen_rw.read_bytes()).hexdigest()
    print(f"frozen rw sha={sha[:16]} fallback={sum(1 for c in ordered if c['fallback'])}", flush=True)

    lines = SE.nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    bm = SE.BM25([e["text"] for e in els])

    def run(qtext_of):
        rows = []
        for d in items:
            qid = str(d["qid"])
            top = bm.rank10(qtext_of(qid, d))
            g = set(mapped[qid]["gold"])
            r = next((i + 1 for i, ix in enumerate(top) if els[ix]["eid"] in g), None)
            rows.append({"qid": qid, "rank": r, "r5": int(bool(r and r <= 5)),
                         "r10": int(bool(r)), "mrr": (1 / r) if r else 0.0})
        return rows

    base = run(lambda qid, d: d["cq"])
    q2 = run(lambda qid, d: rewritten[qid]["rw"])
    dm = [b["mrr"] for b in base]
    d2 = [v["mrr"] - b["mrr"] for b, v in zip(base, q2)]
    d5 = [v["r5"] - b["r5"] for b, v in zip(base, q2)]
    res = {"split": split, "n": len(items),
           "A0_cq(기준)": agg(base), "Q2_rw(LLM 정규화)": agg(q2),
           "paired": {"delta_mrr": round(sum(d2) / len(d2), 4),
                      "mrr_ci95": [round(x, 4) for x in SE.boot_ci(d2)],
                      "delta_r5": round(sum(d5) / len(d5), 4),
                      "r5_ci95": [round(x, 4) for x in SE.boot_ci(d5)],
                      "r5_transitions": {"improve": sum(1 for x in d5 if x > 0),
                                         "regress": sum(1 for x in d5 if x < 0)}},
           "rw_sha": sha}
    (OUT / f"results_q2_{split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
