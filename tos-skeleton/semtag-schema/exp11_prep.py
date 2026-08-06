#!/usr/bin/env python3
"""실험 11 사전 작업 — 트랙 A(gold 모호성 감사) + 트랙 C(게이트·후보 동결·oracle).

전부 결정적, LLM 호출 없음, dev 96만 사용(test 미개봉).
"""
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).parent
OUT = HERE / "out/exp11"
DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
       "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
DEVQ = HERE.parent / "hybrid-enrich/out/qa100_gold.jsonl"

TK = re.compile(r"[가-힣A-Za-z0-9()\[\]·]+특약")


def code_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=HERE).stdout.strip()


def norm_tk(s):
    """특약명 비교용 정규화 — 접두 수식(간편/삭감없음용 등)·공백 제거."""
    s = re.sub(r"[\s]", "", s)
    s = re.sub(r"^\(간편\)|\[[^\]]*\]|\(무\)", "", s)
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = code_commit()
    lines = SE.nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    byid = {e["eid"]: e for e in els}
    mapped = {str(m["qid"]): m for m in
              (json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8"))}
    dev_qids = sorted({str(json.loads(l)["qid"]) for l in open(DEVQ)} & set(mapped),
                      key=int)  # qa100 중 gold 매핑 성공분(96)만

    # ── 트랙 A: 중복 정의 2종 (병기 고정)
    ns = lambda s: re.sub(r"\s+", "", s)
    sigA = Counter(ns(e["text"])[:300] for e in els if len(ns(e["text"])) >= 300)
    dupA_sigs = {s for s, c in sigA.items() if c > 1}
    dupA_eids = {e["eid"] for e in els
                 if len(ns(e["text"])) >= 300 and ns(e["text"])[:300] in dupA_sigs}
    sigB = Counter(ns(e["text"]) for e in els)
    dupB_sigs = {s for s, c in sigB.items() if c > 1}
    dupB_eids = {e["eid"] for e in els if ns(e["text"]) in dupB_sigs}

    def gold_in(qid, eidset):
        return any(g in eidset for g in mapped[qid]["gold"])

    dupA_q = [q for q in dev_qids if gold_in(q, dupA_eids)]
    dupB_q = [q for q in dev_qids if gold_in(q, dupB_eids)]

    # DUP_A 문항 판정: 질문이 지목한 특약 == gold element의 [특약]?
    rows = []
    for qid in dupA_q:
        q = mapped[qid]["q"]
        q_tks = [norm_tk(t) for t in TK.findall(q)]
        gold_scopes = [norm_tk(byid[g].get("scope") or "") for g in mapped[qid]["gold"]
                       if g in byid]
        if not q_tks:
            amb, reason = False, "질문에 특약 명시 없음 — 특약 불일치형 모호 아님(판정 불가)"
        else:
            match = any(qt in gs or gs in qt for qt in q_tks for gs in gold_scopes if gs)
            amb = not match
            reason = ("질문 특약과 gold [특약] 일치" if match else
                      f"질문 특약 {q_tks} vs gold [특약] {sorted(set(gold_scopes))[:3]} 불일치"
                      " — 동일 상용구의 타 특약 사본이 gold일 가능성")
        rows.append({"qid": qid, "q": q[:80], "dup": "DUP_A",
                     "gold_ambiguous": amb, "reason": reason})
    amb_n = sum(1 for r in rows if r["gold_ambiguous"])

    ga = {"code_commit": commit,
          "definitions": {
              "DUP_A": {"rule": "정규화(공백 제거) 앞 300자 동일, 길이>=300자 대상",
                        "elements": len(dupA_eids), "pct": round(len(dupA_eids)/len(els), 3),
                        "groups": sum(1 for c in sigA.values() if c > 1),
                        "max_group": max([c for c in sigA.values() if c > 1], default=0),
                        "gold_dev_questions": len(dupA_q)},
              "DUP_B": {"rule": "정규화 본문 전체 동일",
                        "elements": len(dupB_eids), "pct": round(len(dupB_eids)/len(els), 3),
                        "gold_dev_questions": len(dupB_q)}},
          "note": "규모 인용은 element 비율이 아니라 gold 기준 문항 비율로 한다. gold는 수정하지 않음(플래그만).",
          "judgements": rows, "gold_ambiguous_count": amb_n}
    p = OUT / "gold_ambiguity_dev.json"
    p.write_text(json.dumps(ga, ensure_ascii=False, indent=1))
    print(f"[트랙A] DUP_A: el {len(dupA_eids)}({len(dupA_eids)/len(els)*100:.1f}%) "
          f"gold문항 {len(dupA_q)} | DUP_B: el {len(dupB_eids)}({len(dupB_eids)/len(els)*100:.1f}%) "
          f"gold문항 {len(dupB_q)} | gold_ambiguous {amb_n}건")
    print(f"[트랙A] sha {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")

    # ── 트랙 C: 기준선 게이트
    bm = SE.BM25([e["text"] for e in els])
    def full_order(q):
        qb = list(Counter(SE.bigrams(q)))
        sc = []
        for i in range(bm.N):
            s, tfi = 0.0, bm.tf[i]
            for t in qb:
                f = tfi.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (bm.N - bm.df[t] + 0.5) / (bm.df[t] + 0.5))
                B = 1 - bm.b + bm.b * (bm.dl[i] / bm.avgdl)
                s += idf * f * (bm.k1 + 1) / (f + bm.k1 * B)
            sc.append(s)
        return sorted(range(bm.N), key=lambda i: -sc[i])

    orders = {}
    r5 = r10 = mrr = 0.0
    rec = {20: 0, 50: 0, 100: 0}
    for qid in dev_qids:
        order = full_order(mapped[qid]["q"])
        orders[qid] = order
        g = set(mapped[qid]["gold"])
        r = next((i + 1 for i, ix in enumerate(order[:10]) if els[ix]["eid"] in g), None)
        r5 += bool(r and r <= 5); r10 += bool(r); mrr += (1 / r) if r else 0.0
        for K in rec:
            rec[K] += any(els[ix]["eid"] in g for ix in order[:K])
    n = len(dev_qids)
    gate = {"r5": round(r5 / n, 4), "r10": round(r10 / n, 4), "mrr10": round(mrr / n, 4)}
    ok = gate == {"r5": 0.3125, "r10": 0.4792, "mrr10": 0.2647}
    print(f"[트랙C] 게이트 A0 dev: {gate} → {'통과' if ok else '불일치 — 중단'}")
    oracle = {K: round(rec[K] / n, 4) for K in rec}
    print(f"[트랙C] oracle recall: {oracle} (참조 .5208/.6771/.7917)")
    if not ok:
        raise SystemExit("게이트 불일치 — 전 트랙 중단")

    # 후보 동결 (K=100 상위를 저장, K=20/50은 그 접두)
    shas = {}
    for K in (20, 50, 100):
        f = OUT / f"candidates_K{K}.jsonl"
        with f.open("w") as fh:
            for qid in dev_qids:
                cand = [els[ix]["eid"] for ix in orders[qid][:K]]
                fh.write(json.dumps({"qid": qid, "candidates": cand},
                                    ensure_ascii=False) + "\n")
        shas[K] = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"[트랙C] candidates_K{K} sha {shas[K][:16]}")
    meta = {"code_commit": commit, "gate": gate, "gate_pass": ok,
            "oracle_recall": oracle, "candidate_sha": {str(k): v for k, v in shas.items()}}
    (OUT / "prep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
