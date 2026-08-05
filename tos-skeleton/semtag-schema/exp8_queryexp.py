#!/usr/bin/env python3
"""실험 8 — 질의 측 어휘 변환 (query-side aliasing). 전 과정 결정적, LLM 없음.

배경: 별칭 사전(Q_BRIDGE)을 문서 측에 부착한 실험 3·4는 기각/미확정.
기제 분석의 시사 = "어휘 간극 해소는 질의 측이 구조적으로 우월(문서 불변 → 선별성 보존)".
본 실험은 같은 사전을 **역방향**(고객 표현 → 문서 용어)으로 질의에만 적용한다.

- 문서·element·BM25 인덱스: A0(원문만)에서 완전 불변. 질의 문자열만 변형.
- V1 용어 추가: q' = q + " " + 매치된 문서 용어들(공백 연결, 중복 없이)
- V2 표현 치환: q에서 고객 표현 부분 문자열을 문서 용어로 치환(원 표현 제거)
- 평가: dev 96만. test 채점·보고 금지(상위 세션 판단).
- 게이트: A0 재현(R@5=.3125, MRR@10=.2647) 불일치 시 중단.
- 결정성: 동일 프로세스 내 전체 계산 2회 일치 확인 + 외부 재실행 해시 비교용
  results.json에 비결정 요소(타임스탬프 등) 미포함.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import semtag_experiment as SE          # noqa: E402  (기존 파일 수정 없음 — import만)
from semtag_h3 import Q_BRIDGE          # noqa: E402  (key=문서 용어, value=고객 표현;…)

DOC = Path(
    "/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
    "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
GOLD = HERE / "out" / "gold_mapped.jsonl"
DEVQ = HERE / ".." / "hybrid-enrich" / "out" / "qa100_gold.jsonl"
OUT = HERE / "out" / "exp8"

BASELINE_R5 = 0.3125
BASELINE_MRR = 0.2647

# 역방향 사전: (문서 용어, [고객 표현…]) — Q_BRIDGE 삽입 순서 보존(결정적)
ALIAS = [(SE.nfc(term), [SE.nfc(x.strip()) for x in exprs.split(";") if x.strip()])
         for term, exprs in Q_BRIDGE.items()]


def matches(q):
    """원 질의에서 발동한 (문서 용어, 매치된 고객 표현들). 용어당 1회(중복 없이)."""
    out = []
    for term, exprs in ALIAS:
        hit = [x for x in exprs if x in q]
        if hit:
            out.append((term, hit))
    return out


def expand_v1(q, m):
    terms = [t for t, _ in m]
    return (q + " " + " ".join(terms)) if terms else q


def expand_v2(q, m):
    w = q
    for term, exprs in m:
        for x in exprs:
            w = w.replace(x, term)      # 원 표현 제거 후 문서 용어로 치환
    return w


def eval_rows(bm, els, items, qmod, cache):
    rows = []
    for it in items:
        q = qmod(it)
        if q not in cache:
            cache[q] = bm.rank10(q)
        top10 = [els[i]["eid"] for i in cache[q]]
        g = set(it["gold"])
        rk = next((k + 1 for k, x in enumerate(top10) if x in g), None)
        rows.append({"qid": it["qid"], "r5": int(bool(rk and rk <= 5)),
                     "r10": int(bool(rk)), "mrr": (1 / rk) if rk else 0.0})
    return rows


def agg(rows):
    n = len(rows)
    return {"r5": round(sum(r["r5"] for r in rows) / n, 4),
            "r10": round(sum(r["r10"] for r in rows) / n, 4),
            "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}


def compare(base_rows, var_rows, fired_qids):
    b = {r["qid"]: r for r in base_rows}
    deltas = [r["mrr"] - b[r["qid"]]["mrr"] for r in var_rows]
    imp = [r["qid"] for r in var_rows if r["r5"] > b[r["qid"]]["r5"]]
    reg = [r["qid"] for r in var_rows if r["r5"] < b[r["qid"]]["r5"]]
    fired = [r for r in var_rows if r["qid"] in fired_qids]
    fdeltas = [r["mrr"] - b[r["qid"]]["mrr"] for r in fired]
    unfired_nonzero = [r["qid"] for r in var_rows if r["qid"] not in fired_qids
                       and abs(r["mrr"] - b[r["qid"]]["mrr"]) > 1e-12]
    out = {"dmrr": round(sum(deltas) / len(deltas), 4),
           "dmrr_ci95": [round(x, 4) for x in SE.boot_ci(deltas)],
           "r5_improved": sorted(imp, key=int), "r5_regressed": sorted(reg, key=int),
           "fired_n": len(fired),
           "fired_dmrr": round(sum(fdeltas) / len(fdeltas), 4) if fdeltas else 0.0,
           "fired_dmrr_ci95_exploratory":
               [round(x, 4) for x in SE.boot_ci(fdeltas)] if fdeltas else None,
           "fired_improved": sorted([q for q in imp if q in fired_qids], key=int),
           "fired_regressed": sorted([q for q in reg if q in fired_qids], key=int),
           "unfired_with_nonzero_delta": unfired_nonzero}
    out["all_effects_within_fired"] = (
        set(imp) | set(reg)) <= fired_qids and not unfired_nonzero
    return out


def run_once(els, dev):
    bm = SE.BM25([e["text"] for e in els])          # A0 인덱스 — 전 변형 공유·불변
    cache = {}
    a0 = eval_rows(bm, els, dev, lambda it: it["q"], cache)
    minfo = {it["qid"]: matches(it["q"]) for it in dev}
    fired_qids = {q for q, m in minfo.items() if m}
    v1 = eval_rows(bm, els, dev, lambda it: expand_v1(it["q"], minfo[it["qid"]]), cache)
    v2 = eval_rows(bm, els, dev, lambda it: expand_v2(it["q"], minfo[it["qid"]]), cache)
    return a0, v1, v2, minfo, fired_qids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = DOC.read_text(encoding="utf-8", errors="ignore")
    doc_sha = hashlib.sha256(raw.encode()).hexdigest()
    lines = SE.nfc(raw).splitlines()
    els = SE.split_elements(lines)                   # A0 = 원문 element만(컨텍스트 불요)

    mapped = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l]
    dev_qids = {json.loads(l)["qid"] for l in DEVQ.read_text(encoding="utf-8").splitlines() if l}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    print(f"elements={len(els)} mapped={len(mapped)} dev={len(dev)} "
          f"(test는 로드만·채점 안 함)", flush=True)
    assert len(dev) == 96, f"dev != 96 (={len(dev)})"

    # ── 결정성: 전체 계산 2회 ──
    pass1 = run_once(els, dev)
    pass2 = run_once(els, dev)
    det_ok = json.dumps(pass1[:3], sort_keys=True) == json.dumps(pass2[:3], sort_keys=True)
    print(f"determinism(2-pass identical)={det_ok}", flush=True)
    a0, v1, v2, minfo, fired_qids = pass1

    # ── 게이트: A0 기준선 재현 ──
    a0m = agg(a0)
    gate = (a0m["r5"] == BASELINE_R5 and a0m["mrr10"] == BASELINE_MRR)
    print(f"A0 dev: R@5={a0m['r5']} R@10={a0m['r10']} MRR@10={a0m['mrr10']} "
          f"gate={'PASS' if gate else 'FAIL'}", flush=True)
    if not gate:
        (OUT / "results.json").write_text(json.dumps(
            {"gate": "FAIL", "a0_observed": a0m,
             "a0_expected": {"r5": BASELINE_R5, "mrr10": BASELINE_MRR}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(1)

    v1m, v2m = agg(v1), agg(v2)
    cmp1 = compare(a0, v1, fired_qids)
    cmp2 = compare(a0, v2, fired_qids)

    # 발동 분석 상세(문항별): 매치 용어·표현 + 변형 질의 원문
    qtext = {it["qid"]: it["q"] for it in dev}
    fired_detail = {q: {"matches": [{"term": t, "exprs": hit} for t, hit in m],
                        "q": qtext[q],
                        "q_v1": expand_v1(qtext[q], m),
                        "q_v2": expand_v2(qtext[q], m)}
                    for q, m in sorted(minfo.items(), key=lambda kv: int(kv[0])) if m}
    term_freq = {}
    for m in minfo.values():
        for t, _ in m:
            term_freq[t] = term_freq.get(t, 0) + 1

    # 어휘 측면성(기술 통계): dev 질의가 문서 용어(Q_BRIDGE 키)를 이미 포함하는가.
    # 발동률이 낮은 기제 설명용 — 채점에 미사용.
    key_hits = {}
    n_any_key = 0
    for it in dev:
        hit = [t for t, _ in ALIAS if t in it["q"]]
        if hit:
            n_any_key += 1
        for t in hit:
            key_hits[t] = key_hits.get(t, 0) + 1
    expr_within_term = sum(
        1 for m in minfo.values() for t, hits in m if all(t in x for x in hits))
    n_matches_total = sum(len(m) for m in minfo.values())

    payload = {
        "exp": "exp8 query-side aliasing (Q_BRIDGE 역방향, 결정적 질의 확장)",
        "doc_sha256": doc_sha, "n_elements": len(els), "n_dev": len(dev),
        "index": "A0 element 원문 BM25(bigram) — 전 변형 공유·불변",
        "gate_A0_reproduced": True,
        "determinism_2pass_identical": det_ok,
        "A0": a0m,
        "V1_append_terms": {**v1m, "vs_A0": cmp1},
        "V2_replace_exprs": {**v2m, "vs_A0": cmp2},
        "firing": {"n_fired": len(fired_qids), "n_dev": len(dev),
                   "fired_qids": sorted(fired_qids, key=int),
                   "term_fire_counts": dict(sorted(term_freq.items(),
                                                   key=lambda kv: (-kv[1], kv[0]))),
                   "matches_where_expr_contains_term":
                       f"{expr_within_term}/{n_matches_total}",
                   "per_query": fired_detail},
        "vocab_sidedness": {
            "note": "dev 질의가 이미 문서 용어를 쓰는지(발동률 저조 기제 설명, 채점 미사용)",
            "n_dev_queries_containing_doc_term": n_any_key,
            "n_dev_queries_containing_customer_expr": len(fired_qids),
            "doc_term_counts": dict(sorted(key_hits.items(),
                                           key=lambda kv: (-kv[1], kv[0])))},
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"V1": v1m, "V1_vs_A0": {k: cmp1[k] for k in
                                              ("dmrr", "dmrr_ci95", "r5_improved", "r5_regressed")},
                      "V2": v2m, "V2_vs_A0": {k: cmp2[k] for k in
                                              ("dmrr", "dmrr_ci95", "r5_improved", "r5_regressed")},
                      "fired": len(fired_qids)}, ensure_ascii=False), flush=True)
    print("saved", OUT / "results.json", flush=True)


if __name__ == "__main__":
    main()
