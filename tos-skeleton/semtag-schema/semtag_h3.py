#!/usr/bin/env python3
"""가설 3·4 — 질문 어휘 브리지 태그(s2q) · 희소 태그(s2s) · 결합(s2qs).

값 생성 규칙(결정적·원문/도메인 사전만, 평가셋 질문 미참조 — 누수 차단):
- Q_BRIDGE: 문서 구조 표현 → 질문형 표현 매핑. 출처 = 동료 QA 생성기 PARAPHRASES
  (도메인 지식으로 독립 작성) + 역할→상담 표현 확장(일반 보험 상담 어휘).
- 희소화: 조 제목 보유 또는 역할 적중 element에만 태그(선별성 복원).
채널: concat / any-verify / 선택적 verify (기존 하네스 재사용, dev/test 동일).
"""
import json, sys, re
from pathlib import Path
import semtag_experiment as SE

# 문서 표현 → 질문형 표현 (도메인 사전; 동료 PARAPHRASES 계열 + 상담 어휘 확장)
Q_BRIDGE = {
    "지급사유": "언제 보험금 받는지;지급 조건;보장 되는 경우",
    "보험금을 지급하지 않는": "보장 안 되는 경우;보험금 못 받는 경우",
    "면책": "보장 제외;보장 안됨",
    "부담보": "보장 제외 기간;부담보 해제",
    "납입면제": "납입면제 가능;보험료 안 내도 되는 경우",
    "정의": "무슨 뜻;어떤 의미;란 무엇",
    "청구": "청구 방법;청구 절차;서류 제출",
    "제출서류": "필요한 서류;구비서류;증명서 제출",
    "갱신": "갱신 가능;갱신 조건;재가입",
    "해지": "해지 방법;해지하면",
    "해지환급금": "해지하면 돌려받는 돈;환급금 얼마",
    "무효": "계약 무효;효력 없음",
    "감액": "보험금 줄어드는 경우;감액 지급",
    "보장개시일": "언제부터 보장;보장 시작일",
    "진단확정": "진단 기준;확정 방법;검사 근거",
    "보험료 납입": "보험료 내는 방법;납입 기준",
    "계약 전 알릴 의무": "가입 전 고지;알려야 하는 것",
    "보험기간": "보장 기간;언제까지 보장",
}


def bridge_terms(e):
    src = (e["jo"] or "") + " " + e["text"][:1500]
    out = []
    for k, v in Q_BRIDGE.items():
        if k in src:
            out.append(v)
        if len(out) == 3:
            break
    return ";".join(out)


def is_core(e):
    return bool(e["jo"]) or bool(SE.roles_of(e["text"][:1500]))


def build_tag_v2(e, schema):
    if schema == "A0":
        return ""
    core = is_core(e)
    if schema in ("s2s", "s2qs") and not core:
        return ""                       # 희소화: 핵심 element만 태그
    f = [("유형", e["type"]), ("특약", e["scope"]), ("조항", e["jo"])]
    role = SE.roles_of(e["text"][:1500])
    f.append(("역할", role[0] if role else ""))
    if schema in ("s2q", "s2qs"):
        f.append(("묻는말", bridge_terms(e)))   # 질문 어휘 브리지 (원문에 없는 어휘)
    return SE.serialize(f)


def main():
    doc, devq = sys.argv[1], sys.argv[2]
    lines = SE.nfc(Path(doc).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = SE.annotate_context(SE.split_elements(lines))
    mapped = [json.loads(l) for l in open("out/gold_mapped.jsonl", encoding="utf-8")]
    dev_qids = {json.loads(l)["qid"] for l in open(devq)}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    test = [m for m in mapped if m["qid"] not in dev_qids]
    import math
    from collections import Counter

    def eval_items(bm, els_, items):
        r5 = mrr = 0.0
        for it in items:
            top10 = [els_[i]["eid"] for i in bm.rank10(it["q"])]
            g = set(it["gold"])
            rk = next((k+1 for k, x in enumerate(top10) if x in g), None)
            r5 += int(bool(rk and rk <= 5)); mrr += (1/rk) if rk else 0
        return round(r5/len(items), 4), round(mrr/len(items), 4)

    results = {}
    # A0 + s0 대조군(기존과 동일해야 함 — 재현 검증) + 신규 3종
    for schema in ("A0", "s0", "s2q", "s2s", "s2qs"):
        if schema in ("A0",):
            tags = ["" for _ in els]
        elif schema == "s0":
            tags = [SE.build_tag(e, "tag.s0") for e in els]
        else:
            tags = [build_tag_v2(e, schema) for e in els]
        tagged = sum(1 for t in tags if t)
        tlen = sorted(len(t) for t in tags if t)
        reps = [(t + " ||| " + e["text"]) if t else e["text"] for t, e in zip(tags, els)]
        bm = SE.BM25(reps)
        dr5, dmrr = eval_items(bm, els, dev)
        entry = {"tagged": tagged, "len_p50": tlen[len(tlen)//2] if tlen else 0,
                 "concat_dev_r5": dr5, "concat_dev_mrr": dmrr}
        # verify 채널: 원문 순위 + 태그 멤버십 (any / bridge 우선)
        bm_a0 = SE.BM25([e["text"] for e in els]) if schema == "A0" else bm_a0  # noqa
        results[schema] = entry
        print(f"{schema:6s} tagged={tagged:5d} len_p50={entry['len_p50']:3d} "
              f"concat dev R@5={dr5} MRR={dmrr}", flush=True)
    # verify 채널 별도 패스 (A0 순위 공유)
    bm_a0 = SE.BM25([e["text"] for e in els])
    a0rank = {}
    def a0_top(q, n=200):
        if q not in a0rank:
            a0rank[q] = bm_a0.rank10(q) if False else None
        return a0rank[q]
    # rank10은 top10만 — verify에는 top200 필요 → 직접 계산
    def full_rank(q, n=200):
        qb = list(Counter(SE.bigrams(q)))
        scores = []
        for i in range(bm_a0.N):
            s = 0.0; tfi = bm_a0.tf[i]
            for t in qb:
                fq = tfi.get(t, 0)
                if not fq: continue
                idf = math.log(1 + (bm_a0.N - bm_a0.df[t] + 0.5) / (bm_a0.df[t] + 0.5))
                B = 1 - bm_a0.b + bm_a0.b * (bm_a0.dl[i] / bm_a0.avgdl)
                s += idf * fq * (bm_a0.k1 + 1) / (fq + bm_a0.k1 * B)
            scores.append(s)
        return sorted(range(bm_a0.N), key=lambda i: -scores[i])[:n]
    cache = {}
    for schema in ("s0", "s2q", "s2s", "s2qs"):
        if schema == "s0":
            tags = [SE.build_tag(e, "tag.s0") for e in els]
        else:
            tags = [build_tag_v2(e, schema) for e in els]
        r5 = mrr = 0.0
        for it in dev:
            if it["qid"] not in cache:
                cache[it["qid"]] = full_rank(it["q"])
            base = cache[it["qid"]]
            toks = [t for t in re.split(r"[\s,?.!·()\[\]]+", it["q"])
                    if 2 <= len(t) <= 14 and re.search(r"[가-힣]", t)]
            head = base[:20]
            passed = [i for i in head if any(t in tags[i] for t in toks)]
            order = (passed + [i for i in head if i not in set(passed)] + base[20:]) if passed else base
            top10 = [els[i]["eid"] for i in order[:10]]
            g = set(it["gold"])
            rk = next((k+1 for k, x in enumerate(top10) if x in g), None)
            r5 += int(bool(rk and rk <= 5)); mrr += (1/rk) if rk else 0
        results[schema]["verify_dev_r5"] = round(r5/len(dev), 4)
        results[schema]["verify_dev_mrr"] = round(mrr/len(dev), 4)
        print(f"{schema:6s} verify dev R@5={results[schema]['verify_dev_r5']} "
              f"MRR={results[schema]['verify_dev_mrr']}", flush=True)
    Path("out/semtag_h3_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved out/semtag_h3_results.json", flush=True)


if __name__ == "__main__":
    main()
