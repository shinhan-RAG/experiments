#!/usr/bin/env python3
"""실험 5a — 페이지 헤더 노이즈 제거 (전처리). 전 과정 결정적, LLM 없음.

가설: PDF→md 페이지 경계 블록("# SHINHAN LIFE" 줄, 독립 쪽번호 숫자 줄,
독립 "---" 수평선)이 element 텍스트에 끼어 문자 bigram 매칭에 노이즈.
제거 시 dev 검색 성능 상승 여부를 A0(원문) vs A0'(정제) paired 비교.

- 분할·eid·gold 불변: split_elements는 원문 그대로 수행, 정제는 element
  text에서 라인 필터만 (문서 라인 번호 기준, 원문 문맥으로 판정).
- 게이트: A0 dev 96에서 R@5=0.3125, MRR@10=0.2647 재현 실패 시 즉시 중단.

usage: python3 exp5a_pagehdr.py
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semtag_experiment import BM25, bigrams, boot_ci, nfc, split_elements  # noqa: E402

DOC = Path("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710"
           "/원본문서/md/판매약관_신한(간편가입)통합건강보험 원(ONE)"
           "(무배당, 해약환급금 미지급형)_260507.md")
GOLD = HERE / "out" / "gold_mapped.jsonl"
DEVQ = HERE.parent / "hybrid-enrich" / "out" / "qa100_gold.jsonl"
PRIOR = HERE / "out" / "semtag_results.json"
OUT = HERE / "out" / "exp5a"

GATE_R5, GATE_MRR = 0.3125, 0.2647

TITLE = "신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)"
HDR_EXACT = "# SHINHAN LIFE"
ANCHOR_STRS = {"---", HDR_EXACT, "SHINHAN LIFE", TITLE}
NUM = re.compile(r"\d+")


def noise_lines(lines):
    """원문 라인(1-based) 중 페이지 경계 노이즈 라인 번호 집합 + 규칙별 카운트.

    R1 header : strip == "# SHINHAN LIFE" (정확 일치만).
    R2 pagenum: 숫자만 있는 줄 + 앞뒤 라인 공백(또는 문서 경계) +
                ±2번째 라인에 페이지 경계 앵커(독립 "---" / "# SHINHAN LIFE" /
                "SHINHAN LIFE" / 문서 제목 반복줄) 존재. 앵커 없는 독립 숫자는 보존.
    R3 hr     : strip == "---" + 직전 라인이 공백/문서 경계이거나 이미 노이즈로
                판정된 라인 (setext 밑줄 오인 방지). 표 구분행("|---|")은
                strip이 "|"로 시작하므로 애초에 매치 불가.
    """
    n = len(lines)

    def s(i):
        return lines[i - 1].strip() if 1 <= i <= n else ""

    def blank(i):
        return not s(i)

    def anchor(i):
        return s(i) in ANCHOR_STRS

    noise, by_rule = set(), Counter()
    for i in range(1, n + 1):
        t = s(i)
        if t == HDR_EXACT:
            noise.add(i)
            by_rule["R1_header"] += 1
        elif t and NUM.fullmatch(t) and blank(i - 1) and blank(i + 1) \
                and (anchor(i - 2) or anchor(i + 2)):
            noise.add(i)
            by_rule["R2_pagenum"] += 1
        elif t == "---" and (blank(i - 1) or (i - 1) in noise):
            noise.add(i)
            by_rule["R3_hr"] += 1
    return noise, by_rule


def clean_elements(els, noise):
    """element text에서 노이즈 라인 제거 (분할·eid·start/end 불변)."""
    cleaned, stats = [], {"changed": 0, "emptied": 0, "chars_removed": 0,
                          "reduction_per_changed": []}
    for e in els:
        el_lines = e["text"].split("\n")
        span = range(e["start"], e["end"] + 1)
        assert len(el_lines) == len(span), \
            f"line-count mismatch {e['eid']}: {len(el_lines)} != {len(span)}"
        kept = [l for l, li in zip(el_lines, span) if li not in noise]
        new = "\n".join(kept)
        if new != e["text"]:
            stats["changed"] += 1
            d = len(e["text"]) - len(new)
            stats["chars_removed"] += d
            stats["reduction_per_changed"].append(d)
            if not new.strip():
                stats["emptied"] += 1
        cleaned.append(new)
    return cleaned, stats


def evaluate(texts, dev, els):
    bm = BM25(texts)
    rows = []
    for it in dev:
        top10 = [els[i]["eid"] for i in bm.rank10(it["q"])]
        g = set(it["gold"])
        r = next((k + 1 for k, x in enumerate(top10) if x in g), None)
        rows.append({"qid": it["qid"], "r5": int(bool(r and r <= 5)),
                     "r10": int(bool(r)), "mrr": (1 / r) if r else 0.0})
    n = len(rows)
    agg = {"r5": round(sum(r["r5"] for r in rows) / n, 4),
           "r10": round(sum(r["r10"] for r in rows) / n, 4),
           "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}
    return agg, rows


def main():
    raw = DOC.read_text(encoding="utf-8", errors="ignore")
    doc_sha = hashlib.sha256(raw.encode()).hexdigest()
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if prior["doc_sha256"] != doc_sha:
        sys.exit(f"FAIL-LOUD: doc sha mismatch prior={prior['doc_sha256'][:12]} "
                 f"now={doc_sha[:12]}")

    lines = nfc(raw).splitlines()
    els = split_elements(lines)
    if len(els) != prior["n_elements"]:
        sys.exit(f"FAIL-LOUD: n_elements {len(els)} != {prior['n_elements']}")

    mapped = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l]
    dev_qids = {json.loads(l)["qid"] for l in DEVQ.read_text(encoding="utf-8").splitlines() if l}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    if len(dev) != 96:
        sys.exit(f"FAIL-LOUD: dev size {len(dev)} != 96")
    eids = {e["eid"] for e in els}
    missing = [g for m in dev for g in m["gold"] if g not in eids]
    if missing:
        sys.exit(f"FAIL-LOUD: gold eids missing from split: {missing[:5]}")

    # ── 게이트: A0 재현 ─────────────────────────────────────────────
    a0, rows_a0 = evaluate([e["text"] for e in els], dev, els)
    if a0["r5"] != GATE_R5 or a0["mrr10"] != GATE_MRR:
        sys.exit(f"FAIL-LOUD: baseline gate mismatch — got R@5={a0['r5']} "
                 f"MRR@10={a0['mrr10']}, expected R@5={GATE_R5} MRR@10={GATE_MRR}")
    print(f"gate OK: A0 dev96 R@5={a0['r5']} R@10={a0['r10']} MRR@10={a0['mrr10']}",
          flush=True)

    # ── 변형: 페이지 블록 제거 → A0' ────────────────────────────────
    noise, by_rule = noise_lines(lines)
    texts_c, cst = clean_elements(els, noise)
    a0p, rows_a0p = evaluate(texts_c, dev, els)
    print(f"noise lines={len(noise):,} by_rule={dict(by_rule)}", flush=True)
    print(f"elements changed={cst['changed']:,}/{len(els):,} "
          f"({cst['changed']/len(els):.1%}) emptied={cst['emptied']}", flush=True)
    print(f"A0' dev96 R@5={a0p['r5']} R@10={a0p['r10']} MRR@10={a0p['mrr10']}",
          flush=True)

    # ── paired 비교 ─────────────────────────────────────────────────
    base = {r["qid"]: r for r in rows_a0}
    deltas = [r["mrr"] - base[r["qid"]]["mrr"] for r in rows_a0p]
    ci = [round(x, 4) for x in boot_ci(deltas)]
    imp = [r["qid"] for r in rows_a0p if r["r5"] == 1 and base[r["qid"]]["r5"] == 0]
    reg = [r["qid"] for r in rows_a0p if r["r5"] == 0 and base[r["qid"]]["r5"] == 1]
    dmrr = round(sum(deltas) / len(deltas), 4)
    print(f"ΔMRR={dmrr:+.4f} CI95={ci} R@5 전환: +{len(imp)} / -{len(reg)}",
          flush=True)

    red = cst["reduction_per_changed"]
    payload = {
        "exp": "5a_pagehdr", "doc_sha256": doc_sha,
        "n_elements": len(els), "n_dev": len(dev),
        "gate": {"expected": {"r5": GATE_R5, "mrr10": GATE_MRR},
                 "observed": a0, "passed": True},
        "A0": a0, "A0_prime": a0p,
        "paired": {"dmrr_mean": dmrr, "dmrr_ci95": ci,
                   "r5_improved": imp, "r5_regressed": reg,
                   "r5_net": len(imp) - len(reg)},
        "cleaning": {
            "noise_lines_total": len(noise), "by_rule": dict(by_rule),
            "elements_changed": cst["changed"],
            "elements_changed_pct": round(cst["changed"] / len(els), 4),
            "elements_emptied": cst["emptied"],
            "total_chars_removed": cst["chars_removed"],
            "mean_char_reduction_per_changed_element":
                round(sum(red) / len(red), 1) if red else 0.0},
        "per_qid": {"A0": rows_a0, "A0_prime": rows_a0p},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
