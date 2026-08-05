#!/usr/bin/env python3
"""quote 우선 gold 매핑 + 층화 100건 샘플.
quote를 문서 전문에서 검색 → 겹치는 청크가 gold. line은 다중 매치 선별용."""
import json, re, unicodedata, os, random, collections, bisect

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
QA = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/정답셋_359_최종_v4.jsonl"
DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
SEED = 20260804
N_SAMPLE = 100
N_SMOKE = 12


def norm_ws(s):
    return re.sub(r"\s+", " ", s)


def main():
    raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    # 정규화 텍스트와 원본 offset 매핑 (공백 축약)
    norm_chars, norm2raw = [], []
    prev_ws = False
    for i, ch in enumerate(raw):
        if ch.isspace():
            if not prev_ws:
                norm_chars.append(" ")
                norm2raw.append(i)
            prev_ws = True
        else:
            norm_chars.append(ch)
            norm2raw.append(i)
            prev_ws = False
    norm_text = "".join(norm_chars)

    chunks = [json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"))]
    elements = [json.loads(l) for l in open(os.path.join(OUT, "elements.jsonl"))]

    def overlapping_chunks(r0, r1):
        return [c["chunk_id"] for c in chunks if c["char_start"] < r1 and c["char_end"] > r0]

    def overlapping_elements(r0, r1):
        return [e["element_id"] for e in elements if e["char_start"] < r1 and e["char_end"] > r0]

    # line -> char offset (다중 매치 선별용)
    line_starts = []
    pos = 0
    for l in raw.split("\n"):
        line_starts.append(pos)
        pos += len(l) + 1

    qa = [json.loads(l) for l in open(QA)]
    gold_rows, audit = [], []
    for q in qa:
        if q.get("신뢰도") != "확정":
            continue
        srcs = q.get("출처") or []
        gold_ids, src_results = set(), []
        gold_eids, gold_spans = set(), []
        for s in srcs:
            quote = norm_ws(unicodedata.normalize("NFC", str(s.get("quote", "")))).strip()
            status, matched_at = "failed", None
            if quote:
                probe = quote if len(quote) <= 300 else quote[:300]
                hits = [m.start() for m in re.finditer(re.escape(probe), norm_text)]
                if not hits and len(probe) > 80:
                    probe = quote[:80]
                    hits = [m.start() for m in re.finditer(re.escape(probe), norm_text)]
                if hits:
                    if len(hits) > 1:
                        try:
                            ln = int(s.get("line"))
                            target = line_starts[min(ln - 1, len(line_starts) - 1)]
                            hits.sort(key=lambda h: abs(norm2raw[h] - target))
                        except (TypeError, ValueError):
                            pass
                    h = hits[0]
                    r0 = norm2raw[h]
                    r1 = norm2raw[min(h + len(probe) - 1, len(norm2raw) - 1)] + 1
                    ids = overlapping_chunks(r0, r1)
                    if ids:
                        gold_ids.update(ids)
                        gold_eids.update(overlapping_elements(r0, r1))
                        gold_spans.append([r0, r1])
                        status, matched_at = "ok", r0
            src_results.append({"line": s.get("line"), "quote_head": quote[:60], "status": status, "raw_offset": matched_at})
        row_status = "ok" if gold_ids else "mapping_failed"
        audit.append({"no": q["no"], "status": row_status, "sources": src_results})
        if gold_ids:
            etypes = {c["element_type"] for c in chunks if c["chunk_id"] in gold_ids}
            gold_rows.append({
                "qid": str(q["no"]), "question": q["질문"], "business_type": q.get("사업구분", ""),
                "gold_chunk_ids": sorted(gold_ids),
                "gold_element_ids": sorted(gold_eids),
                "gold_spans": gold_spans,
                "gold_element_types": sorted(etypes),
            })

    with open(os.path.join(OUT, "gold_mapping_audit.jsonl"), "w", encoding="utf-8") as f:
        for a in audit:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # 층화 샘플 100: business_type x 대표 element_type
    random.seed(SEED)
    def stratum(r):
        et = "table" if "table" in r["gold_element_types"] else ("formula" if "formula" in r["gold_element_types"] else "text")
        return (r["business_type"], et)
    by_s = collections.defaultdict(list)
    for r in gold_rows:
        by_s[stratum(r)].append(r)
    for v in by_s.values():
        random.shuffle(v)
    total = len(gold_rows)
    sample, quotas = [], []
    keys = sorted(by_s.keys())
    for k in keys:
        quotas.append((k, max(1, round(len(by_s[k]) / total * N_SAMPLE))))
    picked = []
    for k, qn in quotas:
        picked.extend(by_s[k][:qn])
    # 정확히 100으로 조정
    if len(picked) > N_SAMPLE:
        random.shuffle(picked)
        picked = picked[:N_SAMPLE]
    else:
        rest = [r for r in gold_rows if r not in picked]
        random.shuffle(rest)
        picked.extend(rest[:N_SAMPLE - len(picked)])
    sample = picked

    # smoke 12: element_type별 최소 3
    smoke = []
    for et in ["table", "formula", "text"]:
        cands = [r for r in sample if et in r["gold_element_types"]]
        smoke.extend(cands[:3])
    seen = {r["qid"] for r in smoke}
    for r in sample:
        if len(smoke) >= N_SMOKE:
            break
        if r["qid"] not in seen:
            smoke.append(r)
            seen.add(r["qid"])

    with open(os.path.join(OUT, "qa100_gold.jsonl"), "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "seed": SEED, "total_confirmed": sum(1 for q in qa if q.get("신뢰도") == "확정"),
        "mapped": len(gold_rows), "mapping_failed": sum(1 for a in audit if a["status"] != "ok"),
        "sample": [r["qid"] for r in sample], "smoke": [r["qid"] for r in smoke][:N_SMOKE],
        "business_dist": dict(collections.Counter(r["business_type"] for r in sample)),
        "etype_dist": dict(collections.Counter(stratum(r)[1] for r in sample)),
        "avg_gold_chunks": round(sum(len(r["gold_chunk_ids"]) for r in sample) / len(sample), 2),
    }
    json.dump(manifest, open(os.path.join(OUT, "split_manifest.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sample", "smoke")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
