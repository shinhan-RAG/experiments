#!/usr/bin/env python3
"""공식 평가셋 이관 — 정답셋 train 350 / test 150 (유형내용균형 v1, 사용자 확정).

- gold 매핑: 출처의 line(문서 좌표 동일 확인) → element 스팬 직결 + quote 검증
  (정규화 부분 문자열, 해당 element ±1 이웃까지 허용) — 결정적.
- 오염 대장: 신규 test 150을 3층으로 분류
  강오염 = 구 dev 96에 포함(튜닝·오답분석·사전제작에 반복 사용)
  약오염 = 구 test 193에 포함(집계 채점 4회 개봉, 문항 단위 분석 없음)
  청정   = 어느 쪽에도 미포함
- 기준선: train 350 매핑분에서 A0 실측. **test 150은 완전 봉인 — 채점 0회.**
"""
import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).parent
OUT = HERE / "out/qa500"
BASE = Path("/Users/donggyu/Documents/논문/신한라이프")
DOC = BASE / ("신한RAG_QA셋_원본문서_20260710/원본문서/md/"
              "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
FILES = {"train": "정답셋_350_train_유형내용균형_v1.csv",
         "test": "정답셋_150_test_유형내용균형_v1.csv"}
DEVQ = HERE.parent / "hybrid-enrich/out/qa100_gold.jsonl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=HERE).stdout.strip()
    lines = SE.nfc(DOC.read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    line2el = {}
    for e in els:
        for ln in range(e["start"], e["end"] + 1):
            line2el[ln] = e
    ns = SE.ns

    # 발췌 매핑 폴백용: 정규화 문서 + (정규화 위치→라인) 맵
    doc_ns_parts, pos2line = [], []
    for i, l in enumerate(lines, 1):
        nl = ns(l)
        doc_ns_parts.append(nl)
        pos2line.extend([i] * len(nl))
    doc_ns = "".join(doc_ns_parts)

    import re as _re
    TITLE = "신한(간편가입)통합건강보험"

    def excerpt_lines(raw):
        """비구조 발췌(검수필요)에서 유효 세그먼트 → 매칭 라인들.
        상용구 배제: 세그먼트가 문서에 4회 이상 나타나면 스킵."""
        found = []
        for seg in raw.split("\n"):
            seg = seg.strip()
            if (not seg or seg == "---" or seg.isdigit() or TITLE in seg
                    or _re.fullmatch(r"[#|\-\s\d]+", seg)):
                continue
            key = ns(seg)[:60]
            if len(key) < 20:
                continue
            hits, start = [], 0
            while len(hits) < 4:
                p = doc_ns.find(key, start)
                if p < 0:
                    break
                hits.append(p)
                start = p + 1
            if 1 <= len(hits) <= 3:
                found.extend(pos2line[p] for p in hits)
        return found

    old_mapped = {str(json.loads(l)["qid"]) for l in
                  open(HERE / "out/gold_mapped.jsonl", encoding="utf-8")}
    old_dev = {str(json.loads(l)["qid"]) for l in open(DEVQ)} & old_mapped
    old_test = old_mapped - old_dev

    rows_out, stats = [], Counter()
    for split, fname in FILES.items():
        with open(BASE / fname, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                qid = row["no"].strip()
                if not qid:
                    continue
                map_mode = "line"
                try:
                    srcs = ast.literal_eval(row["출처"])
                except Exception:
                    srcs = []
                gold, verified, spans = [], 0, []
                for s in srcs:
                    raw_ln = str(s.get("line", "0"))
                    m = __import__("re").search(r"\d+", raw_ln)
                    ln = int(m.group()) if m else 0
                    if not ln:  # line 필드 부재 → quote 텍스트로 폴백 탐색
                        for fl in excerpt_lines(str(s.get("quote", ""))):
                            e = line2el.get(fl)
                            if e and e["eid"] not in gold:
                                gold.append(e["eid"])
                                spans.append(fl)
                        map_mode = "line+quote폴백"
                        continue
                    e = line2el.get(ln)
                    if not e:
                        continue
                    spans.append(ln)
                    if e["eid"] not in gold:
                        gold.append(e["eid"])
                    q60 = ns(str(s.get("quote", "")))[:60]
                    ctx = ns(e["text"])
                    idx = els.index(e)
                    for j in (idx - 1, idx + 1):
                        if 0 <= j < len(els):
                            ctx += ns(els[j]["text"])
                    if q60 and q60 in ctx:
                        verified += 1
                if not srcs and row["출처"].strip():  # 비구조 발췌(검수필요) 폴백
                    map_mode = "발췌매핑"
                    for fl in excerpt_lines(row["출처"]):
                        e = line2el.get(fl)
                        if e and e["eid"] not in gold:
                            gold.append(e["eid"])
                            spans.append(fl)
                contam = ("강오염(구dev)" if qid in old_dev else
                          "약오염(구test개봉)" if qid in old_test else "청정")
                rows_out.append({"qid": qid, "split": split, "q": row["질문"],
                                 "신뢰도": row["신뢰도"], "방법": row["방법"],
                                 "map_mode": map_mode,
                                 "gold": gold, "lines": spans,
                                 "quotes_total": len(srcs), "quotes_verified": verified,
                                 "contamination": contam if split == "test" else
                                 ("구dev포함" if qid in old_dev else
                                  "구test포함" if qid in old_test else "신규")})
                stats[f"{split}_mapped" if gold else f"{split}_unmapped"] += 1
                if split == "test":
                    stats[f"test_{contam}"] += 1

    f = OUT / "gold_mapped_500.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    vq = sum(r["quotes_verified"] for r in rows_out)
    tq = sum(r["quotes_total"] for r in rows_out)
    print(f"매핑: {dict(stats)} | quote 검증 {vq}/{tq} ({vq/max(1,tq)*100:.1f}%) | sha {sha[:16]}")

    # train 350 매핑분 A0 기준선 + recall 곡선 (test는 봉인 — 채점 0회)
    bm = SE.BM25([e["text"] for e in els])
    import math
    def full_order(q):
        qb = list(Counter(SE.bigrams(q)))
        sc = []
        for i in range(bm.N):
            s, tfi = 0.0, bm.tf[i]
            for t in qb:
                fq = tfi.get(t, 0)
                if not fq:
                    continue
                idf = math.log(1 + (bm.N - bm.df[t] + 0.5) / (bm.df[t] + 0.5))
                B = 1 - bm.b + bm.b * (bm.dl[i] / bm.avgdl)
                s += idf * fq * (bm.k1 + 1) / (fq + bm.k1 * B)
            sc.append(s)
        return sorted(range(bm.N), key=lambda i: -sc[i])

    train = [r for r in rows_out if r["split"] == "train" and r["gold"]]
    hit = Counter()
    mrr = 0.0
    for r in train:
        order = full_order(r["q"])
        g = set(r["gold"])
        rank = next((i + 1 for i, ix in enumerate(order[:100])
                     if els[ix]["eid"] in g), None)
        mrr += (1 / rank) if rank and rank <= 10 else 0.0
        for K in (5, 10, 20, 50, 100):
            if rank and rank <= K:
                hit[K] += 1
    n = len(train)
    base = {f"recall@{K}": round(hit[K] / n, 4) for K in (5, 10, 20, 50, 100)}
    base["mrr10"] = round(mrr / n, 4)
    print(f"train 기준선(A0, n={n}): {base}")
    meta = {"code_commit": commit, "gold_sha": sha, "stats": dict(stats),
            "quote_verify": f"{vq}/{tq}", "train_baseline_A0": base, "train_n": n,
            "test_sealed": True}
    (OUT / "migration_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
