#!/usr/bin/env python3
"""컬렉션 라우팅 A/B — frontmatter 유/무로 올바른 컬렉션을 찾는가. LLM 미사용.

질의 합성(규칙): ①상품명 변형 ②속성 조합(연도+보종+문서종류) ③목차 내용(고유 헤딩)
A: raw (폴더명+파일명+본문 앞부분)로 검색  /  B: 컬렉션 frontmatter 텍스트로 검색
지표: 정답 컬렉션 top-1 / top-5 적중률 (질의 유형별)
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BM25  # char-bigram BM25 재사용

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
rng = random.Random(42)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def clean_name(name):
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    s = s.replace("무배당", " ").replace("(무)", " ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    fms = [json.loads(l) for l in open(OUT / "collection_fm.jsonl")]
    colls = [f["collection"] for f in fms]

    # ── 코퍼스 2벌
    raw_texts, fm_texts = [], []
    for f in fms:
        d = ROOT / f["collection"]
        body = ""
        rep = (f["representative"].get("판매약관") or
               f["representative"].get("사업방법서") or "")
        if rep and (d / rep).exists():
            body = (d / rep).read_text(encoding="utf-8", errors="ignore")[:1500]
        raw_texts.append(f["collection"] + " " +
                         " ".join(iv["file"] for iv in f["inventory"]) + " " + nfc(body))
        fm_texts.append(" ".join([
            f["collection"],
            " ".join(f["identity"]["flags"] + f["identity"]["kind_hints"]),
            " ".join(f"{iv['doc_type']}{(iv['date'] or '')[:4]}" for iv in f["inventory"]),
            " ".join(f["representative"].values()),
            " ".join(f["structure"])]))
    bmA, bmB = BM25(raw_texts), BM25(fm_texts)

    # ── 질의 합성
    heading_count = Counter(h for f in fms for h in set(f["structure"]))
    queries = []
    sample = [f for f in fms if f["representative"] and f["structure"]]
    rng.shuffle(sample)
    for f in sample[:200]:  # ① 상품명 변형
        cn = clean_name(f["collection"])
        if len(cn) >= 4:
            words = cn.split()
            frag = " ".join(words[:2]) if len(words) > 2 else cn
            queries.append({"type": "상품명변형", "q": f"{frag} 약관 찾아줘",
                            "gold": f["collection"]})
    for f in sample[100:200]:  # ② 속성 조합
        dated = [iv for iv in f["inventory"] if iv["date"] and iv["doc_type"] != "미분류"]
        if dated and f["identity"]["kind_hints"]:
            iv = rng.choice(dated)
            kind = f["identity"]["kind_hints"][0]
            queries.append({"type": "속성조합",
                            "q": f"{iv['date'][:4]}년 {kind}보험 {iv['doc_type']}",
                            "gold": f["collection"]})
    for f in sample[:150]:  # ③ 목차 내용 (그 컬렉션에만 있는 헤딩)
        uniq = [h for h in f["structure"] if heading_count[h] == 1 and len(h) >= 6]
        if uniq:
            h = rng.choice(uniq)
            queries.append({"type": "목차내용", "q": f"{h} 내용이 있는 상품",
                            "gold": f["collection"]})
    print(f"질의 {len(queries)}개: " + str(Counter(q['type'] for q in queries)))

    # ── A/B 평가
    res = defaultdict(lambda: defaultdict(Counter))
    for item in queries:
        for arm, bm in (("A", bmA), ("B", bmB)):
            ranked = bm.rank(item["q"], range(len(colls)))[:5]
            names = [colls[i] for i in ranked]
            res[item["type"]][arm]["n"] += 1
            if names[:1] == [item["gold"]]:
                res[item["type"]][arm]["top1"] += 1
            if item["gold"] in names:
                res[item["type"]][arm]["top5"] += 1

    print(f"\n{'유형':10s}{'':2s}{'A top1':>8s}{'A top5':>8s}{'B top1':>8s}{'B top5':>8s}")
    tot = defaultdict(Counter)
    for t, d in res.items():
        n = d["A"]["n"]
        print(f"{t:12s}" + "".join(f"{d[a][k]/n:8.2f}" for a in ("A", "B") for k in ("top1", "top5")))
        for a in ("A", "B"):
            for k in ("n", "top1", "top5"):
                tot[a][k] += d[a][k]
    n = tot["A"]["n"]
    print(f"{'전체':12s}" + "".join(f"{tot[a][k]/n:8.2f}" for a in ("A", "B") for k in ("top1", "top5")))
    json.dump({t: {a: dict(d[a]) for a in d} for t, d in res.items()},
              open(OUT / "collection_ab.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
