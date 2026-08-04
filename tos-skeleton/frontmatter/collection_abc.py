#!/usr/bin/env python3
"""컬렉션 라우팅 3자 비교: 미사용 / 구조화 frontmatter / 요약 frontmatter.

collection_ab.py와 동일 질의(seed 42)로 C(요약) arm 추가.
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BM25

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
    summ = {json.loads(l)["collection"]: json.loads(l)["summary"]
            for l in open(OUT / "collection_summary.jsonl")}
    colls = [f["collection"] for f in fms]
    print(f"요약 확보 {len(summ)}/{len(colls)}")

    raw_texts, fm_texts, sm_texts = [], [], []
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
        sm_texts.append(f["collection"] + " " + summ.get(f["collection"], ""))
    arms = {"미사용": BM25(raw_texts), "구조화": BM25(fm_texts), "요약": BM25(sm_texts)}

    heading_count = Counter(h for f in fms for h in set(f["structure"]))
    queries = []
    sample = [f for f in fms if f["representative"] and f["structure"]]
    rng.shuffle(sample)
    for f in sample[:200]:
        cn = clean_name(f["collection"])
        if len(cn) >= 4:
            w = cn.split()
            frag = " ".join(w[:2]) if len(w) > 2 else cn
            queries.append({"type": "상품명변형", "q": f"{frag} 약관 찾아줘",
                            "gold": f["collection"]})
    for f in sample[100:200]:
        dated = [iv for iv in f["inventory"] if iv["date"] and iv["doc_type"] != "미분류"]
        if dated and f["identity"]["kind_hints"]:
            iv = rng.choice(dated)
            queries.append({"type": "속성조합",
                            "q": f"{iv['date'][:4]}년 {f['identity']['kind_hints'][0]}보험 {iv['doc_type']}",
                            "gold": f["collection"]})
    for f in sample[:150]:
        uniq = [h for h in f["structure"] if heading_count[h] == 1 and len(h) >= 6]
        if uniq:
            queries.append({"type": "목차내용", "q": f"{rng.choice(uniq)} 내용이 있는 상품",
                            "gold": f["collection"]})

    res = defaultdict(lambda: defaultdict(Counter))
    for item in queries:
        for name, bm in arms.items():
            ranked = [colls[i] for i in bm.rank(item["q"], range(len(colls)))[:5]]
            res[item["type"]][name]["n"] += 1
            if ranked[:1] == [item["gold"]]:
                res[item["type"]][name]["top1"] += 1
            if item["gold"] in ranked:
                res[item["type"]][name]["top5"] += 1

    order = ["미사용", "구조화", "요약"]
    print(f"\n{'유형':10s}" + "".join(f"{a} t1{'':2s}{a} t5{'':2s}" for a in order))
    tot = defaultdict(Counter)
    for t, d in res.items():
        n = d[order[0]]["n"]
        print(f"{t:12s}" + "".join(f"{d[a]['top1']/n:7.2f}{d[a]['top5']/n:7.2f}" for a in order))
        for a in order:
            for k in ("n", "top1", "top5"):
                tot[a][k] += d[a][k]
    n = tot[order[0]]["n"]
    print(f"{'전체':12s}" + "".join(f"{tot[a]['top1']/n:7.2f}{tot[a]['top5']/n:7.2f}" for a in order))
    json.dump({t: {a: dict(d[a]) for a in d} for t, d in res.items()},
              open(OUT / "collection_abc.json", "w"), ensure_ascii=False)
    print("ABC_DONE")


if __name__ == "__main__":
    main()
