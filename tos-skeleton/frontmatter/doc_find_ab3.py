#!/usr/bin/env python3
"""문서 찾기 3조건 실험 — 컬렉션(전체 9,417건)에서 정답 '파일'에 도달하는가.

조건: ①없음(폴더+파일명) ②구조화(정규화 엔트리) ③요약(상품 요약문+파일명)
질의: 파일 시스템 사실에서 합성(순환 완화), 다중 정답 허용, 모호 날짜 제외
유형: 최신 정본 / 연도+종류 / 종류
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

OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
rng = random.Random(7)


def clean_name(name):
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    s = s.replace("무배당", " ").replace("(무)", " ")
    return re.sub(r"\s+", " ", s).strip()


def ambiguous(date):  # 6자리인데 20/19로 시작 → 2008-05 vs 2020-08-05 모호
    return date and len(date) == 8 and date[:2] in ("19", "20") and \
        date[2:4] in ("19", "20")


def main():
    fms = [json.loads(l) for l in open(OUT / "collection_fm.jsonl")]
    summ = {json.loads(l)["collection"]: json.loads(l)["summary"]
            for l in open(OUT / "collection_summary.jsonl")}

    docs = []  # 문서 = 채점 단위
    for f in fms:
        flags = " ".join(f["identity"]["flags"] + f["identity"]["kind_hints"])
        reps = set(f["representative"].values())
        for iv in f["inventory"]:
            date = iv["date"]
            entry = {
                "coll": f["collection"], "file": iv["file"],
                "doc_type": iv["doc_type"], "date": date,
                "is_rep": iv["file"] in reps,
                "repA": f["collection"] + " " + iv["file"],
                "repB": " ".join([
                    f["collection"], flags, iv["doc_type"],
                    (date[:4] + "년 " + date[:8] if date else ""),
                    "최신판 정본 현행" if iv["file"] in reps else "구판"]),
                "repC": f["collection"] + " " + iv["file"] + " " +
                        summ.get(f["collection"], "")[:600],
            }
            docs.append(entry)
    print(f"문서 {len(docs):,}건 인덱싱")
    arms = {"없음": BM25([d["repA"] for d in docs]),
            "구조화": BM25([d["repB"] for d in docs]),
            "요약": BM25([d["repC"] for d in docs])}

    # ── 질의 합성 (파일 시스템 사실 기반, gold = 파일 집합)
    by_coll = defaultdict(list)
    for i, d in enumerate(docs):
        by_coll[d["coll"]].append(i)
    queries = []
    colls = [f for f in fms if len(f["inventory"]) >= 3]
    rng.shuffle(colls)
    for f in colls[:250]:
        name = clean_name(f["collection"])
        if len(name) < 4:
            continue
        idxs = by_coll[f["collection"]]
        # 유형1: 최신 정본 (약관이 2벌 이상 + 날짜 모호 없음)
        yak = [i for i in idxs if docs[i]["doc_type"] == "판매약관" and docs[i]["date"]]
        if len(yak) >= 2 and not any(ambiguous(docs[i]["date"]) for i in yak):
            mx = max(docs[i]["date"] for i in yak)
            gold = [i for i in yak if docs[i]["date"] == mx]
            queries.append({"type": "최신정본", "q": f"{name} 최신 판매약관",
                            "gold": set(gold)})
        # 유형2: 연도+종류
        dated = [i for i in idxs if docs[i]["date"] and docs[i]["doc_type"] != "미분류"
                 and not ambiguous(docs[i]["date"])]
        if dated:
            pick = rng.choice(dated)
            y, t = docs[pick]["date"][:4], docs[pick]["doc_type"]
            gold = [i for i in dated if docs[i]["date"][:4] == y
                    and docs[i]["doc_type"] == t]
            queries.append({"type": "연도종류", "q": f"{name} {y}년 {t}",
                            "gold": set(gold)})
        # 유형3: 종류만
        types = {docs[i]["doc_type"] for i in idxs} - {"미분류"}
        if types:
            t = rng.choice(sorted(types))
            gold = [i for i in idxs if docs[i]["doc_type"] == t]
            queries.append({"type": "종류", "q": f"{name} {t} 보여줘",
                            "gold": set(gold)})
    print(f"질의 {len(queries)}개: {Counter(q['type'] for q in queries)}")

    res = defaultdict(lambda: defaultdict(Counter))
    all_idx = list(range(len(docs)))
    for item in queries:
        for name, bm in arms.items():
            ranked = bm.rank(item["q"], all_idx)[:5]
            res[item["type"]][name]["n"] += 1
            if ranked[:1] and ranked[0] in item["gold"]:
                res[item["type"]][name]["top1"] += 1
            if item["gold"] & set(ranked):
                res[item["type"]][name]["top5"] += 1

    order = ["없음", "구조화", "요약"]
    print(f"\n{'유형':8s}" + "".join(f"  {a}t1 {a}t5" for a in order))
    tot = defaultdict(Counter)
    for t in ("최신정본", "연도종류", "종류"):
        d = res[t]
        n = d[order[0]]["n"]
        print(f"{t:10s}" + "".join(f"{d[a]['top1']/n:7.2f}{d[a]['top5']/n:7.2f}" for a in order))
        for a in order:
            for k in ("n", "top1", "top5"):
                tot[a][k] += d[a][k]
    n = tot[order[0]]["n"]
    print(f"{'전체':10s}" + "".join(f"{tot[a]['top1']/n:7.2f}{tot[a]['top5']/n:7.2f}" for a in order))
    json.dump({t: {a: dict(res[t][a]) for a in res[t]} for t in res},
              open(OUT / "doc_find_ab3.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
