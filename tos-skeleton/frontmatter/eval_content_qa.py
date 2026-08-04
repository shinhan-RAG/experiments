#!/usr/bin/env python3
"""4단계: frontmatter 단독 실험 — 내용 질의로 문서군 찾기.

조건: 없음(상품명+파일명) / 구조화 fm / 요약 fm(있으면)
정답 = 문서군(집합), 지표 = hit@1/5/10 (top-k 안에 정답군 문서가 하나라도)
사용: python3 eval_content_qa.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BM25

OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")


def flatten_fm(r):
    fm = r["fm"]
    parts = []
    for k in ("골격", "조", "부표", "항목", "서식", "헤딩", "표캡션", "표헤더", "첫줄"):
        parts += fm.get(k, [])
    parts += [t["name"] for t in fm.get("특약구성", [])]
    return " ".join(parts)


def main():
    docs = [json.loads(l) for l in open(OUT / "doc_frontmatter.jsonl")]
    files = [d["file"] for d in docs]
    fidx = {f: i for i, f in enumerate(files)}

    arms = {}
    arms["없음"] = BM25([d["product"] + " " + Path(d["file"]).name for d in docs])
    arms["구조화fm"] = BM25([d["product"] + " " + Path(d["file"]).name + " " + flatten_fm(d)
                          for d in docs])
    summ_path = OUT / "doc_summary.jsonl"
    if summ_path.exists():
        summ = {json.loads(l)["file"]: json.loads(l)["summary"] for l in open(summ_path)}
        if len(summ) >= len(docs) * 0.95:
            arms["요약fm"] = BM25([d["product"] + " " + Path(d["file"]).name + " " +
                                 summ.get(d["file"], "") for d in docs])
        else:
            print(f"(요약 {len(summ)}/{len(docs)} — 95% 미만이라 요약 조건 보류)")

    qa = [json.loads(l) for l in open(OUT / "content_qa.jsonl")]
    all_idx = list(range(len(docs)))
    KS = (1, 5, 10)
    res = defaultdict(lambda: defaultdict(Counter))
    for item in qa:
        gold = {fidx[g] for g in item["gold"] if g in fidx}
        if not gold:
            continue
        for name, bm in arms.items():
            ranked = bm.rank(item["q"], all_idx)[:10]
            res[item["type"]][name]["n"] += 1
            for k in KS:
                if gold & set(ranked[:k]):
                    res[item["type"]][name][f"hit{k}"] += 1

    order = list(arms)
    print(f"{'유형':6s}" + "".join(f"{a:>12s} h1/h5/h10" for a in order))
    tot = defaultdict(Counter)
    for t in ("A", "B", "C"):
        d = res[t]
        n = d[order[0]]["n"]
        line = f"{t:6s}"
        for a in order:
            line += f"   {d[a]['hit1']/n:.2f}/{d[a]['hit5']/n:.2f}/{d[a]['hit10']/n:.2f}"
        print(line + f"   (n={n})")
        for a in order:
            for k in ("n", "hit1", "hit5", "hit10"):
                tot[a][k] += d[a][k]
    n = tot[order[0]]["n"]
    line = f"{'전체':6s}"
    for a in order:
        line += f"   {tot[a]['hit1']/n:.2f}/{tot[a]['hit5']/n:.2f}/{tot[a]['hit10']/n:.2f}"
    print(line + f"   (n={n})")
    json.dump({t: {a: dict(res[t][a]) for a in res[t]} for t in res},
              open(OUT / "content_qa_eval.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
