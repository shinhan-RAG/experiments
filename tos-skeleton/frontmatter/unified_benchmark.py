#!/usr/bin/env python3
"""통합 벤치마크 — 고정 문제지(신원 582 + 내용 205 + 혼합 123)로 4조건 일괄 채점.

조건: 파일명 / Index(신원) / fm(내용) / Index+fm
지표: hit@1 / hit@5 (정답=집합)
출력: out/unified_benchmark.json + 마스터 표
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BM25
from eval_content_qa import flatten_fm

OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")


def main():
    docs = [json.loads(l) for l in open(OUT / "doc_frontmatter.jsonl")]
    fidx = {d["file"]: i for i, d in enumerate(docs)}

    ci = json.loads((OUT / "collection_index.json").read_text())
    ident, meta = {}, {}
    for p in ci["products"]:
        for dd in p["documents"]:
            key = p["product_name"] + "/" + dd["file_name"]
            ident[key] = " ".join([
                p["product_name"], dd.get("title", ""), dd["doc_type"],
                ((dd.get("effective_date") or "")[:4] + "년") if dd.get("effective_date") else "",
                "최신판 정본 현행" if dd.get("is_representative") else "구판"])
            meta[key] = dd

    def key_of(d):
        return d["product"] + "/" + Path(d["file"]).name

    t_file, t_idx, t_fm, t_both = [], [], [], []
    for d in docs:
        k = key_of(d)
        base = d["product"] + " " + Path(d["file"]).name
        iv = ident.get(k, base)
        fv = base + " " + flatten_fm(d)
        t_file.append(base)
        t_idx.append(iv)
        t_fm.append(fv)
        t_both.append(iv + " " + flatten_fm(d))
    arms = {"파일명": BM25(t_file), "Index": BM25(t_idx),
            "fm": BM25(t_fm), "Index+fm": BM25(t_both)}

    # ── 고정 문제지 로드 (세 세트 합본)
    queries = []
    # 신원 582 (doc_find_qa_582: gold가 파일 경로 리스트)
    for item in json.loads((OUT / "doc_find_qa_582.json").read_text())["queries"]:
        queries.append({"suite": "신원:" + item.get("type", "?"), "q": item["q"],
                        "gold": item.get("gold_paths", item.get("gold", []))})
    for l in open(OUT / "content_qa.jsonl"):
        item = json.loads(l)
        queries.append({"suite": "내용:" + item["type"], "q": item["q"],
                        "gold": item.get("gold_paths", item.get("gold", []))})
    # 혼합 123 재구성 (eval_combined과 동일 로직)
    for l in open(OUT / "content_qa.jsonl"):
        item = json.loads(l)
        gold_by_type = defaultdict(list)
        for g in item["gold"]:
            if g in fidx:
                m = meta.get(key_of(docs[fidx[g]]))
                if m:
                    gold_by_type[m["doc_type"]].append((g, bool(m.get("is_representative"))))
        for dt in ("판매약관", "사업방법서", "공시약관"):
            if not gold_by_type.get(dt):
                continue
            ng = [g for g, rep in gold_by_type[dt] if rep]
            if 1 <= len(ng) <= 5:
                queries.append({"suite": "혼합", "q": f"{item['term']} 내용이 있는 최신 {dt}",
                                "gold": ng})
            break
    print(f"고정 문제지 {len(queries)}문항: " +
          str(Counter(q["suite"].split(":")[0] for q in queries)))

    all_idx = list(range(len(docs)))
    res = defaultdict(lambda: defaultdict(Counter))
    for item in queries:
        gold = {fidx[g] for g in item["gold"] if g in fidx}
        if not gold:
            continue
        suite = item["suite"].split(":")[0]
        for name, bm in arms.items():
            ranked = bm.rank(item["q"], all_idx)[:5]
            res[suite][name]["n"] += 1
            if ranked[:1] and ranked[0] in gold:
                res[suite][name]["hit1"] += 1
            if gold & set(ranked):
                res[suite][name]["hit5"] += 1

    order = list(arms)
    print(f"\n{'문제 유형':10s}" + "".join(f"{a:>16s}" for a in order))
    print(f"{'':10s}" + "".join(f"{'h1 / h5':>16s}" for _ in order))
    tot = defaultdict(Counter)
    for suite in ("신원", "내용", "혼합"):
        d = res[suite]
        n = d[order[0]]["n"]
        line = f"{suite:10s}"
        for a in order:
            line += f"{d[a]['hit1']/n:8.2f}/{d[a]['hit5']/n:.2f}"
        print(line + f"  (n={n})")
        for a in order:
            for k in ("n", "hit1", "hit5"):
                tot[a][k] += d[a][k]
    n = tot[order[0]]["n"]
    line = f"{'전체':10s}"
    for a in order:
        line += f"{tot[a]['hit1']/n:8.2f}/{tot[a]['hit5']/n:.2f}"
    print(line + f"  (n={n})")
    json.dump({s: {a: dict(res[s][a]) for a in res[s]} for s in res},
              open(OUT / "unified_benchmark.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
