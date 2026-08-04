#!/usr/bin/env python3
"""5단계: Index + frontmatter 결합 실험 — 혼합 질의(내용+신원)로 문서 1개 찍기.

혼합 질의: content_qa의 용어 + "최신 {doc_type}" 제약
정답: 내용 문서군 ∩ (해당 종류 & 정본) — 1~소수
조건: Index만 / fm만 / 둘 다 (BM25, 표현만 교체)
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

    # Index(신원) 표현: collection_index.json에서
    ci = json.loads((OUT / "collection_index.json").read_text())
    ident = {}
    meta = {}
    for p in ci["products"]:
        for dd in p["documents"]:
            key = p["product_name"] + "/" + dd["file_name"]
            # doc_frontmatter의 file은 "폴더/파일" — 매칭 시도
            ident[key] = " ".join([
                p["product_name"], dd.get("title", ""), dd["doc_type"],
                (dd.get("effective_date") or "")[:4] + "년" if dd.get("effective_date") else "",
                "최신판 정본 현행" if dd.get("is_representative") else "구판"])
            meta[key] = {"doc_type": dd["doc_type"],
                         "is_rep": bool(dd.get("is_representative"))}

    def key_of(d):
        return d["product"] + "/" + Path(d["file"]).name

    texts_idx, texts_fm, texts_both = [], [], []
    for d in docs:
        k = key_of(d)
        iv = ident.get(k, d["product"] + " " + Path(d["file"]).name)
        fv = d["product"] + " " + Path(d["file"]).name + " " + flatten_fm(d)
        texts_idx.append(iv)
        texts_fm.append(fv)
        texts_both.append(iv + " " + fv)
    arms = {"Index만": BM25(texts_idx), "fm만": BM25(texts_fm), "둘다": BM25(texts_both)}

    # 혼합 질의 생성
    qa = [json.loads(l) for l in open(OUT / "content_qa.jsonl")]
    mixed = []
    for item in qa:
        gold_types = Counter()
        for g in item["gold"]:
            d = docs[fidx[g]] if g in fidx else None
            if d:
                m = meta.get(key_of(d))
                if m:
                    gold_types[m["doc_type"]] += 1
        for dt in ("판매약관", "사업방법서", "공시약관"):
            if not gold_types.get(dt):
                continue
            new_gold = [g for g in item["gold"] if g in fidx
                        and meta.get(key_of(docs[fidx[g]]), {}).get("doc_type") == dt
                        and meta.get(key_of(docs[fidx[g]]), {}).get("is_rep")]
            if 1 <= len(new_gold) <= 5:
                mixed.append({"q": f"{item['term']} 내용이 있는 최신 {dt}",
                              "gold": new_gold, "src_type": item["type"]})
            break
    print(f"혼합 질의 {len(mixed)}문항 (정답 중앙 "
          f"{sorted(len(m['gold']) for m in mixed)[len(mixed)//2]}건)")

    all_idx = list(range(len(docs)))
    res = defaultdict(Counter)
    for item in mixed:
        gold = {fidx[g] for g in item["gold"]}
        for name, bm in arms.items():
            ranked = bm.rank(item["q"], all_idx)[:5]
            res[name]["n"] += 1
            if ranked[:1] and ranked[0] in gold:
                res[name]["hit1"] += 1
            if gold & set(ranked):
                res[name]["hit5"] += 1

    print(f"\n{'조건':8s}{'hit@1':>8s}{'hit@5':>8s}")
    for name in arms:
        n = res[name]["n"]
        print(f"{name:8s}{res[name]['hit1']/n:8.2f}{res[name]['hit5']/n:8.2f}")
    json.dump({k: dict(v) for k, v in res.items()},
              open(OUT / "combined_eval.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
