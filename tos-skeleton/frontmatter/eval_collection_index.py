#!/usr/bin/env python3
"""Evaluate whether collection_index.json improves document selection.

Arms:
  A filename: product folder + filename only
  B json_index: normalized fields from collection_index.json

The query set is deterministic synthetic regression data. It proves field utility,
not production effect size; real user queries remain required.
"""
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from gate2 import BM25

OUT = Path(__file__).parent / "out"
INDEX = OUT / "collection_index.json"
RESULT = OUT / "collection_index_eval.json"
QA_FILE = OUT / "doc_find_qa_582.json"


def clean_product(name):
    value = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    value = value.replace("무배당", " ").replace("(무)", " ")
    return re.sub(r"\s+", " ", value).strip()


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def searchable_text(product, doc):
    date = doc["effective_date"] or ""
    year = date[:4]
    status = "최신 최신판 정본 현행 대표" if doc["is_representative"] else "구판 이전판"
    return " ".join(filter(None, [
        product["product_name"], doc["title"],
        " ".join(product["flags"]), " ".join(product["kind_hints"]),
        doc["doc_type"], year + "년" if year else "", date, status,
    ]))


def nfc(value):
    return unicodedata.normalize("NFC", value)


def load_frozen_queries(docs):
    """Reuse the exact 582 QA items from the previous doc_find_ab3 test."""
    by_path = {nfc(doc["file_path"]): i for i, doc in enumerate(docs)}
    by_name = defaultdict(set)
    for i, doc in enumerate(docs):
        by_name[nfc(doc["file_name"])].add(i)
    frozen = []
    payload = json.loads(QA_FILE.read_text(encoding="utf-8"))
    for row in payload["queries"]:
        gold = set()
        for path in row["gold_paths"]:
            normalized = nfc(path)
            if normalized in by_path:
                gold.add(by_path[normalized])
            else:
                gold.update(by_name.get(nfc(Path(normalized).name), set()))
        if not gold:
            raise ValueError(f"No gold document mapped for frozen QA: {row['q']}")
        frozen.append({"type": row["type"], "q": row["q"], "gold": gold,
                       "gold_paths": row["gold_paths"]})
    if len(frozen) != 582:
        raise ValueError(f"Expected 582 frozen QA items, got {len(frozen)}")
    return frozen


def main():
    started = time.perf_counter()
    source = json.loads(INDEX.read_text(encoding="utf-8"))
    summaries = {row["collection"]: row["summary"] for row in
                 (json.loads(line) for line in (OUT / "collection_summary.jsonl").read_text().splitlines())}
    docs, products_by_id, by_product = [], {}, defaultdict(list)
    for product in source["products"]:
        products_by_id[product["product_id"]] = product
        for doc in product["documents"]:
            index = len(docs)
            docs.append({**doc, "product_id": product["product_id"],
                         "product_name": product["product_name"]})
            by_product[product["product_id"]].append(index)

    baseline = BM25([f"{doc['product_name']} {doc['file_name']}" for doc in docs])
    indexed = BM25([searchable_text(products_by_id[doc["product_id"]], doc) for doc in docs])
    summarized = BM25([f"{doc['product_name']} {doc['file_name']} "
                       f"{summaries.get(doc['product_name'], '')[:600]}" for doc in docs])
    queries = load_frozen_queries(docs)
    arms = {"filename": baseline, "json_index": indexed, "summary": summarized}
    hits = defaultdict(lambda: defaultdict(Counter))
    paired = {1: Counter(), 5: Counter()}
    query_rows = []

    for item in queries:
        arm_hits = {}
        for arm, engine in arms.items():
            ranked = engine.rank(item["q"], range(len(docs)))[:5]
            h1 = bool(ranked and ranked[0] in item["gold"])
            h5 = bool(item["gold"] & set(ranked))
            arm_hits[arm] = {1: h1, 5: h5}
            bucket = hits[item["type"]][arm]
            bucket["n"] += 1
            bucket["top1"] += h1
            bucket["top5"] += h5
        for k in (1, 5):
            a, b = arm_hits["filename"][k], arm_hits["json_index"][k]
            paired[k]["a_only"] += a and not b
            paired[k]["b_only"] += b and not a
        query_rows.append({"type": item["type"], "q": item["q"],
                           **{f"{arm}_hit{k}": arm_hits[arm][k]
                              for arm in arms for k in (1, 5)},
                           "gold_paths": item["gold_paths"]})

    totals = {arm: Counter() for arm in arms}
    for by_arm in hits.values():
        for arm, values in by_arm.items():
            totals[arm].update(values)
    print(f"documents={len(docs):,} queries={len(queries):,} {dict(Counter(q['type'] for q in queries))}")
    for dtype in ("최신정본", "연도종류", "종류"):
        print(f"\n[{dtype}]")
        for arm in arms:
            value = hits[dtype][arm]
            print(f"  {arm:10s} top1={value['top1']/value['n']:.3f} top5={value['top5']/value['n']:.3f} n={value['n']}")
    print("\n[overall]")
    for arm, value in totals.items():
        print(f"  {arm:10s} top1={value['top1']/value['n']:.3f} top5={value['top5']/value['n']:.3f} n={value['n']}")
    tests = {}
    for k in (1, 5):
        a_only, b_only = paired[k]["a_only"], paired[k]["b_only"]
        p = exact_mcnemar(a_only, b_only)
        tests[f"top{k}"] = {"filename_only": a_only, "json_index_only": b_only, "p": p}
        print(f"  McNemar top{k}: filename_only={a_only} json_only={b_only} p={p:.6g}")

    elapsed_seconds = time.perf_counter() - started
    result = {
        "schema_version": source["schema_version"],
        "qa_source": str(QA_FILE),
        "n_documents": len(docs),
        "n_queries": len(queries),
        "elapsed_seconds": elapsed_seconds,
        "query_types": dict(Counter(q["type"] for q in queries)),
        "results": {dtype: {arm: dict(value) for arm, value in by_arm.items()}
                    for dtype, by_arm in hits.items()},
        "totals": {arm: dict(value) for arm, value in totals.items()},
        "mcnemar": tests,
        "limitations": [
            "Queries are deterministic synthetic regression data generated from indexable facts.",
            "The test measures structured-field utility, not JSON syntax or production effect size.",
            "Real user queries and an independent gold set are still required.",
        ],
        "queries": query_rows,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {RESULT}")
    print(f"elapsed_seconds={elapsed_seconds:.3f}")


if __name__ == "__main__":
    main()
