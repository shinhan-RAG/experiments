#!/usr/bin/env python3
"""Freeze the exact 582 QA items used by doc_find_ab3.py (seed=7)."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
DEST = OUT / "doc_find_qa_582.json"


def clean_name(name):
    value = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    value = value.replace("무배당", " ").replace("(무)", " ")
    return re.sub(r"\s+", " ", value).strip()


def ambiguous(date):
    return date and len(date) == 8 and date[:2] in ("19", "20") and date[2:4] in ("19", "20")


def main():
    rng = random.Random(7)
    fms = [json.loads(line) for line in (OUT / "collection_fm.jsonl").read_text().splitlines()]
    docs, by_collection = [], defaultdict(list)
    for fm in fms:
        for item in fm["inventory"]:
            index = len(docs)
            docs.append({"collection": fm["collection"], "file": item["file"],
                         "doc_type": item["doc_type"], "date": item["date"]})
            by_collection[fm["collection"]].append(index)

    queries = []
    collections = [fm for fm in fms if len(fm["inventory"]) >= 3]
    rng.shuffle(collections)
    for fm in collections[:250]:
        name = clean_name(fm["collection"])
        if len(name) < 4:
            continue
        indices = by_collection[fm["collection"]]
        policies = [i for i in indices if docs[i]["doc_type"] == "판매약관" and docs[i]["date"]]
        if len(policies) >= 2 and not any(ambiguous(docs[i]["date"]) for i in policies):
            latest = max(docs[i]["date"] for i in policies)
            gold = [i for i in policies if docs[i]["date"] == latest]
            queries.append({"type": "최신정본", "q": f"{name} 최신 판매약관",
                            "gold_paths": [f"{docs[i]['collection']}/{docs[i]['file']}" for i in gold]})
        dated = [i for i in indices if docs[i]["date"] and docs[i]["doc_type"] != "미분류"
                 and not ambiguous(docs[i]["date"])]
        if dated:
            chosen = rng.choice(dated)
            year, dtype = docs[chosen]["date"][:4], docs[chosen]["doc_type"]
            gold = [i for i in dated if docs[i]["date"][:4] == year and docs[i]["doc_type"] == dtype]
            queries.append({"type": "연도종류", "q": f"{name} {year}년 {dtype}",
                            "gold_paths": [f"{docs[i]['collection']}/{docs[i]['file']}" for i in gold]})
        types = {docs[i]["doc_type"] for i in indices} - {"미분류"}
        if types:
            dtype = rng.choice(sorted(types))
            gold = [i for i in indices if docs[i]["doc_type"] == dtype]
            queries.append({"type": "종류", "q": f"{name} {dtype} 보여줘",
                            "gold_paths": [f"{docs[i]['collection']}/{docs[i]['file']}" for i in gold]})

    payload = {"source": "doc_find_ab3.py", "seed": 7, "n": len(queries),
               "types": dict(Counter(q["type"] for q in queries)), "queries": queries}
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {DEST}")
    print(f"queries={len(queries)} types={payload['types']}")


if __name__ == "__main__":
    main()
