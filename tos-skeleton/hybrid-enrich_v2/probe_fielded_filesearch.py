#!/usr/bin/env python3
"""Gold-blind query probe for the new fielded FileSearch on frozen first 25."""
from __future__ import annotations

import json
from pathlib import Path

from run_retriever_ab25 import FieldedFileIndex

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def main():
    elements = load("elements_repaired_v1.jsonl")
    element_text = {row["element_id"]: row["text"] for row in elements}
    spans = {row["element_id"]: (row["char_start"], row["char_end"]) for row in elements}
    index = FieldedFileIndex(load("element_tags_new_repaired_full_v1.jsonl"), element_text,
                             load("grep_tag_new_repaired_v1.jsonl"))
    gold = load("train350_gold_repaired_v1.jsonl")[:25]
    rows = []
    for item in gold:
        # Use the full question for structured ranking; deliberately provide a
        # regex that cannot add gold candidates, isolating field-index coverage.
        result = index.search(item["question"], r"(?!)")
        rank = None
        for position, candidate in enumerate(result["results"], 1):
            span = spans[candidate["id"]]
            if any(span[0] < end and span[1] > start for start, end in item["gold_spans"]):
                rank = position
                break
        rows.append({"qid": item["qid"], "rank": rank, "top_ids": [row["id"] for row in result["results"][:10]]})
    n = len(rows)
    report = {
        "scope": "first25, structured fields only, no regex candidate contribution",
        "n": n,
        "recall@1": sum(row["rank"] == 1 for row in rows) / n,
        "recall@5": sum(row["rank"] is not None and row["rank"] <= 5 for row in rows) / n,
        "recall@10": sum(row["rank"] is not None and row["rank"] <= 10 for row in rows) / n,
        "recall@20": sum(row["rank"] is not None for row in rows) / n,
        "misses": [row["qid"] for row in rows if row["rank"] is None],
        "rows": rows,
    }
    (OUT / "probe_fielded_filesearch_first25.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
