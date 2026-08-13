#!/usr/bin/env python3
"""Remap existing gold spans to corrected chunks and repaired elements."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def overlap(row, spans):
    return any(row["char_start"] < end and row["char_end"] > start for start, end in spans)


def main():
    gold = load("train350_gold.jsonl")
    chunks = load("chunks.jsonl")
    elements = load("elements_repaired_v1.jsonl")
    output = []
    for row in gold:
        spans = row["gold_spans"]
        updated = dict(row)
        updated["gold_chunk_ids"] = [item["chunk_id"] for item in chunks if overlap(item, spans)]
        updated["gold_element_ids"] = [item["element_id"] for item in elements if overlap(item, spans)]
        if not updated["gold_chunk_ids"] or not updated["gold_element_ids"]:
            raise RuntimeError(f"unmapped after repair: {row['qid']}")
        output.append(updated)
    with (OUT / "train350_gold_repaired_v1.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "questions": len(output), "all_mapped": True,
        "unique_chunks": len({value for row in output for value in row["gold_chunk_ids"]}),
        "unique_elements": len({value for row in output for value in row["gold_element_ids"]}),
        "questions_touching_repaired_child": sum(any("__r" in value for value in row["gold_element_ids"]) for row in output),
        "mean_gold_chunks": round(sum(len(row["gold_chunk_ids"]) for row in output) / len(output), 2),
        "mean_gold_elements": round(sum(len(row["gold_element_ids"]) for row in output) / len(output), 2),
    }
    (OUT / "train350_gold_repaired_v1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
