#!/usr/bin/env python3
"""Keep unchanged tags intact and splice tags for repaired children into two views."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "indexing"))
from build_views_v2 import ser_tag

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(value, total):
    return round(value / total * 100, 2) if total else 0.0


def main():
    elements = load("elements_repaired_v1.jsonl")
    child_ids = {row["element_id"] for row in elements if row.get("parent_element_id")}
    old_original = {row["element_id"]: row for row in load("element_tags_v2.jsonl")}
    new_original = {row["element_id"]: row for row in load("element_semantic_fields_v4.jsonl")}
    old_generated = {row["element_id"]: row for row in load("element_tags_old_repaired_full_v1.jsonl")}
    new_generated = {row["element_id"]: row for row in load("element_tags_new_repaired_full_v1.jsonl")}

    old_delta = [old_generated[element_id] for element_id in sorted(child_ids)]
    new_delta = [new_generated[element_id] for element_id in sorted(child_ids)]
    old_full, new_full = [], []
    for element in elements:
        element_id = element["element_id"]
        old_full.append(old_generated[element_id] if element_id in child_ids else old_original[element_id])
        new_full.append(new_generated[element_id] if element_id in child_ids else new_original[element_id])

    write("element_tags_old_repaired_delta_v1.jsonl", old_delta)
    write("element_tags_new_repaired_delta_v1.jsonl", new_delta)
    write("element_tags_old_repaired_full_v1.jsonl", old_full)
    write("element_tags_new_repaired_full_v1.jsonl", new_full)

    element_by = {row["element_id"]: row for row in elements}
    write("grep_tag_old_repaired_v1.jsonl", [
        {"element_id": row["element_id"], "g": ser_tag(row) + " ||| " + element_by[row["element_id"]]["text"].replace("\n", " ")}
        for row in old_full
    ])
    write("grep_tag_new_repaired_v1.jsonl", [
        {"element_id": row["element_id"], "g": row["search_text"] + " ||| " + element_by[row["element_id"]]["text"].replace("\n", " ")}
        for row in new_full
    ])
    write("grep_base_repaired_v1.jsonl", [
        {"element_id": row["element_id"], "g": row["text"].replace("\n", " ")}
        for row in elements
    ])

    def address(row):
        locator = row["locator"]
        return (row["schema_tag"], row["contract_key"], tuple(row["subject_key"]), tuple(row["role"]),
                locator.get("article", ""), locator.get("article_title", ""), locator.get("section", ""),
                tuple(locator.get("table_headers", [])), tuple(locator.get("row_keys", [])), tuple(row["qualifier"]))

    addresses = collections.Counter(address(row) for row in new_delta if row["contract_key"] and (row["subject_key"] or row["role"]))
    gold_rows = load("train350_gold_repaired_v1.jsonl") if (OUT / "train350_gold_repaired_v1.jsonl").exists() else []
    gold_ids = {value for row in gold_rows for value in row["gold_element_ids"]}
    gold_tag_rows = [row for row in new_full if row["element_id"] in gold_ids]
    gold_addresses = collections.Counter(address(row) for row in gold_tag_rows if row["contract_key"] and (row["subject_key"] or row["role"]))
    subjects = collections.Counter(value for row in new_delta for value in row["subject_key"])
    roles = collections.Counter(value for row in new_delta for value in row["role"])
    report = {
        "scope": {"unchanged_elements": len(elements) - len(child_ids), "repaired_children": len(child_ids), "full_rows": len(elements)},
        "alignment": {"old_delta": len(old_delta), "new_delta": len(new_delta), "old_full": len(old_full), "new_full": len(new_full)},
        "old_delta_coverage": {
            "article": pct(sum(bool(row["article"]) for row in old_delta), len(old_delta)),
            "role": pct(sum(bool(row["semantic_role"]) for row in old_delta), len(old_delta)),
            "values": pct(sum(bool(row["values"]) for row in old_delta), len(old_delta)),
            "conditions": pct(sum(bool(row["conditions"]) for row in old_delta), len(old_delta)),
        },
        "new_delta_coverage": {
            "contract": pct(sum(bool(row["contract_key"]) for row in new_delta), len(new_delta)),
            "subject": pct(sum(bool(row["subject_key"]) for row in new_delta), len(new_delta)),
            "role": pct(sum(bool(row["role"]) for row in new_delta), len(new_delta)),
            "article": pct(sum(bool(row["locator"].get("article")) for row in new_delta), len(new_delta)),
            "qualifier": pct(sum(bool(row["qualifier"]) for row in new_delta), len(new_delta)),
        },
        "new_delta_address": {
            "eligible_rows": sum(addresses.values()), "distinct": len(addresses),
            "unique_rows_pct": pct(sum(count for count in addresses.values() if count == 1), sum(addresses.values())),
            "collision_groups": sum(count > 1 for count in addresses.values()),
        },
        "train_gold_address": {
            "rows": len(gold_tag_rows), "eligible_rows": sum(gold_addresses.values()), "distinct": len(gold_addresses),
            "unique_rows_pct": pct(sum(count for count in gold_addresses.values() if count == 1), sum(gold_addresses.values())),
            "collision_groups": sum(count > 1 for count in gold_addresses.values()),
        },
        "new_delta_repetition": {"subjects_top20": subjects.most_common(20), "roles": roles.most_common()},
        "qa_accessed_by_tag_generation": False,
    }
    (OUT / "repaired_tag_variants_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
