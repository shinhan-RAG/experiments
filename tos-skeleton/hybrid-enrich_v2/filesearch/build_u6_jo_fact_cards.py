#!/usr/bin/env python3
"""Build generic JO-level fact cards from source elements and U4 tags.

No QA, Gold or retrieval artifact is read.  Adjacent answer-bearing anchors inside
the same legal article are bundled in source order so fragmented definitions,
enumerations, conditions and table rows can be searched as one evidence unit.
Every synthetic card remains a member of its original JO and preserves provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
VERSION = "semtag-u6-jo-fact-card-1.0"
LIST_FIELDS = (
    "subject_key", "role", "qualifier", "reference", "answer_values",
    "benefit_aliases", "fact_roles", "explicit_table_references",
)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(values, limit=None):
    output = list(dict.fromkeys(value for value in values if value not in (None, "")))
    return output[:limit] if limit else output


def anchor_blocks(members, tags, max_chars=1600):
    """Bundle adjacent member anchors without dropping any accepted source anchor."""
    blocks, current, current_members, size = [], [], [], 0
    for member in members:
        anchors = unique(str(value).strip() for value in tags[member].get("evidence_anchor", []))
        if not anchors:
            continue
        member_text = "\n".join(anchors)
        if current and size + 1 + len(member_text) > max_chars:
            blocks.append((current_members, current))
            current, current_members, size = [], [], 0
        # A single source anchor is never truncated; provenance matters more than
        # the soft block budget.  Large table rows form their own card.
        current.extend(anchors)
        current_members.append(member)
        size += (1 if size else 0) + len(member_text)
    if current:
        blocks.append((current_members, current))
    return blocks


def aggregate_tag(card_id, parent, member_ids, tags, anchors):
    source_tags = [tags[member] for member in member_ids]
    contract_keys = unique(tag.get("contract_key", "") for tag in source_tags)
    document_keys = unique(tag.get("document_key", "") for tag in source_tags)
    if len(contract_keys) > 1:
        raise ValueError(f"multiple contract keys inside {parent['element_id']}: {contract_keys}")
    if len(document_keys) > 1:
        raise ValueError(f"multiple document keys inside {parent['element_id']}: {document_keys}")
    output = {
        "element_id": card_id,
        "schema_version": source_tags[0].get("schema_version", ""),
        "schema_tag": "jo_fact_card",
        "contract_key": contract_keys[0] if contract_keys else parent.get("contract_scope", ""),
        "subject_key": [],
        "role": [],
        "locator": {
            "article": "",
            "article_title": parent.get("title", ""),
            "table_headers": [],
            "row_keys": [],
            "section": "",
        },
        "qualifier": [],
        "reference": [],
        "fact_tag_version": VERSION,
        "parent_jo": parent["element_id"],
        "fact_card_source_element_ids": member_ids,
        # One co-located unit is the treatment: split source members remain listed
        # separately in provenance, while BM25F sees their complete answer kernel.
        "evidence_anchor": ["\n".join(anchors)],
    }
    if document_keys:
        output["document_key"] = document_keys[0]
    for field in LIST_FIELDS:
        values = []
        for tag in source_tags:
            raw = tag.get(field, [])
            values.extend(raw if isinstance(raw, list) else [raw])
        if values:
            output[field] = unique(values, 80)
    output["search_text"] = " | ".join(filter(None, (
        "[schema] jo_fact_card",
        "[contract] " + output["contract_key"] if output["contract_key"] else "",
        "[article] " + parent.get("title", "") if parent.get("title") else "",
        "[subject] " + " | ".join(output.get("subject_key", [])),
        "[role] " + " | ".join(output.get("role", [])),
        "[fact] " + " | ".join(output.get("fact_roles", [])),
    )))
    return output


def build(elements, tags_rows, jo_rows, max_chars=1600):
    if len(elements) != len(tags_rows):
        raise ValueError("element/tag length mismatch")
    element_ids = [row["element_id"] for row in elements]
    if element_ids != [row["element_id"] for row in tags_rows]:
        raise ValueError("element/tag id order mismatch")
    if len(element_ids) != len(set(element_ids)):
        raise ValueError("duplicate source element id")
    elements_by_id = {row["element_id"]: row for row in elements}
    tags = {row["element_id"]: row for row in tags_rows}
    source_members = [member for jo in jo_rows for member in jo.get("members", [])]
    if len(source_members) != len(set(source_members)) or set(source_members) != set(element_ids):
        raise ValueError("source JO membership is not an exact element partition")

    out_elements = [copy.deepcopy(row) for row in elements]
    out_tags = [copy.deepcopy(row) for row in tags_rows]
    out_jo = [copy.deepcopy(row) for row in jo_rows]
    next_id = max(int(element_id[1:]) for element_id in element_ids) + 1
    card_count = 0
    for parent in out_jo:
        original_members = list(parent["members"])
        for member_ids, anchors in anchor_blocks(original_members, tags, max_chars=max_chars):
            # A lone already-atomic source element adds no co-location value.
            if len(member_ids) < 2:
                continue
            card_id = f"e{next_id:05d}"
            next_id += 1
            card_tag = aggregate_tag(card_id, parent, member_ids, tags, anchors)
            source_rows = [elements_by_id[member] for member in member_ids]
            card_text = "\n".join(filter(None, (
                parent.get("contract_scope", ""), parent.get("title", ""), *anchors,
            )))
            card = {
                "element_id": card_id,
                "element_type": "jo_fact_card",
                "title": parent.get("title", ""),
                "text": card_text,
                "char_start": min(row["char_start"] for row in source_rows),
                "char_end": max(row["char_end"] for row in source_rows),
                "line_start": min(row.get("line_start", 0) for row in source_rows),
                "line_end": max(row.get("line_end", 0) for row in source_rows),
                "contract_scope": parent.get("contract_scope", ""),
                "source_element_ids": member_ids,
                "parent_jo": parent["element_id"],
            }
            out_elements.append(card)
            out_tags.append(card_tag)
            parent["members"].append(card_id)
            card_count += 1

    all_members = [member for jo in out_jo for member in jo["members"]]
    output_ids = [row["element_id"] for row in out_elements]
    if len(all_members) != len(set(all_members)) or set(all_members) != set(output_ids):
        raise ValueError("output JO membership is not an exact element partition")
    if output_ids != [row["element_id"] for row in out_tags]:
        raise ValueError("output element/tag order mismatch")
    return out_elements, out_tags, out_jo, card_count


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default=str(HERE / "out/elements_u3.jsonl"))
    parser.add_argument("--tags", default=str(HERE / "out/tags_u4_fact_rules.jsonl"))
    parser.add_argument("--jo", default=str(HERE / "out/elements_u3jo.jsonl"))
    parser.add_argument("--out-elements", default=str(HERE / "out/elements_u6_fact_cards.jsonl"))
    parser.add_argument("--out-tags", default=str(HERE / "out/tags_u6_fact_cards.jsonl"))
    parser.add_argument("--out-jo", default=str(HERE / "out/elements_u6_fact_cards_jo.jsonl"))
    parser.add_argument("--out-card-elements", default=str(HERE / "out/elements_u6_fact_cards_only.jsonl"))
    parser.add_argument("--out-card-tags", default=str(HERE / "out/tags_u6_fact_cards_only.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1600)
    args = parser.parse_args()

    paths = [Path(args.elements), Path(args.tags), Path(args.jo)]
    elements, tags, jo = map(load_jsonl, paths)
    out_elements, out_tags, out_jo, card_count = build(
        elements, tags, jo, max_chars=args.max_chars)
    card_elements = [row for row in out_elements if row.get("element_type") == "jo_fact_card"]
    card_ids = {row["element_id"] for row in card_elements}
    card_tags = [row for row in out_tags if row["element_id"] in card_ids]
    outputs = [Path(args.out_elements), Path(args.out_tags), Path(args.out_jo),
               Path(args.out_card_elements), Path(args.out_card_tags)]
    for path, rows in zip(outputs, (out_elements, out_tags, out_jo, card_elements, card_tags)):
        write_jsonl(path, rows)
    stats = {
        "version": VERSION,
        "policy": "adjacent answer-bearing source anchors bundled within exact parent JO",
        "inputs": {path.name: sha256(path) for path in paths},
        "generator_sha256": sha256(Path(__file__)),
        "counts": {
            "source_elements": len(elements), "fact_cards": card_count,
            "output_elements": len(out_elements), "jo": len(out_jo),
        },
        "outputs": {path.name: sha256(path) for path in outputs},
        "invariants": {
            "qa_gold_or_results_read": False,
            "source_element_prefix_preserved": True,
            "source_tag_prefix_preserved": True,
            "jo_membership_exact": True,
            "fact_card_parent_jo_explicit": True,
        },
    }
    stats_path = outputs[1].with_name(outputs[1].stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
