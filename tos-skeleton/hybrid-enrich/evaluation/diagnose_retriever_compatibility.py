#!/usr/bin/env python3
"""Diagnose whether current vector/rg retrievers can consume repaired enrichment."""
from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
STOP = {"무엇", "어떻게", "인가요", "있나요", "되나요", "경우", "대한", "관련", "해당",
        "알려", "주세요", "에서는", "으로", "하는", "되는", "정한", "보험", "특약"}


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def tokens(text):
    return {word.lower() for word in re.findall(r"[가-힣A-Za-z0-9·]+", str(text))
            if len(word) >= 2 and word not in STOP}


def pct(value, total):
    return round(100 * value / total, 2) if total else 0.0


def quantiles(values):
    values = sorted(values)
    def q(p):
        return values[min(len(values) - 1, round((len(values) - 1) * p))]
    return {"min": values[0], "p50": q(.5), "p90": q(.9), "p99": q(.99), "max": values[-1],
            "mean": round(statistics.mean(values), 2)}


def flatten(row):
    values = []
    for key, value in row.items():
        if key in {"element_id", "schema_version", "field_sources", "search_text"}:
            continue
        if isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return " ".join(values)


def sample20(rows):
    if len(rows) <= 20:
        return rows
    step = len(rows) / 20.0
    return [rows[int(index * step)] for index in range(20)]


def scope(gold_rows, elements, tags, raw_rows):
    element_by = {row["element_id"]: row for row in elements}
    tag_by = {row["element_id"]: row for row in tags}
    ordered_ids = [row["element_id"] for row in elements]
    tag_text_by = {row["element_id"]: flatten(tag_by[row["element_id"]]) for row in elements}
    raw_text_by = {row["element_id"]: element_by[row["element_id"]]["text"] for row in elements}

    token_tag_postings = collections.defaultdict(list)
    token_raw_postings = collections.defaultdict(list)
    for element_id in ordered_ids:
        for token in tokens(tag_text_by[element_id]):
            token_tag_postings[token].append(element_id)
        for token in tokens(raw_text_by[element_id]):
            token_raw_postings[token].append(element_id)

    rows = []
    for gold in gold_rows:
        qtokens = tokens(gold["question"])
        gold_ids = set(gold["gold_element_ids"])
        raw_hits = {token for token in qtokens if any(token in tokens(raw_text_by.get(eid, "")) for eid in gold_ids)}
        tag_hits = {token for token in qtokens if any(token in tokens(tag_text_by.get(eid, "")) for eid in gold_ids)}
        added = tag_hits - raw_hits
        candidates = []
        for token in tag_hits:
            postings = token_tag_postings[token]
            if gold_ids & set(postings):
                sampled = {value for value in sample20(postings)}
                candidates.append((len(postings), token, bool(gold_ids & sampled)))
        best = min(candidates) if candidates else None
        rows.append({
            "qid": gold["qid"], "raw_overlap": len(raw_hits), "tag_overlap": len(tag_hits),
            "tag_added": sorted(added), "best_tag_token": best[1] if best else "",
            "best_tag_matches": best[0] if best else 0, "best_tag_gold_survives_sample20": best[2] if best else False,
        })
    return {
        "questions": len(rows),
        "tag_adds_query_evidence": sum(bool(row["tag_added"]) for row in rows),
        "tag_adds_query_evidence_pct": pct(sum(bool(row["tag_added"]) for row in rows), len(rows)),
        "no_gold_tag_overlap": sum(row["tag_overlap"] == 0 for row in rows),
        "best_single_tag_token_over20": sum(row["best_tag_matches"] > 20 for row in rows),
        "best_single_tag_token_gold_lost_by_sample20": sum(row["best_tag_matches"] > 20 and not row["best_tag_gold_survives_sample20"] for row in rows),
        "examples_tag_added": [row for row in rows if row["tag_added"]][:15],
        "examples_sampling_loss": [row for row in rows if row["best_tag_matches"] > 20 and not row["best_tag_gold_survives_sample20"]][:15],
    }


def main():
    chunks = load("chunks.jsonl")
    metas = {row["chunk_id"]: row for row in load("chunk_metadata_v3.jsonl")}
    vector_rows = load("vec_meta_v3_concat_texts.jsonl")
    elements = load("elements_repaired_v1.jsonl")
    tags = load("element_tags_new_repaired_full_v1.jsonl")
    raw_rows = load("grep_base_repaired_v1.jsonl")
    gold = load("train350_gold_repaired_v1.jsonl")

    chunk_by = {row["chunk_id"]: row for row in chunks}
    meta_lengths = [len(metas[row["chunk_id"]].get("embedding_text", "")) for row in chunks]
    raw_lengths = [len(row["text"]) for row in chunks]
    concat_lengths = [len(row["text"]) for row in vector_rows]
    ratios = [m / max(1, m + r) for m, r in zip(meta_lengths, raw_lengths)]
    vector = {
        "rows_aligned": [row["chunk_id"] for row in vector_rows] == [row["chunk_id"] for row in chunks],
        "metadata_chars": quantiles(meta_lengths), "raw_chars": quantiles(raw_lengths),
        "concat_chars": quantiles(concat_lengths), "metadata_share": quantiles(ratios),
        "over_embed_max_chars_6000": sum(value > 6000 for value in concat_lengths),
        "empty_metadata": sum(not metas[row["chunk_id"]].get("embedding_text") for row in chunks),
        "interpretation": "current vector sees labels as text but does not parse JSON fields",
    }
    tag_ids = [row["element_id"] for row in tags]
    element_ids = [row["element_id"] for row in elements]
    file_search = {
        "rows_aligned": tag_ids == element_ids,
        "tag_rows": len(tags), "raw_rows": len(raw_rows),
        "current_behavior": "regex over flattened tag+raw; over 20 matches uses uniform document-order sample",
        "first25": scope(gold[:25], elements, tags, raw_rows),
        "train350": scope(gold, elements, tags, raw_rows),
    }
    result = {
        "vector": vector,
        "file_search": file_search,
        "verdict": {
            "vector_structure_acceptance": "partial_textual_only",
            "filesearch_structure_acceptance": "insufficient_for_structured_tags",
            "old_vs_new_diagnostic_needed": True,
        },
        "limitations": [
            "Single-token sampling analysis is a conservative proxy; the Codex agent may issue multi-token regexes.",
            "Retriever candidate quality must be confirmed by the frozen first-25 paired Codex run.",
        ],
    }
    (OUT / "retriever_compatibility_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
