#!/usr/bin/env python3
"""Audit repaired recursive chunks and selectively repaired elements."""
from __future__ import annotations

import collections
import json
import re
import statistics
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
ARTICLE_RE = re.compile(r"(?:^|\n)#{0,6}\s*제\s?\d+(?:-\d+)?조(?:의\s?\d+)?", re.M)
CONTRACT_RE = re.compile(
    r"(?m)^(?:[^|\n]{2,180}특약[^()\n]{0,40}\(무배당[^)\n]*\)|"
    r"신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^\n]*\))\s*$"
)


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def distribution(values):
    ordered = sorted(values)
    def at(p):
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]
    return {"min": ordered[0], "p50": at(.5), "p90": at(.9), "p99": at(.99),
            "max": ordered[-1], "mean": round(statistics.mean(ordered), 2)}


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    chunks = load("chunks.jsonl")
    elements = load("elements_repaired_v1.jsonl")
    manifest = json.loads((OUT / "element_repair_manifest_v1.json").read_text(encoding="utf-8"))
    parents = set(manifest["parent_map"])

    chunk_text_mismatch = [row["chunk_id"] for row in chunks if raw[row["char_start"]:row["char_end"]] != row["text"]]
    chunk_core_gaps = []
    for previous, current in zip(chunks, chunks[1:]):
        if previous["char_end"] != current["core_char_start"]:
            chunk_core_gaps.append((previous["chunk_id"], current["chunk_id"], previous["char_end"], current["core_char_start"]))
    core_lengths = [row["char_end"] - row["core_char_start"] for row in chunks]
    stored_lengths = [len(row["text"]) for row in chunks]
    chunk_contract_mix = [row["chunk_id"] for row in chunks if len(CONTRACT_RE.findall(row["text"])) > 1]
    chunk_article_mix = [row["chunk_id"] for row in chunks if len(ARTICLE_RE.findall(raw[row["core_char_start"]:row["char_end"]])) > 2]
    duplicate_texts = collections.Counter(row["text"] for row in chunks)

    element_text_mismatch = [row["element_id"] for row in elements if raw[row["char_start"]:row["char_end"]] != row["text"]]
    element_giant = [row["element_id"] for row in elements if len(row["text"]) > 4000 or row["line_end"] - row["line_start"] + 1 > 80]
    element_article_mix = [row["element_id"] for row in elements if len(ARTICLE_RE.findall(row["text"])) >= 3]
    element_contract_mix = [row["element_id"] for row in elements if len(CONTRACT_RE.findall(row["text"])) >= 2]
    unchanged = [row for row in elements if "parent_element_id" not in row]
    source_ids = {row["element_id"] for row in load("elements.jsonl")}
    unchanged_integrity = all(row["element_id"] in source_ids and row["element_id"] not in parents for row in unchanged)

    result = {
        "chunk": {
            "count": len(chunks), "core_length": distribution(core_lengths), "stored_length": distribution(stored_lengths),
            "source_text_mismatch": len(chunk_text_mismatch), "core_gap_or_overlap": len(chunk_core_gaps),
            "core_over_600": sum(value > 600 for value in core_lengths), "stored_over_700": sum(value > 700 for value in stored_lengths),
            "tiny_core_under_100": sum(value < 100 for value in core_lengths), "blank": sum(not row["text"].strip() for row in chunks),
            "multiple_contract_headers": len(chunk_contract_mix), "three_plus_article_headers": len(chunk_article_mix),
            "exact_duplicate_groups": sum(count > 1 for count in duplicate_texts.values()),
            "scope_filled": sum(bool(row.get("contract_scope")) for row in chunks),
            "examples": {"source_mismatch": chunk_text_mismatch[:20], "core_gap": chunk_core_gaps[:20], "article_mix": chunk_article_mix[:20]},
        },
        "element": {
            "count": len(elements), "repaired_parents": len(parents), "children": manifest["child_count"],
            "unchanged": len(unchanged), "unchanged_id_integrity": unchanged_integrity,
            "source_text_mismatch": len(element_text_mismatch), "giant_or_80plus": len(element_giant),
            "three_plus_article_headers": len(element_article_mix), "multiple_contract_headers": len(element_contract_mix),
            "examples": {"article_mix": element_article_mix[:20], "contract_mix": element_contract_mix[:20]},
        },
    }
    result["verdict"] = {
        "chunk_integrity": "pass" if not chunk_text_mismatch and not chunk_core_gaps and not result["chunk"]["core_over_600"] and not result["chunk"]["stored_over_700"] and not chunk_contract_mix else "fail",
        "chunk_semantic_granularity": "conditional" if chunk_article_mix else "pass",
        "element_integrity": "pass" if not element_text_mismatch and not element_giant and not element_article_mix and not element_contract_mix and unchanged_integrity else "fail",
    }
    (OUT / "repaired_structure_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
