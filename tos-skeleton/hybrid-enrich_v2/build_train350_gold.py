#!/usr/bin/env python3
"""Build gold chunk/element mappings for the balanced Train-350 split."""
from __future__ import annotations

import bisect
import collections
import json
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
MANUAL_LINE_ANCHORS = {"field-128": (244615, 244683)}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    norm_chars, norm_to_raw = [], []
    previous_space = False
    for index, char in enumerate(raw):
        if char.isspace():
            if not previous_space:
                norm_chars.append(" ")
                norm_to_raw.append(index)
            previous_space = True
        else:
            norm_chars.append(char)
            norm_to_raw.append(index)
            previous_space = False
    normalized = "".join(norm_chars)

    line_offsets, position = [], 0
    for line in raw.split("\n"):
        line_offsets.append(position)
        position += len(line) + 1

    chunks = rows(OUT / "chunks.jsonl")
    elements = rows(OUT / "elements.jsonl")
    chunk_starts = [row["char_start"] for row in chunks]
    element_starts = [row["char_start"] for row in elements]

    def overlaps(items, starts, start, end, key):
        index = max(0, bisect.bisect_left(starts, start) - 3)
        found = []
        while index < len(items) and items[index]["char_start"] < end:
            if items[index]["char_end"] > start:
                found.append(items[index][key])
            index += 1
        return found

    def locate(source):
        quote = norm(source.get("quote", ""))
        if not quote:
            return None, "empty"
        probes = [quote[:300], quote[:160], quote[:80]]
        hits = []
        for probe in probes:
            if len(probe) < 12:
                continue
            hits = [match.start() for match in re.finditer(re.escape(probe), normalized)]
            if hits:
                break
        method = "exact"
        # OCR-tolerant fallback: cluster matching 24-character quote fragments.
        if not hits:
            fragment_hits = []
            for offset in range(0, max(1, len(quote) - 23), 12):
                fragment = quote[offset:offset + 24].strip()
                if len(fragment) < 16:
                    continue
                occurrences = [match.start() for match in re.finditer(re.escape(fragment), normalized)]
                if len(occurrences) <= 20:
                    fragment_hits.extend((hit, offset) for hit in occurrences)
            candidates = []
            for hit, offset in fragment_hits:
                estimated = hit - offset
                support = sum(abs((other_hit - other_offset) - estimated) <= 120 for other_hit, other_offset in fragment_hits)
                candidates.append((support, estimated, hit))
            if candidates:
                best_support = max(value[0] for value in candidates)
                if best_support >= 2 or len(fragment_hits) == 1:
                    hits = [max(0, value[1]) for value in candidates if value[0] == best_support]
                    probe = quote[: min(len(quote), 160)]
                    method = f"fragment_{best_support}"
        if not hits:
            return None, "failed"
        try:
            line = int(source.get("line"))
            target = line_offsets[min(max(0, line - 1), len(line_offsets) - 1)]
            hits.sort(key=lambda hit: abs(norm_to_raw[min(hit, len(norm_to_raw) - 1)] - target))
        except (TypeError, ValueError):
            pass
        hit = min(hits[0], len(norm_to_raw) - 1)
        raw_start = norm_to_raw[hit]
        norm_end = min(hit + max(1, len(probe)) - 1, len(norm_to_raw) - 1)
        raw_end = norm_to_raw[norm_end] + 1
        return (raw_start, raw_end), method

    split = {row["qid"]: row for row in rows(OUT / "qa500_train_test_split_manifest.jsonl") if row["split"] == "train"}
    qa = {row["qid"]: row for row in rows(OUT / "qa503_field_plus_document_coverage_v3_reviewed.jsonl")}
    output, audit = [], []
    for qid in split:
        row = qa[qid]
        spans = []
        methods = []
        for source in row.get("sources") or []:
            located, method = locate(source)
            methods.append(method)
            if located:
                spans.append(list(located))
        if not spans and qid in MANUAL_LINE_ANCHORS:
            start_line, end_line = MANUAL_LINE_ANCHORS[qid]
            spans.append([line_offsets[start_line - 1], line_offsets[min(end_line, len(line_offsets) - 1)]])
            methods.append("manual_line_anchor")
        # Curated document-coverage rows already have narrow authoritative IDs;
        # keep those IDs while retaining source spans for overlap evaluation.
        chunk_ids = set(row.get("gold_chunk_ids") or [])
        element_ids = set(row.get("gold_element_ids") or [])
        for start, end in spans:
            chunk_ids.update(overlaps(chunks, chunk_starts, start, end, "chunk_id"))
            element_ids.update(overlaps(elements, element_starts, start, end, "element_id"))
        status = "ok" if chunk_ids and element_ids else "failed"
        audit.append({"qid": qid, "status": status, "methods": methods, "spans": spans, "gold_chunks": len(chunk_ids), "gold_elements": len(element_ids)})
        if status == "ok":
            output.append({
                "qid": qid, "question": row["question"], "business_type": row.get("business_type", ""),
                "question_types": row.get("question_types") or split[qid].get("question_types", []),
                "content_labels": split[qid].get("content_labels", []),
                "origin": split[qid]["origin"], "gold_chunk_ids": sorted(chunk_ids),
                "gold_element_ids": sorted(element_ids), "gold_spans": spans,
            })
    for path, data in ((OUT / "train350_gold.jsonl", output), (OUT / "train350_gold_audit.jsonl", audit)):
        with path.open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "requested": len(split), "mapped": len(output), "failed": [row["qid"] for row in audit if row["status"] != "ok"],
        "origin": dict(collections.Counter(row["origin"] for row in output)),
        "mapping_methods": dict(collections.Counter(method for row in audit for method in row["methods"])),
    }
    (OUT / "train350_gold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
