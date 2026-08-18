#!/usr/bin/env python3
"""Selectively re-segment only structurally suspect proxy elements."""
from __future__ import annotations

import bisect
import collections
import json
import os
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
_DEFAULT_DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
DOC = Path(os.environ.get("DOC_PATH", _DEFAULT_DOC))
MAX_CHARS = 3000
MAX_LINES = 60
ARTICLE_RE = re.compile(r"(?:^|\n)#{0,6}\s*제\s?\d+(?:-\d+)?조(?:의\s?\d+)?", re.M)
CONTRACT_RE = re.compile(r"(?m)^(?=.{2,180}$)[^|\n]*특약[^()\n]{0,40}\(무배당[^)\n]*\)\s*$")


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def suspect(element):
    text = element["text"]
    giant = len(text) > 4000 or element["line_end"] - element["line_start"] + 1 > 80
    mixed_articles = len(ARTICLE_RE.findall(text)) >= 3
    mixed_contracts = len(CONTRACT_RE.findall(text)) >= 2
    bad_formula = element["element_type"] == "formula" and (
        len(text) > 4000 or len(ARTICLE_RE.findall(text)) >= 2 or text.count("|") > 30
    )
    return giant or mixed_articles or mixed_contracts or bad_formula


def classify(text):
    stripped = text.strip()
    lines = stripped.splitlines()
    table_lines = sum(line.lstrip().startswith("|") for line in lines)
    if re.match(r"^#{1,6}\s", stripped) and len(lines) == 1:
        return "heading"
    if len(stripped) <= 160 and (re.match(r"^제\s?\d+(?:-\d+)?조", stripped) or re.search(r"특약(?:\([^)]*\))?$", stripped)):
        return "heading"
    if table_lines >= max(2, len(lines) // 2) or re.search(r"<(?:table|tr|td|th)\b", stripped, re.I):
        return "table"
    math_tokens = stripped.count("$$") + len(re.findall(r"\\(?:begin|frac|sum|times|underline)", stripped))
    if math_tokens >= 2 and len(ARTICLE_RE.findall(stripped)) == 0 and table_lines < 3:
        return "formula"
    return "paragraph"


def split_long(start, text):
    """Split a nonblank block without overlap while retaining exact offsets."""
    if len(text) <= MAX_CHARS and text.count("\n") + 1 <= MAX_LINES:
        return [(start, start + len(text), text)]
    # HTML tables are split on row boundaries when possible.
    if re.search(r"<tr\b", text, re.I):
        boundaries = [0] + [match.start() for match in re.finditer(r"(?=<tr\b)", text, re.I)] + [len(text)]
    else:
        boundaries = [0] + [match.end() for match in re.finditer(r"\n", text)] + [len(text)]
    boundaries = sorted(set(boundaries))
    pieces = []
    cursor = 0
    while cursor < len(text):
        target = min(len(text), cursor + MAX_CHARS)
        candidates = [value for value in boundaries if cursor < value <= target]
        end = max(candidates) if candidates else target
        # Enforce line cap inside the selected character window.
        segment = text[cursor:end]
        if segment.count("\n") + 1 > MAX_LINES:
            line_positions = [match.end() for match in re.finditer(r"\n", segment)]
            end = cursor + line_positions[MAX_LINES - 1]
            segment = text[cursor:end]
        if not segment:
            end = min(len(text), cursor + MAX_CHARS)
            segment = text[cursor:end]
        pieces.append((start + cursor, start + end, segment))
        cursor = end
    return pieces


def split_structural(start, text):
    """Cut at article/contract headers before applying size limits."""
    boundary_re = re.compile(
        r"(?m)(?=^#{0,6}\s*제\s?\d+(?:-\d+)?조(?:의\s?\d+)?|"
        r"^[^|\n]{2,180}특약[^()\n]{0,40}\(무배당[^)\n]*\)\s*$|"
        r"^신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^\n]*\)\s*$)"
    )
    boundaries = sorted(set([0] + [match.start() for match in boundary_re.finditer(text)] + [len(text)]))
    output = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right > left:
            output.extend(split_long(start + left, text[left:right]))
    return output


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    elements = load("elements.jsonl")
    parents = {element["element_id"] for element in elements if suspect(element)}

    line_starts, position = [], 0
    for line in raw.split("\n"):
        line_starts.append(position)
        position += len(line) + 1
    def line_of(offset):
        return bisect.bisect_right(line_starts, offset)

    # Recompute scope independently of the broken parent segmentation.
    lines = raw.split("\n")
    rider = re.compile(r"^(?=.{2,180}$)[^|\n]*특약[^()\n]{0,40}\(무배당[^)\n]*\)\s*$")
    main_contract = re.compile(r"^신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^)]*\)\s*$")
    bounds = []
    for index, line in enumerate(lines):
        value = line.strip()
        if "|" in value:
            continue
        if rider.match(value):
            bounds.append((index + 1, value))
        elif main_contract.match(value) and index + 1 > 200:
            bounds.append((index + 1, "주계약(" + value + ")"))
    bound_lines = [item[0] for item in bounds]
    def scope_at(line_no, fallback):
        index = bisect.bisect_right(bound_lines, line_no) - 1
        return bounds[index][1] if index >= 0 else fallback

    repaired, delta, parent_map = [], [], {}
    for element in elements:
        if element["element_id"] not in parents:
            updated = dict(element)
            updated["contract_scope"] = scope_at(element["line_start"], element.get("contract_scope", ""))
            repaired.append(updated)
            continue
        children = []
        # Split first on blank-line structural boundaries.
        for match in re.finditer(r"\S(?:[\s\S]*?\S)?(?=\n{2,}|$)", element["text"]):
            block = match.group(0)
            absolute_start = element["char_start"] + match.start()
            for start, end, text in split_structural(absolute_start, block):
                if not text.strip():
                    continue
                child_index = len(children)
                child = {
                    "element_id": f"{element['element_id']}__r{child_index:03d}",
                    "parent_element_id": element["element_id"],
                    "element_type": classify(text), "text": text,
                    "char_start": start, "char_end": end,
                    "line_start": line_of(start), "line_end": line_of(max(start, end - 1)),
                    "contract_scope": scope_at(line_of(start), element.get("contract_scope", "")),
                }
                children.append(child)
        if not children:
            raise RuntimeError(f"no children for {element['element_id']}")
        repaired.extend(children)
        delta.extend(children)
        parent_map[element["element_id"]] = [child["element_id"] for child in children]

    repaired.sort(key=lambda row: (row["char_start"], row["char_end"], row["element_id"]))
    for name, rows in (("elements_repaired_v1.jsonl", repaired), ("elements_repaired_delta_v1.jsonl", delta)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source_elements": len(elements), "repaired_parent_count": len(parents),
        "unchanged_count": len(elements) - len(parents), "child_count": len(delta),
        "result_elements": len(repaired), "parent_map": parent_map,
        "rules": {"max_chars": MAX_CHARS, "max_lines": MAX_LINES, "qa_accessed": False},
    }
    (OUT / "element_repair_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "parent_map"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
