#!/usr/bin/env python3
"""범용 Semantic Tag 적재 전 감사.

gold 없이도 문서 유형/태거별 key 이식성 문제를 찾는다. 성능 합격을 주장하지 않으며,
성능은 별도 QA로 평가해야 한다.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from schema_adapter import AXES, adapt_tag


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(here / "out/elements_u2.jsonl"))
    ap.add_argument("--tags", default=str(here / "out/tags_u2_rules.jsonl"))
    ap.add_argument("--out", default=str(here / "out/structured_schema_audit.json"))
    a = ap.parse_args()

    elements = [json.loads(line) for line in open(a.elements, encoding="utf-8")]
    tags = [json.loads(line) for line in open(a.tags, encoding="utf-8")]
    element_ids = [row.get("element_id") for row in elements]
    tag_by_id = {row.get("element_id"): row for row in tags}
    missing = [eid for eid in element_ids if eid not in tag_by_id]
    extra_ids = sorted(set(tag_by_id) - set(element_ids))

    coverage = collections.Counter()
    distinct = {axis: set() for axis in AXES}
    recognized = collections.Counter()
    unknown = collections.Counter()
    lengths = []
    rows_with_unknown = 0
    for element in elements:
        tag = tag_by_id.get(element.get("element_id"), {})
        row = adapt_tag(tag, element)
        rows_with_unknown += bool(row["_unknown_paths"])
        recognized.update(row["_recognized_paths"])
        unknown.update(row["_unknown_paths"])
        lengths.append(sum(len(value) for axis in AXES for value in row[axis]))
        for axis in AXES:
            if row[axis]:
                coverage[axis] += 1
                distinct[axis].update(row[axis])

    n = len(elements)
    report = {
        "elements": n,
        "tags": len(tags),
        "alignment": {"missing_tag_ids": len(missing), "extra_tag_ids": len(extra_ids),
                      "missing_examples": missing[:10], "extra_examples": extra_ids[:10]},
        "axis_coverage": {axis: round(coverage[axis] / max(1, n), 4) for axis in AXES},
        "axis_distinct": {axis: len(distinct[axis]) for axis in AXES},
        "unknown": {"row_rate": round(rows_with_unknown / max(1, n), 4),
                    "paths": unknown.most_common(30)},
        "recognized_paths": recognized.most_common(30),
        "tag_chars": {"median": int(statistics.median(lengths)) if lengths else 0,
                      "p95": percentile(lengths, .95), "p99": percentile(lengths, .99),
                      "max": max(lengths, default=0)},
        "gates": {
            "id_alignment": not missing and not extra_ids,
            "unknown_row_rate_le_5pct": rows_with_unknown / max(1, n) <= .05,
            "p99_chars_le_512": percentile(lengths, .99) <= 512,
        },
        "note": "coverage는 이식성/품질 진단이며 검색 성능 근거가 아니다. 문서유형별 QA 확정이 별도로 필요하다.",
    }
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
