#!/usr/bin/env python3
"""Map retrieval-blind official source spans to submit-compatible JO units.

The mapper uses coordinates only.  It never reads questions, Gold IDs, retrieval
results, or experiment outputs.  A source block contained by one JO becomes one
provisional group.  A block crossing JO boundaries is conservatively split into
required groups, one per overlapping JO; independent review must approve or
replace that provisional structure before scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_span(c0, c1, units):
    containing = [unit for unit in units
                  if int(unit["char_start"]) <= c0 < c1 <= int(unit["char_end"])]
    if containing:
        unit = min(containing,
                   key=lambda row: (int(row["char_end"]) - int(row["char_start"]),
                                    row["element_id"]))
        return [(unit, c0, c1, "contained")]
    overlaps = []
    for unit in units:
        left = max(c0, int(unit["char_start"]))
        right = min(c1, int(unit["char_end"]))
        if left < right:
            overlaps.append((unit, left, right, "cross_jo_overlap"))
    return overlaps


def convert_rows(source_rows, units):
    output = []
    for row in source_rows:
        groups, unmapped = [], list(row.get("unmapped") or [])
        for group_index, source_group in enumerate(row.get("groups") or []):
            c0, c1 = int(source_group["c0"]), int(source_group["c1"])
            mapped = map_span(c0, c1, units)
            if not mapped:
                unmapped.append({"c0": c0, "c1": c1, "reason": "no_jo_overlap"})
                continue
            for part_index, (unit, left, right, policy) in enumerate(mapped):
                groups.append({
                    "key": f"source:{row['qid']}:{group_index}:{part_index}",
                    "members": [{
                        "jo": unit["element_id"], "c0": left, "c1": right,
                        "src": "official_source_coordinate_map",
                        "evidence_role": "provisional_official_source_span",
                    }],
                    "c0": left, "c1": right,
                    "policy": policy,
                    "source_group": group_index,
                })
        status = ("provisional_source_to_jo" if groups and not unmapped
                  else "partial" if groups else "unmapped")
        output.append({
            "qid": row["qid"], "q": row.get("q", ""),
            "task_type": row.get("task_type", ""),
            "core_retrieval": row.get("core_retrieval", ""),
            "groups": groups, "unmapped": unmapped, "status": status,
            "rule": "official_source_coordinates;smallest_containing_jo;cross_jo_split_and;provisional_review_required",
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_rows = load_jsonl(args.source_gold)
    units = sorted(load_jsonl(args.jo),
                   key=lambda row: (int(row["char_start"]), int(row["char_end"]),
                                    row["element_id"]))
    rows = convert_rows(source_rows, units)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    manifest = {
        "policy": "coordinate-only provisional source Gold to JO mapping",
        "retrieval_blind": True,
        "inputs": {
            "source_gold": {"path": str(Path(args.source_gold).resolve()),
                            "sha256": sha256(args.source_gold)},
            "jo": {"path": str(Path(args.jo).resolve()), "sha256": sha256(args.jo)},
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "python": sys.version,
        "counts": {
            "input": len(source_rows), "output": len(rows),
            "provisional": sum(row["status"] == "provisional_source_to_jo" for row in rows),
            "partial": sum(row["status"] == "partial" for row in rows),
            "unmapped": sum(row["status"] == "unmapped" for row in rows),
        },
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"] | {"output_sha256": sha256(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
