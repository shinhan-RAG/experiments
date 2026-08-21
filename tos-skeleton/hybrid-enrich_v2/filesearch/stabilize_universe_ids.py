#!/usr/bin/env python3
"""Preserve opaque IDs across schema-only universe rebuilds.

Rows with identical source spans and text reuse the reference universe ID.
New rows receive a deterministic content-derived ID in a disjoint namespace.
No searchable field or ordering is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(row):
    return (int(row["char_start"]), int(row["char_end"]), row.get("text", ""))


def novel_id(prefix, key):
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_id_map(reference, rebuilt, novel_prefix):
    reference_by_key = {row_key(row): row["element_id"] for row in reference}
    if len(reference_by_key) != len(reference):
        raise ValueError("reference contains duplicate source/text keys")
    mapping = {}
    used = set(reference_by_key.values())
    reused = 0
    for row in rebuilt:
        old = row["element_id"]
        key = row_key(row)
        new = reference_by_key.get(key)
        if new is None:
            new = novel_id(novel_prefix, key)
            if new in used:
                raise ValueError(f"content ID collision: {new}")
        else:
            reused += 1
        if old in mapping:
            raise ValueError(f"duplicate rebuilt ID: {old}")
        mapping[old] = new
        used.add(new)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("stable IDs are not unique")
    return mapping, reused


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-elements", required=True)
    parser.add_argument("--reference-jo", required=True)
    parser.add_argument("--rebuilt-elements", required=True)
    parser.add_argument("--rebuilt-tags", required=True)
    parser.add_argument("--rebuilt-jo", required=True)
    parser.add_argument("--output-elements", required=True)
    parser.add_argument("--output-tags", required=True)
    parser.add_argument("--output-jo", required=True)
    args = parser.parse_args()

    ref_elements = load_jsonl(args.reference_elements)
    ref_jo = load_jsonl(args.reference_jo)
    elements = load_jsonl(args.rebuilt_elements)
    tags = load_jsonl(args.rebuilt_tags)
    jo = load_jsonl(args.rebuilt_jo)
    element_map, reused_elements = make_id_map(ref_elements, elements, "e3n")
    jo_map, reused_jo = make_id_map(ref_jo, jo, "j3n")

    stable_elements = []
    for row in elements:
        out = dict(row)
        out["element_id"] = element_map[row["element_id"]]
        stable_elements.append(out)
    stable_tags = []
    for row in tags:
        out = dict(row)
        out["element_id"] = element_map[row["element_id"]]
        stable_tags.append(out)
    stable_jo = []
    for row in jo:
        out = dict(row)
        out["element_id"] = jo_map[row["element_id"]]
        out["members"] = [element_map[item] for item in row["members"]]
        stable_jo.append(out)

    element_ids = {row["element_id"] for row in stable_elements}
    if {row["element_id"] for row in stable_tags} != element_ids:
        raise ValueError("stable tags do not cover the stable element universe exactly")
    member_ids = {item for row in stable_jo for item in row["members"]}
    if not member_ids <= element_ids:
        raise ValueError("stable JO contains unknown member IDs")

    write_jsonl(args.output_elements, stable_elements)
    write_jsonl(args.output_tags, stable_tags)
    write_jsonl(args.output_jo, stable_jo)
    manifest = {
        "algorithm": "exact (char_start,char_end,text) reuse; SHA-256 ID for novel rows",
        "searchable_fields_changed": False,
        "row_order_changed": False,
        "reference": {
            "elements": {"path": str(Path(args.reference_elements).resolve()), "sha256": sha256(args.reference_elements)},
            "jo": {"path": str(Path(args.reference_jo).resolve()), "sha256": sha256(args.reference_jo)},
        },
        "rebuilt": {
            "elements": {"path": str(Path(args.rebuilt_elements).resolve()), "sha256": sha256(args.rebuilt_elements)},
            "tags": {"path": str(Path(args.rebuilt_tags).resolve()), "sha256": sha256(args.rebuilt_tags)},
            "jo": {"path": str(Path(args.rebuilt_jo).resolve()), "sha256": sha256(args.rebuilt_jo)},
        },
        "counts": {
            "elements": len(elements), "elements_reused": reused_elements,
            "elements_novel": len(elements) - reused_elements,
            "jo": len(jo), "jo_reused": reused_jo, "jo_novel": len(jo) - reused_jo,
        },
        "outputs": {},
    }
    for key, path in (("elements", args.output_elements), ("tags", args.output_tags), ("jo", args.output_jo)):
        manifest["outputs"][key] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    manifest_path = Path(args.output_elements).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
