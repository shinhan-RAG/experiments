#!/usr/bin/env python3
"""Overlay a retrieval-blind reviewed subset onto the scoped train Gold."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def unique_by_qid(rows, label):
    result = {row["qid"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate qid in {label}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--reviewed-qids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-qids", required=True)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    reviewed_rows = load_jsonl(args.reviewed)
    reviewed_scope = load_json(args.reviewed_qids)
    base = unique_by_qid(base_rows, "base")
    reviewed = unique_by_qid(reviewed_rows, "reviewed")
    if len(set(reviewed_scope)) != len(reviewed_scope):
        raise ValueError("duplicate qid in reviewed scope")
    missing_from_base = sorted(set(reviewed_scope) - set(base))
    if missing_from_base:
        raise ValueError(f"reviewed scope not present in base: {missing_from_base}")
    outside_scope = sorted(set(reviewed) - set(reviewed_scope))
    if outside_scope:
        raise ValueError(f"reviewed rows outside frozen scope: {outside_scope}")

    output_rows = []
    replaced = []
    excluded = []
    for row in base_rows:
        qid = row["qid"]
        if qid not in reviewed_scope:
            output_rows.append(row)
        elif qid in reviewed:
            output_rows.append(reviewed[qid])
            replaced.append(qid)
        else:
            excluded.append(qid)

    zero_group_unscorable = [row["qid"] for row in output_rows if not row.get("groups")]
    output_rows = [row for row in output_rows if row.get("groups")]
    completeness_reviewed = [
        row["qid"] for row in output_rows
        if row.get("status") == "retrieval_blind_completeness_reviewed"
        or row.get("completeness_review")
    ]

    output = Path(args.output)
    output_qids = Path(args.output_qids)
    write_jsonl(output, output_rows)
    output_qids.write_text(
        json.dumps([row["qid"] for row in output_rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy": "preserve scoped-train order; replace reviewed-scope rows; drop reviewed exclusions",
        "quality_scope": {
            "retrieval_blind_reviewed_total": len(completeness_reviewed),
            "newly_replaced_in_this_overlay": len(replaced),
            "scoped_not_completeness_reviewed": len(output_rows) - len(completeness_reviewed),
            "excluded_after_review": excluded,
            "zero_group_unscorable": zero_group_unscorable,
        },
        "inputs": {
            "base": {"path": str(Path(args.base).resolve()), "sha256": sha256(args.base)},
            "reviewed": {"path": str(Path(args.reviewed).resolve()), "sha256": sha256(args.reviewed)},
            "reviewed_qids": {
                "path": str(Path(args.reviewed_qids).resolve()),
                "sha256": sha256(args.reviewed_qids),
            },
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "python": platform.python_version(),
        "counts": {
            "base": len(base_rows),
            "reviewed_scope": len(reviewed_scope),
            "replaced": len(replaced),
            "excluded": len(excluded),
            "zero_group_unscorable": len(zero_group_unscorable),
            "output": len(output_rows),
        },
        "outputs": {
            "gold": {"path": str(output.resolve()), "sha256": sha256(output)},
            "qids": {"path": str(output_qids.resolve()), "sha256": sha256(output_qids)},
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
