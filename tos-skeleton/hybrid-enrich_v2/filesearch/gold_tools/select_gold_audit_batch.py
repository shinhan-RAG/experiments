#!/usr/bin/env python3
"""Select a deterministic, retrieval-blind audit batch from unreviewed Gold rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_reviewed(row):
    return (row.get("status") == "retrieval_blind_completeness_reviewed"
            or bool(row.get("completeness_review")))


def question_gate(question, gate="all"):
    """Apply a pre-declared question-only gate without retrieval observations."""
    if gate == "all":
        return True
    if gate != "classification_membership":
        raise ValueError(f"unknown question gate: {gate}")
    text = str(question)
    if re.search(r"분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드|분류표|부표|별첨",
                 text):
        return True
    if not re.search(r"(?:포함|해당)\s*(?:되|되는지|되나|되나요)", text):
        return False
    return not bool(re.search(
        r"(?:치료|수술|입원|통원|특약)\s*(?:이|가|은|는|도|에|에는)?\s*"
        r"(?:포함|해당)", text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-gate", default="all",
                        choices=("all", "classification_membership"))
    parser.add_argument("--exclude-qid", action="append", default=[])
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")

    rows = load_jsonl(args.gold)
    excluded = set(args.exclude_qid)
    unreviewed = [row for row in rows if not is_reviewed(row)]
    eligible = [row for row in unreviewed
                if row["qid"] not in excluded and question_gate(row.get("q", ""), args.question_gate)]
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{args.seed}|{row['qid']}".encode("utf-8")).hexdigest(),
    )
    selected = [row["qid"] for row in ranked[:args.size]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "policy": "sha256(seed|qid) ordering over retrieval-blind unreviewed Gold rows after a question-only gate",
        "seed": args.seed,
        "question_gate": args.question_gate,
        "excluded_qids": sorted(excluded),
        "inputs": {"gold": {"path": str(Path(args.gold).resolve()), "sha256": sha256(args.gold)}},
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "counts": {"gold": len(rows), "unreviewed": len(unreviewed),
                   "eligible": len(eligible), "selected": len(selected)},
        "selected_task_types": {
            task_type: sum(row.get("task_type") == task_type for row in ranked[:args.size])
            for task_type in sorted({row.get("task_type", "") for row in ranked[:args.size]})
        },
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"] | {"task_types": manifest["selected_task_types"],
                                          "output_sha256": sha256(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
