#!/usr/bin/env python3
"""Freeze a reviewed holdout without consulting retrieval or agent results."""
import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-qids", required=True)
    parser.add_argument("--reserve-qids", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--output-qids", required=True)
    parser.add_argument("--output-ledger", required=True)
    args = parser.parse_args()

    primary = load_json(args.primary_qids)
    reserve = load_json(args.reserve_qids)
    ledger = load_jsonl(args.ledger)
    by_qid = {row["qid"]: row for row in ledger}
    if len(by_qid) != len(ledger):
        raise ValueError("duplicate qid in ledger")
    missing = sorted((set(primary) | set(reserve)) - set(by_qid))
    if missing:
        raise ValueError(f"unreviewed qids: {missing}")

    accepted_primary = [qid for qid in primary if by_qid[qid]["decision"] != "exclude"]
    needed = args.size - len(accepted_primary)
    accepted_reserve = [qid for qid in reserve if by_qid[qid]["decision"] != "exclude"]
    if needed < 0 or len(accepted_reserve) < needed:
        raise ValueError(
            f"cannot reach size={args.size}: primary={len(accepted_primary)}, "
            f"reserve={len(accepted_reserve)}"
        )
    selected = accepted_primary + accepted_reserve[:needed]
    selected_ledger = [by_qid[qid] for qid in selected]

    Path(args.output_qids).write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_ledger, selected_ledger)
    manifest = {
        "policy": "all non-excluded primary qids, then first non-excluded reserve qids in frozen input order",
        "retrieval_blind": True,
        "target_size": args.size,
        "accepted_primary": len(accepted_primary),
        "reserve_needed": needed,
        "selected_reserve": accepted_reserve[:needed],
        "inputs": {
            "primary_qids": {"path": str(Path(args.primary_qids).resolve()), "sha256": sha256(args.primary_qids)},
            "reserve_qids": {"path": str(Path(args.reserve_qids).resolve()), "sha256": sha256(args.reserve_qids)},
            "ledger": {"path": str(Path(args.ledger).resolve()), "sha256": sha256(args.ledger)},
        },
        "outputs": {
            "qids": {"path": str(Path(args.output_qids).resolve()), "sha256": sha256(args.output_qids)},
            "ledger": {"path": str(Path(args.output_ledger).resolve()), "sha256": sha256(args.output_ledger)},
        },
    }
    manifest_path = Path(args.output_qids).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
