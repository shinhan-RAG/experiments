#!/usr/bin/env python3
"""Apply only independently reviewed Gold OR-members from an immutable ledger."""
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    ledger = load_jsonl(args.ledger)
    jo_rows = {row["element_id"]: row for row in load_jsonl(args.jo)}
    gold_by_qid = {row["qid"]: row for row in gold}
    if len(gold_by_qid) != len(gold):
        raise ValueError("duplicate qid in Gold")

    audit = []
    for repair in ledger:
        if repair.get("decision") != "approved_or":
            continue
        row = gold_by_qid[repair["qid"]]
        group_index = int(repair["group"])
        group = row["groups"][group_index]
        existing = {(m.get("c0"), m.get("c1")) for m in group.get("members", [])}
        added = []
        for requested in repair["members"]:
            jo = jo_rows[requested["jo"]]
            for key, source_key in (("c0", "char_start"), ("c1", "char_end"),
                                    ("contract_scope", "contract_scope"), ("title", "title")):
                if requested[key] != jo[source_key]:
                    raise ValueError(f"ledger/source mismatch: {repair['qid']} {requested['jo']} {key}")
            span = (requested["c0"], requested["c1"])
            if span in existing:
                raise ValueError(f"duplicate member span: {repair['qid']} {span}")
            member = {
                "c0": requested["c0"], "c1": requested["c1"],
                "src": "reviewed_variant_equivalent", "jo": requested["jo"],
                "policy": repair["policy"], "evidence_role": requested["evidence_role"],
            }
            group.setdefault("members", []).append(member)
            existing.add(span)
            added.append(requested["jo"])
        row.setdefault("reviewed_gold_repairs", []).append({
            "group": group_index, "policy": repair["policy"], "members": added,
        })
        audit.append({"qid": repair["qid"], "group": group_index, "members": added})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in gold:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    command = [
        "python3", str(Path(__file__).resolve()), "--gold", str(Path(args.gold).resolve()),
        "--jo", str(Path(args.jo).resolve()), "--ledger", str(Path(args.ledger).resolve()),
        "--output", str(output.resolve()),
    ]
    manifest = {
        "input_gold": str(Path(args.gold).resolve()), "input_gold_sha256": sha256(args.gold),
        "jo": str(Path(args.jo).resolve()), "jo_sha256": sha256(args.jo),
        "ledger": str(Path(args.ledger).resolve()), "ledger_sha256": sha256(args.ledger),
        "generator": str(Path(__file__).resolve()), "generator_sha256": source_sha(),
        "command": command,
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "repair_count": len(audit), "added_members": sum(len(x["members"]) for x in audit),
        "audit": audit,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
