#!/usr/bin/env python3
"""Apply explicit independent Gold PASS/FIX/EXCLUDE adjudications.

The ledger is authoritative and human/agent reviewed; this program does not
infer equivalence.  Replacement members must be contained by the declared JO.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qids", help="if supplied, require one adjudication per qid")
    args = parser.parse_args()

    rows = load_jsonl(args.gold)
    by_qid = {row["qid"]: row for row in rows}
    jo = {row["element_id"]: row for row in load_jsonl(args.jo)}
    decisions = load_jsonl(args.ledger)
    decision_by_qid = {row["qid"]: row for row in decisions}
    if len(decision_by_qid) != len(decisions):
        raise ValueError("duplicate qid in adjudication ledger")
    selected = (json.loads(Path(args.qids).read_text(encoding="utf-8"))
                if args.qids else [row["qid"] for row in rows])
    if args.qids:
        missing = sorted(set(selected) - set(decision_by_qid))
        extra = sorted(set(decision_by_qid) - set(selected))
        if missing or extra:
            raise ValueError(f"ledger coverage mismatch: missing={missing}, extra={extra}")

    output_rows, audit = [], []
    for qid in selected:
        source = by_qid[qid]
        has_decision = qid in decision_by_qid
        decision = decision_by_qid.get(qid, {"qid": qid, "decision": "pass"})
        kind = decision["decision"]
        if kind == "exclude":
            audit.append({"qid": qid, "decision": kind, "reason": decision["reason"]})
            continue
        row = dict(source)
        if kind == "replace_groups":
            groups = []
            for group_index, requested_group in enumerate(decision["groups"]):
                members = []
                for requested in requested_group["members"]:
                    unit = jo[requested["jo"]]
                    c0, c1 = int(requested["c0"]), int(requested["c1"])
                    if not (unit["char_start"] <= c0 < c1 <= unit["char_end"]):
                        raise ValueError(
                            f"span outside JO: {qid} {requested['jo']} {c0}:{c1}")
                    members.append({
                        "c0": c0, "c1": c1, "jo": requested["jo"],
                        "src": "independent_adjudication",
                        "evidence_role": requested.get("evidence_role", ""),
                    })
                if not members:
                    raise ValueError(f"empty replacement group: {qid} {group_index}")
                groups.append({
                    "key": requested_group.get("key", f"reviewed:{qid}:{group_index}"),
                    "members": members,
                    "c0": min(member["c0"] for member in members),
                    "c1": max(member["c1"] for member in members),
                    "policy": requested_group["policy"],
                })
            row["groups"] = groups
        elif kind != "pass":
            raise ValueError(f"unknown decision: {kind}")
        if has_decision or args.qids:
            row["status"] = "independently_reviewed"
            row["adjudication"] = {key: value for key, value in decision.items()
                                   if key not in {"groups"}}
        output_rows.append(row)
        if has_decision or args.qids:
            audit.append({"qid": qid, "decision": kind,
                          "groups": len(row.get("groups", []))})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "inputs": {
            "gold": {"path": str(Path(args.gold).resolve()), "sha256": sha256(args.gold)},
            "jo": {"path": str(Path(args.jo).resolve()), "sha256": sha256(args.jo)},
            "ledger": {"path": str(Path(args.ledger).resolve()), "sha256": sha256(args.ledger)},
            **({"qids": {"path": str(Path(args.qids).resolve()), "sha256": sha256(args.qids)}}
               if args.qids else {}),
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "python": sys.version,
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "n_input": len(selected), "n_output": len(output_rows),
        "audit": audit,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("output", "output_sha256", "n_input", "n_output")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
