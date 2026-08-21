#!/usr/bin/env python3
"""Apply retrieval-blind Gold completeness/span review operations.

The ledger contains only explicit independent decisions. This program performs no
similarity inference and validates every requested span against the declared JO.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_member(requested, jo, qid):
    unit = jo[requested["jo"]]
    c0, c1 = int(requested["c0"]), int(requested["c1"])
    if not (unit["char_start"] <= c0 < c1 <= unit["char_end"]):
        raise ValueError(f"span outside JO: {qid} {requested['jo']} {c0}:{c1}")
    return {
        "c0": c0,
        "c1": c1,
        "jo": requested["jo"],
        "src": "retrieval_blind_completeness_review",
        "evidence_role": requested.get("evidence_role", "reviewed_direct_evidence"),
    }


def refresh_group(group):
    if not group.get("members"):
        raise ValueError("empty Gold group")
    group["c0"] = min(member["c0"] for member in group["members"])
    group["c1"] = max(member["c1"] for member in group["members"])


def validate_output_rows(output_rows, jo, expected_qids):
    actual_qids = [row["qid"] for row in output_rows]
    if len(actual_qids) != len(set(actual_qids)):
        raise ValueError("duplicate qid in output")
    if actual_qids != expected_qids:
        raise ValueError(
            f"output qid/order mismatch: expected={expected_qids} actual={actual_qids}")
    for row in output_rows:
        qid = row["qid"]
        if not row.get("groups"):
            raise ValueError(f"no Gold groups: {qid}")
        for group_index, group in enumerate(row["groups"]):
            members = group.get("members", [])
            if not members:
                raise ValueError(f"empty Gold group: {qid} group={group_index}")
            seen = set()
            for member in members:
                jo_id = member["jo"]
                if jo_id in seen:
                    raise ValueError(
                        f"duplicate JO member: {qid} group={group_index} jo={jo_id}")
                seen.add(jo_id)
                if jo_id not in jo:
                    raise ValueError(f"unknown JO: {qid} group={group_index} jo={jo_id}")
                unit = jo[jo_id]
                c0, c1 = int(member["c0"]), int(member["c1"])
                if not (unit["char_start"] <= c0 < c1 <= unit["char_end"]):
                    raise ValueError(
                        f"span outside JO: {qid} group={group_index} "
                        f"{jo_id} {c0}:{c1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--qids", help="optional exact review subset; omitted applies sparse ledger to all Gold rows")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = {row["qid"]: row for row in load_jsonl(args.gold)}
    jo = {row["element_id"]: row for row in load_jsonl(args.jo)}
    ledger = load_jsonl(args.ledger)
    decisions = {row["qid"]: row for row in ledger}
    qids = (json.loads(Path(args.qids).read_text(encoding="utf-8"))
            if args.qids else [row["qid"] for row in load_jsonl(args.gold)])
    if len(decisions) != len(ledger):
        raise ValueError("duplicate qid in ledger")
    if args.qids and set(decisions) != set(qids):
        raise ValueError(
            f"ledger coverage mismatch: missing={sorted(set(qids)-set(decisions))} "
            f"extra={sorted(set(decisions)-set(qids))}")
    if not set(decisions) <= set(rows):
        raise ValueError(f"ledger qids absent from Gold: {sorted(set(decisions)-set(rows))}")

    output_rows, audit = [], []
    for qid in qids:
        if qid not in decisions:
            output_rows.append(copy.deepcopy(rows[qid]))
            continue
        decision = decisions[qid]
        status = decision["decision"]
        if status == "exclude":
            audit.append({"qid": qid, "decision": status,
                          "reason": decision["reason"]})
            continue
        row = copy.deepcopy(rows[qid])
        if status == "fix":
            for op in decision["ops"]:
                kind = op["op"]
                group = row["groups"][int(op["group"])]
                if kind == "add_members":
                    existing = {member["jo"] for member in group["members"]}
                    additions = [checked_member(item, jo, qid) for item in op["members"]]
                    duplicate = existing.intersection(member["jo"] for member in additions)
                    if duplicate:
                        raise ValueError(f"duplicate JO member: {qid} {sorted(duplicate)}")
                    group["members"].extend(additions)
                    refresh_group(group)
                elif kind == "replace_group_members":
                    group["members"] = [checked_member(item, jo, qid)
                                        for item in op["members"]]
                    if len({member["jo"] for member in group["members"]}) != len(group["members"]):
                        raise ValueError(f"duplicate replacement JO: {qid}")
                    refresh_group(group)
                elif kind == "replace_span":
                    matches = [member for member in group["members"] if member["jo"] == op["jo"]]
                    if len(matches) != 1:
                        raise ValueError(f"replace_span target count: {qid} {op['jo']} {len(matches)}")
                    replacement = checked_member(op, jo, qid)
                    matches[0].update(replacement)
                    refresh_group(group)
                else:
                    raise ValueError(f"unknown operation: {kind}")
            row["status"] = "retrieval_blind_completeness_reviewed"
            row["completeness_review"] = {
                "decision": "fix", "review": decision.get("review", "independent_sol_xhigh"),
                "reason": decision["reason"], "operation_count": len(decision["ops"]),
            }
        elif status != "pass":
            raise ValueError(f"unknown decision: {status}")
        else:
            row["status"] = "retrieval_blind_completeness_reviewed"
            row["completeness_review"] = {
                "decision": "pass", "review": decision.get("review", "independent_sol_xhigh"),
            }
        output_rows.append(row)
        audit.append({"qid": qid, "decision": status,
                      "groups": len(row["groups"]),
                      "members": sum(len(group["members"]) for group in row["groups"])})

    expected_output_qids = [qid for qid in qids
                            if qid not in decisions or decisions[qid]["decision"] != "exclude"]
    validate_output_rows(output_rows, jo, expected_output_qids)

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
        "n_input": len(qids), "n_output": len(output_rows),
        "counts": {kind: sum(1 for item in audit if item["decision"] == kind)
                   for kind in ("pass", "fix", "exclude")},
        "audit": audit,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("output", "output_sha256", "n_input", "n_output", "counts")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
