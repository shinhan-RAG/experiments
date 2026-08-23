#!/usr/bin/env python3
"""Validate a structured retrieval-blind audit and emit an apply-ready ledger.

The validator deliberately knows nothing about retrieval results.  It checks batch
coverage, operation semantics, group indexes, JO membership and absolute spans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


from apply_gold_completeness_ledger import existing_group_keys, group_key

ALLOWED_OPS = {"add_members", "replace_group_members", "replace_span",
               "add_required_group"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_member(member, jo, label):
    required = {"jo", "c0", "c1", "evidence_role"}
    missing = required - set(member)
    if missing:
        raise ValueError(f"missing member fields {sorted(missing)}: {label}")
    jo_id = member["jo"]
    if jo_id not in jo:
        raise ValueError(f"unknown JO {jo_id}: {label}")
    c0, c1 = member["c0"], member["c1"]
    if not isinstance(c0, int) or not isinstance(c1, int):
        raise ValueError(f"non-integer span: {label}")
    unit = jo[jo_id]
    if not (unit["char_start"] <= c0 < c1 <= unit["char_end"]):
        raise ValueError(f"span outside JO {jo_id} {c0}:{c1}: {label}")
    if not str(member["evidence_role"]).strip():
        raise ValueError(f"empty evidence role: {label}")


def validate_decision(decision, gold, jo):
    qid = decision["qid"]
    if qid not in gold:
        raise ValueError(f"qid absent from Gold: {qid}")
    status = decision["decision"]
    ops = decision["ops"]
    if status not in {"pass", "fix", "exclude"}:
        raise ValueError(f"unknown decision {status}: {qid}")
    if status != "fix" and ops:
        raise ValueError(f"{status} decision must have no operations: {qid}")
    if status == "fix" and not ops:
        raise ValueError(f"fix decision must have operations: {qid}")

    # An excluded row is deliberately removed from the scoreable overlay.  Its
    # pre-existing Gold may be malformed (duplicate groups are one reason to
    # exclude it), so validating invariants of a final row that will never be
    # emitted would make a valid quarantine decision impossible.  Pass/fix
    # rows still have to satisfy every final-group invariant below.
    if status == "exclude":
        return

    final_group_members = [list(group["members"]) for group in gold[qid]["groups"]]
    final_group_keys = existing_group_keys(gold[qid]["groups"], jo)
    for op_index, op in enumerate(ops):
        label = f"{qid} op={op_index}"
        kind = op["op"]
        if kind not in ALLOWED_OPS:
            raise ValueError(f"unknown operation {kind}: {label}")
        if kind == "add_required_group":
            # A brand new required AND group: its members are the OR set. It has
            # no target group index, and it must not restate an existing group
            # (that claim belongs in the existing group as an OR member).
            members = op.get("members") or []
            if not members:
                raise ValueError(f"empty member operation: {label}")
            for member_index, member in enumerate(members):
                validate_member(member, jo, f"{label} member={member_index}")
            ids = [member["jo"] for member in members]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate JO in operation: {label}")
            key = group_key(members, jo)
            if key in final_group_keys:
                raise ValueError(f"duplicate required group key {key!r}: {label}")
            final_group_keys.add(key)
            final_group_members.append(list(members))
            continue
        group_index = op["group"]
        if not isinstance(group_index, int) or not 0 <= group_index < len(final_group_members):
            raise ValueError(f"invalid group {group_index}: {label}")
        if kind == "replace_span":
            member = {key: op[key] for key in ("jo", "c0", "c1", "evidence_role")}
            validate_member(member, jo, label)
            matches = [item for item in final_group_members[group_index]
                       if item["jo"] == op["jo"]]
            if len(matches) != 1:
                raise ValueError(f"replace_span target count={len(matches)}: {label}")
            matches[0].update(member)
        else:
            members = op["members"]
            if not members:
                raise ValueError(f"empty member operation: {label}")
            for member_index, member in enumerate(members):
                validate_member(member, jo, f"{label} member={member_index}")
            ids = [member["jo"] for member in members]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate JO in operation: {label}")
            if kind == "replace_group_members":
                final_group_members[group_index] = list(members)
            else:
                existing = {member["jo"] for member in final_group_members[group_index]}
                duplicate = existing.intersection(ids)
                if duplicate:
                    raise ValueError(f"duplicate added JO {sorted(duplicate)}: {label}")
                final_group_members[group_index].extend(members)

    # Independent claims may legitimately live in different spans of the same
    # JO (for example two different classification rows in one appendix card).
    # Only byte-identical member sets are duplicate required groups.
    signatures = [tuple(sorted((member["jo"], int(member["c0"]), int(member["c1"]))
                               for member in members))
                  for members in final_group_members]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"duplicate final required groups: {qid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--prior-audit")
    parser.add_argument("--qids", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audit = load_json(args.audit)
    qids = load_json(args.qids)
    gold_rows = load_jsonl(args.gold)
    jo_rows = load_jsonl(args.jo)
    gold = {row["qid"]: row for row in gold_rows}
    jo = {row["element_id"]: row for row in jo_rows}
    decisions = audit["decisions"]
    actual_qids = [decision["qid"] for decision in decisions]
    if len(actual_qids) != len(set(actual_qids)):
        raise ValueError("duplicate qid in audit")
    if actual_qids != qids:
        raise ValueError(f"audit qid/order mismatch: expected={qids} actual={actual_qids}")
    for decision in decisions:
        validate_decision(decision, gold, jo)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions),
        encoding="utf-8",
    )
    manifest = {
        "policy": "retrieval-blind audit; exact batch coverage; JO/span/group validation",
        "inputs": {
            "audit": {"path": str(Path(args.audit).resolve()), "sha256": sha256(args.audit)},
            **({"prior_audit": {"path": str(Path(args.prior_audit).resolve()),
                                 "sha256": sha256(args.prior_audit)}}
               if args.prior_audit else {}),
            "qids": {"path": str(Path(args.qids).resolve()), "sha256": sha256(args.qids)},
            "gold": {"path": str(Path(args.gold).resolve()), "sha256": sha256(args.gold)},
            "jo": {"path": str(Path(args.jo).resolve()), "sha256": sha256(args.jo)},
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "counts": {
            "qids": len(qids),
            **{kind: sum(decision["decision"] == kind for decision in decisions)
               for kind in ("pass", "fix", "exclude")},
        },
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"] | {"output_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
