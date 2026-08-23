#!/usr/bin/env python3
"""Adjudicate two retrieval-blind Gold audits with an explicit resolution file.

The program does not inspect retrieval output.  It validates both independent
audits against the same frozen Gold/JO inputs, requires an explicit source choice
for every qid, and emits the selected apply-ready ledger plus a hash manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from finalize_gold_audit_batch import load_json, load_jsonl, validate_decision


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed_decisions(audit, qids, label):
    decisions = audit["decisions"]
    actual = [item["qid"] for item in decisions]
    if actual != qids:
        raise ValueError(f"{label} qid/order mismatch: expected={qids} actual={actual}")
    if len(actual) != len(set(actual)):
        raise ValueError(f"duplicate qid in {label}")
    return {item["qid"]: item for item in decisions}


def indexed_decisions_relaxed(audit, qids, label):
    """Index a structurally invalid audit without making it selectable.

    Cross-review sometimes exists precisely because the first model omitted or
    duplicated a qid. Preserve that failure in the manifest instead of losing
    the otherwise valid review. A placeholder keeps comparison rows complete;
    the attached error prevents the invalid source from being selected.
    """
    decisions = audit.get("decisions", [])
    indexed = {}
    errors = {}
    expected = set(qids)
    for position, item in enumerate(decisions):
        qid = item.get("qid") if isinstance(item, dict) else None
        if qid not in expected:
            errors[f"<extra:{position}>"] = f"{label} unexpected qid: {qid!r}"
            continue
        if qid in indexed:
            errors[qid] = f"duplicate qid in {label}: {qid}"
            continue
        indexed[qid] = item
    for qid in qids:
        if qid not in indexed:
            errors[qid] = f"missing qid in {label}: {qid}"
            indexed[qid] = {"qid": qid, "decision": "invalid", "ops": []}
    return indexed, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-audit", required=True)
    parser.add_argument("--review-audit", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--qids", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--jo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    qids = load_json(args.qids)
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate qid in frozen batch")
    gold = {row["qid"]: row for row in load_jsonl(args.gold)}
    jo = {row["element_id"]: row for row in load_jsonl(args.jo)}
    primary, primary_structure_errors = indexed_decisions_relaxed(
        load_json(args.primary_audit), qids, "primary audit")
    review, review_structure_errors = indexed_decisions_relaxed(
        load_json(args.review_audit), qids, "review audit")
    validation_errors = {
        "primary": dict(primary_structure_errors),
        "review": dict(review_structure_errors),
    }
    for source, decisions in (("primary", primary), ("review", review)):
        for decision in decisions.values():
            try:
                validate_decision(decision, gold, jo)
            except (KeyError, TypeError, ValueError) as exc:
                validation_errors[source][decision.get("qid", "<missing>")] = str(exc)

    resolution = load_json(args.resolution)
    rows = resolution["resolutions"]
    actual = [item["qid"] for item in rows]
    if actual != qids:
        raise ValueError(f"resolution qid/order mismatch: expected={qids} actual={actual}")
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate qid in resolution")

    selected = []
    comparison = []
    for item in rows:
        qid = item["qid"]
        source = item["selected_source"]
        if source not in {"primary", "review"}:
            raise ValueError(f"invalid selected_source for {qid}: {source}")
        if qid in validation_errors[source]:
            raise ValueError(
                f"selected {source} decision is invalid for {qid}: "
                f"{validation_errors[source][qid]}")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"empty adjudication reason: {qid}")
        choice = primary[qid] if source == "primary" else review[qid]
        selected.append(choice)
        comparison.append({
            "qid": qid,
            "primary_decision": primary[qid]["decision"],
            "review_decision": review[qid]["decision"],
            "decision_agreement": primary[qid]["decision"] == review[qid]["decision"],
            "selected_source": source,
            "selected_decision": choice["decision"],
            "primary_validation_error": validation_errors["primary"].get(qid, ""),
            "review_validation_error": validation_errors["review"].get(qid, ""),
            "reason": reason,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    manifest = {
        "policy": resolution.get("policy", "explicit retrieval-blind cross-audit adjudication"),
        "inputs": {
            "primary_audit": {"path": str(Path(args.primary_audit).resolve()),
                              "sha256": sha256(args.primary_audit)},
            "review_audit": {"path": str(Path(args.review_audit).resolve()),
                             "sha256": sha256(args.review_audit)},
            "resolution": {"path": str(Path(args.resolution).resolve()),
                           "sha256": sha256(args.resolution)},
            "qids": {"path": str(Path(args.qids).resolve()), "sha256": sha256(args.qids)},
            "gold": {"path": str(Path(args.gold).resolve()), "sha256": sha256(args.gold)},
            "jo": {"path": str(Path(args.jo).resolve()), "sha256": sha256(args.jo)},
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "counts": {
            "qids": len(qids),
            "decision_agreements": sum(item["decision_agreement"] for item in comparison),
            "decision_disagreements": sum(not item["decision_agreement"] for item in comparison),
            "primary_validation_errors": len(validation_errors["primary"]),
            "review_validation_errors": len(validation_errors["review"]),
            **{kind: sum(item["decision"] == kind for item in selected)
               for kind in ("pass", "fix", "exclude")},
        },
        "comparison": comparison,
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"] | {"output_sha256": sha256(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
