#!/usr/bin/env python3
"""Produce reproducible slice, exposure, and paired diagnostics for the 281-qid run."""
from __future__ import annotations

import collections
import json
import random
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
RUN = HERE / "out" / "c29_dual_tool_train281_luna_medium_w4"
PRIOR = HERE.parent / "0819" / "out" / "host_agent" / "gpt56luna_medium_c29_evaluabletrain281x1_overlayv1_20260821" / "results_itt.jsonl"
sys.path.insert(0, str(FS))
from scoring import overlaps
from units import Units


METRICS = ("R@1", "R@5", "R@10", "R@20", "suff@5", "suff@10", "RR@10")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        **{key: mean(rows, key) for key in METRICS},
        "R@5_full": sum(row["R@5"] == 1 for row in rows),
        "R@5_partial": sum(0 < row["R@5"] < 1 for row in rows),
        "R@5_zero": sum(row["R@5"] == 0 for row in rows),
        "mean_submitted": sum(len(row["submitted"]) for row in rows) / len(rows) if rows else 0.0,
        "empty_submissions": sum(not row["submitted"] for row in rows),
    }


def grouped(rows: list[dict], key_fn) -> dict:
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def exposure(units: Units, ids: list[str], groups: list[dict]) -> tuple[float, bool]:
    candidates = units.resolve(ids)
    hits = [any(overlaps(candidate, group) for candidate in candidates) for group in groups]
    return sum(hits) / len(groups), all(hits)


def ordered_union(*lists: list[str]) -> list[str]:
    seen = set()
    result = []
    for values in lists:
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def bootstrap_ci(values: list[float], draws: int = 10_000) -> list[float]:
    rng = random.Random(20260821)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [estimates[int(draws * 0.025)], estimates[int(draws * 0.975)]]


def main() -> None:
    results = jsonl(RUN / "results.jsonl")
    gold = {row["qid"]: row for row in jsonl(FS / "out" / "gold_train_scoped_u3_reviewed_overlay_v1.jsonl")}
    reviewed = {row["qid"] for row in jsonl(FS / "out" / "gold_c17_reviewed_final_s29_v7.jsonl")}
    units = Units(FS / "out" / "elements_u3jo.jsonl")
    if len(results) != 281 or len({row["qid"] for row in results}) != 281:
        raise SystemExit("expected exactly 281 unique result rows")
    for row in results:
        row.update(
            task_type=gold[row["qid"]]["task_type"],
            status=gold[row["qid"]]["status"],
            groups_n=len(gold[row["qid"]]["groups"]),
            core=str(gold[row["qid"]].get("core_retrieval")),
            reviewed=row["qid"] in reviewed,
        )

    channel_rows = []
    tool_counts = collections.Counter()
    all_timestamps = []
    for row in results:
        calls_path = RUN / "sessions" / f"{row['qid']}_r{row['rep']}" / "calls.jsonl"
        calls = jsonl(calls_path)
        tool_counts.update(call.get("cmd", "unknown") for call in calls)
        all_timestamps.extend(call["t"] for call in calls if isinstance(call.get("t"), (int, float)))
        semantic_calls = [[item[0] for item in call.get("returned", [])] for call in calls if call.get("cmd") == "search"]
        meta_calls = [[item[0] for item in call.get("returned", [])] for call in calls if call.get("cmd") == "msearch"]
        groups = gold[row["qid"]]["groups"]
        first_sem = exposure(units, semantic_calls[0], groups)
        first_meta = exposure(units, meta_calls[0], groups)
        first_union = exposure(units, ordered_union(semantic_calls[0], meta_calls[0]), groups)
        all_sem = ordered_union(*semantic_calls)
        all_meta = ordered_union(*meta_calls)
        all_union = exposure(units, ordered_union(all_sem, all_meta), groups)
        channel_rows.append({
            "qid": row["qid"],
            "first_semantic_recall": first_sem[0], "first_semantic_sufficient": first_sem[1],
            "first_meta_recall": first_meta[0], "first_meta_sufficient": first_meta[1],
            "first_union_recall": first_union[0], "first_union_sufficient": first_union[1],
            "all_union_recall": all_union[0], "all_union_sufficient": all_union[1],
            "submitted_sufficient_10": bool(row["suff@10"]),
        })

    channel_summary = {}
    for prefix in ("first_semantic", "first_meta", "first_union", "all_union"):
        channel_summary[prefix] = {
            "mean_fractional_recall": mean(channel_rows, f"{prefix}_recall"),
            "zero_exposure_qids": sum(row[f"{prefix}_recall"] == 0 for row in channel_rows),
            "sufficient_qids": sum(row[f"{prefix}_sufficient"] for row in channel_rows),
            "sufficient_rate": sum(row[f"{prefix}_sufficient"] for row in channel_rows) / len(channel_rows),
        }
    channel_summary["agent_selection"] = {
        "candidate_sufficient_but_submit_insufficient_qids": sum(
            row["all_union_sufficient"] and not row["submitted_sufficient_10"] for row in channel_rows
        ),
        "candidate_insufficient_qids": sum(not row["all_union_sufficient"] for row in channel_rows),
    }

    prior_rows = {row["qid"]: row for row in jsonl(PRIOR)}
    paired = {}
    for key in ("R@1", "R@5", "R@10", "suff@5", "suff@10"):
        diffs = [row[key] - prior_rows[row["qid"]][key] for row in results]
        paired[key] = {
            "current": mean(results, key),
            "prior": sum(prior_rows[row["qid"]][key] for row in results) / len(results),
            "delta": statistics.mean(diffs),
            "bootstrap_95ci_delta": bootstrap_ci(diffs),
            "wins": sum(value > 0 for value in diffs),
            "losses": sum(value < 0 for value in diffs),
            "ties": sum(value == 0 for value in diffs),
        }

    analysis = {
        "overall": summarize(results),
        "bootstrap_95ci": {key: bootstrap_ci([row[key] for row in results]) for key in ("R@5", "R@10", "suff@5", "suff@10")},
        "quality_slice": grouped(results, lambda row: "reviewed_29" if row["reviewed"] else "scoped_unreviewed_252"),
        "task_type": grouped(results, lambda row: row["task_type"]),
        "gold_groups": grouped(results, lambda row: "1" if row["groups_n"] == 1 else "2+"),
        "core_retrieval": grouped(results, lambda row: row["core"]),
        "status": grouped(results, lambda row: row["status"]),
        "channels": channel_summary,
        "tool_counts": dict(tool_counts),
        "wall_clock_from_tool_timestamps_seconds": max(all_timestamps) - min(all_timestamps),
        "paired_vs_prior_host_c29": paired,
        "retry": {
            "qid": "v3-offline-0465",
            "reason": "blank line in concurrent calls.jsonl caused submit JSON parse failure",
            "clean_retry_errors": next(row["error"] for row in results if row["qid"] == "v3-offline-0465"),
            "failed_attempt_preserved": str(RUN / "sessions" / "v3-offline-0465_r0" / "result_failed_attempt1.json"),
        },
    }
    (RUN / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
