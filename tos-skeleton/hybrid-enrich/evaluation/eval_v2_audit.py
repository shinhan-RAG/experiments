#!/usr/bin/env python3
"""Audit the completed v2 100-question, four-arm agentic retrieval run."""

import collections
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
ARMS = ("BASE", "META", "TAG", "BOTH")


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_rank(row, gold, spans):
    for rank, unit_id in enumerate(row.get("ranked_chunk_ids", []), 1):
        unit_span = spans.get(unit_id)
        if unit_span and any(
            unit_span[0] < gold_end and unit_span[1] > gold_start
            for gold_start, gold_end in gold["gold_spans"]
        ):
            return rank
    return None


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def paired_bootstrap(per_a, per_b, metric, iterations=10_000, seed=20260804):
    qids = sorted(set(per_a) & set(per_b))
    rng = random.Random(seed)

    def value(rank):
        if metric.startswith("recall@"): 
            k = int(metric.split("@")[1])
            return float(rank is not None and rank <= k)
        if metric == "mrr@10":
            return 1.0 / rank if rank is not None and rank <= 10 else 0.0
        raise ValueError(metric)

    diffs = []
    for _ in range(iterations):
        sample = [rng.choice(qids) for _ in qids]
        diffs.append(sum(value(per_a[q]) - value(per_b[q]) for q in sample) / len(sample))
    return [round(percentile(diffs, 0.025), 4), round(percentile(diffs, 0.975), 4)]


def summarize(rows, per):
    n = len(rows)
    calls = collections.Counter(call["tool"] for row in rows for call in row.get("tool_calls", []))
    first = collections.Counter(
        row["tool_calls"][0]["tool"] if row.get("tool_calls") else "none" for row in rows
    )
    return {
        "n": n,
        "status": dict(collections.Counter(row.get("status", "missing") for row in rows)),
        "recall@1": sum(rank is not None and rank <= 1 for rank in per.values()) / n,
        "recall@5": sum(rank is not None and rank <= 5 for rank in per.values()) / n,
        "recall@10": sum(rank is not None and rank <= 10 for rank in per.values()) / n,
        "mrr@10": sum(1 / rank for rank in per.values() if rank is not None and rank <= 10) / n,
        "avg_tool_calls": sum(row.get("n_tool_calls", 0) for row in rows) / n,
        "tool_calls": dict(calls),
        "sessions_using_vector": sum(
            any(call["tool"] == "vector_search" for call in row.get("tool_calls", [])) for row in rows
        ),
        "sessions_using_keyword": sum(
            any(call["tool"] == "grep_search" for call in row.get("tool_calls", [])) for row in rows
        ),
        "first_tool": dict(first),
        "ranked_id_prefixes": dict(
            collections.Counter(uid[:1] for row in rows for uid in row.get("ranked_chunk_ids", []))
        ),
    }


def slice_recall(per, gold, field):
    groups = collections.defaultdict(list)
    for qid, rank in per.items():
        if field == "element_type":
            types = gold[qid]["gold_element_types"]
            group = "table" if "table" in types else "formula" if "formula" in types else "text"
        else:
            group = gold[qid]["business_type"]
        groups[group].append(rank)
    return {
        group: {
            "hits": sum(rank is not None and rank <= 5 for rank in ranks),
            "n": len(ranks),
            "recall@5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        }
        for group, ranks in sorted(groups.items())
    }


def main():
    rows = load_jsonl(OUT / "retrieval_v2_final.jsonl")
    gold = {row["qid"]: row for row in load_jsonl(OUT / "qa100_gold.jsonl")}
    spans = {}
    for filename, key in (("chunks.jsonl", "chunk_id"), ("elements.jsonl", "element_id")):
        for unit in load_jsonl(OUT / filename):
            spans[unit[key]] = (unit["char_start"], unit["char_end"])

    duplicate_pairs = len(rows) - len({(row["qid"], row["arm"]) for row in rows})
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    per = {
        arm: {row["qid"]: first_rank(row, gold[row["qid"]], spans) for row in arm_rows}
        for arm, arm_rows in by_arm.items()
    }
    report = {
        "source": "out/retrieval_v2_final.jsonl",
        "rows": len(rows),
        "unique_qid_arm_pairs": len(rows) - duplicate_pairs,
        "duplicate_qid_arm_pairs": duplicate_pairs,
        "arms": {arm: summarize(by_arm[arm], per[arm]) for arm in ARMS},
        "paired_differences": {},
        "recall@5_by_element_type": {
            arm: slice_recall(per[arm], gold, "element_type") for arm in ARMS
        },
        "recall@5_by_business_type": {
            arm: slice_recall(per[arm], gold, "business_type") for arm in ARMS
        },
    }

    for arm, baseline in (
        ("META", "BASE"), ("TAG", "BASE"), ("BOTH", "BASE"),
        ("BOTH", "META"), ("BOTH", "TAG"),
    ):
        a5 = report["arms"][arm]["recall@5"]
        b5 = report["arms"][baseline]["recall@5"]
        transitions = collections.Counter()
        for qid in sorted(gold):
            base_hit = per[baseline][qid] is not None and per[baseline][qid] <= 5
            arm_hit = per[arm][qid] is not None and per[arm][qid] <= 5
            transitions[f"base_{'hit' if base_hit else 'miss'}__arm_{'hit' if arm_hit else 'miss'}"] += 1
        report["paired_differences"][f"{arm}-{baseline}"] = {
            "recall@5_difference": a5 - b5,
            "recall@5_ci95": paired_bootstrap(per[arm], per[baseline], "recall@5"),
            "mrr@10_difference": report["arms"][arm]["mrr@10"] - report["arms"][baseline]["mrr@10"],
            "mrr@10_ci95": paired_bootstrap(per[arm], per[baseline], "mrr@10"),
            "transitions": dict(transitions),
        }

    row_by_pair = {(row["qid"], row["arm"]): row for row in rows}
    complete_qids = [
        qid for qid in gold
        if all(row_by_pair[(qid, arm)].get("status") == "ranked" for arm in ARMS)
    ]
    report["complete_case_sensitivity"] = {
        "n": len(complete_qids),
        "recall@5": {
            arm: sum(per[arm][qid] is not None and per[arm][qid] <= 5 for qid in complete_qids) / len(complete_qids)
            for arm in ARMS
        },
    }

    best_single = max(report["arms"]["META"]["recall@5"], report["arms"]["TAG"]["recall@5"])
    report["gates"] = {
        "G1_META_plus_10pt": report["arms"]["META"]["recall@5"] >= report["arms"]["BASE"]["recall@5"] + 0.10,
        "G2_TAG_plus_10pt": report["arms"]["TAG"]["recall@5"] >= report["arms"]["BASE"]["recall@5"] + 0.10,
        "G3_BOTH_above_best_single": report["arms"]["BOTH"]["recall@5"] > best_single,
    }

    old_eval = json.loads((OUT / "eval_main.json").read_text(encoding="utf-8"))
    report["repeatability_warning"] = {
        "previous_BASE_recall@5": old_eval["BASE"]["recall@5"],
        "v2_BASE_recall@5": report["arms"]["BASE"]["recall@5"],
        "difference": report["arms"]["BASE"]["recall@5"] - old_eval["BASE"]["recall@5"],
        "note": "Same 100-question sample, but independent agent sessions; this run-to-run shift is larger than the observed enrichment gains.",
    }
    smoke = json.loads((OUT / "split_manifest.json").read_text(encoding="utf-8"))["smoke"]
    report["audit_flags"] = {
        "smoke_entries": len(smoke),
        "smoke_unique_qids": len(set(smoke)),
        "all_final_ranked_ids_are_elements": all(
            uid.startswith("e") for row in rows for uid in row.get("ranked_chunk_ids", [])
        ),
        "authoritative_tool_outputs_logged": False,
        "sampling_settings_recorded": False,
        "main_100_was_reused_for_v2_iteration": True,
    }

    target = OUT / "eval_v2_audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
