#!/usr/bin/env python3
"""Train-only deterministic screen: evidence_unit / locator_coverage / portfolio tuning."""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
sys.path.insert(0, str(FS))

from clm_search import SlotSearch  # noqa: E402
from scoring import score  # noqa: E402

DEFAULT_GOLD = HERE.parent / "0824" / "v2" / "gold_v21_train213.jsonl"
METRICS = ("R@1", "R@5", "R@10", "R@20", "suff@5", "suff@10", "RR@10")


def load_gold(path: Path) -> list[dict]:
    return [
        row for row in map(json.loads, path.open(encoding="utf-8"))
        if row.get("groups")
    ]


def jo_rank(elements: list[tuple[dict, float]], m2j: dict, jid: dict) -> list[dict]:
    """Element-level ranked list to JO-level (best element per JO, first occurrence wins)."""
    out, seen = [], set()
    for element, _ in elements:
        jo_id = m2j.get(element["element_id"], "")
        if jo_id and jo_id not in seen and jo_id in jid:
            seen.add(jo_id)
            out.append(jid[jo_id])
    return out


def run_arm(search: SlotSearch, arm: dict, gold: list[dict],
            m2j: dict, jid: dict) -> tuple[dict, list[dict]]:
    """Run one arm across all gold questions. Returns (agg_metrics, per_question_rows)."""
    sfw = arm.get("sfw")
    profile = arm.get("profile", "core")
    portfolio_mode = arm.get("portfolio")
    limit = 200

    values = collections.defaultdict(list)
    rows = []

    for g in gold:
        slots, toks = search.router.route(g["q"])
        _, L = search.match_table(slots, toks)

        if portfolio_mode:
            res = search.rank_structured_portfolio(
                slots, toks, g["q"],
                weights=sfw,
                profile=profile,
                mode=portfolio_mode,
                limit=400,
                window=int(arm.get("portfolio_window", 40)),
                seed_quota=int(arm.get("seed_quota", 20)),
                lexical_counts=L,
                rrf_k=int(arm.get("rrf_k", 60)),
                coverage_seed=int(arm.get("coverage_seed", 3)),
                coverage_per_role=int(arm.get("coverage_per_role", 1)),
            )
        else:
            res = search.rank_structured(
                slots, toks, g["q"],
                weights=sfw,
                profile=profile,
                limit=limit,
                lexical_counts=L,
            )

        ranked_jo = jo_rank(res, m2j, jid)
        metrics = score(ranked_jo, g["groups"], ks=(1, 5, 10, 20))
        for key in METRICS:
            values[key].append(metrics[key])

        rows.append({
            "qid": g["qid"],
            "ranked_jo": [u["element_id"] for u in ranked_jo[:40]],
            **metrics,
        })

    n = len(gold)
    agg = {key: sum(values[key]) / n for key in METRICS}
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--arms", default="", help="comma-separated; default=all")
    ap.add_argument("--n", type=int, default=-1, help="head N questions; -1=all")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "det")
    args = ap.parse_args()

    if "test" in args.gold.name.lower():
        raise SystemExit("holdout protection: test gold deterministic eval forbidden")

    arms_all = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm_names = [x for x in args.arms.split(",") if x] or [
        k for k in arms_all if not k.startswith("_")
    ]
    unknown = [x for x in arm_names if x not in arms_all]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    gold = load_gold(args.gold)
    if args.n > 0:
        gold = gold[:args.n]
    print(f"gold={args.gold.name} n={len(gold)} arms={arm_names}", flush=True)

    elements_path = FS / "out" / arms_all[arm_names[0]].get("elements", "elements_u3.jsonl")
    tags_path = FS / "out" / arms_all[arm_names[0]].get("tags", "tags_u4_fact_rules.jsonl")
    jo_path = FS / "out" / arms_all[arm_names[0]].get("jo", "elements_u3jo.jsonl")

    print("loading search engine...", end=" ", flush=True)
    t0 = time.time()
    search = SlotSearch(str(elements_path), str(tags_path))
    search.ensure_structured()
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    jo_rows = [json.loads(l) for l in jo_path.open(encoding="utf-8")]
    jid = {u["element_id"]: u for u in jo_rows}
    m2j = {m: u["element_id"] for u in jo_rows for m in u["members"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for name in arm_names:
        arm = arms_all[name]
        print(f"  {name} ...", end=" ", flush=True)
        t1 = time.time()
        agg, rows = run_arm(search, arm, gold, m2j, jid)
        elapsed = time.time() - t1
        summaries[name] = agg

        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        r5 = agg["R@5"]
        s5 = agg["suff@5"]
        print(f"R@5={r5:.4f} suff@5={s5:.4f} ({elapsed:.1f}s)", flush=True)

    candidates = [x for x in arm_names if x != "baseline"]
    winner = max(
        candidates,
        key=lambda name: (summaries[name]["R@5"], summaries[name]["suff@10"],
                          summaries[name]["R@10"], name),
    ) if candidates else "baseline"

    report = {
        "gold": str(args.gold.resolve()),
        "n": len(gold),
        "selection_metric": "jo fractional R@5",
        "tie_break": ["suff@10", "R@10"],
        "selected_arm": winner,
        "arms": {
            name: {k: round(v, 6) for k, v in metrics.items()}
            for name, metrics in summaries.items()
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"{'arm':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'suff@5':>7} {'suff@10':>8} {'RR@10':>6}")
    print("-" * 80)
    for name in arm_names:
        m = summaries[name]
        marker = " <--" if name == winner else ""
        print(f"{name:<20} {m['R@1']:>6.4f} {m['R@5']:>6.4f} {m['R@10']:>6.4f}"
              f" {m['suff@5']:>7.4f} {m['suff@10']:>8.4f} {m['RR@10']:>6.4f}{marker}")
    print("=" * 80)
    print(f"selected: {winner}")


if __name__ == "__main__":
    main()
