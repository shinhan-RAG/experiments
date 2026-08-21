#!/usr/bin/env python3
"""Recompute paired host-agent metrics with qid-cluster bootstrap intervals."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


METRICS = ("R@1", "R@5", "R@10", "suff@5", "suff@10")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--results", default="results.jsonl")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_jsonl(run_dir / args.results)
    by_key = {(r["qid"], r["rep"], r["arm"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})
    reps = sorted({r["rep"] for r in rows})
    missing = [(q, rep, arm) for q in qids for rep in reps
               for arm in (args.baseline, args.candidate) if (q, rep, arm) not in by_key]
    if missing:
        raise ValueError(f"missing paired cells: {missing[:10]}")

    result = {
        "run": run_dir.name,
        "results": args.results,
        "n_qids": len(qids),
        "reps": len(reps),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "fatal_errors": {arm: sum(by_key[q, rep, arm]["errors"] for q in qids for rep in reps)
                         for arm in (args.baseline, args.candidate)},
        "protocol_errors": {arm: sum(by_key[q, rep, arm]["protocol_errors"] for q in qids for rep in reps)
                            for arm in (args.baseline, args.candidate)},
        "empty_submissions": {arm: sum(by_key[q, rep, arm]["empty_submission"] for q in qids for rep in reps)
                              for arm in (args.baseline, args.candidate)},
        "metrics": {},
    }
    rng = random.Random(args.seed)
    for metric in METRICS:
        qmeans = {}
        for qid in qids:
            base = sum(by_key[qid, rep, args.baseline][metric] for rep in reps) / len(reps)
            cand = sum(by_key[qid, rep, args.candidate][metric] for rep in reps) / len(reps)
            qmeans[qid] = {"baseline": base, "candidate": cand, "delta": cand - base}
        samples = []
        for _ in range(args.samples):
            sampled = [qids[rng.randrange(len(qids))] for _ in qids]
            samples.append(sum(qmeans[q]["delta"] for q in sampled) / len(sampled))
        deltas = [qmeans[q]["delta"] for q in qids]
        wins = [q for q in qids if qmeans[q]["delta"] > 0]
        losses = [q for q in qids if qmeans[q]["delta"] < 0]
        result["metrics"][metric] = {
            "baseline": sum(qmeans[q]["baseline"] for q in qids) / len(qids),
            "candidate": sum(qmeans[q]["candidate"] for q in qids) / len(qids),
            "delta": sum(deltas) / len(deltas),
            "qid_bootstrap_ci95": [percentile(samples, .025), percentile(samples, .975)],
            "wins": wins,
            "losses": losses,
            "ties": len(qids) - len(wins) - len(losses),
            "hard_improvements": [q for q in qids if qmeans[q]["baseline"] == 0 and qmeans[q]["candidate"] == 1],
            "hard_regressions": [q for q in qids if qmeans[q]["baseline"] == 1 and qmeans[q]["candidate"] == 0],
            "qid_means": qmeans,
        }
    if args.results == "results.jsonl":
        output = run_dir / "paired_analysis.json"
    else:
        label = Path(args.results).stem
        output = run_dir / f"paired_analysis_{label}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
