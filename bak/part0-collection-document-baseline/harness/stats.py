"""Paired statistics: exact McNemar, paired bootstrap CI, 2x2 factorial effects."""
import math
from collections import defaultdict
from random import Random

from .config import ARMS

METRICS = ("hit1", "hit5", "mrr10", "ndcg10", "recall5", "recall10")
BINARY = ("hit1", "hit5")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = sum(math.comb(n, k) for k in range(0, lo + 1)) / (2 ** n) * 2
    return min(1.0, p)


def paired_bootstrap_ci(deltas: list[float], seed: int, n_boot: int = 10000,
                        alpha: float = 0.05) -> dict:
    rng = Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean_delta": sum(deltas) / n, "ci95": [lo, hi], "n_boot": n_boot}


def per_query_table(rows: list[dict]) -> dict:
    """qa_id -> arm -> metric row"""
    t = defaultdict(dict)
    for r in rows:
        t[r["qa_id"]][r["arm"]] = r
    return t


def aggregate(rows: list[dict]) -> dict:
    """Macro means by (group, arm) for overall/suite/type/cardinality."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for gkey in ("overall",
                     f"suite:{r['suite']}",
                     f"type:{r['structure_type']}",
                     f"gold:{r['gold_cardinality']}"):
            groups[gkey][r["arm"]].append(r)
    out = {}
    for gkey, by_arm in groups.items():
        out[gkey] = {}
        for arm, rs in by_arm.items():
            out[gkey][arm] = {"n": len(rs)}
            for m in METRICS:
                out[gkey][arm][m] = sum(r[m] for r in rs) / len(rs)
    return out


def paired_tests(rows: list[dict], seed: int) -> dict:
    table = per_query_table(rows)
    qa_ids = sorted(table.keys())
    out = {}
    baseline = "B00"
    for arm in ARMS:
        if arm == baseline:
            continue
        entry = {"vs": baseline}
        for m in BINARY:
            b = sum(1 for q in qa_ids
                    if table[q][baseline][m] == 1 and table[q][arm][m] == 0)
            c = sum(1 for q in qa_ids
                    if table[q][baseline][m] == 0 and table[q][arm][m] == 1)
            entry[f"mcnemar_{m}"] = {"baseline_only": b, "arm_only": c,
                                     "p": mcnemar_exact(b, c)}
        for m in METRICS:
            deltas = [table[q][arm][m] - table[q][baseline][m] for q in qa_ids]
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            ties = len(deltas) - wins - losses
            entry[f"delta_{m}"] = paired_bootstrap_ci(deltas, seed)
            entry[f"wtl_{m}"] = {"win": wins, "tie": ties, "loss": losses}
        out[arm] = entry
    # all-pairs exact McNemar on the binary metrics
    pairs = {}
    for i, a in enumerate(ARMS):
        for bm_arm in ARMS[i + 1:]:
            key = f"{a}_vs_{bm_arm}"
            pairs[key] = {}
            for m in BINARY:
                b = sum(1 for q in qa_ids
                        if table[q][a][m] == 1 and table[q][bm_arm][m] == 0)
                c = sum(1 for q in qa_ids
                        if table[q][a][m] == 0 and table[q][bm_arm][m] == 1)
                pairs[key][m] = {"a_only": b, "b_only": c,
                                 "p": mcnemar_exact(b, c)}
    out["all_pairs_mcnemar"] = pairs
    return out


def factorial_effects(agg_overall: dict) -> dict:
    """Strict 2x2 concat-ablation effects (valid because B11=concat)."""
    out = {}
    for m in METRICS:
        b00 = agg_overall["B00"][m]
        b10 = agg_overall["B10"][m]
        b01 = agg_overall["B01"][m]
        b11 = agg_overall["B11"][m]
        out[m] = {
            "index_main": (b10 + b11) / 2 - (b00 + b01) / 2,
            "fm_main": (b01 + b11) / 2 - (b00 + b10) / 2,
            "interaction": b11 - b10 - b01 + b00,
        }
    return out


def percentile(values: list[float], q: float) -> float:
    vs = sorted(values)
    if not vs:
        return 0.0
    idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
    return vs[idx]
