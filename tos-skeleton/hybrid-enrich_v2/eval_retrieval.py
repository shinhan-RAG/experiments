#!/usr/bin/env python3
"""any-hit Recall@k / MRR@10, arm 비교, paired bootstrap, 도구 프로파일."""
import json, os, sys, random, collections

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
ARMS = ["BASE", "META", "TAG", "BOTH"]


def load_spans():
    spans = {}
    for fname, key in (("chunks.jsonl", "chunk_id"), ("elements.jsonl", "element_id")):
        for l in open(os.path.join(OUT, fname)):
            c = json.loads(l)
            spans[c[key]] = (c["char_start"], c["char_end"])
    return spans


UNIT_SPANS = {}

def first_rank(ranked, gold_spans):
    for i, uid in enumerate(ranked):
        sp = UNIT_SPANS.get(uid)
        if sp and any(sp[0] < g1 and sp[1] > g0 for g0, g1 in gold_spans):
            return i + 1
    return None


def metrics(rows, gold):
    per = {}
    for r in rows:
        gs = gold[r["qid"]]["gold_spans"]
        fr = first_rank(r["ranked_chunk_ids"], gs)
        per[r["qid"]] = fr
    n = len(per)
    return {
        "n": n,
        "recall@1": round(sum(1 for v in per.values() if v and v <= 1) / n, 4),
        "recall@5": round(sum(1 for v in per.values() if v and v <= 5) / n, 4),
        "recall@10": round(sum(1 for v in per.values() if v and v <= 10) / n, 4),
        "mrr@10": round(sum(1 / v for v in per.values() if v) / n, 4),
    }, per


def bootstrap_diff(perA, perB, k=5, iters=10000, seed=7):
    qids = sorted(set(perA) & set(perB))
    random.seed(seed)
    diffs = []
    hitA = {q: 1 if perA[q] and perA[q] <= k else 0 for q in qids}
    hitB = {q: 1 if perB[q] and perB[q] <= k else 0 for q in qids}
    for _ in range(iters):
        s = [random.choice(qids) for _ in qids]
        diffs.append(sum(hitA[q] - hitB[q] for q in s) / len(s))
    diffs.sort()
    return round(diffs[int(0.025 * iters)], 4), round(diffs[int(0.975 * iters)], 4)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "main"
    global UNIT_SPANS
    UNIT_SPANS.update(load_spans())
    gold = {r["qid"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, "qa100_gold.jsonl")))}
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"retrieval_{split}.jsonl"))]
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    report = {"split": split}
    pers = {}
    for arm in ARMS:
        if arm not in by_arm:
            continue
        m, per = metrics(by_arm[arm], gold)
        m["errors"] = sum(1 for r in by_arm[arm] if r["status"] == "error")
        m["not_found"] = sum(1 for r in by_arm[arm] if r["status"] == "not_found")
        m["avg_tool_calls"] = round(sum(r["n_tool_calls"] for r in by_arm[arm]) / len(by_arm[arm]), 2)
        tools = collections.Counter(t["tool"] for r in by_arm[arm] for t in r["tool_calls"])
        m["tool_mix"] = dict(tools)
        report[arm] = m
        pers[arm] = per
    for a, b in [("META", "BASE"), ("TAG", "BASE"), ("BOTH", "META"), ("BOTH", "TAG"), ("BOTH", "BASE")]:
        if a in pers and b in pers:
            lo, hi = bootstrap_diff(pers[a], pers[b])
            report[f"{a}-{b}"] = {
                "recall@5_diff": round(report[a]["recall@5"] - report[b]["recall@5"], 4),
                "ci95": [lo, hi]}
    # element_type별 recall@5
    et_slice = {}
    for arm in pers:
        d = collections.defaultdict(lambda: [0, 0])
        for qid, fr in pers[arm].items():
            et = "table" if "table" in gold[qid]["gold_element_types"] else ("formula" if "formula" in gold[qid]["gold_element_types"] else "text")
            d[et][1] += 1
            if fr and fr <= 5:
                d[et][0] += 1
        et_slice[arm] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(d.items())}
    report["recall@5_by_etype"] = et_slice
    json.dump(report, open(os.path.join(OUT, f"eval_{split}.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
