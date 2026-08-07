"""Part 0.5 runner — dev-tune → freeze → test-run → paired stats → publish.

Protocol (preregistered in PART0_5_IMPROVEMENT_DESIGN_20260804):
- tuning ONLY on the dev QA (targets disjoint from the frozen test 500)
- tuned parameters frozen (SHA) before the single test pass
- acceptance vs C1(index_only): overall MRR@10 CI low > 0  AND  identity Hit@1
  non-inferior (CI low > -0.02)  AND  content-or-mixed significant gain
"""
import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config import Config, sha256_file, sha256_text
from . import pr_artifacts, score, stats, improve
from .runner import TargetLock, check_existing, publish
from .source_facts import scan_universe

FIELDS = ("base", "idx", "fm")
GRID_W_IDX = (0.5, 1.0, 2.0, 4.0)
GRID_W_FM = (0.1, 0.25, 0.5, 1.0)
CASCADE_KS = (20, 50, 100)
RRF_K = 60          # Cormack et al. 2009 default
NONINF_MARGIN = 0.02


def build_fields(mods, ident, fm_docs):
    flatten_fm = mods["eval_content_qa"].flatten_fm
    fields, keys = [], []
    for d in fm_docs:
        base = d["product"] + " " + Path(d["file"]).name
        key = d["product"] + "/" + Path(d["file"]).name
        fields.append({"base": base, "idx": ident.get(key, ""),
                       "fm": flatten_fm(d)})
        keys.append(d["file"])
    return fields, keys


def load_qa(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def qa_metrics(items, rank_fn, keys, kidx, condition):
    rows = []
    for item in items:
        gold = {kidx[p] for p in item["gold_source_paths"]}
        t0 = time.perf_counter()
        top10 = rank_fn(item)[:10]
        dt = time.perf_counter() - t0
        m = score.metrics_for(top10, gold)
        rows.append({"qa_id": item["qa_id"], "arm": condition,
                     "suite": item["suite"],
                     "structure_type": item["document_structure_type"],
                     "gold_cardinality": "multi" if len(gold) > 1 else "single",
                     "n_gold": len(gold),
                     "top10_keys": [keys[i] for i in top10],
                     "latency_sec": dt, **m})
    return rows


def mean(rows, metric, suite=None):
    sel = [r for r in rows if suite is None or r["suite"] == suite]
    return sum(r[metric] for r in sel) / len(sel) if sel else 0.0


def paired_vs(rows_a, rows_b, seed):
    """rows_a = condition, rows_b = baseline; same qa_id sets."""
    a = {r["qa_id"]: r for r in rows_a}
    b = {r["qa_id"]: r for r in rows_b}
    qs = sorted(a)
    out = {}
    for m in ("hit1", "hit5"):
        x = sum(1 for q in qs if b[q][m] == 1 and a[q][m] == 0)
        y = sum(1 for q in qs if b[q][m] == 0 and a[q][m] == 1)
        out[f"mcnemar_{m}"] = {"baseline_only": x, "cond_only": y,
                               "p": stats.mcnemar_exact(x, y)}
    for m in ("mrr10", "ndcg10", "recall5", "recall10", "hit1", "hit5"):
        deltas = [a[q][m] - b[q][m] for q in qs]
        out[f"delta_{m}"] = stats.paired_bootstrap_ci(deltas, seed)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        out[f"wtl_{m}"] = {"win": wins, "tie": len(deltas) - wins - losses,
                           "loss": losses}
    # per-suite deltas for the acceptance rule
    for suite in ("identity", "content", "mixed"):
        sq = [q for q in qs if a[q]["suite"] == suite]
        for m in ("hit1", "hit5", "mrr10"):
            deltas = [a[q][m] - b[q][m] for q in sq]
            out[f"{suite}_delta_{m}"] = stats.paired_bootstrap_ci(deltas, seed)
    return out


def acceptance(paired):
    c1 = paired
    overall_ok = c1["delta_mrr10"]["ci95"][0] > 0
    identity_ok = c1["identity_delta_hit1"]["ci95"][0] > -NONINF_MARGIN
    content_ok = (c1["content_delta_hit5"]["ci95"][0] > 0
                  or c1["mixed_delta_hit5"]["ci95"][0] > 0
                  or c1["content_delta_mrr10"]["ci95"][0] > 0
                  or c1["mixed_delta_mrr10"]["ci95"][0] > 0)
    return {"overall_mrr_ci_low_gt0": overall_ok,
            "identity_hit1_noninferior": identity_ok,
            "content_or_mixed_gain": content_ok,
            "accepted": overall_ok and identity_ok and content_ok}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--test-qa", required=True)
    ap.add_argument("--dev-qa", required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--expected-docs", type=int, default=9417)
    ap.add_argument("--source-zip-sha256", default="")
    args = ap.parse_args(argv)

    work = Path(args.work_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cfg = Config(source_root=Path(args.source_root).resolve(),
                 repo_root=Path(args.repo_root).resolve(),
                 work_dir=work, results_dir=results_dir,
                 seed=args.seed, expected_docs=args.expected_docs,
                 source_zip_sha256=args.source_zip_sha256)

    t_all = time.perf_counter()
    universe = scan_universe(cfg.source_root)
    if len(universe) != cfg.expected_docs:
        raise SystemExit(f"universe {len(universe)} != {cfg.expected_docs}")

    mods = pr_artifacts.load_pr_modules(cfg)
    art = pr_artifacts.build_artifacts(cfg, mods)
    fm_docs = [json.loads(l) for l in open(art["doc_frontmatter_jsonl"],
                                           encoding="utf-8")]
    ci = json.loads(Path(art["collection_index_json"]).read_text(encoding="utf-8"))
    ident = pr_artifacts.build_ident_text(ci)
    fields, keys = build_fields(mods, ident, fm_docs)
    kidx = {k: i for i, k in enumerate(keys)}
    all_idx = list(range(len(keys)))
    bigrams = mods["gate2"].bigrams
    BM25 = mods["gate2"].BM25

    # base arms (shared by RRF / cascade / router / references)
    print("[build] base arms", flush=True)
    t_b00 = [f["base"] for f in fields]
    t_b10 = [" ".join(x for x in (f["base"], f["idx"]) if x) for f in fields]
    t_b01 = [" ".join(x for x in (f["base"], f["fm"]) if x) for f in fields]
    arm00, arm10, arm01 = BM25(t_b00), BM25(t_b10), BM25(t_b01)
    print("[build] BM25F fields", flush=True)
    bm25f = improve.BM25F([{**f} for f in fields], bigrams, FIELDS)

    dev = load_qa(args.dev_qa)
    test = load_qa(args.test_qa)
    for item in dev + test:
        missing = [p for p in item["gold_source_paths"] if p not in kidx]
        if missing:
            raise SystemExit(f"gold unmapped: {item['qa_id']}")

    # ---------- stage 1: dev tuning ----------
    print("[tune] dev rankings for reference arms", flush=True)
    dev_rank10 = {}
    for name, arm in (("B10", arm10), ("B01", arm01)):
        dev_rank10[name] = {it["qa_id"]: arm.rank(it["query"], all_idx)
                            for it in dev}
    ref_rows = qa_metrics(dev, lambda it: dev_rank10["B10"][it["qa_id"]],
                          keys, kidx, "B10")
    ref_id_h1 = mean(ref_rows, "hit1", "identity")

    b_all = {f: 0.75 for f in FIELDS}
    grid_log = []
    best = None
    for wi in GRID_W_IDX:
        for wf in GRID_W_FM:
            w = {"base": 1.0, "idx": wi, "fm": wf}
            rows = qa_metrics(
                dev, lambda it: bm25f.rank(it["query"], all_idx, w, b_all),
                keys, kidx, f"bm25f_{wi}_{wf}")
            entry = {"w_idx": wi, "w_fm": wf,
                     "mrr10": mean(rows, "mrr10"),
                     "identity_hit1": mean(rows, "hit1", "identity"),
                     "content_hit5": mean(rows, "hit5", "content"),
                     "mixed_hit5": mean(rows, "hit5", "mixed")}
            entry["feasible"] = entry["identity_hit1"] >= ref_id_h1 - NONINF_MARGIN
            grid_log.append(entry)
            if entry["feasible"] and (best is None or entry["mrr10"] > best["mrr10"]):
                best = entry
            print(f"  bm25f w_idx={wi} w_fm={wf} mrr={entry['mrr10']:.3f} "
                  f"idh1={entry['identity_hit1']:.3f}", flush=True)
    if best is None:
        best = max(grid_log, key=lambda e: e["mrr10"])
        best["feasible_fallback"] = True

    cas_log = []
    best_k = None
    for K in CASCADE_KS:
        rows = qa_metrics(
            dev, lambda it: improve.cascade_rank(
                dev_rank10["B10"][it["qa_id"]], arm01, it["query"], K),
            keys, kidx, f"cascade_{K}")
        entry = {"K": K, "mrr10": mean(rows, "mrr10"),
                 "identity_hit1": mean(rows, "hit1", "identity")}
        entry["feasible"] = entry["identity_hit1"] >= ref_id_h1 - NONINF_MARGIN
        cas_log.append(entry)
        if entry["feasible"] and (best_k is None or entry["mrr10"] > best_k["mrr10"]):
            best_k = entry
        print(f"  cascade K={K} mrr={entry['mrr10']:.3f} "
              f"idh1={entry['identity_hit1']:.3f}", flush=True)
    if best_k is None:
        best_k = max(cas_log, key=lambda e: e["mrr10"])
        best_k["feasible_fallback"] = True

    tuned = {"bm25f": {"w": {"base": 1.0, "idx": best["w_idx"],
                             "fm": best["w_fm"]},
                       "b": b_all, "k1": 1.2},
             "cascade_k": best_k["K"], "rrf_k": RRF_K,
             "dev_ref_identity_hit1_B10": ref_id_h1,
             "grid_log": grid_log, "cascade_log": cas_log,
             "dev_qa_sha256": sha256_file(Path(args.dev_qa)),
             "test_qa_sha256": sha256_file(Path(args.test_qa))}
    tuned_path = work / "tuned_params.json"
    tuned_path.write_text(json.dumps(tuned, ensure_ascii=False, indent=2,
                                     sort_keys=True), encoding="utf-8")
    tuned_sha = sha256_file(tuned_path)
    print(f"[tune] frozen: w_idx={best['w_idx']} w_fm={best['w_fm']} "
          f"K={best_k['K']} sha={tuned_sha[:12]}", flush=True)

    # ---------- stage 2: single test pass ----------
    w = tuned["bm25f"]["w"]
    K = tuned["cascade_k"]
    print("[test] base rankings", flush=True)
    full = {}
    for name, arm in (("C0", arm00), ("C1", arm10), ("B01", arm01)):
        full[name] = {it["qa_id"]: arm.rank(it["query"], all_idx) for it in test}

    conditions = {}
    conditions["C0_filename"] = qa_metrics(
        test, lambda it: full["C0"][it["qa_id"]], keys, kidx, "C0_filename")
    conditions["C1_index"] = qa_metrics(
        test, lambda it: full["C1"][it["qa_id"]], keys, kidx, "C1_index")
    conditions["C2_bm25f"] = qa_metrics(
        test, lambda it: bm25f.rank(it["query"], all_idx, w, b_all),
        keys, kidx, "C2_bm25f")
    conditions["C3_cascade"] = qa_metrics(
        test, lambda it: improve.cascade_rank(
            full["C1"][it["qa_id"]], arm01, it["query"], K),
        keys, kidx, "C3_cascade")

    def router_rank(it):
        r = improve.route(it["query"])
        if r == "identity":
            return full["C1"][it["qa_id"]]
        if r == "content":
            return full["B01"][it["qa_id"]]
        return improve.cascade_rank(full["C1"][it["qa_id"]], arm01,
                                    it["query"], K)
    conditions["C4_router"] = qa_metrics(test, router_rank, keys, kidx,
                                         "C4_router")
    route_pred = {it["qa_id"]: improve.route(it["query"]) for it in test}
    route_acc = sum(1 for it in test
                    if route_pred[it["qa_id"]] == it["suite"]) / len(test)

    conditions["C5_rrf"] = qa_metrics(
        test, lambda it: improve.rrf_fuse(
            [full["C0"][it["qa_id"]], full["C1"][it["qa_id"]],
             full["B01"][it["qa_id"]]], RRF_K),
        keys, kidx, "C5_rrf")

    # C6: morphological tokenizer (optional dependency)
    c6_note = "skipped: kiwipiepy unavailable"
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        keep = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "SH")
        def morph(text):
            return [t.form for t in kiwi.tokenize(text)
                    if t.tag.startswith(keep)]
        print("[test] C6 morphological index", flush=True)
        tok10 = improve.TokBM25([morph(t) for t in t_b10])
        conditions["C6_index_morph"] = qa_metrics(
            test, lambda it: tok10.rank(morph(it["query"]), all_idx),
            keys, kidx, "C6_index_morph")
        c6_note = "kiwipiepy Kiwi, POS keep=" + ",".join(keep)
    except Exception as e:  # noqa
        print(f"[test] C6 skipped: {e}", flush=True)

    # ---------- stage 3: stats + acceptance ----------
    paired = {}
    verdicts = {}
    for name, rows in conditions.items():
        if name in ("C0_filename", "C1_index"):
            continue
        paired[name] = {"vs_C1": paired_vs(rows, conditions["C1_index"],
                                           cfg.seed),
                        "vs_C0": paired_vs(rows, conditions["C0_filename"],
                                           cfg.seed)}
        verdicts[name] = acceptance(paired[name]["vs_C1"])

    agg = {}
    for name, rows in conditions.items():
        agg[name] = {"overall": {m: mean(rows, m) for m in stats.METRICS}}
        for suite in ("identity", "content", "mixed"):
            agg[name][suite] = {m: mean(rows, m, suite) for m in stats.METRICS}
        agg[name]["latency_p50"] = stats.percentile(
            [r["latency_sec"] for r in rows], 0.5)
        agg[name]["latency_p95"] = stats.percentile(
            [r["latency_sec"] for r in rows], 0.95)

    results = {
        "schema": "shinhan.part0_5-improvements.result.v1",
        "design_doc": "PART0_5_IMPROVEMENT_DESIGN_20260804.md",
        "tuned_params_sha256": tuned_sha,
        "tuned": {k: tuned[k] for k in ("bm25f", "cascade_k", "rrf_k")},
        "router_suite_agreement": route_acc,
        "c6_note": c6_note,
        "aggregate": agg,
        "paired": paired,
        "acceptance": verdicts,
        "n_test": len(test), "n_dev": len(dev),
        "elapsed_sec": time.perf_counter() - t_all,
        "pr_file_sha256": pr_artifacts.pr_file_hashes(cfg),
    }

    ident_payload = {"part": "0.5", "tuned_sha": tuned_sha,
                     "test_qa_sha": tuned["test_qa_sha256"],
                     "dev_qa_sha": tuned["dev_qa_sha256"],
                     "seed": cfg.seed}
    ident_sha = hashlib.sha256(json.dumps(
        ident_payload, sort_keys=True).encode()).hexdigest()
    target = results_dir / f"part05_{ident_sha[:16]}"
    staging = results_dir / f".staging05_{ident_sha[:16]}"
    results_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    (staging / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    with open(staging / "per_query_rows.jsonl", "w", encoding="utf-8") as f:
        for rows in conditions.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    shutil.copy2(tuned_path, staging / "tuned_params.json")
    files = ["results.json", "per_query_rows.jsonl", "tuned_params.json"]

    class _C:
        def identity_sha(self):
            return ident_sha
        def identity_payload(self):
            return ident_payload
    with TargetLock(results_dir):
        if target.exists():
            raise SystemExit(f"target exists: {target} — refuse overwrite")
        publish(_C(), staging, target, files)
    print(f"published: {target}", flush=True)
    for name, v in verdicts.items():
        print(f"  {name}: accepted={v['accepted']} ({v})", flush=True)


if __name__ == "__main__":
    main()
