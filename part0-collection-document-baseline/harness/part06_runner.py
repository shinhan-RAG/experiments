"""Part 0.6 R1 runner — T1(BM25F round-2) + T2(verify combination) on the
frozen dev/test protocol. Baseline for acceptance: C1(index_only).

Preregistered (PART0_6_NEXT_DESIGN_20260805):
- T1 grid: w_idx=4.0 (round-1 tuned), w_fm in {0.05,0.1,0.25},
  b_fm in {0.75,0.9}, b_idx in {0.4,0.75}, b_base=0.75, k1=1.2 — dev only.
- T2: router mixed path -> verify_partition(Index ranking, fm membership),
  depth in {20,50,100} — dev only (identity/content paths unchanged).
- Acceptance: same 3 criteria as Part 0.5 (vs C1). C4b additionally paired
  against round-1 C4 (cascade-mixed router) for the record.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from .config import Config, sha256_file
from . import pr_artifacts, score, stats, improve
from .part05_runner import (FIELDS, RRF_K, NONINF_MARGIN, build_fields,
                            load_qa, qa_metrics, mean, paired_vs, acceptance)
from .runner import TargetLock, publish
from .source_facts import scan_universe

GRID_W_FM = (0.05, 0.1, 0.25)
GRID_B_FM = (0.75, 0.9)
GRID_B_IDX = (0.4, 0.75)
W_IDX = 4.0            # round-1 dev-tuned, carried forward
CASCADE_K_R1 = 20      # round-1 tuned (needed to reproduce round-1 C4)
VERIFY_DEPTHS = (20, 50, 100)


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

    print("[build] arms + BM25F", flush=True)
    t_b10 = [" ".join(x for x in (f["base"], f["idx"]) if x) for f in fields]
    t_b01 = [" ".join(x for x in (f["base"], f["fm"]) if x) for f in fields]
    arm10, arm01 = BM25(t_b10), BM25(t_b01)
    bm25f = improve.BM25F([{**f} for f in fields], bigrams, FIELDS)
    fm_texts = [f["fm"] for f in fields]

    dev = load_qa(args.dev_qa)
    test = load_qa(args.test_qa)
    for item in dev + test:
        if any(p not in kidx for p in item["gold_source_paths"]):
            raise SystemExit(f"gold unmapped: {item['qa_id']}")

    # ---------- dev tuning ----------
    print("[tune] dev reference rankings", flush=True)
    dev_r10 = {it["qa_id"]: arm10.rank(it["query"], all_idx) for it in dev}
    ref_rows = qa_metrics(dev, lambda it: dev_r10[it["qa_id"]], keys, kidx, "B10")
    ref_id_h1 = mean(ref_rows, "hit1", "identity")

    grid_log, best = [], None
    for wf in GRID_W_FM:
        for bf in GRID_B_FM:
            for bi in GRID_B_IDX:
                w = {"base": 1.0, "idx": W_IDX, "fm": wf}
                b = {"base": 0.75, "idx": bi, "fm": bf}
                rows = qa_metrics(
                    dev, lambda it: bm25f.rank(it["query"], all_idx, w, b),
                    keys, kidx, "g")
                e = {"w_fm": wf, "b_fm": bf, "b_idx": bi,
                     "mrr10": mean(rows, "mrr10"),
                     "identity_hit1": mean(rows, "hit1", "identity"),
                     "content_hit5": mean(rows, "hit5", "content"),
                     "mixed_hit5": mean(rows, "hit5", "mixed")}
                e["feasible"] = e["identity_hit1"] >= ref_id_h1 - NONINF_MARGIN
                grid_log.append(e)
                if e["feasible"] and (best is None or e["mrr10"] > best["mrr10"]):
                    best = e
                print(f"  bm25f wf={wf} bf={bf} bi={bi} "
                      f"mrr={e['mrr10']:.3f} idh1={e['identity_hit1']:.3f}",
                      flush=True)
    if best is None:
        best = max(grid_log, key=lambda e: e["mrr10"])
        best["feasible_fallback"] = True

    vlog, best_v = [], None
    for depth in VERIFY_DEPTHS:
        def vrank(it, d=depth):
            r = improve.route(it["query"])
            if r == "identity":
                return dev_r10[it["qa_id"]]
            if r == "content":
                return arm01.rank(it["query"], all_idx)
            toks = improve.content_tokens(it["query"])
            return improve.verify_partition(dev_r10[it["qa_id"]], fm_texts,
                                            toks, d)
        rows = qa_metrics(dev, vrank, keys, kidx, f"v{depth}")
        e = {"depth": depth, "mrr10": mean(rows, "mrr10"),
             "mixed_mrr10": mean(rows, "mrr10", "mixed"),
             "identity_hit1": mean(rows, "hit1", "identity")}
        e["feasible"] = e["identity_hit1"] >= ref_id_h1 - NONINF_MARGIN
        vlog.append(e)
        if e["feasible"] and (best_v is None or
                              e["mixed_mrr10"] > best_v["mixed_mrr10"]):
            best_v = e
        print(f"  verify depth={depth} mixed_mrr={e['mixed_mrr10']:.3f}",
              flush=True)
    if best_v is None:
        best_v = max(vlog, key=lambda e: e["mixed_mrr10"])
        best_v["feasible_fallback"] = True

    tuned = {"bm25f": {"w": {"base": 1.0, "idx": W_IDX, "fm": best["w_fm"]},
                       "b": {"base": 0.75, "idx": best["b_idx"],
                             "fm": best["b_fm"]}, "k1": 1.2},
             "verify_depth": best_v["depth"],
             "cascade_k_r1": CASCADE_K_R1,
             "dev_ref_identity_hit1_B10": ref_id_h1,
             "grid_log": grid_log, "verify_log": vlog,
             "dev_qa_sha256": sha256_file(Path(args.dev_qa)),
             "test_qa_sha256": sha256_file(Path(args.test_qa))}
    tuned_path = work / "tuned_params_r1.json"
    tuned_path.write_text(json.dumps(tuned, ensure_ascii=False, indent=2,
                                     sort_keys=True), encoding="utf-8")
    tuned_sha = sha256_file(tuned_path)
    print(f"[tune] frozen: w_fm={best['w_fm']} b_fm={best['b_fm']} "
          f"b_idx={best['b_idx']} verify_depth={best_v['depth']} "
          f"sha={tuned_sha[:12]}", flush=True)

    # ---------- test pass ----------
    print("[test] rankings", flush=True)
    full10 = {it["qa_id"]: arm10.rank(it["query"], all_idx) for it in test}
    full01 = {it["qa_id"]: arm01.rank(it["query"], all_idx) for it in test}

    w, b = tuned["bm25f"]["w"], tuned["bm25f"]["b"]
    D = tuned["verify_depth"]
    conditions = {}
    conditions["C1_index"] = qa_metrics(
        test, lambda it: full10[it["qa_id"]], keys, kidx, "C1_index")
    conditions["C2b_bm25f2"] = qa_metrics(
        test, lambda it: bm25f.rank(it["query"], all_idx, w, b),
        keys, kidx, "C2b_bm25f2")

    def route_r0(it):   # round-1 C4 (mixed -> cascade K=20), for the record
        r = improve.route(it["query"])
        if r == "identity":
            return full10[it["qa_id"]]
        if r == "content":
            return full01[it["qa_id"]]
        return improve.cascade_rank(full10[it["qa_id"]], arm01,
                                    it["query"], CASCADE_K_R1)
    conditions["C4_router_r0"] = qa_metrics(test, route_r0, keys, kidx,
                                            "C4_router_r0")

    def route_verify(it):
        r = improve.route(it["query"])
        if r == "identity":
            return full10[it["qa_id"]]
        if r == "content":
            return full01[it["qa_id"]]
        toks = improve.content_tokens(it["query"])
        return improve.verify_partition(full10[it["qa_id"]], fm_texts, toks, D)
    conditions["C4b_router_verify"] = qa_metrics(test, route_verify, keys,
                                                 kidx, "C4b_router_verify")

    # ---------- stats ----------
    paired, verdicts = {}, {}
    for name in ("C2b_bm25f2", "C4_router_r0", "C4b_router_verify"):
        paired[name] = {"vs_C1": paired_vs(conditions[name],
                                           conditions["C1_index"], cfg.seed)}
        verdicts[name] = acceptance(paired[name]["vs_C1"])
    paired["C4b_vs_C4"] = paired_vs(conditions["C4b_router_verify"],
                                    conditions["C4_router_r0"], cfg.seed)

    agg = {}
    for name, rows in conditions.items():
        agg[name] = {"overall": {m: mean(rows, m) for m in stats.METRICS}}
        for suite in ("identity", "content", "mixed"):
            agg[name][suite] = {m: mean(rows, m, suite) for m in stats.METRICS}

    results = {
        "schema": "shinhan.part0_6-r1.result.v1",
        "design_doc": "PART0_6_NEXT_DESIGN_20260805.md",
        "tuned_params_sha256": tuned_sha,
        "tuned": {k: tuned[k] for k in ("bm25f", "verify_depth")},
        "aggregate": agg, "paired": paired, "acceptance": verdicts,
        "n_test": len(test), "n_dev": len(dev),
        "elapsed_sec": time.perf_counter() - t_all,
        "pr_file_sha256": pr_artifacts.pr_file_hashes(cfg),
    }

    ident_payload = {"part": "0.6-r1", "tuned_sha": tuned_sha,
                     "test_qa_sha": tuned["test_qa_sha256"],
                     "dev_qa_sha": tuned["dev_qa_sha256"], "seed": cfg.seed}
    ident_sha = hashlib.sha256(json.dumps(
        ident_payload, sort_keys=True).encode()).hexdigest()
    target = results_dir / f"part06r1_{ident_sha[:16]}"
    staging = results_dir / f".staging06_{ident_sha[:16]}"
    results_dir.mkdir(parents=True, exist_ok=True)
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
    shutil.copy2(tuned_path, staging / "tuned_params_r1.json")
    files = ["results.json", "per_query_rows.jsonl", "tuned_params_r1.json"]

    class _C:
        def identity_sha(self):
            return ident_sha
        def identity_payload(self):
            return ident_payload
    with TargetLock(results_dir):
        if target.exists():
            raise SystemExit(f"target exists: {target}")
        publish(_C(), staging, target, files)
    print(f"published: {target}", flush=True)
    for name, v in verdicts.items():
        print(f"  {name}: accepted={v['accepted']} ({v})", flush=True)


if __name__ == "__main__":
    main()
