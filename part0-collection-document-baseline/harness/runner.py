"""Part 0 runner: build → QA → validate → freeze → score → publish.

Publication contract:
- same-target lock (O_CREAT|O_EXCL lockfile)
- sibling staging on the same filesystem, validation before atomic rename
- existing accepted target: same identity → revalidate and reuse untouched;
  different identity → fail loudly; never overwrite/merge/repair
- failure still writes a failure archive with a machine-readable reason
"""
import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from .config import ARMS, Config, Quotas, sha256_file
from . import pr_artifacts, qa_build, validate_qa, score, stats
from .source_facts import scan_universe

ACCEPTED_MARKER = "ACCEPTED.json"
LOCK_NAME = ".part0.lock"


class TargetLock:
    def __init__(self, target_parent: Path):
        target_parent.mkdir(parents=True, exist_ok=True)
        self.path = target_parent / LOCK_NAME
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, str(os.getpid()).encode())
        except FileExistsError:
            raise SystemExit(f"lock held: {self.path} — refuse concurrent run")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.path.unlink(missing_ok=True)


def check_existing(cfg: Config, target: Path) -> str | None:
    marker = target / ACCEPTED_MARKER
    if not target.exists():
        return None
    if not marker.exists():
        raise SystemExit(
            f"target {target} exists without {ACCEPTED_MARKER} — refuse to touch")
    accepted = json.loads(marker.read_text(encoding="utf-8"))
    if accepted["identity_sha256"] != cfg.identity_sha():
        raise SystemExit(
            "different input identity at same target — refuse "
            f"(existing {accepted['identity_sha256'][:12]}, "
            f"new {cfg.identity_sha()[:12]})")
    for rel, sha in accepted["files_sha256"].items():
        now = sha256_file(target / rel)
        if now != sha:
            raise SystemExit(f"accepted target corrupted: {rel}")
    return "reuse"


def publish(cfg: Config, payload_dir: Path, target: Path, files: list[str]) -> None:
    files_sha = {rel: sha256_file(payload_dir / rel) for rel in files}
    marker = {"identity_sha256": cfg.identity_sha(),
              "identity": cfg.identity_payload(),
              "files_sha256": files_sha}
    (payload_dir / ACCEPTED_MARKER).write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    for rel in files:
        if not (payload_dir / rel).exists():
            raise SystemExit(f"staging incomplete: {rel} missing — not publishing")
    os.rename(payload_dir, target)


def run(cfg: Config) -> dict:
    t_start = time.perf_counter()
    target = cfg.results_dir / f"part0_{cfg.identity_sha()[:16]}"
    with TargetLock(cfg.results_dir):
        if check_existing(cfg, target) == "reuse":
            print(f"accepted target verified — reuse untouched: {target}")
            return {"status": "reused", "target": str(target)}
    staging = cfg.results_dir / f".staging_{cfg.identity_sha()[:16]}_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        # 1. universe + coverage gate
        universe = scan_universe(cfg.source_root)
        if len(universe) != cfg.expected_docs:
            raise SystemExit(
                f"universe={len(universe)} != expected {cfg.expected_docs}")
        uman = cfg.work_dir / "universe_manifest.jsonl"
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        with open(uman, "w", encoding="utf-8") as f:
            for d in universe:
                f.write(json.dumps(
                    {"doc_id": d.doc_id, "rel_path": d.rel_path,
                     "product_id": d.product_id, "sha256": d.sha256,
                     "bytes": d.size_bytes},
                    ensure_ascii=False, sort_keys=True) + "\n")

        # 2. rebuild PR artifacts (byte-unmodified modules, paths injected)
        mods = pr_artifacts.load_pr_modules(cfg)
        art = pr_artifacts.build_artifacts(cfg, mods)
        fm_docs = [json.loads(l) for l in
                   open(art["doc_frontmatter_jsonl"], encoding="utf-8")]
        ftype = {}
        rel_to_id = {d.rel_path: d.doc_id for d in universe}
        for row in fm_docs:
            if row["file"] not in rel_to_id:
                raise SystemExit(f"fm row not in universe: {row['file'][:80]}")
            ftype[rel_to_id[row["file"]]] = row["ftype"]

        # 3. QA 500 (source-only)
        builder = qa_build.QABuilder(cfg, universe, ftype)
        qa_result = builder.build()
        qa_out = qa_build.write_qa(cfg, qa_result)
        n = qa_out["report"]["n"]
        want = (cfg.quotas.identity + cfg.quotas.content + cfg.quotas.mixed)
        if n != want:
            raise SystemExit(
                f"QA count {n} != {want}; shortages={qa_out['report']['shortages']}")

        # 4. mechanical validation + review ledger (500/500)
        val = validate_qa.validate(cfg, universe,
                                   cfg.work_dir / "qa" / "qa_500.jsonl")
        validate_qa.write_validation(cfg, val)

        # 5. representations + freeze BEFORE scoring
        ci = json.loads(Path(art["collection_index_json"]).read_text(
            encoding="utf-8"))
        ident = pr_artifacts.build_ident_text(ci)
        rep = score.build_representations(cfg, mods, ident, fm_docs)
        frozen = score.freeze(cfg, {
            "universe_manifest_sha256": sha256_file(uman),
            "artifacts": {k: v for k, v in art.items() if "sha" in k or k == "n_docs"},
            "pr_file_sha256": pr_artifacts.pr_file_hashes(cfg),
            "representations": rep["info"],
        })

        # 6. score (verify freeze first)
        score.verify_freeze(cfg)
        qa_items = [json.loads(l) for l in
                    open(cfg.work_dir / "qa" / "qa_500.jsonl", encoding="utf-8")]
        res = score.run_arms(cfg, mods, rep["reps"], rep["keys"], qa_items)

        # 7. aggregate + paired stats
        agg = stats.aggregate(res["rows"])
        paired = stats.paired_tests(res["rows"], cfg.seed)
        effects = stats.factorial_effects(agg["overall"])
        lat = {arm: {"p50": stats.percentile(res["latency"][arm], 0.50),
                     "p95": stats.percentile(res["latency"][arm], 0.95),
                     "mean": sum(res["latency"][arm]) / len(res["latency"][arm])}
               for arm in ARMS}

        # 8. per-query cardinality gate: every qa_id × arm exactly once
        counts = {}
        for r in res["rows"]:
            counts[(r["qa_id"], r["arm"])] = counts.get((r["qa_id"], r["arm"]), 0) + 1
        if len(counts) != len(qa_items) * len(ARMS) or set(counts.values()) != {1}:
            raise SystemExit("per-query cardinality violation")

        elapsed = time.perf_counter() - t_start
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        id_to_rel = {d.doc_id: d.rel_path for d in universe}
        d_docs = sorted(id_to_rel[i] for i, t in ftype.items() if t == "D")
        results = {
            "type_D_excluded_from_scored_set": {
                "count": len(d_docs), "docs": d_docs},
            "schema": "shinhan.part0-baseline.result.v1",
            "identity_sha256": cfg.identity_sha(),
            "freeze": frozen,
            "qa_report": qa_out["report"],
            "aggregate": agg,
            "paired": paired,
            "factorial_effects_concat_ablation": effects,
            "latency_sec": lat,
            "index_build_secs": res["index_build_secs"],
            "operational": {
                "elapsed_sec_total": elapsed,
                "peak_rss_bytes": peak_rss,
                "complexity_note": (
                    "per query per arm: O(D · Q_b) char-bigram BM25 over "
                    "D=9,417 docs (linear scan, no inverted index); "
                    "corpus scan for QA gold: one rg pass per candidate phrase"),
            },
        }
        (staging / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        with open(staging / "per_query_rows.jsonl", "w", encoding="utf-8") as f:
            for r in res["rows"]:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        for rel in ("qa/qa_500.jsonl", "qa/review_ledger.jsonl",
                    "qa/validation_summary.json", "qa/qa_build_report.json",
                    "freeze/freeze_manifest.json",
                    "representations/manifest.json"):
            src = cfg.work_dir / rel
            dst = staging / Path(rel).name
            shutil.copy2(src, dst)
        files = ["results.json", "per_query_rows.jsonl", "qa_500.jsonl",
                 "review_ledger.jsonl", "validation_summary.json",
                 "qa_build_report.json", "freeze_manifest.json",
                 "manifest.json"]
        with TargetLock(cfg.results_dir):
            if check_existing(cfg, target) == "reuse":
                shutil.rmtree(staging)
                return {"status": "reused", "target": str(target)}
            publish(cfg, staging, target, files)
        print(f"published: {target}")
        return {"status": "published", "target": str(target),
                "results": results}
    except BaseException as e:
        fail_dir = cfg.results_dir / f"failed_{cfg.identity_sha()[:16]}_{os.getpid()}"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "FAILURE.json").write_text(json.dumps(
            {"reason": str(e), "type": type(e).__name__,
             "identity_sha256": cfg.identity_sha()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        if fail_dir.exists():
            shutil.rmtree(fail_dir)
        os.rename(staging, fail_dir)
        print(f"FAILED — evidence preserved at {fail_dir}", file=sys.stderr)
        raise


def make_config(args) -> Config:
    quotas = Quotas(**json.loads(args.quotas)) if args.quotas else Quotas()
    return Config(
        source_root=Path(args.source_root).resolve(),
        repo_root=Path(args.repo_root).resolve(),
        work_dir=Path(args.work_dir).resolve(),
        results_dir=Path(args.results_dir).resolve(),
        seed=args.seed,
        expected_docs=args.expected_docs,
        source_zip_sha256=args.source_zip_sha256,
        quotas=quotas,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Part 0 collection→document baseline")
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--expected-docs", type=int, default=9417)
    ap.add_argument("--source-zip-sha256", default="")
    ap.add_argument("--quotas", default="", help="JSON override for Quotas")
    args = ap.parse_args(argv)
    cfg = make_config(args)
    out = run(cfg)
    print(json.dumps({"status": out["status"], "target": out["target"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
