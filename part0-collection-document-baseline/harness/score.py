"""Arm representations, freeze, and scoring.

All four arms: identical candidate universe, identical queries/gold bytes,
identical BM25 (PR #14 gate2.BM25, char-bigram, imported byte-unmodified),
identical top-k. B11 is concatenation (concat_ablation).
"""
import json
import math
import time
from pathlib import Path

from .config import (ARMS, ARM_LABELS, B11_PRIMARY_MEANING, COMPOSITION_RULE,
                     Config, sha256_file, sha256_text)


def build_representations(cfg: Config, mods: dict, ident: dict,
                          fm_docs: list[dict]) -> dict:
    """fm_docs = doc_frontmatter.jsonl rows in file order (canonical order)."""
    flatten_fm = mods["eval_content_qa"].flatten_fm
    reps = {arm: [] for arm in ARMS}
    keys = []
    n_ident_miss = 0
    for d in fm_docs:
        base = d["product"] + " " + Path(d["file"]).name
        key = d["product"] + "/" + Path(d["file"]).name
        idx = ident.get(key)
        if idx is None:
            n_ident_miss += 1
            idx = ""
        fm = flatten_fm(d)
        keys.append(d["file"])
        reps["B00"].append(base)
        reps["B10"].append(" ".join(x for x in (base, idx) if x))
        reps["B01"].append(" ".join(x for x in (base, fm) if x))
        reps["B11"].append(" ".join(x for x in (base, idx, fm) if x))
    rep_dir = cfg.work_dir / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)
    info = {"composition_rule": COMPOSITION_RULE,
            "b11_primary_meaning": B11_PRIMARY_MEANING,
            "n_docs": len(keys), "n_ident_miss": n_ident_miss,
            "arm_labels": ARM_LABELS, "sha256": {}, "bytes": {}}
    for arm in ARMS:
        p = rep_dir / f"{arm}.txt"
        text = "\n".join(reps[arm]) + "\n"
        p.write_text(text, encoding="utf-8")
        info["sha256"][arm] = sha256_file(p)
        info["bytes"][arm] = p.stat().st_size
    (rep_dir / "keys.json").write_text(
        json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    info["keys_sha256"] = sha256_file(rep_dir / "keys.json")
    (rep_dir / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return {"reps": reps, "keys": keys, "info": info}


def freeze(cfg: Config, extra: dict) -> dict:
    """Write the pre-scoring freeze manifest. Scoring refuses to run without it."""
    freeze_dir = cfg.work_dir / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "identity": cfg.identity_payload(),
        "identity_sha256": cfg.identity_sha(),
        "qa_sha256": sha256_file(cfg.work_dir / "qa" / "qa_500.jsonl"),
        "ledger_sha256": sha256_file(cfg.work_dir / "qa" / "review_ledger.jsonl"),
        **extra,
    }
    p = freeze_dir / "freeze_manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                            sort_keys=True), encoding="utf-8")
    return manifest


def verify_freeze(cfg: Config) -> dict:
    p = cfg.work_dir / "freeze" / "freeze_manifest.json"
    if not p.exists():
        raise SystemExit("freeze manifest missing — refuse to score")
    manifest = json.loads(p.read_text(encoding="utf-8"))
    now_qa = sha256_file(cfg.work_dir / "qa" / "qa_500.jsonl")
    if now_qa != manifest["qa_sha256"]:
        raise SystemExit("QA file changed after freeze — refuse to score")
    rep_info = json.loads((cfg.work_dir / "representations" / "manifest.json")
                          .read_text(encoding="utf-8"))
    for arm in ARMS:
        now = sha256_file(cfg.work_dir / "representations" / f"{arm}.txt")
        if now != rep_info["sha256"][arm]:
            raise SystemExit(f"representation {arm} changed after freeze")
    return manifest


# ---------- metrics ----------

def metrics_for(ranked10: list[int], gold: set, k_max: int = 10) -> dict:
    hits = [1 if r in gold else 0 for r in ranked10]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    out = {
        "hit1": 1 if first == 1 else 0,
        "hit5": 1 if (first is not None and first <= 5) else 0,
        "mrr10": (1.0 / first) if (first is not None and first <= 10) else 0.0,
        "rank_first_gold": first,
    }
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg10"] = dcg / idcg if idcg > 0 else 0.0
    out["recall5"] = len(gold & set(ranked10[:5])) / len(gold)
    out["recall10"] = len(gold & set(ranked10[:10])) / len(gold)
    return out


def run_arms(cfg: Config, mods: dict, reps: dict, keys: list[str],
             qa_items: list[dict]) -> dict:
    BM25 = mods["gate2"].BM25
    kidx = {k: i for i, k in enumerate(keys)}
    all_idx = list(range(len(keys)))
    unmapped = []
    for item in qa_items:
        missing = [p for p in item["gold_source_paths"] if p not in kidx]
        if missing:
            unmapped.append({"qa_id": item["qa_id"], "missing": missing})
    if unmapped:
        raise SystemExit(f"gold paths unmapped to universe: {unmapped[:3]}")

    rows = []
    latency = {arm: [] for arm in ARMS}
    build_secs = {}
    arms = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        arms[arm] = BM25(reps[arm])
        build_secs[arm] = time.perf_counter() - t0
    for item in qa_items:
        gold = {kidx[p] for p in item["gold_source_paths"]}
        for arm in ARMS:
            t0 = time.perf_counter()
            ranked10 = arms[arm].rank(item["query"], all_idx)[:10]
            dt = time.perf_counter() - t0
            latency[arm].append(dt)
            m = metrics_for(ranked10, gold)
            rows.append({
                "qa_id": item["qa_id"], "arm": arm, "suite": item["suite"],
                "structure_type": item["document_structure_type"],
                "gold_cardinality": "multi" if len(gold) > 1 else "single",
                "n_gold": len(gold),
                "top10_keys": [keys[i] for i in ranked10],
                "latency_sec": dt, **m,
            })
    return {"rows": rows, "latency": latency, "index_build_secs": build_secs}
