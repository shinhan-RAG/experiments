#!/usr/bin/env python3
"""Train-only deterministic screen and preregistered arm selection."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
DEFAULT_GOLD = ROOT / "out" / "noah" / "gold_v4_train_full.jsonl"
sys.path.insert(0, str(FS))
from scoring import score  # noqa: E402

from tag_hybrid import TagHybridSearch, default_paths

METRICS = ("R@1", "R@5", "R@10", "R@20", "R@40", "suff@5", "suff@10", "RR@10")


def load_qtags(path: Path) -> dict:
    if not path.exists():
        return {}
    return {row["qid"]: row for row in map(json.loads, path.open(encoding="utf-8"))}


def qtag_slots(row: dict | None) -> dict[str, list[str]]:
    return {k: (row or {}).get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}


def jo_rank(rows: list[dict], jid: dict) -> list[dict]:
    out, seen = [], set()
    for row in rows:
        jo = row["jo"]
        if jo and jo not in seen and jo in jid:
            seen.add(jo)
            out.append(jid[jo])
    return out


def main():
    elements, tags, jo, index_dir = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--elements", type=Path, default=elements)
    ap.add_argument("--tags", type=Path, default=tags)
    ap.add_argument("--jo", type=Path, default=jo)
    ap.add_argument("--index-dir", type=Path, default=index_dir)
    ap.add_argument("--arms", default="", help="쉼표 구분; 기본은 arms.json 전체")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n", type=int, default=-1, help="검증용 선두 문항 수; -1은 전수")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "det")
    ap.add_argument("--allow-test-gold", action="store_true", help="최종 실행 후 진단에만 사용")
    args = ap.parse_args()
    if "test" in args.gold.name.lower() and not args.allow_test_gold:
        raise SystemExit("holdout 보호: test gold 결정론 평가 금지 (--allow-test-gold는 최종 실행 후 진단 전용)")
    arms_all = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm_names = [x for x in args.arms.split(",") if x] or list(arms_all)
    unknown = [x for x in arm_names if x not in arms_all]
    if unknown:
        raise SystemExit(f"알 수 없는 arm: {unknown}")
    gold = [r for r in map(json.loads, args.gold.open(encoding="utf-8")) if r.get("groups") and r.get("status", "ok") == "ok"]
    if args.n > 0:
        gold = gold[: args.n]
    engine = TagHybridSearch(args.elements, args.tags, args.jo, args.index_dir)
    qtags = load_qtags(FS / "out" / "qtags_haiku.jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in arm_names}
    values = {name: collections.defaultdict(list) for name in arm_names}
    try:
        for pos, g in enumerate(gold, 1):
            explicit = qtag_slots(qtags.get(g["qid"]))
            for name in arm_names:
                arm = arms_all[name]
                rows, slots, _ = engine.search(
                    g["q"], explicit=explicit, weights=arm["tag_weights"], channel_top_k=args.limit,
                    limit=args.limit, collapse_jo=bool(arm.get("collapse_jo", True)),
                )
                ranked = jo_rank(rows, engine.jid)
                metrics = score(ranked, g["groups"], ks=(1, 5, 10, 20, 40))
                for key in METRICS:
                    values[name][key].append(metrics[key])
                handles[name].write(json.dumps({
                    "qid": g["qid"], "ranked_jo": [x["element_id"] for x in ranked[:40]],
                    "ranked_element": [x["element"]["element_id"] for x in rows[:40]],
                    "slots": slots, **metrics,
                }, ensure_ascii=False) + "\n")
            if pos % 25 == 0:
                print(f"{pos}/{len(gold)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    summaries = {}
    for name in arm_names:
        summaries[name] = {key: sum(values[name][key]) / len(gold) for key in METRICS}
    candidates = [x for x in arm_names if x != "baseline_fs_slot"]
    preference = {"tag_sparse_equal": 3, "tag_sparse_struct": 2, "tag_sparse_lexical": 1}
    winner = max(
        candidates,
        key=lambda name: (
            summaries[name]["R@5"], summaries[name]["suff@10"], summaries[name]["R@10"], preference.get(name, 0), name
        ),
    ) if candidates else "baseline_fs_slot"
    report = {
        "gold": str(args.gold.resolve()), "n": len(gold), "selection_metric": "jo fractional R@5",
        "full_train_screen": len(gold) == 337 and len(arm_names) == len(arms_all),
        "tie_break": ["suff@10", "R@10", "equal-weight simplicity"], "selected_arm": winner,
        "arms": {name: {k: round(v, 6) for k, v in metrics.items()} for name, metrics in summaries.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "selected_arm.txt").write_text(winner + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
