#!/usr/bin/env python3
"""Train-only deterministic screen for the 0823 host_retrieve arm set.

Base evaluation (per arm, tag search only, R@K) is unchanged from 0821.
Additions:
- host_retrieve union evaluation: simulate the host's tag+meta union (as
  agent_runner.pre_search does) and score the union'd ranked list. This is
  the retrieval ceiling the agent can reach in host_retrieve mode.
- --oracle: per-question tag/meta/union hit-rank breakdown and a
  tag_only/meta_only/both/neither classification, written to oracle.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
VS = ROOT / "vector_search"
DEFAULT_GOLD = ROOT / "out" / "noah" / "gold_v4_train_full.jsonl"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(VS))
from scoring import score, overlaps  # noqa: E402
from units import Units  # noqa: E402

from tag_hybrid import TagHybridSearch, default_paths

METRICS = ("R@1", "R@5", "R@10", "R@20", "R@40", "suff@5", "suff@10", "RR@10")


def load_qtags(path: Path) -> dict:
    if not path.exists():
        return {}
    return {row["qid"]: row for row in map(json.loads, path.open(encoding="utf-8"))}


def qtag_slots(row: dict | None) -> dict[str, list[str]]:
    return {k: (row or {}).get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}


def jo_rank(rows: list[dict], jid: dict) -> list[dict]:
    """Dedup a jo-tagged item list (tag rows or meta items) by jo, first occurrence wins."""
    out, seen = [], set()
    for row in rows:
        jo = row.get("jo")
        if jo and jo not in seen and jo in jid:
            seen.add(jo)
            out.append(jid[jo])
    return out


def union_rank(tag_rows: list[dict], meta_items: list[dict], jid: dict, host_top_k: int = 40, rrf_k: int = 60) -> list[dict]:
    """RRF fusion of tag + meta results by jo, bounded to host_top_k."""
    scores: dict[str, float] = {}
    for rank, row in enumerate(tag_rows, 1):
        jo = row.get("jo")
        if jo and jo in jid:
            scores[jo] = scores.get(jo, 0) + 1.0 / (rrf_k + rank)
    for rank, item in enumerate(meta_items, 1):
        jo = item.get("jo", "")
        if jo and jo in jid:
            scores[jo] = scores.get(jo, 0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda j: (-scores[j], j))
    return [jid[jo] for jo in ordered[:host_top_k]]


def meta_search(hs, query: str, units: Units, top_k: int = 200) -> list[dict]:
    """Host-side meta search: V9 ChunkHybridSearch(hybrid), chunk spans mapped to jo."""
    results = hs.search(query, strategy="hybrid", top_k=top_k)
    items = []
    for r in results:
        unit = units.jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
        items.append({"id": r["id"], "jo": unit["element_id"] if unit else ""})
    return items


def hit_ranks(ranked_units: list[dict], groups: list[dict]) -> dict[int, int]:
    """gi -> first 1-indexed rank in ranked_units that overlaps gold group gi."""
    hits: dict[int, int] = {}
    for r, e in enumerate(ranked_units, 1):
        for gi, g in enumerate(groups):
            if gi not in hits and overlaps(e, g):
                hits[gi] = r
    return hits


def classify(tag_rank, meta_rank) -> str:
    if tag_rank is not None and meta_rank is not None:
        return "both"
    if tag_rank is not None:
        return "tag_only"
    if meta_rank is not None:
        return "meta_only"
    return "neither"


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
    ap.add_argument("--meta-top-k", type=int, default=200, help="host meta 채널 top_k")
    ap.add_argument("--n", type=int, default=-1, help="검증용 선두 문항 수; -1은 전수")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "det")
    ap.add_argument("--allow-test-gold", action="store_true", help="최종 실행 후 진단에만 사용")
    ap.add_argument("--oracle", action="store_true", help="tag/meta/union 문항별 채널 기여 분석을 oracle.jsonl에 기록")
    ap.add_argument("--oracle-arm", default="", help="오라클에 쓸 arm 설정; 기본은 첫 host_retrieve arm")
    args = ap.parse_args()
    if "test" in args.gold.name.lower() and not args.allow_test_gold:
        raise SystemExit("holdout 보호: test gold 결정론 평가 금지 (--allow-test-gold는 최종 실행 후 진단 전용)")
    arms_all = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm_names = [x for x in args.arms.split(",") if x] or list(arms_all)
    unknown = [x for x in arm_names if x not in arms_all]
    if unknown:
        raise SystemExit(f"알 수 없는 arm: {unknown}")
    host_arms = [x for x in arm_names if arms_all[x].get("mode") == "host_retrieve"]
    oracle_arm = args.oracle_arm or (host_arms[0] if host_arms else "")
    if args.oracle:
        if not oracle_arm:
            raise SystemExit("--oracle: host_retrieve arm이 없습니다 (--oracle-arm으로 지정)")
        if oracle_arm not in arms_all:
            raise SystemExit(f"알 수 없는 --oracle-arm: {oracle_arm}")

    gold = [r for r in map(json.loads, args.gold.open(encoding="utf-8")) if r.get("groups") and r.get("status", "ok") == "ok"]
    if args.n > 0:
        gold = gold[: args.n]
    engine = TagHybridSearch(args.elements, args.tags, args.jo, args.index_dir)
    units = Units(str(args.jo))
    qtags = load_qtags(FS / "out" / "qtags_haiku.jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hs_cache: dict[str, object] = {}
    meta_cache: dict[tuple[str, str], list[dict]] = {}

    def get_meta_items(query: str, view: str) -> list[dict]:
        key = (query, view)
        if key not in meta_cache:
            if view not in hs_cache:
                from hybrid_search import ChunkHybridSearch

                hs_cache[view] = ChunkHybridSearch(view=view)
            meta_cache[key] = meta_search(hs_cache[view], query, units, top_k=args.meta_top_k)
        return meta_cache[key]

    handles = {name: (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in arm_names}
    values = {name: collections.defaultdict(list) for name in arm_names}
    union_handles = {name: (args.out_dir / f"{name}_union.jsonl").open("w", encoding="utf-8") for name in host_arms}
    union_values = {name: collections.defaultdict(list) for name in host_arms}
    oracle_handle = (args.out_dir / "oracle.jsonl").open("w", encoding="utf-8") if args.oracle else None

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

                need_meta = name in host_arms or (args.oracle and name == oracle_arm)
                if need_meta and arm.get("meta"):
                    meta_items = get_meta_items(g["q"], arm.get("meta_view", "V9"))
                else:
                    meta_items = []
                host_top_k = int(arm.get("host_top_k", 40))

                if name in host_arms:
                    union_ranked = union_rank(rows, meta_items, engine.jid, host_top_k)
                    union_metrics = score(union_ranked, g["groups"], ks=(1, 5, 10, 20, 40))
                    for key in METRICS:
                        union_values[name][key].append(union_metrics[key])
                    union_handles[name].write(json.dumps({
                        "qid": g["qid"], "ranked_jo": [x["element_id"] for x in union_ranked],
                        "host_top_k": host_top_k, **union_metrics,
                    }, ensure_ascii=False) + "\n")

                if args.oracle and name == oracle_arm:
                    tag_hits = hit_ranks(ranked, g["groups"])
                    meta_ranked = jo_rank(meta_items, engine.jid)
                    meta_hits = hit_ranks(meta_ranked, g["groups"])
                    union_ranked_o = union_ranked if name in host_arms else union_rank(rows, meta_items, engine.jid, host_top_k)
                    union_hits = hit_ranks(union_ranked_o, g["groups"])
                    groups_out = []
                    for gi in range(len(g["groups"])):
                        t, m, u = tag_hits.get(gi), meta_hits.get(gi), union_hits.get(gi)
                        groups_out.append({"gi": gi, "tag_rank": t, "meta_rank": m, "union_rank": u, "class": classify(t, m)})
                    has_tag = any(x["tag_rank"] is not None for x in groups_out)
                    has_meta = any(x["meta_rank"] is not None for x in groups_out)
                    if has_tag and has_meta:
                        qclass = "both"
                    elif has_tag:
                        qclass = "tag_only"
                    elif has_meta:
                        qclass = "meta_only"
                    else:
                        qclass = "neither"
                    oracle_handle.write(json.dumps({
                        "qid": g["qid"], "arm": oracle_arm, "n_groups": len(g["groups"]),
                        "groups": groups_out, "classification": qclass,
                        "union_covers_all": all(x["union_rank"] is not None for x in groups_out),
                        "host_top_k": host_top_k,
                    }, ensure_ascii=False) + "\n")
            if pos % 25 == 0:
                print(f"{pos}/{len(gold)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
        for handle in union_handles.values():
            handle.close()
        if oracle_handle:
            oracle_handle.close()

    summaries = {}
    for name in arm_names:
        summaries[name] = {key: sum(values[name][key]) / len(gold) for key in METRICS}
    union_summaries = {}
    for name in host_arms:
        union_summaries[name] = {key: sum(union_values[name][key]) / len(gold) for key in METRICS}

    candidates = [x for x in arm_names if x != "baseline"]
    preference = {"host_retrieve": 3, "host_retrieve_greedy": 2, "expanded_inclusive": 1}
    winner = max(
        candidates,
        key=lambda name: (
            summaries[name]["R@5"], summaries[name]["suff@10"], summaries[name]["R@10"], preference.get(name, 0), name
        ),
    ) if candidates else "baseline"
    report = {
        "gold": str(args.gold.resolve()), "n": len(gold), "selection_metric": "jo fractional R@5",
        "full_train_screen": len(gold) == 337 and len(arm_names) == len(arms_all),
        "tie_break": ["suff@10", "R@10", "equal-weight simplicity"], "selected_arm": winner,
        "arms": {name: {k: round(v, 6) for k, v in metrics.items()} for name, metrics in summaries.items()},
        "union": {name: {k: round(v, 6) for k, v in metrics.items()} for name, metrics in union_summaries.items()},
        "oracle_path": str((args.out_dir / "oracle.jsonl").resolve()) if args.oracle else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "selected_arm.txt").write_text(winner + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
