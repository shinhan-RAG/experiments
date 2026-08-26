#!/usr/bin/env python3
"""결정론 전수 평가 — 0826 LLM v3 태그 실험.

규칙 태그 vs LLM 태그를 동일 gold 분모에서 비교한다. 태그 파일별 인덱스를 자동 빌드/로드.
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
sys.path.insert(0, str(FS))
from scoring import score  # noqa: E402

from tag_hybrid import TagHybridSearch, build_index

METRICS = ("R@1", "R@5", "R@10", "R@20", "R@40", "suff@5", "suff@10", "RR@10")
DEFAULT_GOLD = ROOT / "vector_search" / "noah" / "0824" / "v2" / "gold_v21_train213.jsonl"


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


def resolve_tags_path(tags_name: str) -> Path:
    if Path(tags_name).is_absolute():
        return Path(tags_name)
    for base in [HERE / "out", FS / "out"]:
        p = base / tags_name
        if p.exists():
            return p
    return FS / "out" / tags_name


def ensure_index(elements: Path, tags: Path, jo: Path, arm_name: str) -> Path:
    index_dir = HERE / "out" / f"idx_{arm_name}"
    meta_path = index_dir / "tag_index_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sources = meta.get("sources", {})
        from tag_hybrid import sha256
        try:
            if (sources.get("tags", {}).get("sha256") == sha256(tags)
                    and sources.get("elements", {}).get("sha256") == sha256(elements)):
                return index_dir
        except Exception:
            pass
    print(f"  Building index for {arm_name} ({tags.name})...", flush=True)
    build_index(elements, tags, jo, index_dir)
    return index_dir


def main():
    ap = argparse.ArgumentParser(description="0826 결정론 전수 평가")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--arms", default="", help="쉼표 구분; 기본은 arms.json 전체")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "out" / "det")
    ap.add_argument("--allow-test-gold", action="store_true")
    args = ap.parse_args()

    if "test" in args.gold.name.lower() and not args.allow_test_gold:
        raise SystemExit("holdout 보호: test gold 결정론 평가 금지")

    arms_all = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm_names = [x for x in args.arms.split(",") if x] or list(arms_all)
    unknown = [x for x in arm_names if x not in arms_all]
    if unknown:
        raise SystemExit(f"알 수 없는 arm: {unknown}")

    gold = [r for r in map(json.loads, args.gold.open(encoding="utf-8"))
            if r.get("groups")]
    if args.n > 0:
        gold = gold[:args.n]

    elements = FS / "out" / "elements_u3.jsonl"
    jo = FS / "out" / "elements_u3jo.jsonl"
    qtags = load_qtags(FS / "out" / "qtags_haiku.jsonl")

    engine_pool = {}
    engines = {}
    for name in arm_names:
        arm = arms_all[name]
        tags_path = resolve_tags_path(arm["tags"])
        if not tags_path.exists():
            print(f"  SKIP {name}: 태그 파일 없음 ({tags_path})", flush=True)
            continue
        tags_key = str(tags_path.resolve())
        if tags_key not in engine_pool:
            index_dir = ensure_index(elements, tags_path, jo, name)
            print(f"  Loading engine for {tags_path.name}...", flush=True)
            engine_pool[tags_key] = TagHybridSearch(elements, tags_path, jo, index_dir)
        engines[name] = engine_pool[tags_key]

    active_arms = [n for n in arm_names if n in engines]
    if not active_arms:
        raise SystemExit("활성 arm 없음 — 태그 파일을 먼저 생성하세요")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {name: (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in active_arms}
    values = {name: collections.defaultdict(list) for name in active_arms}

    print(f"Evaluating {len(gold)} questions × {len(active_arms)} arms", flush=True)
    try:
        for pos, g in enumerate(gold, 1):
            explicit = qtag_slots(qtags.get(g["qid"]))
            for name in active_arms:
                arm = arms_all[name]
                engine = engines[name]
                rows, slots, _ = engine.search(
                    g["q"], explicit=explicit,
                    weights=arm["tag_weights"],
                    channel_top_k=args.limit,
                    limit=args.limit,
                    collapse_jo=bool(arm.get("collapse_jo", True)),
                )
                ranked = jo_rank(rows, engine.jid)
                metrics = score(ranked, g["groups"], ks=(1, 5, 10, 20, 40))
                for key in METRICS:
                    values[name][key].append(metrics[key])
                handles[name].write(json.dumps({
                    "qid": g["qid"],
                    "ranked_jo": [x["element_id"] for x in ranked[:40]],
                    **metrics,
                }, ensure_ascii=False) + "\n")
            if pos % 5 == 0 or pos <= 3:
                r5s = " ".join(f"{n}={values[n]['R@5'][-1]:.2f}" for n in active_arms)
                print(f"  {pos}/{len(gold)} {r5s}", flush=True)
                for h in handles.values():
                    h.flush()
    finally:
        for handle in handles.values():
            handle.close()

    summaries = {}
    for name in active_arms:
        summaries[name] = {key: round(sum(values[name][key]) / len(gold), 6) for key in METRICS}

    baseline_r5 = summaries.get("baseline_rules", {}).get("R@5", 0)
    for name in active_arms:
        summaries[name]["delta_R@5"] = round(summaries[name]["R@5"] - baseline_r5, 6)

    candidates = [x for x in active_arms if x != "baseline_rules"]
    winner = max(
        candidates,
        key=lambda name: (summaries[name]["R@5"], summaries[name]["suff@10"],
                          summaries[name]["R@10"], name),
    ) if candidates else active_arms[0]

    report = {
        "gold": str(args.gold.resolve()),
        "n": len(gold),
        "selection_metric": "jo fractional R@5",
        "tie_break": ["suff@10", "R@10"],
        "selected_arm": winner,
        "arms": summaries,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
