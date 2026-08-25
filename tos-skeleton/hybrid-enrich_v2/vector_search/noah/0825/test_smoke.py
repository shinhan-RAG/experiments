#!/usr/bin/env python3
"""Smoke test: arm config parsing, search engine load, baseline vs evidence_unit divergence."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
sys.path.insert(0, str(FS))

GOLD_PATH = HERE.parent / "0824" / "v2" / "gold_v21_train213.jsonl"
REQUIRED_ARMS = ("baseline", "A1_eu_04", "A2_eu_08", "A3_eu_12",
                 "F1_loc_cov", "F2_loc_cov_high", "G1_portfolio_wide", "AF_combined")
REQUIRED_SFW_KEYS = {
    "A1_eu_04": "evidence_unit",
    "F1_loc_cov": "_locator_coverage",
    "G1_portfolio_wide": None,
}


def test_arms_config():
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    for name in REQUIRED_ARMS:
        assert name in arms, f"missing arm: {name}"
        arm = arms[name]
        assert arm.get("ranker") == "bm25f", f"{name}: ranker must be bm25f"
        assert arm.get("profile") == "core", f"{name}: profile must be core"
        assert arm.get("elements") == "elements_u3.jsonl", f"{name}: wrong elements"
        assert arm.get("tags") == "tags_u4_fact_rules.jsonl", f"{name}: wrong tags"
    for name, key in REQUIRED_SFW_KEYS.items():
        if key:
            sfw = arms[name].get("sfw", {})
            assert key in sfw and sfw[key] > 0, f"{name}: sfw.{key} must be positive"
    assert arms["G1_portfolio_wide"].get("portfolio") == "rrf_safe_axes"
    print("  arms config: OK")


def test_gold_exists():
    assert GOLD_PATH.exists(), f"gold not found: {GOLD_PATH}"
    gold = [json.loads(l) for l in GOLD_PATH.open(encoding="utf-8")]
    valid = [g for g in gold if g.get("groups")]
    assert len(valid) >= 200, f"expected >=200 valid questions, got {len(valid)}"
    print(f"  gold: OK ({len(valid)} questions)")


def test_search_engine():
    from clm_search import SlotSearch

    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm = arms["baseline"]
    elements_path = FS / "out" / arm["elements"]
    tags_path = FS / "out" / arm["tags"]

    assert elements_path.exists(), f"elements not found: {elements_path}"
    assert tags_path.exists(), f"tags not found: {tags_path}"

    print("  loading search engine...", end=" ", flush=True)
    t0 = time.time()
    search = SlotSearch(str(elements_path), str(tags_path))
    search.ensure_structured()
    print(f"done ({time.time() - t0:.1f}s)")

    gold = [
        g for g in map(json.loads, GOLD_PATH.open(encoding="utf-8"))
        if g.get("groups")
    ][:3]

    for g in gold:
        slots, toks = search.router.route(g["q"])
        _, L = search.match_table(slots, toks)
        res = search.rank_structured(
            slots, toks, g["q"],
            weights=arm.get("sfw"),
            profile="core",
            limit=20,
            lexical_counts=L,
        )
        assert len(res) > 0, f"no results for {g['qid']}: {g['q'][:40]}"
        assert res[0][1] > 0, f"zero score for {g['qid']}"
        print(f"  {g['qid']}: {len(res)} results, top_score={res[0][1]:.4f}")

    return search, gold


def test_evidence_unit_divergence(search, gold):
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    baseline_sfw = arms["baseline"].get("sfw")
    eu_sfw = arms["A2_eu_08"].get("sfw")
    changed = 0

    for g in gold:
        slots, toks = search.router.route(g["q"])
        _, L = search.match_table(slots, toks)

        base_res = search.rank_structured(slots, toks, g["q"], weights=baseline_sfw,
                                          profile="core", limit=20, lexical_counts=L)
        eu_res = search.rank_structured(slots, toks, g["q"], weights=eu_sfw,
                                        profile="core", limit=20, lexical_counts=L)

        base_ids = [e["element_id"] for e, _ in base_res[:5]]
        eu_ids = [e["element_id"] for e, _ in eu_res[:5]]
        if base_ids != eu_ids:
            changed += 1

    print(f"  evidence_unit divergence: {changed}/{len(gold)} questions changed top-5")
    if changed == 0:
        print("  WARNING: evidence_unit produced identical results - may need tuning")


def test_portfolio_mode(search, gold):
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm = arms["G1_portfolio_wide"]

    g = gold[0]
    slots, toks = search.router.route(g["q"])
    _, L = search.match_table(slots, toks)

    res = search.rank_structured_portfolio(
        slots, toks, g["q"],
        weights=arm.get("sfw"),
        profile="core",
        mode=arm["portfolio"],
        limit=40,
        window=int(arm.get("portfolio_window", 50)),
        seed_quota=int(arm.get("seed_quota", 20)),
        lexical_counts=L,
        rrf_k=int(arm.get("rrf_k", 50)),
        coverage_seed=int(arm.get("coverage_seed", 4)),
        coverage_per_role=int(arm.get("coverage_per_role", 2)),
    )
    assert len(res) > 0, f"portfolio returned empty for {g['qid']}"
    print(f"  portfolio (rrf_safe_axes): {len(res)} results for {g['qid']}")


def main():
    print("=== 0825 smoke test ===")
    errors = 0

    for name, fn in [
        ("arms config", test_arms_config),
        ("gold exists", test_gold_exists),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            errors += 1

    try:
        search, gold = test_search_engine()
    except Exception as e:
        print(f"  FAIL search engine: {e}")
        errors += 1
        search, gold = None, None

    if search and gold:
        for name, fn in [
            ("evidence_unit divergence", lambda: test_evidence_unit_divergence(search, gold)),
            ("portfolio mode", lambda: test_portfolio_mode(search, gold)),
        ]:
            try:
                fn()
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                errors += 1

    print(f"\n{'PASS' if errors == 0 else 'FAIL'}: {errors} error(s)")
    return errors


if __name__ == "__main__":
    sys.exit(main())
