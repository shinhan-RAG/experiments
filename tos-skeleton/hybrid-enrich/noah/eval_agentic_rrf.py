#!/usr/bin/env python3
"""RRF 에이전틱 검색 평가. Recall@K, MRR@10, McNemar, bootstrap CI, 도구 사용 프로파일."""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def first_rank(row: dict, gold: dict, spans: dict) -> int | None:
    for rank, uid in enumerate(row.get("ranked_chunk_ids") or [], 1):
        sp = spans.get(uid)
        if sp and any(sp[0] < end and sp[1] > start for start, end in gold["gold_spans"]):
            return rank
    return None


def mcnemar_exact(wins: int, losses: int) -> float:
    from math import comb
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) * 0.5 ** n for i in range(k + 1)) * 2
    return min(p, 1.0)


def bootstrap_diff(a_ranks: list, b_ranks: list, k: int = 5, n_boot: int = 10000) -> dict:
    n = len(a_ranks)
    diffs = []
    rng = random.Random(42)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_hit = sum(1 for i in idx if a_ranks[i] is not None and a_ranks[i] <= k) / n
        b_hit = sum(1 for i in idx if b_ranks[i] is not None and b_ranks[i] <= k) / n
        diffs.append(a_hit - b_hit)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    mean = sum(diffs) / len(diffs)
    return {"mean_diff": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


def resolve_path(raw: str) -> Path:
    """상대경로는 데이터 루트(hybrid-enrich) 기준으로 해석한다.

    코드는 noah/ 에 있고 실행도 noah/ 에서 하므로, `out/gold_train.jsonl` 을
    그대로 넘기면 noah/out/ 을 찾아 실패한다. 절대경로와 실제로 존재하는
    상대경로는 그대로 두고, 그 외에만 BASE 를 앞에 붙인다."""
    q = Path(raw)
    if q.is_absolute() or q.exists():
        return q
    return BASE / raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT / "agentic_rrf_results.jsonl"))
    ap.add_argument("--gold", default=str(OUT / "gold_train.jsonl"))
    ap.add_argument("--core-only", action="store_true", default=False)
    args = ap.parse_args()

    results = load_jsonl(resolve_path(args.results))
    gold_all = load_jsonl(resolve_path(args.gold))
    # 골드가 raw 좌표가 아니면 채점 자체가 무의미하다 (정규화 좌표와 최대 56,667자 어긋남).
    bad = [g["qid"] for g in gold_all if g.get("coord_space") != "raw"]
    if bad:
        raise SystemExit(f"[FATAL] 골드 좌표계가 raw 가 아니다 ({len(bad)}건, 예: {bad[:3]}). "
                         f"build_unified_gold.py 를 다시 실행하라.")
    if args.core_only:
        gold_all = [g for g in gold_all if g.get("core_retrieval")]
    gold_all = [g for g in gold_all if g.get("gold_spans")]
    gold_by = {g["qid"]: g for g in gold_all}

    # 스팬은 전부 이 폴더 out/ 안의 산출물에서만 읽는다 (raw 좌표).
    # 엘리먼트는 P섹션 코퍼스 하나만. 구 elements*.jsonl 은 동일 ID에 다른 스팬을
    # 써서 함께 로드하면 서로 덮어쓴다 (run_agentic_rrf.load_spans 주석 참고).
    spans: dict[str, tuple[int, int]] = {}
    for name, key in (("chunks.jsonl", "chunk_id"),
                      ("elements_psection.jsonl", "element_id")):
        p = OUT / name
        if p.exists():
            for row in load_jsonl(p):
                spans[row[key]] = (row["char_start"], row["char_end"])

    arms = sorted(set(r["arm"] for r in results))
    arm_data: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    for r in results:
        arm_data[r["arm"]][r["qid"]] = r

    print(f"=== 평가 결과 (n={len(gold_by)}, core_only={args.core_only}) ===\n")
    print(f"{'Arm':<12} {'n':>4} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>7} {'err':>4} {'calls':>5}")

    arm_ranks: dict[str, list] = {}
    for arm in arms:
        rows = arm_data[arm]
        ranks = []
        for qid, gold in gold_by.items():
            r = rows.get(qid, {})
            ranks.append(first_rank(r, gold, spans))
        arm_ranks[arm] = ranks
        n = max(len(ranks), 1)
        r1 = sum(r == 1 for r in ranks) / n
        r5 = sum(r is not None and r <= 5 for r in ranks) / n
        r10 = sum(r is not None and r <= 10 for r in ranks) / n
        mrr = sum(1 / r if r and r <= 10 else 0 for r in ranks) / n
        errs = sum(rows.get(qid, {}).get("status") == "error" for qid in gold_by)
        avg_calls = sum(rows.get(qid, {}).get("n_tool_calls", 0) for qid in gold_by) / n
        print(f"{arm:<12} {len(ranks):>4} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f} {mrr:>7.4f} {errs:>4} {avg_calls:>5.1f}")

    if len(arms) >= 2:
        print(f"\n=== 쌍별 비교 (R@5) ===")
        base = arms[0]
        for other in arms[1:]:
            qids = list(gold_by.keys())
            a_ranks = arm_ranks[base]
            b_ranks = arm_ranks[other]
            wins = sum(1 for a, b in zip(a_ranks, b_ranks)
                       if (b is not None and b <= 5) and not (a is not None and a <= 5))
            losses = sum(1 for a, b in zip(a_ranks, b_ranks)
                         if (a is not None and a <= 5) and not (b is not None and b <= 5))
            p = mcnemar_exact(wins, losses)
            boot = bootstrap_diff(b_ranks, a_ranks, k=5)
            print(f"  {other} vs {base}: win/lose={wins}/{losses} McNemar p={p:.4f} "
                  f"bootstrap_diff={boot['mean_diff']:+.4f} CI[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")

    print(f"\n=== 도구 사용 프로파일 ===")
    for arm in arms:
        rows = arm_data[arm]
        tool_counts: dict[str, int] = collections.Counter()
        total = 0
        for qid in gold_by:
            r = rows.get(qid, {})
            for tc in r.get("tool_calls", []):
                tool_counts[tc["tool"]] += 1
                total += 1
        n = max(len([q for q in gold_by if q in rows]), 1)
        parts = [f"{t}={c / n:.1f}" for t, c in sorted(tool_counts.items())]
        print(f"  {arm:<12} avg_total={total / n:.1f}  {' '.join(parts)}")

    out_path = resolve_path(args.results).with_suffix(".eval.json")
    eval_result = {"n": len(gold_by), "arms": {}}
    for arm in arms:
        ranks = arm_ranks[arm]
        n = max(len(ranks), 1)
        eval_result["arms"][arm] = {
            "R@1": round(sum(r == 1 for r in ranks) / n, 4),
            "R@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 4),
            "R@10": round(sum(r is not None and r <= 10 for r in ranks) / n, 4),
            "MRR@10": round(sum(1 / r if r and r <= 10 else 0 for r in ranks) / n, 4),
        }
    out_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n평가 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
