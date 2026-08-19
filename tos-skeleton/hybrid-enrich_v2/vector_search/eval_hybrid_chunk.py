#!/usr/bin/env python3
"""결정론 평가: 청크 기반 하이브리드 검색기(BM25+Dense RRF)를 gold span에 대해 채점.

filesearch/scoring.py의 score(), overlaps()를 재사용한다.
arm: bm25:V9, dense:V9, rrf:V9, bm25:BASE 등 (세미콜론 구분)

Usage:
    python eval_hybrid_chunk.py --gold ../filesearch/out/gold_spans_lsh_train.jsonl
    python eval_hybrid_chunk.py --gold ../filesearch/out/gold_spans_lsh_train.jsonl --arms "bm25:V9;dense:V9;rrf:V9"
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILESEARCH = HERE.parent / "filesearch"
sys.path.insert(0, str(FILESEARCH))
sys.path.insert(0, str(HERE))

from scoring import score, overlaps
from hybrid_search import ChunkHybridSearch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_arm(s: str) -> tuple[str, str]:
    parts = s.split(":")
    strategy = parts[0]
    view = parts[1] if len(parts) > 1 else "V9"
    return strategy, view


def main():
    ap = argparse.ArgumentParser(description="청크 하이브리드 검색기 결정론 평가")
    ap.add_argument("--gold", required=True, help="gold spans jsonl 경로")
    ap.add_argument("--arms", default="bm25:V9;dense:V9;rrf:V9;bm25:BASE",
                    help="평가 arm (세미콜론 구분, strategy:view)")
    ap.add_argument("--limit", type=int, default=200, help="검색 결과 상한")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"[오류] gold 파일이 없습니다: {gold_path}")

    G = [g for g in read_jsonl(gold_path) if g.get("groups")]
    print(f"n={len(G)}")

    arms = [x.strip() for x in args.arms.split(";") if x.strip()]
    parsed = [parse_arm(a) for a in arms]

    searchers: dict[str, ChunkHybridSearch] = {}
    for _, view in parsed:
        if view not in searchers:
            searchers[view] = ChunkHybridSearch(view=view)

    chunks_by_id: dict[str, dict] = {}
    for hs in searchers.values():
        chunks_by_id.update(hs.chunks)

    ks = (1, 5, 10, 20, 40, 100)
    hdr = [f"R@{k}" for k in ks] + ["S@5", "suff@5", "suff@10", "RR@10"]
    print("\n| arm | " + " | ".join(hdr) + " | 후보0 |")
    print("|---|" + "---|" * (len(hdr) + 1))

    rank_files = {}
    for arm in arms:
        safe_name = arm.replace(":", "_").replace("=", "-")
        rank_files[arm] = open(HERE / "out" / f"chunk_ranks_{safe_name}.jsonl", "w", encoding="utf-8")

    for arm, (strategy, view) in zip(arms, parsed):
        hs = searchers[view]
        agg: dict[str, list[float]] = collections.defaultdict(list)
        zero = 0

        for g in G:
            q = g["q"]

            if strategy == "bm25":
                result_ids = hs.bm25_search(q, args.limit)
            elif strategy == "dense":
                result_ids = [uid for uid, _ in hs.dense_search(q, args.limit)]
            else:
                results = hs.hybrid_search(q, top_k=args.limit)
                result_ids = [r["id"] for r in results]

            ranked_units = []
            for uid in result_ids:
                c = chunks_by_id.get(uid)
                if c:
                    ranked_units.append(c)

            if not ranked_units:
                zero += 1

            sc = score(ranked_units, g["groups"], ks=ks)
            for k, v in sc.items():
                agg[k].append(v)

            rank_files[arm].write(json.dumps({
                "qid": g.get("qid", ""),
                "q": q,
                "n_cand": len(ranked_units),
                "top": [uid for uid in result_ids[:40]],
            }, ensure_ascii=False) + "\n")

        n = len(G)
        row_vals = []
        for k in ks:
            row_vals.append(f"{sum(agg[f'R@{k}']) / n:.3f}")
        row_vals.append(f"{sum(agg['S@5']) / n:.3f}")
        row_vals.append(f"{sum(agg['suff@5']) / n:.3f}")
        row_vals.append(f"{sum(agg['suff@10']) / n:.3f}")
        row_vals.append(f"{sum(agg['RR@10']) / n:.3f}")

        print(f"| {arm} | " + " | ".join(row_vals) + f" | {zero} |")

    for f in rank_files.values():
        f.close()

    print(f"\n순위 파일: {HERE / 'out' / 'chunk_ranks_*.jsonl'}")


if __name__ == "__main__":
    main()
