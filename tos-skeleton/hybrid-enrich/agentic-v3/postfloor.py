#!/usr/bin/env python3
"""F-FLOOR: post-hoc RRF safety net. No additional LLM calls.

Takes agent results (ranked_chunk_ids) and fills remaining positions with
deterministic RRF results. Agent's top-K are preserved, rest filled from RRF
with dedup.
"""
import argparse, json, os, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "hybrid-v7"))
from fuse import displacement


def floor_rescore(agent_result: dict, rrf_result: dict, k_preserve: int = 5,
                  final_topk: int = 10) -> dict:
    """Merge agent top-K with RRF fill."""
    agent_ids = agent_result.get("ranked_chunk_ids", [])[:k_preserve]
    rrf_ids = rrf_result.get("ranked_chunk_ids", [])

    seen = set(agent_ids)
    filled = list(agent_ids)
    for cid in rrf_ids:
        if len(filled) >= final_topk:
            break
        if cid not in seen:
            filled.append(cid)
            seen.add(cid)

    return {
        "qid": agent_result["qid"],
        "ranked_chunk_ids": filled[:final_topk],
        "status": "ok",
        "arm": f"{agent_result.get('arm', 'unknown')}-floor{k_preserve}",
        "agent_contributed": len(agent_ids),
        "rrf_filled": len(filled) - len(agent_ids),
    }


def load_gold_cids(gold_file: str, chunkset_key: str) -> dict:
    """qid -> set(chunk_id). chunkset_mappings[chunkset]는 group dict의 flat
    list이고, 각 group은 'chunk_ids'(flat list)를 갖는다 (eval_v3_table.py의
    gold_first_chunk와 동일한 스키마 가정 — evidence_groups 중첩 구조가 아니다).
    """
    gold_by_qid = {}
    for line in open(gold_file, encoding="utf-8"):
        r = json.loads(line)
        groups = r.get("chunkset_mappings", {}).get(chunkset_key, [])
        gold_cids = set()
        for group in groups:
            gold_cids.update(group.get("chunk_ids", []))
        if gold_cids:
            gold_by_qid[r["qid"]] = gold_cids
    return gold_by_qid


def run_floor(agent_file: str, rrf_file: str, k_preserve: int = 5,
              gold_file: str | None = None, chunkset_key: str = "fixed600") -> list[dict]:
    """Apply FLOOR rescoring to all questions."""
    # Load agent results keyed by qid
    agent_by_qid = {}
    for line in open(agent_file, encoding="utf-8"):
        r = json.loads(line)
        agent_by_qid[r["qid"]] = r

    # Load RRF results keyed by qid
    rrf_by_qid = {}
    for line in open(rrf_file, encoding="utf-8"):
        r = json.loads(line)
        rrf_by_qid[r["qid"]] = r

    # Load gold if provided (for displacement diagnostic)
    gold_by_qid = {}
    if gold_file and os.path.exists(gold_file):
        gold_by_qid = load_gold_cids(gold_file, chunkset_key)

    results = []
    diag = {"gained": 0, "displaced": 0, "base_hit": 0, "fused_hit": 0, "n": 0}

    for qid, agent_r in agent_by_qid.items():
        rrf_r = rrf_by_qid.get(qid)
        if not rrf_r:
            results.append(agent_r)
            continue

        floored = floor_rescore(agent_r, rrf_r, k_preserve)
        results.append(floored)

        if qid in gold_by_qid:
            gold = gold_by_qid[qid]
            d = displacement(agent_r.get("ranked_chunk_ids", []),
                           floored["ranked_chunk_ids"], gold, k=10)
            for key in diag:
                if key != "n":
                    diag[key] += d[key]
            diag["n"] += 1

    return results, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="Agent result JSONL")
    ap.add_argument("--rrf", required=True, help="C0-RRF-DET result JSONL")
    ap.add_argument("--k-preserve", type=int, nargs="+", default=[3, 5, 10],
                    help="K values to sweep (agent slots to preserve)")
    ap.add_argument("--gold", default=None, help="Gold JSONL for displacement diagnostic")
    ap.add_argument("--chunkset", default="fixed600")
    ap.add_argument("--output-dir", default=str(BASE / "out"))
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    agent_name = Path(a.agent).stem

    for k in a.k_preserve:
        results, diag = run_floor(a.agent, a.rrf, k, a.gold, a.chunkset)
        outf = os.path.join(a.output_dir, f"{agent_name}_floor{k}.jsonl")
        with open(outf, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"K={k}: saved {len(results)} → {outf}", flush=True)
        if diag["n"]:
            print(f"  displacement(n={diag['n']}): gained={diag['gained']} displaced={diag['displaced']} "
                  f"base_hit={diag['base_hit']} fused_hit={diag['fused_hit']}", flush=True)


if __name__ == "__main__":
    main()
