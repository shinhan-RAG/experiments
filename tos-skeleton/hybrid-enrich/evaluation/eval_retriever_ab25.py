#!/usr/bin/env python3
"""Evaluate OLD_RET vs NEW_RET on the frozen first 25."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
ARMS = ("OLD_RET", "NEW_RET")


def load_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def first_rank(ids, gold):
    return next((rank for rank, uid in enumerate(ids, 1) if uid in gold), None)


def metrics(ranks):
    n = len(ranks)
    return {
        "n": n,
        "recall@1": sum(rank == 1 for rank in ranks) / n,
        "recall@5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks) / n,
        "mrr@10": sum(1 / rank if rank and rank <= 10 else 0 for rank in ranks) / n,
    }


def exposed_ids(row, tool):
    found = []
    for call in row.get("tool_calls", []):
        if call.get("tool") != tool:
            continue
        result = call.get("result") or {}
        rows = result if isinstance(result, list) else result.get("results", [])
        found.extend(item.get("id") for item in rows if item.get("id"))
    return found


def main():
    manifest = json.loads((OUT / "retriever_ab25_manifest.json").read_text())
    gold_all = {row["qid"]: row for row in load_jsonl("train350_gold_repaired_v1.jsonl")}
    rows = load_jsonl("retrieval_retriever_ab25.jsonl")
    by = {(row["qid"], row["arm"]): row for row in rows}
    result = {"scope": {"qids": manifest["qids"], "arms": list(ARMS)}, "arms": {}, "paired": {}}
    ranks_by = {}
    details = []
    for arm in ARMS:
        combined_ranks = []
        chunk_ranks = []
        element_ranks = []
        vector_exposure = file_exposure = any_exposure = 0
        errors = invalid_final = 0
        arm_ranks = {}
        for qid in manifest["qids"]:
            row = by.get((qid, arm), {"status": "error", "ranked_chunk_ids": []})
            gold = gold_all[qid]
            gc, ge = set(gold["gold_chunk_ids"]), set(gold["gold_element_ids"])
            gu = gc | ge
            final = row.get("ranked_chunk_ids", [])[:10]
            vr = exposed_ids(row, "vector_search")
            fr = exposed_ids(row, "file_search")
            exposed = set(vr) | set(fr)
            cr, er, ur = first_rank(final, gc), first_rank(final, ge), first_rank(final, gu)
            chunk_ranks.append(cr); element_ranks.append(er); combined_ranks.append(ur); arm_ranks[qid] = ur
            vector_exposure += bool(set(vr) & gc)
            file_exposure += bool(set(fr) & ge)
            any_exposure += bool(exposed & gu)
            errors += row.get("status") == "error"
            invalid = [uid for uid in final if uid not in exposed]
            invalid_final += bool(invalid)
            details.append({"qid": qid, "arm": arm, "combined_rank": ur, "chunk_rank": cr,
                            "element_rank": er, "vector_gold_exposed": bool(set(vr) & gc),
                            "file_gold_exposed": bool(set(fr) & ge), "invalid_final_ids": invalid})
        ranks_by[arm] = arm_ranks
        result["arms"][arm] = {
            "combined": metrics(combined_ranks), "chunk": metrics(chunk_ranks),
            "element": metrics(element_ranks), "vector_gold_exposure": vector_exposure / 25,
            "file_gold_exposure": file_exposure / 25, "any_gold_exposure": any_exposure / 25,
            "errors": errors, "sessions_with_unexposed_final_id": invalid_final,
        }
    old, new = ranks_by["OLD_RET"], ranks_by["NEW_RET"]
    improved = [qid for qid in manifest["qids"] if (new[qid] or 999) < (old[qid] or 999)]
    regressed = [qid for qid in manifest["qids"] if (new[qid] or 999) > (old[qid] or 999)]
    old_r5 = result["arms"]["OLD_RET"]["combined"]["recall@5"]
    new_r5 = result["arms"]["NEW_RET"]["combined"]["recall@5"]
    result["paired"] = {"new_minus_old_recall@5": new_r5 - old_r5,
                        "improved_qids": improved, "regressed_qids": regressed}
    (OUT / "eval_retriever_ab25.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "eval_retriever_ab25_details.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
