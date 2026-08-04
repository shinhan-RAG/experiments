#!/usr/bin/env python3
"""검증+평가 — 스키마·격리 검증 후 gold와 결합해 Arm별 Accuracy@1.

사용: python3 eval_claude_document_selection.py --split smoke|dev|test
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
QA = OUT / "document_search_qa_500_candidate_v1.jsonl"


def validate(rec):
    errs = []
    if rec.get("status") not in ("selected", "not_found", "error"):
        errs.append("bad_status")
    arm = rec.get("arm")
    fc = rec.get("frontmatter_checks") or []
    ib = rec.get("index_batches") or []
    if arm == "B" and fc:
        errs.append("B_fm_access")
    if arm == "C" and ib:
        errs.append("C_index_access")
    if arm == "D":
        batch_ids = set()
        for batch in ib:
            if isinstance(batch, list):
                batch_ids.update(doc_id for doc_id in batch if isinstance(doc_id, str))
            elif isinstance(batch, dict):
                batch_ids.update(doc_id for doc_id in batch.get("candidate_document_ids", [])
                                 if isinstance(doc_id, str))
            else:
                errs.append("D_bad_index_batch")
        checked = {c.get("document_id") for c in fc}
        if checked - batch_ids:
            errs.append("D_fm_outside_batch")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(QA)}
    rows = [json.loads(l) for l in open(OUT / f"claude_docselect_{args.split}.jsonl")]
    # (qid,arm) 최신 것만
    latest = {}
    for r in rows:
        latest[(r["qid"], r["arm"])] = r
    rows = list(latest.values())

    res = defaultdict(lambda: defaultdict(Counter))
    viol = Counter()
    for r in rows:
        errs = validate(r)
        for e in errs:
            viol[e] += 1
        g = gold.get(r["qid"])
        if not g:
            continue
        intent = g["intent_type"]
        arm = r["arm"]
        for scope in ("전체", intent):
            c = res[scope][arm]
            c["n"] += 1
            if r.get("status") == "error" or errs:
                c["error"] += 1
            elif r.get("status") == "not_found":
                c["not_found"] += 1
            elif r.get("selected_document_id") == g["gold_document_id"]:
                c["acc1"] += 1
        if arm == "D" and r.get("index_batches"):
            res["_D"]["stats"]["batches"] += len(r["index_batches"])
            res["_D"]["stats"]["fm_reads"] += len(r.get("frontmatter_checks") or [])
            res["_D"]["stats"]["runs"] += 1

    arms = sorted({r["arm"] for r in rows})
    print(f"{'구분':10s}" + "".join(f"{a:>14s}" for a in arms))
    for scope in ("전체", "identity", "content", "mixed"):
        d = res.get(scope)
        if not d:
            continue
        line = f"{scope:10s}"
        for a in arms:
            c = d[a]
            n = c["n"] or 1
            line += f"  {c['acc1']/n:5.2f}(nf{c['not_found']},e{c['error']})"
        print(line + f"  (n={res[scope][arms[0]]['n']})")
    s = res["_D"]["stats"]
    if s["runs"]:
        print(f"\nD: 평균 배치 {s['batches']/s['runs']:.1f}, 평균 FM 확인 {s['fm_reads']/s['runs']:.1f}")
    print(f"격리/스키마 위반: {dict(viol) or '0건'}")
    turns = [r.get("turns") for r in rows if r.get("turns")]
    dur = [r.get("duration_ms") for r in rows if r.get("duration_ms")]
    if turns:
        print(f"평균 turns {sum(turns)/len(turns):.1f} | 평균 {sum(dur)/len(dur)/1000:.0f}s | "
              f"총비용 ${sum(r.get('cost_usd') or 0 for r in rows):.2f}")
    payload = {
        "split": args.split,
        "n_records": len(rows),
        "violations": dict(viol),
        "metrics": {sc: {a: dict(res[sc][a]) for a in res[sc]}
                    for sc in res if sc != "_D"},
        "d_stats": dict(res["_D"]["stats"]),
    }
    with open(OUT / f"claude_docselect_eval_{args.split}.json", "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
