#!/usr/bin/env python3
"""Claude 문서 선택 runner — 문항×Arm별 독립 claude 세션, 증분·재개·병렬.

사용: python3 run_claude_document_selection.py --split smoke|dev|test --arms A,B,C,D
      [--limit N] [--model haiku] [--workers 16]
출력: out/claude_docselect_<split>.jsonl (증분, (qid,arm) 중복 방지)
"""
import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
VIEWS = HERE / "views"
QA = OUT / "document_search_qa_500_candidate_v1.jsonl"
PROMPT_MD = (HERE / "claude_document_selector_prompt.md").read_text(encoding="utf-8")

FORBIDDEN = ("gold_document_id", "gold_source_path", "constraints",
             "evidence", "review_status")
VALID_STATUSES = {"selected", "not_found"}


def sections():
    parts = {}
    cur = None
    for line in PROMPT_MD.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            parts[cur] = []
        elif cur:
            parts[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


SEC = sections()


def build_prompt(qid, arm, query):
    q = SEC["QUESTION"].replace("{qid}", qid).replace("{arm}", arm).replace("{query}", query)
    return SEC["COMMON"] + "\n\n" + SEC[f"ARM_{arm}"] + "\n\n" + q


def run_one(item, arm, model):
    qid, query = item["qid"], item["query"]
    prompt = build_prompt(qid, arm, query)
    for f in FORBIDDEN:
        assert f not in prompt or f == "constraints" and False, f"입력 오염: {f}"
    t0 = time.time()
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json",
             "--max-turns", "40",
             "--allowedTools", "Glob", "Grep", "Read",
             "--disallowedTools", "Bash", "Write", "Edit", "WebSearch", "WebFetch"],
            input=prompt, capture_output=True, text=True, timeout=420,
            cwd=str(VIEWS / arm))
        out = json.loads(r.stdout)
        text = str(out.get("result", ""))
        turns = out.get("num_turns")
        cost = out.get("total_cost_usd")
    except Exception as e:
        return {"qid": qid, "arm": arm, "status": "error",
                "error": str(e)[:200], "duration_ms": int((time.time()-t0)*1000)}
    m = re.search(r"\{.*\}", text, re.S)
    rec = None
    if m:
        try:
            rec = json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    if not isinstance(rec, dict) or rec.get("status") not in VALID_STATUSES:
        rec = {"qid": qid, "arm": arm, "status": "error",
               "error": "schema_parse_fail_or_invalid_status", "raw": text[-300:]}
    rec.setdefault("qid", qid)
    rec["arm"] = arm
    rec["turns"] = turns
    rec["cost_usd"] = cost
    rec["model"] = model
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["smoke", "dev", "test", "mixed100"])
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    qa = [json.loads(l) for l in open(QA)]
    if args.split == "mixed100":
        mx = [q for q in qa if q["intent_type"] == "mixed"]
        items = [q for q in mx if q["split"] == "dev"] + [q for q in mx if q["split"] == "test"]
        items = items[:100]
    elif args.split == "smoke":
        items = []
        for t in ("identity", "content", "mixed"):
            items += [q for q in qa if q["intent_type"] == t and q["split"] == "dev"][:3]
    else:
        items = [q for q in qa if q["split"] == args.split]
    if args.limit:
        items = items[: args.limit]
    arms = args.arms.split(",")

    dest = OUT / f"claude_docselect_{args.split}.jsonl"
    done = set()
    if dest.exists():
        for l in open(dest):
            d = json.loads(l)
            if d.get("status") in VALID_STATUSES:
                done.add((d["qid"], d["arm"]))
    jobs = [(it, a) for it in items for a in arms if (it["qid"], a) not in done]
    print(f"{args.split}: 문항 {len(items)} × arms {arms} → 실행 {len(jobs)} "
          f"(기존 {len(done)}) | model={args.model} workers={args.workers}", flush=True)

    with open(dest, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, it, a, args.model): (it["qid"], a) for it, a in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"RUN_DONE {args.split}", flush=True)


if __name__ == "__main__":
    main()
