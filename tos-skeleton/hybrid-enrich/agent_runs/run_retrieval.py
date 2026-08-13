#!/usr/bin/env python3
"""QA x Arm 별 독립 claude -p 세션 실행. 증분 저장, (qid,arm) 중복 방지, 병렬."""
import json, os, re, subprocess, sys, time, argparse, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
SESS = os.path.join(OUT, "sessions")
ARMS = ["BASE", "META", "TAG", "BOTH"]
MODEL = "claude-sonnet-5"
PROMPT_TMPL = open(os.path.join(BASE, "docs", "retrieval_prompt.md"), encoding="utf-8").read()
PROMPT_HASH = hashlib.sha256(PROMPT_TMPL.encode()).hexdigest()[:12]


def run_one(qid, question, arm, split):
    sdir = os.path.join(SESS, f"{split}_{qid}_{arm}")
    os.makedirs(sdir, exist_ok=True)
    done_f = os.path.join(sdir, "result.json")
    if os.path.exists(done_f):
        return json.load(open(done_f))
    for f in ("calls.jsonl",):
        p = os.path.join(sdir, f)
        if os.path.exists(p):
            os.remove(p)
    prompt = PROMPT_TMPL.replace("{QUESTION}", question)
    env = dict(os.environ, ARM=arm, SESSION_DIR=sdir)
    t0 = time.time()
    try:
        p = subprocess.run(
            ["claude", "-p", prompt, "--model", MODEL,
             "--allowedTools", "Bash(python3 tools.py:*)",
             "--disallowedTools", "Read,Write,Edit,Grep,Glob,WebFetch,WebSearch,Agent,Task",
             "--output-format", "json", "--max-turns", "24"],
            capture_output=True, text=True, timeout=420, cwd=BASE, env=env)
        raw = p.stdout
        try:
            outer = json.loads(raw)
            text = outer.get("result", "")
        except Exception:
            text = raw
        m = re.findall(r"\{[^{}]*\"ranked_chunk_ids\"[\s\S]*?\}", text)
        rec = json.loads(m[-1]) if m else {"status": "error", "ranked_chunk_ids": [], "final_reason": "no json"}
    except subprocess.TimeoutExpired:
        rec = {"status": "error", "ranked_chunk_ids": [], "final_reason": "timeout"}
    except Exception as e:
        rec = {"status": "error", "ranked_chunk_ids": [], "final_reason": str(e)[:200]}
    calls = []
    cf = os.path.join(sdir, "calls.jsonl")
    if os.path.exists(cf):
        calls = [json.loads(l) for l in open(cf)]
    result = {"qid": qid, "arm": arm, "status": rec.get("status", "error"),
              "ranked_chunk_ids": rec.get("ranked_chunk_ids", [])[:10],
              "final_reason": rec.get("final_reason", ""),
              "tool_calls": calls, "n_tool_calls": len(calls),
              "duration_ms": int((time.time() - t0) * 1000),
              "model": MODEL, "prompt_hash": PROMPT_HASH}
    json.dump(result, open(done_f, "w"), ensure_ascii=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["smoke", "main", "v2"], required=True)
    ap.add_argument("--arms", default="BASE,META,TAG,BOTH")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="샘플 앞 N문항만")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(OUT, "split_manifest.json")))
    qids = manifest["smoke"] if a.split == "smoke" else manifest["sample"]  # main/v2 = 동일 100문항
    if a.limit:
        qids = qids[:a.limit]
    gold = {r["qid"]: r for r in (json.loads(l) for l in open(os.path.join(OUT, "qa100_gold.jsonl")))}
    jobs = [(q, gold[q]["question"], arm) for q in qids for arm in a.arms.split(",")]
    os.makedirs(SESS, exist_ok=True)
    outf = os.path.join(OUT, f"retrieval_{a.split}.jsonl")
    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, q, ques, arm, a.split): (q, arm) for q, ques, arm in jobs}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"{i+1}/{len(jobs)} elapsed={int(time.time()-t0)}s errors={sum(1 for x in results if x['status']=='error')}", flush=True)
    with open(outf, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: (x["qid"], x["arm"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", outf)


if __name__ == "__main__":
    main()
