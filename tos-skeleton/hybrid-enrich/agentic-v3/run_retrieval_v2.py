#!/usr/bin/env python3
"""QA x Arm x Rep 별 독립 에이전트 세션 실행. `run_retrieval.py`의 백엔드 추상화 리팩터.

원본과 다른 점
    (a) subprocess를 직접 부르지 않고 `backends.get_backend()`가 반환하는
        AgentBackend(cli/http)에 위임한다. 결과에 토큰 usage가 실린다.
    (b) T0_BASE/T1_V6/T2_V9/T3_STRUCT 별칭 arm을 ARM_MAP으로 실제 ARM 값에 매핑한다.
        T3_STRUCT는 세션 env에 STRUCT_ENABLED=1을 얹어 tools.py의 구조화 검색
        서브커맨드(structured)를 쓸 수 있음을 표시한다 — 다만 ClaudeCliBackend가
        이미 `Bash(python tools.py:*)` 와일드카드를 --allowedTools로 넘기므로
        `structured` 서브커맨드는 STRUCT_ENABLED와 무관하게 항상 허용돼 있다.
        (구조화 검색을 실제로 쓰게 하려면 retrieval_prompt.md에 도구 설명을
        추가해야 한다 — 이 리팩터의 범위 밖.)
    (c) --reps로 같은 (qid,arm) 조합을 N회 반복하고 결과 파일명에 rep{N}을 새긴다.
    (d) --round-robin이면 qid(+rep) 해시로 시드를 고정해 arm 실행 순서를 섞는다
        (재현 가능하되 arm별로 앞/뒤 편중이 생기지 않도록).

문항: docs/QASet/레거시/v3_retrieval/qa_gold_v3.jsonl 에서 split=train & core_retrieval
출력: out/run_fixed600_{arm}_rep{N}_{tag}.jsonl
"""
import json, os, re, time, argparse, hashlib, random
from concurrent.futures import ThreadPoolExecutor, as_completed

from backends import get_backend

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
SESS = os.path.join(OUT, "sessions")
WIKI = os.path.abspath(os.path.join(BASE, "..", "..", ".."))
GOLD = os.path.join(WIKI, "docs", "QASet", "레거시", "v3_retrieval", "qa_gold_v3.jsonl")

ARMS = ["A0_BASE", "J2_JSONPURE", "A6_HIER2", "V7_DENSE",
        "V9_QSURF", "V10_ASKS", "V11_ALIAS"]

# T0..T3 별칭 → 실제 ARM env 값. tools.py는 이 실제 값으로 vec_*/view_* 파일을 고른다.
ARM_MAP = {
    "T0_BASE": "A0_BASE",
    "T1_V6": "J2_JSONPURE",
    "T2_V9": "V9_QSURF",
    "T3_STRUCT": "A0_BASE",  # dense는 base, 어휘 축은 structured_search가 대신한다
}

PROMPT_TMPL = open(os.path.join(BASE, "retrieval_prompt.md"), encoding="utf-8").read()
PROMPT_HASH = hashlib.sha256(PROMPT_TMPL.encode()).hexdigest()[:12]


def run_one(qid, question, arm, rep, backend):
    actual_arm = ARM_MAP.get(arm, arm)
    sdir = os.path.join(SESS, arm, qid, f"rep{rep}")
    os.makedirs(sdir, exist_ok=True)

    done_f = os.path.join(sdir, "result.json")
    if os.path.exists(done_f):
        try:
            return json.load(open(done_f, encoding="utf-8"))
        except Exception:
            os.remove(done_f)

    cf = os.path.join(sdir, "calls.jsonl")
    if os.path.exists(cf):
        os.remove(cf)

    env = dict(os.environ, ARM=actual_arm, SESSION_DIR=sdir)
    if arm == "T3_STRUCT":
        env["STRUCT_ENABLED"] = "1"

    prompt = PROMPT_TMPL.replace("{QUESTION}", question)
    t0 = time.time()
    text, usage = backend.run(prompt, sdir, env, BASE)

    m = re.findall(r"\{[^{}]*\"ranked_chunk_ids\"[\s\S]*?\}", text)
    try:
        rec = json.loads(m[-1]) if m else {"status": "error", "ranked_chunk_ids": [], "final_reason": "no json"}
    except Exception as e:
        rec = {"status": "error", "ranked_chunk_ids": [], "final_reason": f"parse error: {e}"[:200]}

    calls = []
    if os.path.exists(cf):
        calls = [json.loads(l) for l in open(cf, encoding="utf-8")]

    result = {"qid": qid, "arm": arm, "rep": rep, "status": rec.get("status", "error"),
              "ranked_chunk_ids": (rec.get("ranked_chunk_ids") or [])[:10],
              "final_reason": rec.get("final_reason", ""),
              "tool_calls": calls, "n_tool_calls": len(calls),
              "duration_ms": int((time.time() - t0) * 1000),
              "usage": usage, "prompt_hash": PROMPT_HASH}
    # run_retrieval.py와 같은 이유로 성공만 캐시한다: 일시적 실패까지 굳히면
    # 재실행해도 영원히 error로 남는다.
    if result["status"] != "error":
        json.dump(result, open(done_f, "w", encoding="utf-8"), ensure_ascii=False)
    return result


def load_questions(split, core_only):
    rows = []
    for l in open(GOLD, encoding="utf-8"):
        r = json.loads(l)
        if r["split"] != split:
            continue
        if core_only and not r.get("core_retrieval"):
            continue
        rows.append((r["qid"], r["question"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--all-tasks", action="store_true", help="core_retrieval 필터 해제")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="앞 N문항만")
    ap.add_argument("--tag", default="agentic_v3tr")
    ap.add_argument("--backend", default="cli", choices=["cli", "http"])
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--reps", type=int, default=2, help="같은 (qid,arm) 반복 횟수")
    ap.add_argument("--round-robin", action="store_true",
                     help="qid별 arm 실행 순서를 qid 해시로 시드 고정해 섞는다")
    a = ap.parse_args()

    qs = load_questions(a.split, not a.all_tasks)
    if a.limit:
        qs = qs[:a.limit]
    arms = [x for x in a.arms.split(",") if x]
    known = set(ARMS) | set(ARM_MAP)
    for arm in arms:
        assert arm in known, f"unknown arm {arm}"

    backend = get_backend(a.backend, model=a.model)

    jobs = []
    for rep in range(1, a.reps + 1):
        for qid, question in qs:
            order = list(arms)
            if a.round_robin:
                seed = int(hashlib.md5(f"{qid}:{rep}".encode()).hexdigest(), 16)
                random.Random(seed).shuffle(order)
            jobs.extend((qid, question, arm, rep) for arm in order)

    os.makedirs(SESS, exist_ok=True)
    print(f"문항 {len(qs)} x arm {len(arms)} x rep {a.reps} = {len(jobs)}세션 · "
          f"workers={a.workers} · backend={a.backend}/{a.model}", flush=True)

    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, q, ques, arm, rep, backend): (q, arm, rep)
                for q, ques, arm, rep in jobs}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            results.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                el = int(time.time() - t0)
                err = sum(1 for x in results if x["status"] == "error")
                eta = int(el / (i + 1) * (len(jobs) - i - 1))
                print(f"{i+1}/{len(jobs)} elapsed={el}s eta={eta}s errors={err}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for arm in arms:
        for rep in range(1, a.reps + 1):
            rows = sorted([r for r in results if r["arm"] == arm and r["rep"] == rep],
                          key=lambda x: x["qid"])
            outf = os.path.join(OUT, f"run_fixed600_{arm}_rep{rep}_{a.tag}.jsonl")
            with open(outf, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("saved", outf, len(rows))


if __name__ == "__main__":
    main()
