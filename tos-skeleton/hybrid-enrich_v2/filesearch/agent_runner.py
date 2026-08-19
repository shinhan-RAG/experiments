#!/usr/bin/env python3
"""에이전트 실측 러너 — claude -p + agent_tools.py(search/read/submit), S5 예산, ITT 채점, reps, resume.

결과: out/agent/<run>/sessions/<qid>_r<rep>/{calls.jsonl,submit.json,claude.json}, out/agent/<run>/results.jsonl
채점: submit 순위 → 조(u2jo) map-back → fractional evidence-group R@K, Success@K, MRR@10. 미제출/오류 = 0점(ITT).
"""
import argparse, collections, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import score
from units import Units

VSEARCH_DOC = """1b) 메타데이터 검색: python3 agent_tools.py msearch --q "<자연어 질의>" [--page N]
   - 의미 유사도(임베딩)+키워드 결합 검색. 질문 표현이 문서 용어와 달라도 찾습니다. 검색 횟수는 search 와 합쳐 세션당 20회.
   - 태그 검색(search)과 메타데이터 검색(msearch)을 질문 성격에 따라 골라 쓰거나 둘 다 써서 교차 확인하십시오. 결과 id(c…)도 그대로 제출할 수 있습니다.
"""
PROMPT = """당신은 보험 약관 문서 안에서 질문의 근거 조항(element)을 찾는 검색 에이전트입니다.
도구는 아래 셸 명령 3개뿐입니다(다른 파일·명령 사용 금지). 반드시 `python3 agent_tools.py ...` 형태로 호출하십시오.

1) 검색: python3 agent_tools.py search --q "<검색어>" [--page N] [--contract "<특약명>"] [--role "<역할코드>"] [--subject "<대상어>"]
   - 한 페이지 40건(총 후보 수와 남은 호출 수가 함께 나옵니다). 세션당 최대 20회.
   - 검색어는 질문에서 핵심어(특약명·급여금명·질병명·조건)를 골라 쓰고, 결과가 엉뚱하면 다른 표현으로 다시 검색하십시오.
   - 역할코드: exclusion_exception, premium_waiver, payment_trigger, payment_amount, limit_frequency, timing_period, definition, criteria_rule, contract_lifecycle, claim_procedure, code_reference
{vsearch}2) 읽기: python3 agent_tools.py read --id <element_id>   — 해당 조(條) 전체 원문. 세션당 최대 8회. 확신이 서지 않는 후보를 확인할 때 쓰십시오.
3) 제출: python3 agent_tools.py submit --ids <id1>,<id2>,...   — 근거일 가능성이 높은 순서로 **최대 10개**. 반드시 1회 호출하고, 그 뒤에는 어떤 도구도 부르지 마십시오.

원칙: 질문이 특정 특약을 가리키면 그 특약의 element 를 우선하십시오. 질문이 여러 근거(예: 정의 + 지급조건 + 청구절차)를 요구하면 서로 다른 조의 element 를 섞어 제출하십시오.
검색 결과에 facets(특약·조 분포)가 있으면 그것을 보고 범위를 좁혀(--contract, --role) 다시 검색하거나 다른 페이지를 보십시오. 제출은 가능하면 10개를 채우되 같은 조를 중복해 넣지 마십시오. 답변 문장은 쓰지 말고 제출만 하십시오.

[질문]
{question}
"""


def run_one(args, run_dir, g, rep, arm):
    sess = run_dir / "sessions" / f"{g['qid']}_r{rep}"
    done = sess / "claude.json"
    if done.exists() and (sess / "submit.json").exists():
        return json.load(open(sess / "submit.json")), json.load(open(done))
    sess.mkdir(parents=True, exist_ok=True)
    for f in ("calls.jsonl", "submit.json"):
        if (sess / f).exists():
            (sess / f).unlink()
    env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=g["qid"], SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    cmd = ["claude", "-p", "--model", args.model, "--output-format", "json", "--max-turns", str(args.max_turns),
           "--allowedTools", "Bash(python3 agent_tools.py:*)",
           "--disallowedTools", "Read,Edit,Write,Grep,Glob,WebFetch,WebSearch,Agent,NotebookEdit,Task"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=PROMPT.format(question=g["q"], vsearch=VSEARCH_DOC if arm.get("meta") else ""), capture_output=True, text=True, timeout=args.timeout, cwd=str(HERE), env=env)
        try:
            cj = json.loads(r.stdout)
        except Exception:
            cj = {"parse_error": True, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        cj = {"timeout": True}
    cj["_elapsed"] = time.time() - t0
    json.dump(cj, open(done, "w"), ensure_ascii=False)
    sub = json.load(open(sess / "submit.json")) if (sess / "submit.json").exists() else {"qid": g["qid"], "ranked": [], "no_submit": True}
    return sub, cj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 이름 (out/agent/<run>)")
    ap.add_argument("--arm", required=True, help="ARM json")
    ap.add_argument("--gold", default=str(HERE / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--jo", default=str(HERE / "out/elements_u2jo.jsonl"))
    ap.add_argument("--elements", default=str(HERE / "out/elements_u2.jsonl"))
    ap.add_argument("--n", type=int, default=-1, help="문항 수(층화 표본), -1=전수")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    arm = json.loads(a.arm)
    run_dir = HERE / "out" / "agent" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": arm, "args": vars(a)}, open(run_dir / "config.json", "w"), ensure_ascii=False, indent=1)
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"]
    if a.n > 0:  # core/비core 층화 표본(고정 시드)
        rnd = random.Random(a.seed)
        core = [g for g in G if g["core_retrieval"] == "True"]; nc = [g for g in G if g["core_retrieval"] != "True"]
        k_core = round(a.n * len(core) / len(G))
        G = rnd.sample(core, k_core) + rnd.sample(nc, a.n - k_core)
    U = Units(a.jo)
    jobs = [(g, rep) for rep in range(a.reps) for g in G]
    print(f"run={a.run} n={len(G)} reps={a.reps} jobs={len(jobs)} model={a.model}", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for (g, rep), (sub, cj) in zip(jobs, ex.map(lambda gr: run_one(a, run_dir, gr[0], gr[1], arm), jobs)):
            ranked_jo = U.resolve(sub.get("ranked", []))
            sc = score(ranked_jo, g["groups"], ks=(1, 5, 10, 20))
            row = {"qid": g["qid"], "rep": rep, "core": g["core_retrieval"] == "True", "task_type": g["task_type"],
                   "submitted": sub.get("ranked", []), "no_submit": sub.get("no_submit", False),
                   "cost_usd": cj.get("total_cost_usd"), "turns": cj.get("num_turns"), "elapsed": cj.get("_elapsed"),
                   "error": bool(cj.get("is_error") or cj.get("timeout") or cj.get("parse_error")), **sc}
            rows.append(row)
            print(f"{g['qid']} r{rep} R@5={sc['R@5']:.2f} n_sub={len(row['submitted'])} turns={row['turns']} ${row['cost_usd']}", flush=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 집계: 문항별 reps 평균 → 전체 평균
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)
    agg = {}
    for k in ("R@1", "R@5", "R@10", "R@20", "S@5", "suff@5", "suff@10", "RR@10"):
        agg[k] = sum(sum(x[k] for x in v) / len(v) for v in byq.values()) / len(byq)
    core = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if v[0]["core"]]
    nc = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if not v[0]["core"]]
    summary = {"run": a.run, "n_q": len(byq), "reps": a.reps, "model": a.model, "arm": arm,
               "metrics": {k: round(v, 4) for k, v in agg.items()},
               "core_R@5": round(sum(core) / max(1, len(core)), 4), "noncore_R@5": round(sum(nc) / max(1, len(nc)), 4),
               "no_submit": sum(r["no_submit"] for r in rows), "errors": sum(r["error"] for r in rows),
               "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 2),
               "rep_disagreement_R@5": round(sum(1 for v in byq.values() if len({x["R@5"] for x in v}) > 1) / len(byq), 3)}
    json.dump(summary, open(run_dir / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
