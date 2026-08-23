#!/usr/bin/env python3
"""에이전트 실측 러너 — Claude/Codex + agent_tools.py(search/read/submit), S5 예산, ITT 채점.

filesearch/agent_runner.py 사본. 변경점: (1) 데이터·채점 모듈은 filesearch/ 를 참조,
(2) --arm 에 arms.json 의 arm 이름을 그대로 줄 수 있음, (3) PROMPT 에 arm 조건부 {extra}
(scope/browse·폴백 안내 — 신규 플래그가 없으면 원본과 동일 프롬프트).
결과: out/agent/<run>/sessions/<qid>_r<rep>/{calls.jsonl,submit.json,claude.json}, out/agent/<run>/results.jsonl
"""
import argparse, collections, json, os, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score
from units import Units

VSEARCH_DOC = """1b) 메타데이터 검색: python3 agent_tools.py msearch --q "<자연어 질의>" [--page N]
   - 의미 유사도(임베딩)+키워드 결합 검색. 질문 표현이 문서 용어와 달라도 찾습니다. 검색 횟수는 search 와 합쳐 세션당 20회.
   - 태그 검색(search)과 메타데이터 검색(msearch)을 질문 성격에 따라 골라 쓰거나 둘 다 써서 교차 확인하십시오. 결과 id(c…)도 그대로 제출할 수 있습니다.
"""
PROMPT = """당신은 보험 약관 문서 안에서 질문의 근거 조항(element)을 찾는 검색 에이전트입니다.
도구는 아래 셸 명령뿐입니다(다른 파일·명령 사용 금지). 반드시 `python3 agent_tools.py ...` 형태로 호출하십시오.
도구는 한 번에 하나씩 별도 명령으로 호출하고 `&&`, `;`, 파이프, 리다이렉션으로 묶지 마십시오.
명령 실행 자체가 CreateProcess 또는 일시적 도구 오류로 실패하면 질의를 바꾸지 말고 **동일한 명령을 그대로 1회 재시도**하십시오.

1) 검색: python3 agent_tools.py search --q "<검색어>" [--page N] [--contract "<특약명>"] [--role "<역할코드>"] [--subject "<대상어>"]
   - 한 페이지 최대 40건(조 단위 그룹 arm은 표시 그룹 수가 더 적고 raw_page_size가 함께 나옵니다). 세션당 최대 20회.
   - 검색어는 질문에서 핵심어(특약명·급여금명·질병명·조건)를 골라 쓰고, 결과가 엉뚱하면 다른 표현으로 다시 검색하십시오.
   - 역할코드: exclusion_exception, premium_waiver, payment_trigger, payment_amount, limit_frequency, timing_period, definition, criteria_rule, contract_lifecycle, claim_procedure, code_reference
{extra}{vsearch}2) 읽기: python3 agent_tools.py read --id <element_id>   — 해당 조(條) 전체 원문. 세션당 최대 8회. 확신이 서지 않는 후보를 확인할 때 쓰십시오.
3) 제출: python3 agent_tools.py submit --ids <id1>,<id2>,...   — 근거일 가능성이 높은 순서로 **최대 10개**. 반드시 1회 호출하고, 그 뒤에는 어떤 도구도 부르지 마십시오.

원칙: 질문이 특정 특약을 가리키면 그 특약의 element 를 우선하십시오. 질문이 여러 근거(예: 정의 + 지급조건 + 청구절차)를 요구하면 서로 다른 조의 element 를 섞어 제출하십시오.
검색 결과에 facets(특약·조 분포)가 있으면 그것을 보고 범위를 좁혀(--contract, --role) 다시 검색하거나 다른 페이지를 보십시오. 제출은 가능하면 10개를 채우되 같은 조를 중복해 넣지 마십시오. 답변 문장은 쓰지 말고 제출만 하십시오.

[질문]
{question}
"""


def build_extra(arm):
    """arm 플래그 조건부 프롬프트 추가 줄 — 플래그가 전부 꺼져 있으면 빈 문자열(기준선 프롬프트 동일)."""
    lines = []
    if arm.get("ranker") == "bm25f":
        lines.append('   - 이 arm은 단일단계 구조화 검색입니다. 필요하면 범용 축 --identity/--topic/--function/--locator/--constraint/--relation/--structure 로 검색 의도를 명시할 수 있습니다.')
    if arm.get("portfolio") and arm.get("portfolio_prompt", True):
        lines.append('   - 검색 도구가 원질의 뒤에 규칙 후속질의를 자동 수행합니다. 결과의 phase=seed는 기존 순위, 그 밖의 phase는 identity 완화·역할 분해·facet 탐색 후보입니다. 복수 근거 질문에서는 서로 다른 phase의 후보도 읽고 제출하십시오.')
    if arm.get("scope_boost"):
        lines.append('   - 계층 탐색: --scope "<특약>[/<관>[/<조>]]" 를 주면 그 범위의 결과가 가산 부스트됩니다(필터 아님). --q 없이 --scope 만 주면 그 계층의 하위 목록을 보여줍니다(특약 목록은 --scope 생략).')
    if arm.get("search_only") and arm.get("meta"):
        lines.append('   - 변인 통제를 위해 직접 msearch는 사용하지 않습니다. 첫 search 응답에 태그 결과와 고정 메타 결과가 자동 결합됩니다.')
    elif arm.get("fallback") and arm.get("meta"):
        lines.append('   - 태그 검색 결과가 없거나 빈약하면 같은 응답 안에 메타데이터 검색 결과가 자동 포함됩니다(channel: tag+fallback:meta, 항목 src:"meta").')
    if arm.get("fallback_prompt") and arm.get("meta"):
        lines.append('   - search 결과가 0건이거나 질문과 동떨어져 보이면 즉시 msearch 로 전환해 다시 검색하십시오.')
    if arm.get("ref_expand"):
        lines.append('   - 검색·읽기 결과의 ref_jo/refs 는 그 조가 참조하는 다른 조입니다. 근거가 여러 곳에 흩어진 질문이면 참조 조를 함께 확인·제출하십시오.')
    if arm.get("group_jo"):
        lines.append('   - 결과 하나는 같은 조의 후보 묶음입니다. evidence_variants는 그 조의 서로 다른 원문 근거이며 raw_rank 순서입니다. 모두 확인한 뒤 그룹 id(j...) 또는 해당 근거 id(e.../c...) 중 하나만 제출하십시오.')
    return ("\n".join(lines) + "\n") if lines else ""


def resolve_arm(s):
    """--arm 이 JSON 이면 그대로, 아니면 arms.json 의 arm 이름으로 해석."""
    s = s.strip()
    if s.startswith("{"):
        return json.loads(s)
    arms = json.load(open(HERE / "arms.json", encoding="utf-8"))
    if s not in arms:
        raise SystemExit(f"[오류] arms.json 에 없는 arm: {s} (보유: {', '.join(arms)})")
    return arms[s]


def codex_command_violations(commands, arm=None):
    """Codex 셸 래퍼 차이는 허용하되 agent_tools 단일 호출만 통과시킨다."""
    prefixes = tuple(f"/bin/{shell} -{flag} "
                     for shell in ("zsh", "bash", "sh") for flag in ("c", "lc"))
    actions = ("search", "read", "submit") if (arm or {}).get("search_only") else ("search", "msearch", "read", "submit")
    violations = []
    for command in commands:
        prefix = next((p for p in prefixes if command.startswith(p)), None)
        body = command[len(prefix):] if prefix else ""
        if len(body) >= 2 and body[0] in "'\"" and body[-1] == body[0]:
            body = body[1:-1]
        valid_action = any(body.startswith(f"python3 agent_tools.py {action}") for action in actions)
        if not prefix or not valid_action or any(op in body for op in ("&&", ";", "|", ">", "<")):
            violations.append(command)
    return violations


def codex_infra_failure_count(stderr):
    """Codex 최상위 returncode=0에 가려지는 unified-exec 생성 실패 수."""
    return (stderr or "").count("Failed to create unified exec process")


def codex_failed_commands(stderr):
    """stderr에서 unified-exec 생성에 실패한 원 명령을 순서대로 뽑는다."""
    return re.findall(r"exec_command failed for `(.+?)`: CreateProcess", stderr or "")


def codex_unrecovered_infra_error(stderr, submit_exists):
    """submit 실행 생성 실패 후 submit.json도 없으면 복구되지 않은 세션 오류다."""
    return bool(not submit_exists and "agent_tools.py submit" in (stderr or "")
                and codex_infra_failure_count(stderr) > 0)


def run_one(args, run_dir, g, rep, arm):
    sess = run_dir / "sessions" / f"{g['qid']}_r{rep}"
    done = sess / ("gpt.json" if args.provider == "codex" else "claude.json")
    if done.exists() and (sess / "submit.json").exists():
        cj = json.load(open(done))
        if args.provider == "codex":
            cj["tool_policy_violations"] = codex_command_violations(cj.get("commands", []), arm)
            cj["is_error"] = (cj.get("returncode") != 0 or bool(cj["tool_policy_violations"])
                              or bool(cj.get("unrecovered_infra_error")))
            json.dump(cj, open(done, "w"), ensure_ascii=False)
        return json.load(open(sess / "submit.json")), cj
    sess.mkdir(parents=True, exist_ok=True)
    json.dump({"qid": g["qid"], "question": g["q"]}, open(sess / "question.json", "w"),
              ensure_ascii=False)
    for f in ("calls.jsonl", "submit.json"):
        if (sess / f).exists():
            (sess / f).unlink()
    env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=g["qid"],
               SEMTAG_QUESTION=g["q"], SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    if os.environ.get("SEMTAG_PYBIN"):
        env["PATH"] = os.environ["SEMTAG_PYBIN"] + ":" + env.get("PATH", "")
    prompt = PROMPT.format(question=g["q"],
                           vsearch=VSEARCH_DOC if arm.get("meta") and not arm.get("search_only") else "",
                           extra=build_extra(arm))
    if args.provider == "codex":
        cmd = ["codex", "exec", "--model", args.model, "--sandbox", "workspace-write",
               "--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--json", "-"]
    else:
        cmd = ["claude", "-p", "--model", args.model, "--output-format", "json",
               "--max-turns", str(args.max_turns),
               "--allowedTools", "Bash(python3 agent_tools.py:*)",
               "--disallowedTools", "Read,Edit,Write,Grep,Glob,WebFetch,WebSearch,Agent,NotebookEdit,Task"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=args.timeout,
                           cwd=str(HERE), env=env)
        if args.provider == "codex":
            events = []
            for line in r.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            commands = [e["item"]["command"] for e in events
                        if e.get("type") == "item.completed"
                        and e.get("item", {}).get("type") == "command_execution"]
            violations = codex_command_violations(commands, arm)
            usage = next((e.get("usage", {}) for e in reversed(events)
                          if e.get("type") == "turn.completed"), {})
            stderr_full = r.stderr or ""
            failed_commands = codex_failed_commands(stderr_full)
            unrecovered_commands = [command for command in failed_commands if command not in commands]
            cj = {"provider": "codex", "returncode": r.returncode, "commands": commands,
                  "tool_policy_violations": violations, "usage": usage,
                  "num_turns": len(commands), "total_cost_usd": None,
                  "is_error": r.returncode != 0 or bool(violations),
                  "infra_failures": codex_infra_failure_count(stderr_full),
                  "infra_failed_commands": failed_commands,
                  "infra_unrecovered_commands": unrecovered_commands,
                  "failed_submit_exec": ("agent_tools.py submit" in stderr_full and
                                           "Failed to create unified exec process" in stderr_full),
                  "stderr": stderr_full[-8000:]}
            if not events:
                cj.update(parse_error=True, stdout=r.stdout[-2000:])
        else:
            try:
                cj = json.loads(r.stdout)
            except Exception:
                cj = {"parse_error": True, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        cj = {"timeout": True}
    cj["_elapsed"] = time.time() - t0
    submit_exists = (sess / "submit.json").exists()
    sub = json.load(open(sess / "submit.json")) if submit_exists else {"qid": g["qid"], "ranked": [], "no_submit": True}
    if args.provider == "codex":
        cj["unrecovered_infra_error"] = codex_unrecovered_infra_error(cj.get("stderr", ""), submit_exists)
        cj["is_error"] = bool(cj.get("is_error") or cj["unrecovered_infra_error"])
    json.dump(cj, open(done, "w"), ensure_ascii=False)
    return sub, cj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 이름 (out/agent/<run>)")
    ap.add_argument("--arm", required=True, help="ARM json 또는 arms.json 의 arm 이름")
    ap.add_argument("--gold", default=str(FS / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--jo", default=str(FS / "out/elements_u2jo.jsonl"))
    ap.add_argument("--elements", default=str(FS / "out/elements_u2.jsonl"))
    ap.add_argument("--qids", default="", help="고정 qid 목록 json")
    ap.add_argument("--n", type=int, default=-1, help="문항 수(층화 표본), -1=전수")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--provider", default="claude", choices=("claude", "codex"))
    ap.add_argument("--model", default="", help="기본값: claude=sonnet, codex=gpt-5.6-terra")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if not a.model:
        a.model = "gpt-5.6-terra" if a.provider == "codex" else "sonnet"
    arm = resolve_arm(a.arm)
    run_dir = HERE / "out" / "agent" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": arm, "args": vars(a)}, open(run_dir / "config.json", "w"), ensure_ascii=False, indent=1)
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"]
    if a.qids:
        keep = set(json.load(open(a.qids)))
        G = [g for g in G if g["qid"] in keep]
    elif a.n > 0:  # core/비core 층화 표본(고정 시드)
        rnd = random.Random(a.seed)
        core = [g for g in G if g["core_retrieval"] == "True"]; nc = [g for g in G if g["core_retrieval"] != "True"]
        k_core = round(a.n * len(core) / len(G))
        G = rnd.sample(core, k_core) + rnd.sample(nc, a.n - k_core)
    U = Units(a.jo)
    jobs = [(g, rep) for rep in range(a.reps) for g in G]
    print(f"run={a.run} n={len(G)} reps={a.reps} jobs={len(jobs)} provider={a.provider} model={a.model}", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for (g, rep), (sub, cj) in zip(jobs, ex.map(lambda gr: run_one(a, run_dir, gr[0], gr[1], arm), jobs)):
            ranked_jo = U.resolve(sub.get("ranked", []))
            sc = score(ranked_jo, g["groups"], ks=(1, 5, 10, 20))
            row = {"qid": g["qid"], "rep": rep, "core": g["core_retrieval"] == "True", "task_type": g["task_type"],
                   "submitted": sub.get("ranked", []), "no_submit": sub.get("no_submit", False),
                   "cost_usd": cj.get("total_cost_usd"), "turns": cj.get("num_turns"), "elapsed": cj.get("_elapsed"),
                   "infra_failures": cj.get("infra_failures", 0),
                   "infra_unrecovered": len(cj.get("infra_unrecovered_commands", [])),
                   "error": bool(cj.get("is_error") or cj.get("timeout") or cj.get("parse_error")), **sc}
            rows.append(row)
            print(f"{g['qid']} r{rep} R@5={sc['R@5']:.2f} n_sub={len(row['submitted'])} "
                  f"turns={row['turns']} cost={row['cost_usd']}", flush=True)
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
    usage = collections.Counter()
    if a.provider == "codex":
        for sess in (run_dir / "sessions").glob("*_r*/gpt.json"):
            usage.update(json.load(open(sess)).get("usage", {}))
    summary = {"run": a.run, "n_q": len(byq), "reps": a.reps,
               "provider": a.provider, "model": a.model, "arm": arm,
               "metrics": {k: round(v, 4) for k, v in agg.items()},
               "core_R@5": round(sum(core) / max(1, len(core)), 4), "noncore_R@5": round(sum(nc) / max(1, len(nc)), 4),
               "no_submit": sum(r["no_submit"] for r in rows), "errors": sum(r["error"] for r in rows),
               "infra_failures": sum(r.get("infra_failures", 0) for r in rows),
               "infra_degraded_sessions": sum(r.get("infra_failures", 0) > 0 for r in rows),
               "infra_unrecovered_commands": sum(r.get("infra_unrecovered", 0) for r in rows),
               "infra_unrecovered_sessions": sum(r.get("infra_unrecovered", 0) > 0 for r in rows),
               "cost_usd": (None if a.provider == "codex" else round(sum(r["cost_usd"] or 0 for r in rows), 2)),
               **({"usage": dict(usage)} if usage else {}),
               "rep_disagreement_R@5": round(sum(1 for v in byq.values() if len({x["R@5"] for x in v}) > 1) / len(byq), 3)}
    json.dump(summary, open(run_dir / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
