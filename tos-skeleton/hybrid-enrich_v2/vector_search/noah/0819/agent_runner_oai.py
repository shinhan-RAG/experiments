#!/usr/bin/env python3
"""에이전트 실측 러너 (OpenAI 호환 API — KT Cloud vLLM Qwen3-32B 등). 도구·예산·채점은 agent_runner.py 와 동일(agent_tools.py 재사용).

filesearch/agent_runner_oai.py 사본. 변경점: (1) 기본 엔드포인트 = 로컬 vLLM(:8080),
(2) 기본 모델 Qwen/Qwen3-32B-FP8 · 기본 프로토콜 text(+ --no-think 권장),
(3) --arm 에 arms.json arm 이름 허용, (4) 도구 스키마: scope 는 scope_boost arm 에서만 노출,
qualifier/schema/strategy 는 비노출 유지(질의측 생성률 ~0% — 0818).

환경: OPENAI_API_KEY(vLLM 은 임의 문자열 가능) · 예: --base-url http://localhost:8080/v1
"""
import argparse, collections, json, os, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
from scoring import score
from units import Units
from agent_runner import PROMPT, VSEARCH_DOC, build_extra, resolve_arm


def tools_for(arm):
    search_props = {
        "q": {"type": "string", "description": "검색어(특약명·급여금명·질병명·조건 등 핵심어)"},
        "page": {"type": "integer", "description": "페이지(1부터)", "default": 1},
        "contract": {"type": "string", "description": "특약명(쉼표 구분, 선택)"},
        "role": {"type": "string", "description": "역할코드(쉼표 구분, 선택)"},
        "subject": {"type": "string", "description": "대상어(쉼표 구분, 선택)"}}
    search_required = ["q"]
    search_desc = "약관 element 검색. 한 페이지 40건, 세션당 최대 20회."
    if arm.get("scope_boost"):
        search_props["scope"] = {"type": "string", "description": '계층 경로 "<특약>[/<관>[/<조>]]" — 매치 결과 가산 부스트(필터 아님). q 없이 scope 만 주면 하위 목록(browse)'}
        search_required = []
        search_desc += ' q 또는 scope 중 하나는 필수.'
    return [
        {"type": "function", "function": {"name": "search", "description": search_desc,
            "parameters": {"type": "object", "properties": search_props, "required": search_required}}},
        {"type": "function", "function": {"name": "msearch", "description": "메타데이터(임베딩+키워드) 검색. 한 페이지 40건. search 와 합쳐 세션당 최대 20회.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "page": {"type": "integer", "default": 1}}, "required": ["q"]}}},
        {"type": "function", "function": {"name": "read", "description": "element_id 의 조(條) 전체 원문. 세션당 최대 8회.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
        {"type": "function", "function": {"name": "submit", "description": "근거 element_id 순위 제출(최대 10). 1회만, 이후 종료.",
            "parameters": {"type": "object", "properties": {"ids": {"type": "string", "description": "쉼표 구분 id 순위"}}, "required": ["ids"]}}},
    ]


TEXT_PROTOCOL = """
도구는 함수 호출 대신 **한 줄** 로 호출합니다: 반드시 `CALL: {"tool":"search","q":"...","page":1}` / `CALL: {"tool":"read","id":"e00123"}` / `CALL: {"tool":"submit","ids":"e1,e2,e3"}` 형식. 한 번에 하나만. 도구 결과를 받은 뒤 다음 호출을 하십시오."""


def run_tool(env, name, args):
    cmd = [sys.executable, str(HERE / "agent_tools.py"), name]
    for k, v in args.items():
        if v in (None, "", 0) and k != "page":
            continue
        cmd += [f"--{k}", str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(HERE), timeout=120)
    return (r.stdout or r.stderr)[-12000:]


def run_one(args, run_dir, g, rep, arm, client, tools):
    sess = run_dir / "sessions" / f"{g['qid']}_r{rep}"
    done = sess / "llm.json"
    if done.exists() and (sess / "submit.json").exists():
        return json.load(open(sess / "submit.json")), json.load(open(done))
    sess.mkdir(parents=True, exist_ok=True)
    for f in ("calls.jsonl", "submit.json"):
        if (sess / f).exists():
            (sess / f).unlink()
    env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=g["qid"], SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    sysmsg = PROMPT.format(question=g["q"], vsearch=VSEARCH_DOC if arm.get("meta") else "", extra=build_extra(arm)).replace("python3 agent_tools.py ", "")
    if args.protocol == "text":
        sysmsg += TEXT_PROTOCOL
    if args.no_think:
        sysmsg = "/no_think\n" + sysmsg
    msgs = [{"role": "system", "content": "당신은 도구를 사용하는 검색 에이전트입니다."}, {"role": "user", "content": sysmsg}]
    usage = collections.Counter(); t0 = time.time(); turns = 0; err = None; transcript = []
    try:
        for turn in range(args.max_turns):
            turns += 1
            kw = dict(model=args.model, messages=msgs, temperature=args.temperature, max_tokens=2048)
            if args.protocol == "tools":
                kw["tools"] = tools
            resp = client.chat.completions.create(**kw)
            if resp.usage:
                usage["prompt"] += resp.usage.prompt_tokens or 0; usage["completion"] += resp.usage.completion_tokens or 0
            m = resp.choices[0].message
            calls = []
            if args.protocol == "tools" and m.tool_calls:
                msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in m.tool_calls]})
                for tc in m.tool_calls:
                    try:
                        a = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        a = {}
                    calls.append((tc.id, tc.function.name, a))
            else:
                text = m.content or ""
                msgs.append({"role": "assistant", "content": text})
                import re
                for line in text.splitlines():
                    mm = re.match(r"\s*CALL:\s*(\{.*\})\s*$", line)
                    if mm:
                        try:
                            a = json.loads(mm.group(1)); calls.append((None, a.pop("tool", ""), a))
                        except Exception:
                            pass
                if not calls:
                    # 호출 없는 응답: 종료 안내 1회, 그래도 없으면 중단
                    if (sess / "submit.json").exists():
                        break
                    msgs.append({"role": "user", "content": "도구를 호출하거나 submit 으로 제출하십시오."})
                    continue
            for cid, name, a in calls:
                if name not in ("search", "msearch", "read", "submit"):
                    out = json.dumps({"error": "unknown tool"})
                else:
                    out = run_tool(env, name, a)
                transcript.append({"turn": turn, "tool": name, "args": a, "out": out[:400]})
                if args.protocol == "tools" and cid:
                    msgs.append({"role": "tool", "tool_call_id": cid, "content": out})
                else:
                    msgs.append({"role": "user", "content": f"[{name} 결과]\n{out}"})
            if (sess / "submit.json").exists():
                break
    except Exception as e:  # API 오류 → ITT 0점
        err = f"{type(e).__name__}: {str(e)[:300]}"
    meta = {"model": args.model, "turns": turns, "usage": dict(usage), "elapsed": time.time() - t0, "error": err, "transcript": transcript}
    json.dump(meta, open(done, "w"), ensure_ascii=False)
    sub = json.load(open(sess / "submit.json")) if (sess / "submit.json").exists() else {"qid": g["qid"], "ranked": [], "no_submit": True}
    return sub, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--arm", required=True, help="ARM json 또는 arms.json arm 이름")
    ap.add_argument("--gold", default=str(FS / "out/gold_spans_lsh_train.jsonl"))
    ap.add_argument("--jo", default=str(FS / "out/elements_u2jo.jsonl"))
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"))
    ap.add_argument("--protocol", default="text", choices=("tools", "text"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--no-think", action="store_true", help="Qwen3 등 thinking 모델에 /no_think 지시(프롬프트 첫 줄)")
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY") or "dummy", base_url=a.base_url)
    arm = resolve_arm(a.arm)
    tools = tools_for(arm)
    run_dir = HERE / "out" / "agent" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": arm, "args": vars(a)}, open(run_dir / "config.json", "w"), ensure_ascii=False, indent=1)
    G = [g for g in (json.loads(l) for l in open(a.gold)) if g["groups"] and g.get("status", "ok") == "ok"]
    if a.n > 0:
        rnd = random.Random(a.seed)
        core = [g for g in G if g["core_retrieval"] == "True"]; nc = [g for g in G if g["core_retrieval"] != "True"]
        k_core = round(a.n * len(core) / len(G))
        G = rnd.sample(core, k_core) + rnd.sample(nc, a.n - k_core)
    U = Units(a.jo)
    jobs = [(g, rep) for rep in range(a.reps) for g in G]
    print(f"run={a.run} n={len(G)} reps={a.reps} jobs={len(jobs)} model={a.model} base={a.base_url} protocol={a.protocol}", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for (g, rep), (sub, meta) in zip(jobs, ex.map(lambda gr: run_one(a, run_dir, gr[0], gr[1], arm, client, tools), jobs)):
            ranked_jo = U.resolve(sub.get("ranked", []))
            sc = score(ranked_jo, g["groups"], ks=(1, 5, 10, 20))
            row = {"qid": g["qid"], "rep": rep, "core": g["core_retrieval"] == "True", "task_type": g["task_type"],
                   "submitted": sub.get("ranked", []), "no_submit": sub.get("no_submit", False), "turns": meta["turns"],
                   "tokens": meta["usage"], "elapsed": meta["elapsed"], "error": bool(meta["error"]), **sc}
            rows.append(row)
            print(f"{g['qid']} r{rep} R@5={sc['R@5']:.2f} n_sub={len(row['submitted'])} turns={row['turns']} err={meta['error'] or ''}", flush=True)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)
    agg = {k: sum(sum(x[k] for x in v) / len(v) for v in byq.values()) / len(byq) for k in ("R@1", "R@5", "R@10", "R@20", "S@5", "suff@5", "suff@10", "RR@10")}
    core = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if v[0]["core"]]
    nc = [sum(x["R@5"] for x in v) / len(v) for v in byq.values() if not v[0]["core"]]
    summary = {"run": a.run, "n_q": len(byq), "reps": a.reps, "model": a.model, "protocol": a.protocol, "no_think": a.no_think, "arm": arm,
               "metrics": {k: round(v, 4) for k, v in agg.items()},
               "core_R@5": round(sum(core) / max(1, len(core)), 4), "noncore_R@5": round(sum(nc) / max(1, len(nc)), 4),
               "no_submit": sum(r["no_submit"] for r in rows), "errors": sum(r["error"] for r in rows),
               "prompt_tokens": sum(r["tokens"].get("prompt", 0) for r in rows), "completion_tokens": sum(r["tokens"].get("completion", 0) for r in rows),
               "rep_disagreement_R@5": round(sum(1 for v in byq.values() if len({x["R@5"] for x in v}) > 1) / len(byq), 3)}
    json.dump(summary, open(run_dir / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
