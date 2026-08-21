#!/usr/bin/env python3
"""Codex 내부 exec 장애를 피하는 호스트 통제형 search/read/submit agent runner."""
import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score
from units import Units

SCHEMA = HERE / "host_agent_action_schema.json"
SYSTEM = """당신은 보험 약관 검색 에이전트입니다. 셸이나 도구를 직접 호출하지 마십시오.
호스트가 이미 원질의로 첫 하이브리드 검색을 실행했습니다. 아래 도구 기록을 보고 다음 행동 하나만 JSON으로 정하십시오.
- submit: 관련 근거 ID를 순서대로 최대 10개. Recall 평가이므로 서로 다른 관련 조를 폭넓게 포함해 가능하면 10개를 채웁니다.
- read: 확신이 없는 후보 조 전체를 읽습니다. 하나는 id에, 여러 개는 ids에 넣습니다.
- search: 최초 결과가 불충분할 때만 다른 자연어 질의로 재검색합니다. q에 검색어를 넣습니다.
그룹 결과의 evidence_variants는 같은 조의 서로 다른 원문이며 raw_rank는 원 순위입니다.
같은 조는 group id(j...) 또는 evidence id(e.../c...) 중 하나만 제출합니다.
사용하지 않는 q/id/ids 필드는 각각 빈 문자열/빈 배열로 출력합니다.
"""


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_arm(name):
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    return arms[name]


def visible_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"id", "jo"} and isinstance(child, str) and child:
                found.add(child)
            elif key in {"members", "member_ids"} and isinstance(child, list):
                found.update(x for x in child if isinstance(x, str))
            else:
                found.update(visible_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(visible_ids(child))
    return found


def tool_call(sess, qid, question, arm, action, value=""):
    env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=qid,
               SEMTAG_QUESTION=question, SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    if action == "search":
        command = [sys.executable, str(HERE / "agent_tools.py"), "search", "--q", value]
    elif action == "read":
        command = [sys.executable, str(HERE / "agent_tools.py"), "read", "--id", value]
    else:
        command = [sys.executable, str(HERE / "agent_tools.py"), "submit", "--ids", value]
    proc = subprocess.run(command, cwd=str(HERE), env=env, capture_output=True, text=True, timeout=240)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"error": "tool_parse", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    return payload, proc.returncode, proc.stderr


def model_action(model, reasoning_effort, prompt, output, timeout):
    command = ["codex", "exec", "--model", model, "--sandbox", "read-only", "--ephemeral",
               "--ignore-user-config", "--skip-git-repo-check", "--output-schema", str(SCHEMA),
               "--output-last-message", str(output), "--json", "-"]
    if reasoning_effort:
        command[4:4] = ["--config", f'model_reasoning_effort="{reasoning_effort}"']
    proc = subprocess.run(command, input=prompt, text=True, capture_output=True,
                          timeout=timeout, cwd=str(HERE), env=os.environ.copy())
    action = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    return action, proc


def model_usage(stdout):
    """Codex JSONL의 turn.completed usage를 재현 가능한 세션 통계로 보존한다."""
    total = collections.Counter()
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") if event.get("type") == "turn.completed" else None
        if usage:
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    total[key] += value
    return dict(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--qids", required=True)
    parser.add_argument("--gold", default=str(FS / "out/gold_spans_lsh_train.jsonl"))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--rep-start", type=int, default=0,
                        help="first replicate index; used for exact-key clean retries")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        default="medium")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    qids = json.loads(Path(args.qids).read_text(encoding="utf-8"))
    keep = set(qids)
    gold_path = Path(args.gold).resolve()
    gold = {row["qid"]: row for row in load_jsonl(gold_path)
            if row["qid"] in keep}
    missing_qids = sorted(keep - set(gold))
    if missing_qids:
        raise ValueError(f"qids missing from Gold: {missing_qids}")
    arms = {name: resolve_arm(name) for name in args.arms}
    units = {name: Units(FS / "out" / arm.get("jo", "elements_u2jo.jsonl"))
             for name, arm in arms.items()}
    run_dir = HERE / "out/host_agent" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"args": vars(args), "arms": arms,
                "arm_order_policy": "balanced_rotation_by_qid_and_rep_v1",
                "runner_sha256": sha256(__file__),
                "action_schema_sha256": sha256(SCHEMA),
                "system_prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
                "agent_tools_sha256": sha256(HERE / "agent_tools.py"),
                "arms_sha256": sha256(HERE / "arms.json"),
                "clm_search_sha256": sha256(FS / "clm_search.py"),
                "structured_search_sha256": sha256(FS / "structured_search.py"),
                "schema_adapter_sha256": sha256(FS / "schema_adapter.py"),
                "textmatch_sha256": sha256(FS / "textmatch.py"),
                "patterns_sha256": sha256(FS / "patterns.py"),
                "scoring_sha256": sha256(FS / "scoring.py"),
                "units_sha256": sha256(FS / "units.py"),
                "gold": str(gold_path),
                "gold_sha256": sha256(gold_path),
                "qids_sha256": sha256(args.qids),
                "arm_data": {
                    name: {
                        key: {
                            "path": str((FS / "out" / arm.get(key, default)).resolve()),
                            "sha256": sha256(FS / "out" / arm.get(key, default)),
                        }
                        for key, default in (("elements", "elements_u2.jsonl"),
                                             ("tags", "tags_u2_rules.jsonl"),
                                             ("jo", "elements_u2jo.jsonl"))
                    }
                    for name, arm in arms.items()
                }}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    blocks = [(qid, rep) for rep in range(args.rep_start, args.rep_start + args.reps)
              for qid in qids]
    rng.shuffle(blocks)
    jobs = []
    arm_orders = []
    for qid, rep in blocks:
        names = list(args.arms)
        if len(names) > 1:
            qid_offset = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16)
            offset = (qid_offset + rep) % len(names)
            names = names[offset:] + names[:offset]
        arm_orders.append({"qid": qid, "rep": rep, "arms": names})
        jobs.extend((qid, rep, name) for name in names)
    manifest["arm_orders"] = arm_orders
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def one(job):
        qid, rep, arm_name = job
        arm = arms[arm_name]
        question = gold[qid]["q"]
        sess = run_dir / "sessions" / f"{qid}_{arm_name}_r{rep}"
        sess.mkdir(parents=True, exist_ok=True)
        trace = []
        t0 = time.time()
        first, rc, stderr = tool_call(sess, qid, question, arm, "search", question)
        trace.append({"action": "search", "q": question, "response": first,
                      "returncode": rc, "stderr": stderr[-2000:]})
        seen = visible_ids(first)
        submitted = []
        did_submit = False
        fatal_errors = int(bool(rc or first.get("error")))
        protocol_errors = 0
        model_calls = 0
        usage_total = collections.Counter()
        for turn in range(args.max_actions):
            prompt = SYSTEM + "\n[질문]\n" + question + "\n[도구 기록]\n" + json.dumps(trace, ensure_ascii=False)
            if turn == args.max_actions - 1:
                prompt += "\n[마지막 턴]\n추가 검색이나 읽기를 하지 말고 지금까지 본 후보로 반드시 submit 하십시오."
            output = sess / f"action_{turn}.json"
            try:
                action, proc = model_action(args.model, args.reasoning_effort, prompt, output, args.timeout)
            except Exception as exc:
                trace.append({"action": "model_error", "error": f"{type(exc).__name__}:{exc}"})
                fatal_errors += 1; break
            model_calls += 1
            usage = model_usage(proc.stdout)
            usage_total.update(usage)
            if not action or proc.returncode:
                trace.append({"action": "model_error", "returncode": proc.returncode,
                              "stderr": proc.stderr[-2000:], "model_usage": usage})
                fatal_errors += 1; break
            kind = action["action"]
            if kind == "submit":
                did_submit = True
                requested = action.get("ids") or []
                submitted = [item for item in requested if item in seen][:10]
                invalid = [item for item in requested if item not in seen]
                response, rc, stderr = tool_call(sess, qid, question, arm, "submit", ",".join(submitted))
                trace.append({"action": "submit", "requested": requested, "invalid": invalid,
                              "response": response, "returncode": rc, "stderr": stderr[-2000:],
                              "model_usage": usage})
                fatal_errors += int(bool(rc or response.get("error")))
                protocol_errors += len(invalid)
                break
            if kind == "read":
                requested = []
                one_id = action.get("id", "")
                if one_id:
                    requested.append(one_id)
                requested.extend(action.get("ids") or [])
                requested = list(dict.fromkeys(requested))[:10]
                valid = [item for item in requested if item in seen]
                invalid = [item for item in requested if item not in seen]
                responses = []
                for item in valid:
                    response, rc, stderr = tool_call(
                        sess, qid, question, arm, "read", item)
                    responses.append({"id": item, "response": response,
                                      "returncode": rc, "stderr": stderr[-2000:]})
                    seen.update(visible_ids(response))
                    protocol_errors += int(bool(rc or response.get("error")))
                trace.append({"action": "read", "requested": requested,
                              "invalid": invalid, "responses": responses,
                              "model_usage": usage})
                protocol_errors += int(not requested) + len(invalid)
                continue
            else:
                value = action.get("q", "").strip() or question
            response, rc, stderr = tool_call(sess, qid, question, arm, kind, value)
            trace.append({"action": kind, "value": value, "response": response,
                          "returncode": rc, "stderr": stderr[-2000:], "model_usage": usage})
            seen.update(visible_ids(response))
            protocol_errors += int(bool(rc or response.get("error")))
        if not did_submit:
            fatal_errors += 1
        (sess / "host_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = score(units[arm_name].resolve(submitted), gold[qid]["groups"], ks=(1, 5, 10, 20))
        row = {"qid": qid, "rep": rep, "arm": arm_name, "submitted": submitted,
               "model_calls": model_calls, "elapsed": time.time() - t0,
               "model_usage": dict(usage_total),
               "empty_submission": int(did_submit and not submitted),
               "errors": fatal_errors, "protocol_errors": protocol_errors, **metrics}
        (sess / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(qid, arm_name, rep, f"R@5={metrics['R@5']:.2f}", f"n={len(submitted)}",
              f"err={fatal_errors}", f"proto={protocol_errors}", flush=True)
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs))
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"run": args.run, "n_q": len(qids), "reps": args.reps, "model": args.model,
               "reasoning_effort": args.reasoning_effort, "arms": {}}
    for name in args.arms:
        subset = [row for row in rows if row["arm"] == name]
        usage = collections.Counter()
        for row in subset:
            usage.update(row.get("model_usage", {}))
        summary["arms"][name] = {key: sum(row[key] for row in subset) / len(subset)
                                  for key in ("R@1", "R@5", "R@10", "suff@5", "suff@10")}
        summary["arms"][name].update(errors=sum(row["errors"] for row in subset),
                                     protocol_errors=sum(row["protocol_errors"] for row in subset),
                                     empty_submissions=sum(row["empty_submission"] for row in subset),
                                     mean_submitted=sum(len(row["submitted"]) for row in subset) / len(subset),
                                     mean_model_calls=sum(row["model_calls"] for row in subset) / len(subset),
                                     mean_elapsed=sum(row["elapsed"] for row in subset) / len(subset),
                                     model_usage_total=dict(usage))
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
