#!/usr/bin/env python3
"""Codex CLI runner for the frozen-metadata semantic-tag search experiment."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
import sys
sys.path.insert(0, str(FS))
from scoring import score  # noqa: E402
from units import Units  # noqa: E402

PROMPT = """당신은 보험 약관에서 질문의 근거 element를 찾는 검색 에이전트입니다.
반드시 현재 디렉터리에서 아래 명령으로만 검색 도구를 사용하세요.

python agent_tools.py search --q "..." [--page N] [--contract "..."] [--role "..."] [--subject "..."] [--qualifier "..."] [--schema "..."]
python agent_tools.py msearch --q "..." [--page N]
python agent_tools.py read --id <element_id>
python agent_tools.py submit --ids <id1>,<id2>,...

검색 정책:
- 먼저 질문을 서로 다른 근거 축으로 분해하세요.
- search는 임베딩 없는 태그 슬롯 검색입니다. 특약명(contract), 역할(role), 대상(subject), 조건(qualifier), 표/산식(schema)이 명확할 때 사용하세요.
- msearch는 V9 메타데이터 BM25+dense hybrid 벡터 검색입니다. 자연어, 정의, 구어체, 표현 변형이나 슬롯이 불명확할 때 사용하세요.
- 모든 문항에서 슬롯 검색이 놓친 의미 기반 후보가 없는지 확인하기 위해 submit 전에 msearch를 최소 1회 사용하세요.
- 질문이 슬롯 단서와 자연어 의미를 함께 가지면 두 채널을 각각 사용하세요. 어느 채널이든 결과가 동떨어지면 약관 표현으로 바꾸어 재검색하세요.
- 유력 후보는 read로 원문을 확인하고, 여러 근거 축이면 서로 다른 사실을 덮는 조를 찾으세요.
- 같은 조의 중복 ID를 내지 말고 관련도 순으로 최대 10개를 submit한 뒤 즉시 종료하세요.
답변 문장을 작성하지 말고 도구 호출과 submit에 집중하세요.

[질문]
{question}
"""


def is_core(row: dict) -> bool:
    return row.get("core_retrieval") is True or str(row.get("core_retrieval", "")).lower() == "true"


def resolve_arm(name: str) -> dict:
    if name.strip().startswith("{"):
        return json.loads(name)
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    if name not in arms:
        raise SystemExit(f"unknown arm: {name}")
    return arms[name]


def run_one(args, run_dir: Path, gold: dict, rep: int, arm: dict):
    session = run_dir / "sessions" / f"{gold['qid']}_r{rep}"
    done = session / "codex.json"
    submitted = session / "submit.json"
    if args.resume and done.exists() and submitted.exists():
        return json.loads(submitted.read_text(encoding="utf-8")), json.loads(done.read_text(encoding="utf-8"))
    session.mkdir(parents=True, exist_ok=True)
    for name in ("calls.jsonl", "submit.json", "codex.json", "codex_output.txt"):
        p = session / name
        if p.exists():
            p.unlink()
    env = dict(os.environ, SEMTAG_SESSION=str(session), SEMTAG_QID=gold["qid"], SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    output = session / "codex_output.txt"
    cmd = [args.codex_bin, "exec", "--model", args.model, "--sandbox", "workspace-write",
           "--skip-git-repo-check", "--ephemeral", "-o", str(output), "-"]
    started = time.time()
    meta = {"model": args.model, "error": None}
    try:
        result = subprocess.run(cmd, input=PROMPT.format(question=gold["q"]), capture_output=True,
                                text=True, timeout=args.timeout, cwd=HERE, env=env,
                                encoding="utf-8", errors="replace")
        meta.update({"returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]})
        if result.returncode != 0:
            meta["error"] = f"codex_exit:{result.returncode}"
    except subprocess.TimeoutExpired:
        meta["error"] = "timeout"
    except OSError as exc:
        meta["error"] = f"spawn:{exc}"
    meta["elapsed"] = time.time() - started
    done.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    sub = json.loads(submitted.read_text(encoding="utf-8")) if submitted.exists() else {"qid": gold["qid"], "ranked": [], "no_submit": True}
    return sub, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--gold", type=Path, required=True); ap.add_argument("--jo", type=Path, default=FS / "out" / "elements_u2jo.jsonl")
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.6-luna"); ap.add_argument("--timeout", type=int, default=1200)
    default_codex = os.environ.get("CODEX_BIN") or str(Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")
    ap.add_argument("--codex-bin", default=default_codex); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); arm = resolve_arm(args.arm)
    gold_rows = [r for r in map(json.loads, args.gold.open(encoding="utf-8")) if r.get("groups") and r.get("status", "ok") == "ok"]
    if 0 < args.n < len(gold_rows):
        rnd = random.Random(args.seed); core = [r for r in gold_rows if is_core(r)]; other = [r for r in gold_rows if not is_core(r)]
        k = min(round(args.n * len(core) / len(gold_rows)), len(core)); gold_rows = rnd.sample(core, k) + rnd.sample(other, args.n - k); gold_rows.sort(key=lambda r: r["qid"])
    run_dir = HERE / "out" / "agent" / args.run; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"arm_name": args.arm, "arm": arm, "args": {**vars(args), "gold": str(args.gold), "jo": str(args.jo)}}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    units = Units(str(args.jo)); jobs = [(g, rep) for rep in range(args.reps) for g in gold_rows]; rows = []
    print(f"run={args.run} arm={args.arm} n={len(gold_rows)} model={args.model}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outputs = pool.map(lambda job: run_one(args, run_dir, job[0], job[1], arm), jobs)
        for (g, rep), (sub, meta) in zip(jobs, outputs):
            metrics = score(units.resolve(sub.get("ranked", [])), g["groups"], ks=(1, 5, 10, 20))
            rows.append({"qid": g["qid"], "rep": rep, "core": is_core(g), "task_type": g.get("task_type", ""), "submitted": sub.get("ranked", []), "no_submit": bool(sub.get("no_submit")), "elapsed": meta.get("elapsed"), "error": bool(meta.get("error")), **metrics})
            print(f"{g['qid']} r{rep} R@5={metrics['R@5']:.2f} submitted={len(sub.get('ranked', []))}", flush=True)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by_q = collections.defaultdict(list)
    for r in rows: by_q[r["qid"]].append(r)
    keys = ("R@1", "R@5", "R@10", "R@20", "S@5", "suff@5", "suff@10", "RR@10")
    metrics = {k: sum(sum(x[k] for x in v) / len(v) for v in by_q.values()) / len(by_q) for k in keys} if by_q else {k: 0 for k in keys}
    summary = {"run": args.run, "arm_name": args.arm, "n_q": len(by_q), "reps": args.reps, "model": args.model, "metrics": {k: round(v, 4) for k, v in metrics.items()}, "no_submit": sum(r["no_submit"] for r in rows), "errors": sum(r["error"] for r in rows)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
