#!/usr/bin/env python3
"""Run the 281-question C29 dual-tool evaluation with parallel Codex CLI workers."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score
from units import Units


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "medium"
ARM_NAME = "c29_dual_tool_hybrid"
GOLD = FS / "out" / "gold_train_scoped_u3_reviewed_overlay_v1.jsonl"
QIDS = FS / "out" / "gold_train_scoped_u3_reviewed_overlay_v1_qids.json"

PROMPT = """당신은 보험 약관 질문의 근거를 찾는 검색 에이전트입니다.
현재 디렉터리에서 아래 네 명령만 사용하십시오.

python agent_tools.py search --q "..." [--page N] [--contract "..."] [--role "..."] [--subject "..."] [--qualifier "..."] [--schema "..."]
python agent_tools.py msearch --q "..." [--page N]
python agent_tools.py read --id <element_id_or_jo_id>
python agent_tools.py submit --ids <id1>,<id2>,...

검색 정책:
- search는 C29 semantic-tag 파일서치입니다. 원 질문으로 최소 1회 사용하십시오. CLM 어휘/슬롯 후보와 구조화 BM25F fact 후보를 quota로 합치며, 범주 포함·코드 질문에는 evidence-unit 및 membership support 후보를 조건부로 추가합니다.
- msearch는 고정 V9 메타데이터 BM25+dense RRF 하이브리드 검색입니다. 의미 기반 후보를 확인하기 위해 최소 1회 사용하십시오.
- 두 채널의 유력 후보를 read로 검증하십시오. 질문이 여러 근거를 요구하면 서로 다른 evidence group을 모두 덮으십시오.
- Windows 세션 로그 충돌을 피하기 위해 도구 명령은 병렬 호출하지 말고 한 번에 하나씩 순차 실행하십시오.
- 같은 조의 중복 ID를 피하고 관련도 순으로 최대 10개를 submit하십시오.
- Gold, 평가 파일, 다른 세션 또는 소스 코드를 열람하지 마십시오.
- submit이 성공하면 즉시 종료하십시오.

[질문]
{question}
"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path, normalize_text: bool = False) -> str:
    data = path.read_text(encoding="utf-8").encode("utf-8") if normalize_text else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def resolve_arm() -> dict:
    return json.loads((HERE / "arms.json").read_text(encoding="utf-8"))[ARM_NAME]


def codex_command(codex_bin: str, output: Path) -> list[str]:
    return [
        codex_bin, "exec", "--model", MODEL,
        "--config", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--sandbox", "workspace-write", "--ephemeral",
        "--skip-git-repo-check", "--output-last-message", str(output), "-",
    ]


def start_embedding_server(run_dir: Path) -> tuple[subprocess.Popen | None, str]:
    configured = os.environ.get("EMBED_ENDPOINT", "").strip()
    if configured:
        return None, configured.rstrip("/")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/v1"
    log_path = run_dir / "embedding_server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    process = subprocess.Popen(
        [sys.executable, str(HERE / "embedding_server.py"), "--port", str(port)],
        cwd=HERE, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
    )
    process._log_handle = log_handle  # type: ignore[attr-defined]
    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            log_handle.flush()
            raise RuntimeError(f"embedding server exited; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return process, endpoint
        except Exception:
            time.sleep(1)
    process.terminate()
    raise TimeoutError(f"embedding server startup timeout; see {log_path}")


def stop_embedding_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    handle = getattr(process, "_log_handle", None)
    if handle:
        handle.close()


def run_one(args, run_dir: Path, row: dict, rep: int, arm: dict) -> tuple[dict, dict]:
    session = run_dir / "sessions" / f"{row['qid']}_r{rep}"
    result_path = session / "result.json"
    if args.resume and result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        return saved["submission"], saved["codex"]
    session.mkdir(parents=True, exist_ok=True)
    for name in ("calls.jsonl", "submit.json", "codex.json", "codex_output.txt", "result.json"):
        path = session / name
        if path.exists():
            path.unlink()
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SEMTAG_SESSION=str(session),
        SEMTAG_QID=row["qid"],
        SEMTAG_QUESTION=row["q"],
        SEMTAG_ARM=json.dumps(arm, ensure_ascii=False),
    )
    output = session / "codex_output.txt"
    started = time.time()
    meta = {"model": MODEL, "reasoning_effort": REASONING_EFFORT, "error": None}
    try:
        proc = subprocess.run(
            codex_command(args.codex_bin, output),
            input=PROMPT.format(question=row["q"]),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=HERE,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        meta.update(returncode=proc.returncode, stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:])
        if proc.returncode:
            meta["error"] = f"codex_exit:{proc.returncode}"
    except subprocess.TimeoutExpired:
        meta["error"] = "timeout"
    except OSError as exc:
        meta["error"] = f"os_error:{exc}"
    meta["elapsed"] = time.time() - started
    submission_path = session / "submit.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8")) if submission_path.exists() else {"ranked": []}
    calls = load_jsonl(session / "calls.jsonl") if (session / "calls.jsonl").exists() else []
    channels = sorted({call.get("cmd") for call in calls if call.get("cmd")})
    meta["tools_used"] = channels
    meta["required_tools_ok"] = "search" in channels and "msearch" in channels
    if not submission_path.exists():
        meta["error"] = meta["error"] or "no_submit"
    payload = {"submission": submission, "codex": meta}
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return submission, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="c29_dual_tool_train281_luna_medium_w4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--n", type=int, default=-1, help="Smoke subset size; -1 uses all 281 questions")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN") or shutil.which("codex") or "codex")
    args = parser.parse_args()
    if args.workers != 4:
        raise SystemExit("이 실험은 workers=4로 고정합니다.")

    arm = resolve_arm()
    qids = json.loads(QIDS.read_text(encoding="utf-8"))
    rows_by_qid = {row["qid"]: row for row in load_jsonl(GOLD)}
    missing = [qid for qid in qids if qid not in rows_by_qid]
    if missing:
        raise SystemExit(f"Gold에 없는 qid: {missing[:10]}")
    rows = [rows_by_qid[qid] for qid in qids]
    if 0 < args.n < len(rows):
        rows = random.Random(args.seed).sample(rows, args.n)

    run_dir = HERE / "out" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": vars(args),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "arm_name": ARM_NAME,
        "arm": arm,
        "n_q": len(rows),
        "gold": str(GOLD.resolve()),
        "gold_sha256_lf": digest(GOLD, normalize_text=True),
        "qids": str(QIDS.resolve()),
        "qids_sha256_lf": digest(QIDS, normalize_text=True),
        "runner_sha256": digest(Path(__file__)),
        "agent_tools_wrapper_sha256": digest(HERE / "agent_tools.py"),
        "base_agent_tools_sha256": digest(HERE.parent / "0819" / "agent_tools.py"),
        "filesearch_code": {
            name: digest(FS / name)
            for name in ("build_tags_u4_fact.py", "structured_search.py", "schema_adapter.py", "clm_search.py", "scoring.py", "units.py")
        },
        "data": {
            name: digest(FS / "out" / name, normalize_text=True)
            for name in ("elements_u3.jsonl", "elements_u3jo.jsonl", "tags_u4_fact_rules.jsonl")
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    units = Units(str(FS / "out" / arm["jo"]))
    jobs = [(row, rep) for rep in range(args.reps) for row in rows]
    results = []
    embedding_process, embedding_endpoint = start_embedding_server(run_dir)
    os.environ["EMBED_ENDPOINT"] = embedding_endpoint
    manifest["embedding_endpoint"] = embedding_endpoint
    manifest["embedding_server_managed"] = embedding_process is not None
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"run={args.run} n={len(rows)} reps={args.reps} model={MODEL} workers=4 embed={embedding_endpoint}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = pool.map(lambda job: run_one(args, run_dir, job[0], job[1], arm), jobs)
            for (row, rep), (submission, meta) in zip(jobs, outputs):
                ranked = submission.get("ranked", [])
                metrics = score(units.resolve(ranked), row["groups"], ks=(1, 5, 10, 20))
                result = {
                    "qid": row["qid"], "rep": rep, "submitted": ranked,
                    "error": int(bool(meta.get("error"))),
                    "required_tools_ok": bool(meta.get("required_tools_ok")),
                    "elapsed": meta.get("elapsed", 0), **metrics,
                }
                results.append(result)
                print(row["qid"], f"R@5={metrics['R@5']:.3f}", f"n={len(ranked)}",
                      f"tools_ok={result['required_tools_ok']}", f"error={result['error']}", flush=True)
    finally:
        stop_embedding_server(embedding_process)

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    metric_keys = ("R@1", "R@5", "R@10", "R@20", "suff@5", "suff@10", "RR@10")
    summary = {
        "run": args.run, "n_q": len(rows), "reps": args.reps, "model": MODEL,
        "reasoning_effort": REASONING_EFFORT, "workers": 4,
        "metrics": {key: sum(row[key] for row in results) / len(results) for key in metric_keys},
        "errors": sum(row["error"] for row in results),
        "required_tool_violations": sum(not row["required_tools_ok"] for row in results),
        "mean_elapsed": sum(row["elapsed"] for row in results) / len(results),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
