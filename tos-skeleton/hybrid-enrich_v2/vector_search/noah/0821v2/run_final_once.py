#!/usr/bin/env python3
"""Preflight and execute the candidate-only v5 test exactly once."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "gpt-5.6-luna"
ARM = "slot_meta_agent"
RUN = "0821v2_v5_test_slot_meta_luna_workers4_meta_enforced_once"
GOLD_MANIFEST = HERE / "out" / "gold_v5" / "gold_v5_manifest.json"
LOCK = HERE / "out" / "final_once" / "lock_workers4_meta_enforced.json"
REPORT = HERE / "out" / "final_once" / "report_workers4_meta_enforced.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def preflight(codex_bin: str) -> dict:
    subprocess.run([sys.executable, str(HERE / "build_gold_v5.py")], check=True, cwd=HERE)
    manifest = json.loads(GOLD_MANIFEST.read_text(encoding="utf-8"))
    gold = Path(manifest["test"]["path"])
    if manifest["test"]["n"] <= 0 or digest(gold) != manifest["test"]["sha256"]:
        raise SystemExit("v5 test manifest/hash 검증 실패")
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    arm = arms.get(ARM) or {}
    if arm.get("tag_weights") != {"clm": 1.0, "sparse": 0.0}:
        raise SystemExit("slot_meta_agent가 slot-only가 아닙니다.")
    if arm.get("router") != "rules" or arm.get("qtags") or arm.get("meta_strategy") != "hybrid" or not arm.get("require_meta_before_submit"):
        raise SystemExit("slot/meta 라우팅 불변식 위반")
    qtags_path = HERE.parents[2] / "filesearch" / "out" / "qtags_haiku.jsonl"
    if qtags_path.exists():
        test_qids = {json.loads(line)["qid"] for line in gold.open(encoding="utf-8")}
        qtag_qids = {json.loads(line).get("qid") for line in qtags_path.open(encoding="utf-8") if line.strip()}
        if test_qids & qtag_qids:
            raise SystemExit("test qid가 frozen qtags에 포함되어 있습니다.")
    codex = Path(codex_bin)
    if not codex.exists():
        raise SystemExit(f"Codex CLI 없음: {codex}")
    version = subprocess.run([str(codex), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="slot_meta_preflight_") as temp:
        env = dict(os.environ, SEMTAG_SESSION=temp, SEMTAG_QID="preflight", SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
        checks = [
            ("search", [sys.executable, str(HERE / "agent_tools.py"), "search", "--q", "보험료 납입면제 조건", "--role", "premium_waiver"], "tag:slot"),
            ("msearch", [sys.executable, str(HERE / "agent_tools.py"), "msearch", "--q", "보험료를 내지 않아도 되는 조건"], "meta:hybrid"),
        ]
        for name, cmd, channel in checks:
            result = subprocess.run(cmd, cwd=HERE, env=env, check=True, capture_output=True, text=True, encoding="utf-8", timeout=180)
            payload = json.loads(result.stdout)
            if payload.get("channel") != channel or not payload.get("results"):
                raise SystemExit(f"{name} 도구 preflight 실패: {payload}")
    return {"model": MODEL, "arm": ARM, "run": RUN, "gold": str(gold), "gold_sha256": digest(gold), "n": manifest["test"]["n"], "codex": str(codex), "codex_version": version, "tool_preflight": ["tag:slot", "meta:hybrid"], "test_qtag_overlap": 0}


def summarize(run_dir: Path, config: dict, returncode: int) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    channels, commands = Counter(), Counter()
    sessions = run_dir / "sessions"
    if sessions.exists():
        for calls in sessions.glob("*/calls.jsonl"):
            for line in calls.open(encoding="utf-8"):
                call = json.loads(line)
                commands[call.get("cmd", "unknown")] += 1
                if call.get("channel"):
                    channels[call["channel"]] += 1
    successful_meta_qids = set()
    if sessions.exists():
        for calls in sessions.glob("*/calls.jsonl"):
            for line in calls.open(encoding="utf-8"):
                call = json.loads(line)
                if call.get("cmd") == "msearch" and call.get("channel") == "meta:hybrid" and call.get("returned"):
                    successful_meta_qids.add(call.get("qid"))
    return {
        "run_once": True, "returncode": returncode, "config": config, "summary": summary,
        "tool_calls": dict(commands), "channels": dict(channels), "finished_at": now(),
        "successful_meta_qids": len(successful_meta_qids),
        "all_questions_used_meta": len(successful_meta_qids) == config["n"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_codex = os.environ.get("CODEX_BIN") or str(Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd")
    parser.add_argument("--codex-bin", default=default_codex)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.workers != 4:
        raise SystemExit("최종 재실행은 workers=4로 고정되어 있습니다.")
    config = preflight(args.codex_bin)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    run_dir = HERE / "out" / "agent" / RUN
    if LOCK.exists() or run_dir.exists():
        raise SystemExit(f"최종 1회 실행이 이미 시작되었거나 결과가 존재합니다: {LOCK}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = {**config, "status": "started", "started_at": now(), "reps": 1, "baseline": False}
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cmd = [
        sys.executable, str(HERE / "agent_runner.py"), "--run", RUN, "--arm", ARM,
        "--gold", config["gold"], "--n", "-1", "--reps", "1", "--workers", str(args.workers),
        "--model", MODEL, "--codex-bin", args.codex_bin,
    ]
    result = subprocess.run(cmd, cwd=HERE)
    report = summarize(run_dir, config, result.returncode)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verified = result.returncode == 0 and report["all_questions_used_meta"]
    lock.update(status="completed" if verified else "failed", finished_at=now(), returncode=result.returncode,
                all_questions_used_meta=report["all_questions_used_meta"])
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not verified:
        raise SystemExit(result.returncode or 2)


if __name__ == "__main__":
    main()
