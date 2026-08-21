#!/usr/bin/env python3
"""Thin policy wrapper around the reviewed C29 tool implementation in ../0819."""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_TOOL = HERE.parent / "0819" / "agent_tools.py"


def submission_policy_error() -> str | None:
    if len(sys.argv) < 2 or sys.argv[1] != "submit":
        return None
    arm = json.loads(os.environ.get("SEMTAG_ARM", "{}"))
    session = Path(os.environ.get("SEMTAG_SESSION", "."))
    calls_path = session / "calls.jsonl"
    calls = []
    if calls_path.exists():
        calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    commands = {row.get("cmd") for row in calls}
    missing = []
    if arm.get("require_search_before_submit") and "search" not in commands:
        missing.append("search")
    if arm.get("require_meta_before_submit") and "msearch" not in commands:
        missing.append("msearch")
    if missing:
        return "submit 전 필수 도구가 누락되었습니다: " + ", ".join(missing)
    return None


def sanitize_session_log() -> None:
    """Remove blank lines left by overlapping Windows append handles before base-tool reads."""
    session = Path(os.environ.get("SEMTAG_SESSION", "."))
    calls_path = session / "calls.jsonl"
    if not calls_path.exists():
        return
    lines = calls_path.read_text(encoding="utf-8").splitlines()
    valid = [line for line in lines if line.strip()]
    for line in valid:
        json.loads(line)
    if len(valid) != len(lines):
        calls_path.write_text("\n".join(valid) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "submit":
        sanitize_session_log()
    error = submission_policy_error()
    if error:
        print(json.dumps({"error": error}, ensure_ascii=False))
        raise SystemExit(0)
    runpy.run_path(str(BASE_TOOL), run_name="__main__")
