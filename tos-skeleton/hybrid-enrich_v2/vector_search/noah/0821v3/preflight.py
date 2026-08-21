#!/usr/bin/env python3
"""Validate frozen artifacts, Codex CLI, and both retrieval channels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
EXPECTED = {
    "elements_u3.jsonl": "9d13e8d61781a4d39593795b0a30b33d41da7e1134335f04056e3d35540f5a49",
    "elements_u3jo.jsonl": "44d856004c473e186880cbe749c70d701d1ae25f528c6128cc575a004e8d980c",
    "tags_u4_fact_rules.jsonl": "19df8e21d9f9287394cf9d02924d5d0e2c84be7e9bf3ae9fca4f02aaacd18104",
    "gold_train_scoped_u3_reviewed_overlay_v1.jsonl": "d8038733c364f6441dd3d8bfc9bd3ac5bccb1b640b8458f344a96291be0936af",
    "gold_train_scoped_u3_reviewed_overlay_v1_qids.json": "63d35e9f132392a659c34fd004bb60e550bcbbba38c766dd6acc074fe78752ea",
}


def normalized_digest(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def run_tool(args: list[str], arm: dict, session: str) -> dict:
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        SEMTAG_SESSION=session,
        SEMTAG_QID="preflight",
        SEMTAG_QUESTION="보험료 납입면제 조건은?",
        SEMTAG_ARM=json.dumps(arm, ensure_ascii=False),
    )
    proc = subprocess.run(
        [sys.executable, str(HERE / "agent_tools.py"), *args],
        cwd=HERE, env=env, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tool-smoke", action="store_true")
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN") or shutil.which("codex") or "codex")
    args = parser.parse_args()
    report = {"artifacts": {}, "dependencies": {}, "tools": {}}
    for name, expected in EXPECTED.items():
        path = FS / "out" / name
        actual = normalized_digest(path)
        report["artifacts"][name] = {"path": str(path), "sha256_lf": actual, "ok": actual == expected}
        if actual != expected:
            raise SystemExit(f"요청 커밋 artifact SHA 불일치: {name}: {actual} != {expected}")

    gold_rows = [json.loads(line) for line in (FS / "out" / "gold_train_scoped_u3_reviewed_overlay_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    qids = json.loads((FS / "out" / "gold_train_scoped_u3_reviewed_overlay_v1_qids.json").read_text(encoding="utf-8"))
    if len(gold_rows) != 281 or qids != [row["qid"] for row in gold_rows] or not all(row.get("groups") for row in gold_rows):
        raise SystemExit("281 QA/Gold 정렬 또는 group 검증 실패")

    required = [
        VS / "out" / "view_V9.jsonl", VS / "out" / "chunks.jsonl",
        VS / "out" / "emb" / "chunk_V9.npy", VS / "out" / "emb" / "chunk_V9_ids.json",
        HERE.parent / "0819" / "agent_tools.py",
    ]
    for path in required:
        report["dependencies"][str(path)] = path.exists()
        if not path.exists():
            raise SystemExit(f"필수 파일 없음: {path}")

    version = subprocess.run([args.codex_bin, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    auth_proc = subprocess.run([args.codex_bin, "login", "status"], check=True, capture_output=True, text=True)
    auth = (auth_proc.stdout or auth_proc.stderr).strip()
    report["codex"] = {"binary": args.codex_bin, "version": version, "auth": auth,
                       "model": "gpt-5.6-luna", "reasoning_effort": "medium", "workers": 4}

    if not args.skip_tool_smoke:
        arm = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))["c29_dual_tool_hybrid"]
        with tempfile.TemporaryDirectory(prefix="c29_0821v3_preflight_") as session:
            semantic = run_tool(["search", "--q", "보험료 납입면제 조건은?"], arm, session)
            metadata = run_tool(["msearch", "--q", "보험료를 내지 않아도 되는 조건"], arm, session)
        if not semantic.get("results"):
            raise SystemExit(f"semantic search smoke 실패: {semantic}")
        if metadata.get("channel") != "meta:hybrid" or not metadata.get("results"):
            raise SystemExit(f"metadata hybrid smoke 실패: {metadata}")
        report["tools"] = {
            "semantic": {"results": len(semantic["results"]), "ranker": "C29 quota ensemble"},
            "metadata": {"results": len(metadata["results"]), "channel": metadata["channel"]},
        }
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
