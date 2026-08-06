#!/usr/bin/env python3
"""Run frozen first-25 OLD_RET/NEW_RET as isolated Codex CLI sessions."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
FILE_ARMS = ("OLD_RET", "NEW_RET")
VECTOR_ARMS = ("VECTOR_OLD_RET", "VECTOR_NEW_RET")
SESSION_ROOT = "codex_retriever_ab25_sessions"


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def prompt(row, arm, session_dir):
    tool = "python3 codex_retrieval_tool.py"
    common = f"""보험 약관 검색 에이전트다. 질문의 정답 근거 unit을 찾아 최대 10개까지 정확도순으로 제출하라.
질문: {row['question']}
Arm: {arm}
gold/정답/평가 파일은 절대 열지 마라. 아래 검색 명령만 사용하라. 도구는 합계 최대 8회다.
Vector: {tool} --arm {arm} --session-dir {session_dir} vector --query '검색 질의'
Read: {tool} --arm {arm} --session-dir {session_dir} read --id UNIT_ID
"""
    if arm == "OLD_RET":
        common += f"FileSearch는 기존 정규식 검색이다: {tool} --arm {arm} --session-dir {session_dir} file --pattern '정규식'\n"
    else:
        common += f"""FileSearch는 구조 슬롯 AND 검색이다:
{tool} --arm {arm} --session-dir {session_dir} file --filters '{{"contract":"특약명","role":"premium_waiver","subject":"대상"}}'
사용 가능한 슬롯: contract, subject, role, article, table, qualifier, reference, schema.
필드 사이는 AND, 한 필드의 배열 값은 OR이다. total_matches가 20보다 크면 질문에서 추론 가능한 슬롯을 추가해 재질의하라.
role 값: exclusion_exception, premium_waiver, payment_trigger, payment_amount, limit_frequency, timing_period, definition, criteria_rule, contract_lifecycle, claim_procedure, code_reference.
원문 정규식까지 함께 제한하려면 --raw-regex를 사용할 수 있다.
"""
    return common + "후보 원문을 최소 1개 read로 검증한 뒤 지정된 JSON 스키마로만 답하라."


def run_one(row, arm):
    session_dir = OUT / SESSION_ROOT / f"{row['qid']}_{arm}"
    session_dir.mkdir(parents=True, exist_ok=True)
    result_path = session_dir / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") != "error":
            return cached
    command = [
        "codex", "-s", "workspace-write", "-a", "never", "-C", str(BASE),
        "exec", "--ephemeral", "--color", "never", "--output-schema", str(OUT / "codex_retrieval_result_schema.json"),
        "-o", str(result_path), prompt(row, arm, session_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"status": "error", "ranked_unit_ids": [], "reason": (completed.stderr or completed.stdout)[-1000:]}
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    result.update({"qid": row["qid"], "question": row["question"], "arm": arm, "returncode": completed.returncode})
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--experiment", choices=("file", "vector"), default="file")
    args = parser.parse_args()
    global SESSION_ROOT
    arms = FILE_ARMS if args.experiment == "file" else VECTOR_ARMS
    SESSION_ROOT = "codex_retriever_ab25_sessions" if args.experiment == "file" else "codex_vector_retriever_ab25_sessions"
    manifest = json.loads((OUT / "retriever_ab25_manifest.json").read_text())
    by_qid = {row["qid"]: row for row in load("train350_gold_repaired_v1.jsonl")}
    selected = [by_qid[qid] for qid in manifest["qids"][:args.limit]]
    jobs = [(row, arm) for row in selected for arm in arms]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, row, arm) for row, arm in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"{index}/{len(jobs)} errors={sum(row['status']=='error' for row in results)}", flush=True)
    output_name = "codex_retriever_ab25_results.jsonl" if args.experiment == "file" else "codex_vector_retriever_ab25_results.jsonl"
    with (OUT / output_name).open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: (item["qid"], item["arm"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
