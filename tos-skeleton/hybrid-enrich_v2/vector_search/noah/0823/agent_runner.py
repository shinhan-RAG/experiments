#!/usr/bin/env python3
"""Claude CLI runner for the 0823 host_retrieve / agentic dual-mode experiment."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
VS = ROOT / "vector_search"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(VS))
from scoring import score  # noqa: E402
from units import Units  # noqa: E402
from tag_hybrid import TagHybridSearch  # noqa: E402

PROMPT = """당신은 보험 약관 문서에서 질문의 근거 조항을 찾는 검색 에이전트입니다.
사용 가능한 명령은 아래 네 가지뿐입니다. 반드시 명령을 통해 검색하고 마지막에 submit을 한 번 호출하십시오.

1) 태그 검색: {py} agent_tools.py search --q "<검색어>" [--page N] [--contract "<특약명>"] [--role "<역할코드>"] [--subject "<대상어>"] [--qualifier "<조건>"] [--schema "<table|formula>"]
2) 메타 검색: {py} agent_tools.py msearch --q "<자연어 질의>" [--page N]
   - V9 메타데이터의 BM25+dense RRF 검색이며 태그 검색과 독립적입니다.
3) 원문 읽기: {py} agent_tools.py read --id <검색결과 id>
4) 제출: {py} agent_tools.py submit --ids <id1>,<id2>,...  (관련도 순 최대 10개)

search와 msearch를 질문 성격에 맞게 함께 사용하십시오. 결과가 동떨어지면 고객 표현과 약관 표현을 바꾸어 재검색하십시오.
여러 근거를 요구하는 질문은 서로 다른 사실 축과 서로 다른 조를 찾으십시오. 같은 조의 중복 id는 제출하지 마십시오.
답변 문장을 작성하지 말고 근거 id만 submit 하십시오.

[질문]
{question}
"""

PROMPT_INCLUSIVE = """당신은 보험 약관 문서에서 질문의 근거 조항을 찾는 검색 에이전트입니다.
사용 가능한 명령은 아래 네 가지뿐입니다. 반드시 명령을 통해 검색하고 마지막에 submit을 한 번 호출하십시오.

1) 태그 검색: {py} agent_tools.py search --q "<검색어>" [--page N] [--contract "<특약명>"] [--role "<역할코드>"] [--subject "<대상어>"] [--qualifier "<조건>"] [--schema "<table|formula>"]
2) 메타 검색: {py} agent_tools.py msearch --q "<자연어 질의>" [--page N]
3) 원문 읽기: {py} agent_tools.py read --id <검색결과 id>
4) 제출: {py} agent_tools.py submit --ids <id1>,<id2>,...  (관련도 순 최대 10개)

search와 msearch를 질문 성격에 맞게 함께 사용하십시오. 결과가 동떨어지면 고객 표현과 약관 표현을 바꾸어 재검색하십시오.
여러 근거를 요구하는 질문은 서로 다른 사실 축과 서로 다른 조를 찾으십시오. 같은 조의 중복 id는 제출하지 마십시오.
확실하지 않은 후보도 포함하십시오. 미포함(false negative)이 오포함(false positive)보다 훨씬 나쁩니다.
유사한 조항이 여러 특약에 반복되면 모두 포함하십시오.
답변 문장을 작성하지 말고 근거 id만 submit 하십시오.

[질문]
{question}
"""

PROMPT_RERANK = """당신은 보험 약관 문서에서 질문의 근거 조항을 찾는 재순위화 에이전트입니다.
호스트가 미리 검색한 후보 목록이 있습니다. 당신의 역할은:
1) 후보 확인: {py} agent_tools.py candidates [--page N]
2) 유망한 후보 원문 읽기: {py} agent_tools.py read --id <id>
3) 관련 후보 제출: {py} agent_tools.py submit --ids <id1>,<id2>,...  (관련도 순 최대 10개)

search와 msearch는 사용할 수 없습니다. 반드시 candidates 명령으로 후보를 확인하십시오.
확실하지 않은 후보도 포함하십시오. 미포함이 오포함보다 훨씬 나쁩니다.
여러 근거를 요구하는 질문은 서로 다른 사실 축의 후보를 각각 찾으십시오.
답변 문장을 작성하지 말고 근거 id만 submit 하십시오.

[질문]
{question}
"""

PROMPT_RERANK_GREEDY = """당신은 보험 약관 문서에서 질문의 근거 조항을 찾는 재순위화 에이전트입니다.
호스트가 미리 검색한 후보 목록이 있습니다. 당신의 역할은:
1) 후보 확인: {py} agent_tools.py candidates [--page N]
2) 상위 20개 후보의 원문을 모두 읽기: {py} agent_tools.py read --id <id>
3) 질문 주제와 조금이라도 관련된 후보 전부 제출: {py} agent_tools.py submit --ids <id1>,<id2>,...

search와 msearch는 사용할 수 없습니다. 반드시 candidates 명령으로 후보를 확인하십시오.
정밀도보다 재현율이 중요합니다. 조금이라도 관련이 있으면 반드시 포함하십시오.
여러 근거를 요구하는 질문은 서로 다른 사실 축의 후보를 각각 찾으십시오.
답변 문장을 작성하지 말고 근거 id만 submit 하십시오.

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
        raise SystemExit(f"arms.json에 없는 arm: {name}")
    return arms[name]


def _snippet(text: str, qtoks: list[str], chars: int = 360) -> str:
    clean = " ".join((text or "").split())
    positions = [clean.lower().find(t) for t in qtoks if clean.lower().find(t) >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    return clean[start:start + chars]


def pre_search(question: str, qid: str, arm: dict, session: Path) -> dict:
    """Host-controlled search: run tag + meta, union, write candidates.json"""
    elements_path = FS / "out" / arm.get("elements", "elements_u2.jsonl")
    tags_path = FS / "out" / arm.get("tags", "tags_u2_rules.jsonl")
    jo_path = FS / "out" / arm.get("jo", "elements_u2jo.jsonl")
    engine = TagHybridSearch(elements_path, tags_path, jo_path, HERE / "out" / "tag_index")

    qtag_rel = arm.get("qtags")
    explicit = {}
    if arm.get("router") in ("llm", "union") and qtag_rel:
        qtag_path = FS / qtag_rel
        if qtag_path.exists():
            for line in qtag_path.open(encoding="utf-8"):
                row = json.loads(line)
                if row.get("qid") == qid:
                    explicit = {k: row.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
                    break

    tag_rows, slots, qtoks = engine.search(
        question, explicit=explicit, weights=arm.get("tag_weights"),
        channel_top_k=200, limit=200, collapse_jo=bool(arm.get("collapse_jo", True))
    )

    meta_items = []
    if arm.get("meta"):
        from hybrid_search import ChunkHybridSearch
        hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
        meta_results = hs.search(question, strategy="hybrid", top_k=200)
        units = Units(str(jo_path))
        for result in meta_results:
            unit = units.jo_of_span(result["char_start"], result["char_end"]) if result.get("char_start") is not None else None
            meta_items.append({
                "id": result["id"],
                "jo": unit["element_id"] if unit else "",
                "score": 0,
                "source": "meta",
            })

    qtoks_list = [x.lower() for x in re.findall(r"[가-힣A-Za-z0-9%]+", question or "") if len(x) > 1]

    rrf_k = 60
    rrf_scores: dict[str, float] = {}
    tag_info: dict[str, dict] = {}
    for rank, row in enumerate(tag_rows, 1):
        element = row["element"]
        jo = row["jo"]
        if jo:
            rrf_scores[jo] = rrf_scores.get(jo, 0) + 1.0 / (rrf_k + rank)
            if jo not in tag_info:
                unit = engine.jid.get(jo)
                base = (element.get("contract_scope") or "").split("(무배당")[0].strip()
                tag_info[jo] = {
                    "id": element["element_id"],
                    "jo": jo,
                    "source": "tag",
                    "contract": base,
                    "variant": "",
                    "jo_title": ((unit or {}).get("title") or "")[:60],
                    "preview": _snippet(element.get("text", ""), qtoks_list),
                }

    meta_info: dict[str, dict] = {}
    for rank, item in enumerate(meta_items, 1):
        jo = item["jo"]
        if jo:
            rrf_scores[jo] = rrf_scores.get(jo, 0) + 1.0 / (rrf_k + rank)
            if jo not in tag_info and jo not in meta_info:
                unit = engine.jid.get(jo)
                if unit:
                    base = (unit.get("contract_scope") or "").split("(무배당")[0].strip()
                    meta_info[jo] = {
                        "id": item["id"],
                        "jo": jo,
                        "source": "meta",
                        "contract": base,
                        "variant": "",
                        "jo_title": (unit.get("title") or "")[:60],
                        "preview": "",
                    }

    host_top_k = int(arm.get("host_top_k", 40))
    ordered_jos = sorted(rrf_scores, key=lambda j: (-rrf_scores[j], j))[:host_top_k]
    candidates = []
    for jo in ordered_jos:
        info = tag_info.get(jo) or meta_info.get(jo)
        if info:
            info["score"] = round(rrf_scores[jo], 7)
            candidates.append(info)

    payload = {"qid": qid, "query": question, "total": len(candidates), "candidates": candidates}
    (session / "candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_one(args, run_dir: Path, gold: dict, rep: int, arm: dict, arm_name: str):
    session = run_dir / "sessions" / f"{gold['qid']}_r{rep}"
    done = session / "claude.json"
    submitted = session / "submit.json"
    if args.resume and done.exists() and submitted.exists():
        return json.loads(submitted.read_text(encoding="utf-8")), json.loads(done.read_text(encoding="utf-8"))
    session.mkdir(parents=True, exist_ok=True)
    for name in ("calls.jsonl", "submit.json", "claude.json", "candidates.json"):
        p = session / name
        if p.exists():
            p.unlink()

    mode = arm.get("mode", "agentic")
    if mode == "host_retrieve":
        pre_search(gold["q"], gold["qid"], arm, session)
        prompt_tpl = PROMPT_RERANK_GREEDY if arm.get("greedy") else PROMPT_RERANK
    else:
        prompt_tpl = PROMPT_INCLUSIVE if arm_name == "expanded_inclusive" else PROMPT

    env = dict(os.environ)
    env.update(SEMTAG_SESSION=str(session), SEMTAG_QID=gold["qid"], SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    cmd = [
        args.claude_bin,
        "-p",
        "--model",
        args.model,
        "--output-format",
        "json",
        "--max-turns",
        str(args.max_turns),
        "--allowedTools",
        f"Bash({args.pybin} agent_tools.py:*)",
        "--disallowedTools",
        "Read,Edit,Write,Grep,Glob,WebFetch,WebSearch,Agent,NotebookEdit,Task",
    ]
    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=prompt_tpl.format(py=args.pybin, question=gold["q"]),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            cwd=HERE,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        try:
            claude = json.loads(result.stdout)
        except Exception:
            claude = {"parse_error": True, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        claude = {"timeout": True}
    claude["_elapsed"] = time.time() - started
    done.write_text(json.dumps(claude, ensure_ascii=False), encoding="utf-8")
    sub = json.loads(submitted.read_text(encoding="utf-8")) if submitted.exists() else {"qid": gold["qid"], "ranked": [], "no_submit": True}
    return sub, claude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--jo", type=Path, default=FS / "out" / "elements_u2jo.jsonl")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--pybin", default="python")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    arm = resolve_arm(args.arm)
    gold_rows = [
        row for row in map(json.loads, args.gold.open(encoding="utf-8"))
        if row.get("groups") and row.get("status", "ok") == "ok"
    ]
    if args.n > 0 and args.n < len(gold_rows):
        rnd = random.Random(args.seed)
        core = [g for g in gold_rows if is_core(g)]
        other = [g for g in gold_rows if not is_core(g)]
        n_core = round(args.n * len(core) / len(gold_rows))
        gold_rows = rnd.sample(core, min(n_core, len(core))) + rnd.sample(other, args.n - min(n_core, len(core)))
        gold_rows.sort(key=lambda x: x["qid"])
    run_dir = HERE / "out" / "agent" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"arm_name": args.arm, "arm": arm, "args": {**vars(args), "gold": str(args.gold), "jo": str(args.jo)}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    units = Units(str(args.jo))
    jobs = [(g, rep) for rep in range(args.reps) for g in gold_rows]
    rows = []
    print(f"run={args.run} arm={args.arm} n={len(gold_rows)} reps={args.reps} model={args.model}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outputs = pool.map(lambda job: run_one(args, run_dir, job[0], job[1], arm, args.arm), jobs)
        for (g, rep), (submitted, claude) in zip(jobs, outputs):
            metrics = score(units.resolve(submitted.get("ranked", [])), g["groups"], ks=(1, 5, 10, 20))
            row = {
                "qid": g["qid"], "rep": rep, "core": is_core(g), "task_type": g.get("task_type", ""),
                "c3_partial": bool(g.get("c3_partial")), "submitted": submitted.get("ranked", []),
                "no_submit": bool(submitted.get("no_submit")), "cost_usd": claude.get("total_cost_usd"),
                "turns": claude.get("num_turns"), "elapsed": claude.get("_elapsed"),
                "error": bool(claude.get("is_error") or claude.get("timeout") or claude.get("parse_error")), **metrics,
            }
            rows.append(row)
            print(f"{g['qid']} r{rep} R@5={metrics['R@5']:.2f} submitted={len(row['submitted'])}", flush=True)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_q = collections.defaultdict(list)
    for row in rows:
        by_q[row["qid"]].append(row)
    keys = ("R@1", "R@5", "R@10", "R@20", "S@5", "suff@5", "suff@10", "RR@10")
    metrics = {key: sum(sum(x[key] for x in group) / len(group) for group in by_q.values()) / len(by_q) for key in keys}
    strict = [group for group in by_q.values() if not group[0].get("c3_partial")]
    strict_metrics = {key: sum(sum(x[key] for x in group) / len(group) for group in strict) / len(strict) for key in keys} if strict else {}
    summary = {
        "run": args.run, "arm_name": args.arm, "n_q": len(by_q), "reps": args.reps, "model": args.model,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "strict_nonpartial": {"n_q": len(strict), "metrics": {k: round(v, 4) for k, v in strict_metrics.items()}},
        "no_submit": sum(x["no_submit"] for x in rows), "errors": sum(x["error"] for x in rows),
        "cost_usd": round(sum(x["cost_usd"] or 0 for x in rows), 2),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
