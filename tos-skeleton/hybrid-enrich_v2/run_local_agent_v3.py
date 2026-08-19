#!/usr/bin/env python3
"""Run a local tool-using agent on the 72-question Train intersection.

This replaces unavailable Claude capacity for today's within-run 4-arm test.
It logs every returned candidate so retrieval and selection loss are separable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
MODEL = "qwen2.5:3b-instruct-q8_0"
ARMS = {
    "BASE": ("vec_base", "grep_base"),
    "OLD_META": ("vec_meta_v2", "grep_base"),
    "META": ("vec_meta_v3_concat", "grep_base"),
    "OLD_TAG": ("vec_base", "grep_tag_v2"),
    "TAG": ("vec_base", "grep_tag_v3"),
    "BOTH": ("vec_meta_v3_concat", "grep_tag_v3"),
}
SYSTEM = """너는 보험 약관 검색 에이전트다. 질문의 근거가 있는 청크/엘리먼트를 찾아라.
vector_search는 의미 기반 청크(c*) 검색, file_search는 정규식 기반 엘리먼트(e*) 검색이다.
정확한 특약명과 핵심 역할을 짧은 정규식으로 검색하라. 첫 검색 한 번만으로 종료하지 말고
vector_search와 file_search를 교차검증하며, 최종 제출 전에 최소 하나의 후보를 read_unit으로 확인하라.
도구는 최대 8회 사용한다. 마지막 응답은 반드시 다음 JSON 하나만 출력한다:
{"status":"ranked","ranked_chunk_ids":["e00001","c00002"],"final_reason":"..."}
후보 ID는 최대 10개이며 근거 가능성이 높은 순서로 둔다. 검색된 유력 후보가 여러 개면 하나만 버리지 말고 최대 10개까지 유지하라."""
TOOL_DEFS = [
    {"type": "function", "function": {"name": "vector_search", "description": "자연어 의미 검색 top10", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "file_search", "description": "짧은 키워드 또는 정규식으로 엘리먼트 검색", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_unit", "description": "청크 또는 엘리먼트 원문 읽기", "parameters": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]}}},
]


def post(path: str, payload: dict, timeout=300):
    request = urllib.request.Request(
        f"http://localhost:11434{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Search:
    def __init__(self):
        self.vecs = {}
        self.vec_ids = {}
        for stem, _ in set(ARMS.values()):
            self.vecs[stem] = np.load(OUT / f"{stem}.npy", mmap_mode="r")
            self.vec_ids[stem] = json.loads((OUT / f"{stem}_ids.json").read_text())["ids"]
        self.chunks = {row["chunk_id"]: row["text"] for row in load_jsonl(OUT / "chunks.jsonl")}
        self.elements = {row["element_id"]: row["text"] for row in load_jsonl(OUT / "elements.jsonl")}
        self.grep = {}
        for _, stem in set(ARMS.values()):
            self.grep[stem] = load_jsonl(OUT / f"{stem}.jsonl")

    def vector(self, stem, query):
        result = post("/api/embed", {"model": "bge-m3", "input": [query], "keep_alive": "30m"})
        vector = np.asarray(result["embeddings"][0], dtype=np.float32)
        vector /= np.linalg.norm(vector) or 1
        scores = self.vecs[stem] @ vector
        top = np.argsort(-scores)[:10]
        ids = self.vec_ids[stem]
        return [{"id": ids[index], "score": round(float(scores[index]), 4), "preview": self.chunks[ids[index]][:240]} for index in top]

    def file(self, stem, pattern):
        try:
            regex = re.compile(pattern, re.I)
        except re.error:
            regex = re.compile(re.escape(pattern), re.I)
        matched = [row for row in self.grep[stem] if regex.search(row["g"])]
        total = len(matched)
        if total > 20:
            step = total / 20
            matched = [matched[int(index * step)] for index in range(20)]
        return {"total_matches": total, "results": [{"id": row["element_id"], "match": regex.search(row["g"]).group(0)[:180] if regex.search(row["g"]) else ""} for row in matched]}

    def read(self, uid):
        text = self.chunks.get(uid) or self.elements.get(uid)
        return {"id": uid, "text": text[:4000] if text else "", "found": text is not None}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def parse_final(text):
    matches = re.findall(r"\{[\s\S]*?\}", text or "")
    for candidate in reversed(matches):
        try:
            row = json.loads(candidate)
            if "ranked_chunk_ids" in row:
                return row
        except json.JSONDecodeError:
            pass
    return {"status": "error", "ranked_chunk_ids": [], "final_reason": "final JSON parse failed"}


def run_one(search, qid, question, arm, output_dir):
    target = output_dir / f"{qid}_{arm}.json"
    if target.exists():
        return json.loads(target.read_text())
    vector_stem, grep_stem = ARMS[arm]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls = []
    started = time.time()
    final = None
    try:
        for _ in range(9):
            response = post("/api/chat", {"model": MODEL, "messages": messages, "tools": TOOL_DEFS, "stream": False, "keep_alive": "30m", "options": {"temperature": 0}}, timeout=420)
            message = response["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final = parse_final(message.get("content", ""))
                break
            for call in tool_calls:
                if len(calls) >= 8:
                    break
                fn = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if fn == "vector_search":
                    result = search.vector(vector_stem, str(args.get("query", question)))
                elif fn == "file_search":
                    result = search.file(grep_stem, str(args.get("pattern", "")))
                else:
                    result = search.read(str(args.get("unit_id", "")))
                calls.append({"tool": fn, "args": args, "result": result})
                messages.append({"role": "tool", "tool_name": fn, "content": json.dumps(result, ensure_ascii=False)})
            if len(calls) >= 8:
                messages.append({"role": "user", "content": "도구 상한에 도달했다. 지금까지 후보로 최종 JSON만 출력하라."})
        if final is None:
            response = post("/api/chat", {"model": MODEL, "messages": messages + [{"role": "user", "content": "최종 JSON만 출력하라."}], "stream": False, "keep_alive": "30m", "options": {"temperature": 0}})
            final = parse_final(response["message"].get("content", ""))
    except Exception as exc:
        final = {"status": "error", "ranked_chunk_ids": [], "final_reason": repr(exc)[:300]}
    row = {"qid": qid, "question": question, "arm": arm, "status": final.get("status", "error"), "ranked_chunk_ids": final.get("ranked_chunk_ids", [])[:10], "final_reason": final.get("final_reason", ""), "tool_calls": calls, "n_tool_calls": len(calls), "model": MODEL, "duration_ms": int((time.time() - started) * 1000)}
    target.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="BASE,META,TAG,BOTH")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    train = {row["no"] for row in load_jsonl(OUT / "qa500_train_test_split_manifest.jsonl") if row["split"] == "train" and row["origin"] == "field"}
    gold = [row for row in load_jsonl(OUT / "qa100_gold.jsonl") if row["qid"] in train]
    if args.limit:
        gold = gold[:args.limit]
    output_dir = OUT / "local_agent_v3b_sessions"
    output_dir.mkdir(exist_ok=True)
    search = Search()
    jobs = [(row["qid"], row["question"], arm) for row in gold for arm in args.arms.split(",")]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, search, qid, question, arm, output_dir) for qid, question, arm in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"{index}/{len(futures)} errors={sum(row['status'] == 'error' for row in results)}", flush=True)
    with (OUT / "retrieval_local_agent_v3b.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: (item["qid"], item["arm"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
