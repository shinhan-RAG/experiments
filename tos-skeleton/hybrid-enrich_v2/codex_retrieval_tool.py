#!/usr/bin/env python3
"""Isolated retrieval tool used by Codex OLD_RET/NEW_RET sessions."""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

from slot_filesearch import SlotFileSearch

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
MAX_CALLS = 8
VECTOR_ARMS = {"VECTOR_OLD_RET", "VECTOR_NEW_RET", "META_OLD", "META_NEW"}


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def charge(session_dir, tool, arguments):
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "calls.jsonl"
    count = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
    if count >= MAX_CALLS:
        return None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False) + "\n")
    return count + 1


def embed(query):
    request = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": "bge-m3", "input": [query], "keep_alive": "30m"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        vector = np.asarray(json.load(response)["embeddings"][0], dtype=np.float32)
    return vector / (np.linalg.norm(vector) or 1)


def _tokens(value):
    words = set(re.findall(r"[가-힣]+|[a-z]+|\d+(?:\.\d+)?", str(value).lower()))
    expanded = set(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 4:
            expanded.update(word[index:index + 3] for index in range(len(word) - 2))
    return {word for word in expanded if len(word) >= 2}


def _field_text(value):
    if isinstance(value, dict):
        return " ".join(_field_text(child) for child in value.values())
    if isinstance(value, list):
        return " ".join(_field_text(child) for child in value)
    return str(value or "")


def vector_search(query, vector_ret="old", stem="vec_meta_v3_concat", metadata_name="chunk_metadata_v3_0_reconstructed.jsonl"):
    # Retriever A/B is frozen to the same v3.0 metadata vector used by the
    # FileSearch compatibility experiment. Only candidate selection changes.
    matrix = np.load(OUT / f"{stem}.npy", mmap_mode="r")
    ids = json.loads((OUT / f"{stem}_ids.json").read_text())["ids"]
    chunks = {row["chunk_id"]: row["text"] for row in load("chunks.jsonl")}
    scores = matrix @ embed(query)
    if vector_ret == "old":
        top = np.argsort(-scores)[:10]
        rerank = {}
    else:
        metadata = {row["chunk_id"]: row for row in load(metadata_name)}
        qtokens = _tokens(query)
        candidates = np.argsort(-scores)[:200]
        reranked = []
        for cosine_rank, index in enumerate(candidates, 1):
            row = metadata[ids[index]]
            matched_fields = 0
            overlap = 0
            for field in ("scope_context", "topic_context", "canonical_entities", "aliases",
                          "event_relations", "condition_context", "exception_context",
                          "temporal_numeric_context", "reference_context"):
                count = len(qtokens & _tokens(_field_text(row.get(field))))
                overlap += count
                matched_fields += int(count > 0)
            # Cosine remains primary; field breadth only resolves a semantically plausible top-200.
            structural = 0.018 * matched_fields + 0.004 * min(overlap, 12)
            reranked.append((float(scores[index]) + structural, float(scores[index]), -cosine_rank, index,
                             matched_fields, overlap))
        reranked.sort(reverse=True)
        top = np.asarray([item[3] for item in reranked[:10]])
        rerank = {item[3]: {"matched_fields": item[4], "token_overlap": item[5]} for item in reranked[:10]}
    return {"results": [{"id": ids[index], "score": round(float(scores[index]), 4),
                          **rerank.get(int(index), {}), "preview": chunks[ids[index]][:260]} for index in top],
            "vector_ret": vector_ret}


def old_file_search(pattern):
    rows = load("grep_tag_new_repaired_v1.jsonl")
    try:
        regex = re.compile(pattern, re.I)
    except re.error:
        regex = re.compile(re.escape(pattern), re.I)
    matches = [row for row in rows if regex.search(row["g"])]
    total = len(matches)
    if total > 20:
        step = total / 20.0
        matches = [matches[int(index * step)] for index in range(20)]
    return {"total_matches": total, "sampled": total > 20,
            "results": [{"element_id": row["element_id"], "match": regex.search(row["g"]).group(0)[:180]}
                        for row in matches]}


def read_unit(unit_id):
    for name, key in (("chunks.jsonl", "chunk_id"), ("elements_repaired_v1.jsonl", "element_id")):
        for row in load(name):
            if row[key] == unit_id:
                return {"id": unit_id, "text": row["text"][:4000], "found": True}
    return {"id": unit_id, "text": "", "found": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("OLD_RET", "NEW_RET", "VECTOR_OLD_RET", "VECTOR_NEW_RET",
                                           "META_OLD", "META_NEW", "ST_OLD", "ST_NEW"), required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    vector = sub.add_parser("vector")
    vector.add_argument("--query", required=True)
    file = sub.add_parser("file")
    file.add_argument("--pattern", default="")
    file.add_argument("--filters", default="{}")
    file.add_argument("--raw-regex", default="")
    read = sub.add_parser("read")
    read.add_argument("--id", required=True)
    args = parser.parse_args()
    arguments = vars(args).copy()
    arguments["session_dir"] = str(arguments["session_dir"])
    call = charge(args.session_dir, args.action, arguments)
    if call is None:
        print(json.dumps({"error": "8-call limit reached"}, ensure_ascii=False)); return
    if args.action == "vector":
        if args.arm == "META_OLD":
            result = vector_search(args.query, stem="vec_meta_v2_current_chunks")
        elif args.arm == "META_NEW":
            result = vector_search(args.query, stem="vec_meta_v3_1_vector_separated")
        else:
            result = vector_search(args.query, "new" if args.arm == "VECTOR_NEW_RET" else "old")
    elif args.action == "file":
        if args.arm == "OLD_RET":
            result = old_file_search(args.pattern)
        else:
            variant = "old" if args.arm == "ST_OLD" else "new"
            result = SlotFileSearch(variant).search(json.loads(args.filters), args.raw_regex)
    else:
        result = read_unit(args.id)
    result["calls_used"] = call
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
