"""Derive parent and offset-bound chunk qrels for AIHub text-chunk smoke data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
AIHUB = BASE / "data" / "aihub"
MIN_OVERLAP_CHARS = 8


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _occurrences(span: dict) -> list[tuple[int, int]]:
    raw = span.get("occurrences")
    if raw is None and "char_start" in span and "char_end" in span:
        raw = [[span["char_start"], span["char_end"]]]
    result = []
    for value in raw or []:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or type(value[0]) is not int
            or type(value[1]) is not int
            or value[0] < 0
            or value[1] <= value[0]
        ):
            raise ValueError(f"invalid supporting-span occurrence: {value!r}")
        if value[1] - value[0] < MIN_OVERLAP_CHARS:
            raise ValueError(f"supporting span shorter than {MIN_OVERLAP_CHARS} chars")
        result.append((value[0], value[1]))
    if not result:
        raise ValueError("supporting span has no source offsets")
    return result


def derive_qrels(
    corpus: list[dict],
    qa_meta: list[dict],
    queries: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return filtered queries, parent qrels, and graded chunk qrels.

    A full evidence occurrence contained in one chunk receives grade 2. When an
    occurrence crosses a chunk boundary, every chunk covering at least eight
    evidence characters receives grade 1. One-character overlap is never gold.
    """
    chunks_by_parent: dict[str, list[dict]] = defaultdict(list)
    chunk_ids = set()
    for chunk in corpus:
        chunk_id = str(chunk["_id"])
        if chunk_id in chunk_ids:
            raise ValueError(f"duplicate chunk ID: {chunk_id}")
        chunk_ids.add(chunk_id)
        parent_id = str(chunk["parent_id"])
        start, end = chunk.get("char_start"), chunk.get("char_end")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError(f"invalid chunk offsets: {chunk_id}")
        if len(str(chunk.get("text", ""))) != end - start:
            raise ValueError(f"chunk text/offset length mismatch: {chunk_id}")
        chunks_by_parent[parent_id].append(chunk)

    query_by_id = {str(row["_id"]): row for row in queries}
    if len(query_by_id) != len(queries):
        raise ValueError("duplicate query ID")
    qa_by_id = {str(row["qid"]): row for row in qa_meta}
    if len(qa_by_id) != len(qa_meta):
        raise ValueError("duplicate QA metadata qid")
    if set(query_by_id) != set(qa_by_id):
        raise ValueError("queries and QA metadata IDs differ")

    parent_qrels = []
    chunk_qrels = []
    for qid in sorted(qa_by_id):
        meta = qa_by_id[qid]
        parent_id = str(meta["parent_id"])
        parent_chunks = chunks_by_parent.get(parent_id)
        if not parent_chunks:
            raise ValueError(f"query {qid} references absent parent {parent_id}")

        gains: dict[str, int] = {}
        for span in meta.get("supporting_spans") or []:
            for span_start, span_end in _occurrences(span):
                occurrence_has_gold = False
                for chunk in parent_chunks:
                    chunk_start = int(chunk["char_start"])
                    chunk_end = int(chunk["char_end"])
                    overlap = max(
                        0,
                        min(span_end, chunk_end) - max(span_start, chunk_start),
                    )
                    if overlap < MIN_OVERLAP_CHARS:
                        continue
                    chunk_id = str(chunk["_id"])
                    grade = 2 if chunk_start <= span_start and span_end <= chunk_end else 1
                    gains[chunk_id] = max(gains.get(chunk_id, 0), grade)
                    occurrence_has_gold = True
                if not occurrence_has_gold:
                    raise ValueError(
                        f"query {qid} evidence [{span_start},{span_end}) "
                        f"is not represented by any chunk"
                    )

        if not gains:
            raise ValueError(f"query {qid} has no offset-bound chunk gold")
        parent_qrels.append(
            {"query-id": qid, "corpus-id": parent_id, "score": 2}
        )
        chunk_qrels.extend(
            {"query-id": qid, "corpus-id": chunk_id, "score": gains[chunk_id]}
            for chunk_id in sorted(gains)
        )

    filtered_queries = [query_by_id[qid] for qid in sorted(query_by_id)]
    return filtered_queries, parent_qrels, chunk_qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("smoke", "element"))
    parser.add_argument("--aihub-dir", type=Path, default=AIHUB)
    args = parser.parse_args()

    corpus_dir = args.aihub_dir / args.variant
    qa_dir = args.aihub_dir / "qa"
    corpus_path = corpus_dir / "corpus.jsonl"
    qa_path = qa_dir / "qa_meta.jsonl"
    query_path = qa_dir / "queries.jsonl"
    queries, parent_qrels, chunk_qrels = derive_qrels(
        _load_jsonl(corpus_path),
        _load_jsonl(qa_path),
        _load_jsonl(query_path),
    )

    _write_jsonl(corpus_dir / "queries.jsonl", queries)
    _write_jsonl(corpus_dir / "qrels.jsonl", parent_qrels)
    _write_jsonl(corpus_dir / "chunk_qrels.jsonl", chunk_qrels)
    manifest = {
        "contract": "aihub-text-chunk-qrels-v2",
        "evaluation_track": "s0_exploratory_chunk_smoke",
        "confirmatory_gate_status": "not_advanced",
        "variant": args.variant,
        "parent_qrel_count": len(parent_qrels),
        "chunk_qrel_count": len(chunk_qrels),
        "query_count": len(queries),
        "minimum_chunk_overlap_chars": MIN_OVERLAP_CHARS,
        "inputs": {
            "corpus_sha256": _sha256(corpus_path),
            "qa_meta_sha256": _sha256(qa_path),
            "queries_sha256": _sha256(query_path),
        },
    }
    (corpus_dir / "qrel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{args.variant}] queries={len(queries)} "
        f"parent_qrels={len(parent_qrels)} chunk_qrels={len(chunk_qrels)}"
    )


if __name__ == "__main__":
    main()
