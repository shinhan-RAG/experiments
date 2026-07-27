"""Build a deterministic, lossless AIHub 512-character text-chunk fixture."""

from __future__ import annotations

import argparse
from collections import Counter
import codecs
import hashlib
import json
from pathlib import Path
import re
import zipfile


BASE = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE / "data" / "aihub" / "smoke"
DEFAULT_PARENTS = BASE / "data" / "aihub" / "parents.jsonl"
DEFAULT_CHUNK_CHARS = 512
DEFAULT_MIN_CHARS = 80
SENT_SPLIT = re.compile(r"(?<=[다요음\.。」』】])\s+|\n{1,}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_member(archive: zipfile.ZipFile) -> str:
    members = sorted(
        name for name in archive.namelist()
        if not name.endswith("/") and name.lower().endswith(".json")
    )
    if len(members) != 1:
        raise ValueError(
            f"expected exactly one JSON member, found {len(members)}: {members[:5]}"
        )
    return members[0]


def stream_records(zip_path: Path, limit: int):
    """Stream the first ``limit`` records from one explicit JSON member."""
    with zipfile.ZipFile(zip_path) as archive:
        member = _json_member(archive)
        with archive.open(member) as stream:
            decoder = json.JSONDecoder()
            byte_decoder = codecs.getincrementaldecoder("utf-8")()
            buffer = ""
            started = False
            count = 0
            while count < limit:
                block = stream.read(1 << 20)
                buffer += byte_decoder.decode(block, final=not block)
                if not started:
                    marker = buffer.find('"data"')
                    start = buffer.find("[", marker)
                    if start == -1:
                        if not block:
                            raise ValueError(f"JSON member has no data array: {member}")
                        buffer = buffer[-64:]
                        continue
                    buffer = buffer[start + 1 :]
                    started = True
                while count < limit:
                    buffer = buffer.lstrip(" \t\r\n,")
                    if not buffer or buffer[0] == "]":
                        break
                    try:
                        row, end = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(row, dict):
                        raise ValueError("AIHub data member must contain JSON objects")
                    yield row
                    count += 1
                    buffer = buffer[end:]
                if not block:
                    break


def sentence_spans(raw: str) -> list[tuple[int, int]]:
    spans = []
    previous = 0
    for match in SENT_SPLIT.finditer(raw):
        if match.start() > previous:
            spans.append((previous, match.start()))
        previous = match.end()
    if previous < len(raw):
        spans.append((previous, len(raw)))
    return spans


def _retain_short_fragments(
    chunks: list[tuple[int, int]],
    min_chars: int,
) -> list[tuple[int, int]]:
    """Keep short fragments as explicit chunks so max-size and evidence survive."""
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    return list(chunks)


def chunk_offsets(
    raw: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[tuple[int, int]]:
    """Return deterministic offsets without dropping non-whitespace text."""
    if chunk_chars < 1 or min_chars < 1:
        raise ValueError("chunk_chars and min_chars must be positive")
    chunks = []
    current_start = current_end = None
    for start, end in sentence_spans(raw):
        if end - start > chunk_chars:
            if current_start is not None:
                chunks.append((current_start, current_end))
                current_start = current_end = None
            chunks.extend(
                (index, min(index + chunk_chars, end))
                for index in range(start, end, chunk_chars)
            )
        elif current_start is None:
            current_start, current_end = start, end
        elif end - current_start > chunk_chars:
            chunks.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = end
    if current_start is not None:
        chunks.append((current_start, current_end))
    chunks = _retain_short_fragments(chunks, min_chars)

    cursor = 0
    for start, end in chunks:
        if raw[cursor:start].strip():
            raise ValueError("chunking dropped non-whitespace source text")
        cursor = end
    if raw[cursor:].strip():
        raise ValueError("chunking dropped non-whitespace source tail")
    return chunks


def parent_id(domain: str, record: dict, record_index: int) -> str:
    source_id = str(record.get("book_id") or "record")
    return f"{domain}:{source_id}:{record_index:06d}"


def build(
    sources: dict[str, Path],
    *,
    output_dir: Path,
    parents_output: Path,
    n_per_domain: int,
    chunk_chars: int,
    min_chars: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    parents_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    meta_path = output_dir / "parent_meta.jsonl"
    parent_rows = []
    meta_rows = []
    stats = {
        "contract": "aihub-text-chunk-corpus-v2",
        "representation": "fixed_text_chunk",
        "parser_derived_element": False,
        "chunk_chars": chunk_chars,
        "min_chars": min_chars,
        "dropped_non_whitespace_chars": 0,
        "short_fragment_count": 0,
        "docs": 0,
        "chunks": 0,
        "by_domain": {},
        "sources": {},
    }
    seen_parent_ids = set()
    with corpus_path.open("w", encoding="utf-8") as corpus_stream:
        for domain, source in sorted(sources.items()):
            docs = chunks = 0
            for record_index, record in enumerate(stream_records(source, n_per_domain)):
                pid = parent_id(domain, record, record_index)
                if pid in seen_parent_ids:
                    raise ValueError(f"duplicate parent ID: {pid}")
                seen_parent_ids.add(pid)
                raw = str(record.get("text") or "")
                offsets = chunk_offsets(
                    raw,
                    chunk_chars=chunk_chars,
                    min_chars=min_chars,
                )
                if not offsets:
                    continue
                stats["short_fragment_count"] += sum(
                    end - start < min_chars for start, end in offsets
                )
                category = str(record.get("category") or "")
                for chunk_index, (start, end) in enumerate(offsets):
                    row = {
                        "_id": f"{pid}:c{chunk_index:04d}",
                        "parent_id": pid,
                        "title": f"{domain} · {category}",
                        "text": raw[start:end],
                        "char_start": start,
                        "char_end": end,
                        "domain": domain,
                        "category": category,
                        "split_method": "sentence_bounded_512",
                    }
                    corpus_stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    chunks += 1
                ne_types = Counter(
                    str(item.get("type", "?")) for item in (record.get("NE") or [])
                    if isinstance(item, dict)
                )
                meta_rows.append(
                    {
                        "parent_id": pid,
                        "domain": domain,
                        "category": category,
                        "keyword": record.get("keyword") or [],
                        "ne_types": dict(ne_types.most_common(10)),
                        "n_chunks": len(offsets),
                    }
                )
                parent_rows.append(
                    {"parent_id": pid, "domain": domain, "category": category, "text": raw}
                )
                docs += 1
            stats["by_domain"][domain] = {"docs": docs, "chunks": chunks}
            stats["docs"] += docs
            stats["chunks"] += chunks
            stats["sources"][domain] = {
                "path_name": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }

    with meta_path.open("w", encoding="utf-8") as stream:
        for row in meta_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with parents_output.open("w", encoding="utf-8") as stream:
        for row in parent_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stats["outputs"] = {
        "corpus_sha256": sha256_file(corpus_path),
        "parent_meta_sha256": sha256_file(meta_path),
        "parents_sha256": sha256_file(parents_output),
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-zip", type=Path, required=True)
    parser.add_argument("--legal-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parents-output", type=Path, default=DEFAULT_PARENTS)
    parser.add_argument("--n-per-domain", type=int, default=5000)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    args = parser.parse_args()
    stats = build(
        {"의료": args.medical_zip, "법률": args.legal_zip},
        output_dir=args.output_dir,
        parents_output=args.parents_output,
        n_per_domain=args.n_per_domain,
        chunk_chars=args.chunk_chars,
        min_chars=args.min_chars,
    )
    print(f"parents={stats['docs']} chunks={stats['chunks']} output={args.output_dir}")


if __name__ == "__main__":
    main()
