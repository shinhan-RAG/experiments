"""Build an S0 parsed-Markdown/text chunk corpus.

This is a chunking utility, not a parser-derived element extractor. Markdown
tables remain text-mediated representations, and the source data is never
copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CHUNK_CHARS = 1400
MIN_CHARS = 120
HEADER_RE = re.compile(r"(?m)^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
FORMULA_RE = re.compile(r"\$\$|\\\(|\\\[")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_chunk_stats(stats: dict | None) -> dict:
    stats = stats if stats is not None else {}
    for key in ("dropped_fragment_count", "dropped_char_count", "oversized_table_chunk_count"):
        stats.setdefault(key, 0)
    return stats


def doc_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]


def chunk_representation_type(text: str) -> str:
    """A text representation hint, explicitly not a parser-derived element type."""
    if len(TABLE_ROW_RE.findall(text)) >= 3:
        return "markdown_table_text"
    return "formula_text" if FORMULA_RE.search(text) else "text"


def split_sections(markdown: str) -> list[tuple[str, str]]:
    matches = list(HEADER_RE.finditer(markdown))
    if not matches:
        return [("", markdown)]
    sections = []
    if matches[0].start() > 0:
        preamble = markdown[:matches[0].start()].strip()
        if preamble:
            sections.append(("(preamble)", preamble))
    for index, match in enumerate(matches):
        body = markdown[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(markdown)].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def _blocks(body: str) -> list[tuple[bool, str]]:
    blocks, current, current_is_table = [], [], None
    for line in body.split("\n"):
        is_table = bool(TABLE_ROW_RE.fullmatch(line))
        if current and is_table != current_is_table:
            blocks.append((bool(current_is_table), "\n".join(current).strip("\n")))
            current = []
        current.append(line)
        current_is_table = is_table
    if current:
        blocks.append((bool(current_is_table), "\n".join(current).strip("\n")))
    return [(is_table, text) for is_table, text in blocks if text]


def _split_non_table_block(text: str, max_chars: int) -> list[str]:
    remaining, pieces = text.strip(), []
    while len(remaining) > max_chars:
        cut = max(remaining.rfind("\n", 0, max_chars + 1), remaining.rfind(" ", 0, max_chars + 1))
        if cut <= 0:
            cut = max_chars
        piece = remaining[:cut].strip() or remaining[:max_chars]
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def pack_chunks(body: str, *, max_chars: int = CHUNK_CHARS, min_chars: int = MIN_CHARS, stats: dict | None = None) -> list[str]:
    """Deterministically split text; retain oversized Markdown tables intact."""
    if max_chars <= 0 or min_chars < 0:
        raise ValueError("max_chars must be positive and min_chars non-negative")
    stats = _ensure_chunk_stats(stats)
    packed, buffer = [], ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            packed.append(buffer.strip())
        buffer = ""

    for is_table, block in _blocks(body):
        if is_table and len(block) > max_chars:
            flush()
            packed.append(block)
            stats["oversized_table_chunk_count"] += 1
            continue
        for piece in ([block] if is_table else _split_non_table_block(block, max_chars)):
            if not piece:
                continue
            if buffer and len(buffer) + len(piece) + 1 > max_chars:
                flush()
            if not buffer and len(piece) > max_chars:
                raise AssertionError("non-table chunk exceeds max_chars")
            buffer = f"{buffer}\n{piece}" if buffer else piece
    flush()
    chunks = []
    for chunk in packed:
        if len(chunk) < min_chars:
            stats["dropped_fragment_count"] += 1
            stats["dropped_char_count"] += len(chunk)
        else:
            chunks.append(chunk)
    return chunks


def _input_manifest(input_dir: Path, files: list[Path]) -> tuple[dict, str]:
    manifest = {
        "schema_version": "s0-input-markdown-manifest-v1",
        "root_name": input_dir.name,
        "files": [{"relative_path": path.relative_to(input_dir).as_posix(), "byte_size": path.stat().st_size, "sha256": _file_sha256(path)} for path in files],
    }
    return manifest, _canonical_sha256(manifest)


def build_corpus(input_dir: Path | str, output_dir: Path | str, *, max_chars: int = CHUNK_CHARS, min_chars: int = MIN_CHARS) -> dict:
    input_path, output_path = Path(input_dir), Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"S0 Markdown input directory is missing: {input_path}")
    files = sorted(path for path in input_path.rglob("*.md") if path.is_file())
    if not files:
        raise FileNotFoundError(f"S0 Markdown input directory has no .md files: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    input_manifest, input_hash = _input_manifest(input_path, files)
    stats = {"schema_version": "s0-chunk-stats-v1", "chunk_max_chars": max_chars, "short_fragment_min_chars": min_chars, "docs": 0, "chunks": 0, "zero_chunk_document_count": 0, "dropped_fragment_count": 0, "dropped_char_count": 0, "oversized_table_chunk_count": 0, "by_representation_type": {}, "per_doc": [], "input_corpus_manifest_sha256": input_hash}
    corpus = []
    for source in files:
        document_name = source.stem
        document_id = doc_id_from_name(source.relative_to(input_path).as_posix())
        doc_chunks = 0
        for section_index, (section_title, body) in enumerate(split_sections(source.read_text(encoding="utf-8"))):
            for chunk_index, chunk in enumerate(pack_chunks(body, max_chars=max_chars, min_chars=min_chars, stats=stats)):
                representation = chunk_representation_type(chunk)
                corpus.append({"_id": f"{document_id}_{section_index:03d}_{chunk_index:02d}", "title": document_name[:80] + (" — " + section_title[:60] if section_title else ""), "text": chunk, "doc": document_name, "section": section_title, "chunk_representation_type": representation})
                stats["by_representation_type"][representation] = stats["by_representation_type"].get(representation, 0) + 1
                doc_chunks += 1
        stats["docs"] += 1
        stats["zero_chunk_document_count"] += int(doc_chunks == 0)
        stats["per_doc"].append({"doc": document_name, "doc_id": document_id, "chunks": doc_chunks})
    stats["chunks"] = len(corpus)
    corpus_path = output_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as stream:
        for record in corpus:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    output_manifest = {"schema_version": "s0-output-chunk-manifest-v1", "input_corpus_manifest_sha256": input_hash, "row_count": len(corpus), "records": [{"chunk_id": row["_id"], "text_char_count": len(row["text"]), "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest()} for row in corpus]}
    output_hash = _canonical_sha256(output_manifest)
    output_manifest["manifest_sha256"] = output_hash
    stats["output_corpus_manifest_sha256"] = output_hash
    stats["output_corpus_file_sha256"] = _file_sha256(corpus_path)
    (output_path / "input_manifest.json").write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "corpus_manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a local S0 Markdown/text chunk corpus")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=CHUNK_CHARS)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = parser.parse_args(argv)
    print(build_corpus(args.input_dir, args.output_dir, max_chars=args.max_chars, min_chars=args.min_chars))


if __name__ == "__main__":
    main()
