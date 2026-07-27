"""Build parser-unverified text-mediated structural chunks for AIHub smoke tests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

try:
    from scripts.build_aihub_corpus import (
        _retain_short_fragments,
        parent_id,
        sha256_file,
        stream_records,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_aihub_corpus import (  # type: ignore
        _retain_short_fragments,
        parent_id,
        sha256_file,
        stream_records,
    )


BASE = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE / "data" / "aihub" / "element"
SECTION_MARK = re.compile(r"【[^】]{1,20}】")
SENTENCE_BOUNDARY = re.compile(r"(?<=[다요음\.。」』】])\s+|\n{1,}")
DEFAULT_SENTENCE_GROUP = 3
DEFAULT_MAX_CHARS = 900
DEFAULT_MIN_CHARS = 80


def _cap(start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    return [
        (index, min(index + max_chars, end))
        for index in range(start, end, max_chars)
    ]


def legal_offsets(
    raw: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[tuple[int, int]]:
    starts = [match.start() for match in SECTION_MARK.finditer(raw)]
    bounds = sorted(set([0, *starts, len(raw)]))
    segments = [
        (bounds[index], bounds[index + 1])
        for index in range(len(bounds) - 1)
        if bounds[index + 1] > bounds[index]
    ]
    capped = [
        piece
        for start, end in segments
        for piece in _cap(start, end, max_chars)
    ]
    return _retain_short_fragments(capped, min_chars)


def _sentence_spans(raw: str) -> list[tuple[int, int]]:
    spans = []
    previous = 0
    for match in SENTENCE_BOUNDARY.finditer(raw):
        if match.start() > previous:
            spans.append((previous, match.start()))
        previous = match.end()
    if previous < len(raw):
        spans.append((previous, len(raw)))
    return spans


def medical_offsets(
    raw: str,
    *,
    sentence_group: int = DEFAULT_SENTENCE_GROUP,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[tuple[int, int]]:
    sentences = _sentence_spans(raw)
    grouped = []
    for index in range(0, len(sentences), sentence_group):
        group = sentences[index : index + sentence_group]
        grouped.extend(_cap(group[0][0], group[-1][1], max_chars))
    return _retain_short_fragments(grouped, min_chars)


def build(
    sources: dict[str, Path],
    *,
    output_dir: Path,
    n_per_domain: int,
    sentence_group: int,
    max_chars: int,
    min_chars: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    meta_path = output_dir / "parent_meta.jsonl"
    metadata = []
    stats = {
        "contract": "aihub-text-mediated-structural-chunk-v2",
        "representation": "text_mediated_structural_chunk",
        "parser_derived_element": False,
        "multimodal_element": False,
        "sentence_group": sentence_group,
        "max_chars": max_chars,
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
            lengths = []
            docs = chunks = 0
            for record_index, record in enumerate(stream_records(source, n_per_domain)):
                pid = parent_id(domain, record, record_index)
                if pid in seen_parent_ids:
                    raise ValueError(f"duplicate parent ID: {pid}")
                seen_parent_ids.add(pid)
                raw = str(record.get("text") or "")
                offsets = (
                    legal_offsets(raw, max_chars=max_chars, min_chars=min_chars)
                    if domain == "법률"
                    else medical_offsets(
                        raw,
                        sentence_group=sentence_group,
                        max_chars=max_chars,
                        min_chars=min_chars,
                    )
                )
                if not offsets:
                    continue
                stats["short_fragment_count"] += sum(
                    end - start < min_chars for start, end in offsets
                )
                cursor = 0
                for start, end in offsets:
                    if raw[cursor:start].strip():
                        raise ValueError("structural chunking dropped source text")
                    cursor = end
                if raw[cursor:].strip():
                    raise ValueError("structural chunking dropped source tail")

                category = str(record.get("category") or "")
                split_method = (
                    "legal_section_regex" if domain == "법률"
                    else "medical_sentence_group"
                )
                for chunk_index, (start, end) in enumerate(offsets):
                    row = {
                        "_id": f"{pid}:e{chunk_index:04d}",
                        "parent_id": pid,
                        "title": f"{domain} · {category}",
                        "text": raw[start:end],
                        "char_start": start,
                        "char_end": end,
                        "domain": domain,
                        "category": category,
                        "split_method": split_method,
                        "element_claim": "text_mediated_parser_unverified",
                    }
                    corpus_stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    lengths.append(end - start)
                    chunks += 1
                ne_types = Counter(
                    str(item.get("type", "?")) for item in (record.get("NE") or [])
                    if isinstance(item, dict)
                )
                metadata.append(
                    {
                        "parent_id": pid,
                        "domain": domain,
                        "category": category,
                        "keyword": record.get("keyword") or [],
                        "ne_types": dict(ne_types.most_common(10)),
                        "n_elements": len(offsets),
                    }
                )
                docs += 1
            stats["by_domain"][domain] = {
                "docs": docs,
                "chunks": chunks,
                "average_chars": int(statistics.mean(lengths)) if lengths else 0,
            }
            stats["docs"] += docs
            stats["chunks"] += chunks
            stats["sources"][domain] = {
                "path_name": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }

    with meta_path.open("w", encoding="utf-8") as stream:
        for row in metadata:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stats["outputs"] = {
        "corpus_sha256": sha256_file(corpus_path),
        "parent_meta_sha256": sha256_file(meta_path),
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
    parser.add_argument("--n-per-domain", type=int, default=5000)
    parser.add_argument("--sentence-group", type=int, default=DEFAULT_SENTENCE_GROUP)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    args = parser.parse_args()
    stats = build(
        {"의료": args.medical_zip, "법률": args.legal_zip},
        output_dir=args.output_dir,
        n_per_domain=args.n_per_domain,
        sentence_group=args.sentence_group,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    print(f"parents={stats['docs']} chunks={stats['chunks']} output={args.output_dir}")


if __name__ == "__main__":
    main()
