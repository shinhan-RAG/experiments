"""Write a normalized AIHub corpus without ever mutating the raw corpus.

The input is an immutable source artifact.  Header normalization is written to
an explicit, different output path together with a small provenance manifest.
It is intentionally not an in-place repair utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


DUP_HEADER = re.compile(r"\[([^\]\n]{1,20})\]\n【\s*\1\s*】[ \t]*\n?")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = DUP_HEADER.sub(r"[\1]\n", text)
    return text


def normalize_corpus(input_path: Path | str, output_path: Path | str) -> dict:
    """Normalize headers from one immutable JSONL input to a separate output."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"raw corpus is missing: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("normalized output must not overwrite the raw corpus")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    source_sha256 = _sha256_file(source)
    rows = changed = 0
    with source.open(encoding="utf-8") as source_stream, temporary.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line in source_stream:
            if not line.strip():
                continue
            row = json.loads(line)
            original = str(row.get("text", ""))
            normalized = clean_text(original)
            if normalized != original:
                row["text"] = normalized
                changed += 1
            rows += 1
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
    report = {
        "schema_version": "aihub-header-normalization-v1",
        "input_path": source.name,
        "input_sha256": source_sha256,
        "output_path": destination.name,
        "output_sha256": _sha256_file(destination),
        "rows": rows,
        "changed_rows": changed,
    }
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="immutable raw corpus.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="separate normalized corpus.jsonl")
    args = parser.parse_args(argv)
    report = normalize_corpus(args.input, args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
