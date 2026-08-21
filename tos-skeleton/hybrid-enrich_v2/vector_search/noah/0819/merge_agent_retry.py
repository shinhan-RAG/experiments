#!/usr/bin/env python3
"""Replace invalid agent rows with clean paired retry rows by exact key."""
import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row["qid"], int(row["rep"]), row["arm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--retry", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load_jsonl(args.base)
    retry = [row for path in args.retry for row in load_jsonl(path)]
    if len({key(row) for row in base}) != len(base):
        raise ValueError("duplicate key in base")
    retry_by_key = {key(row): row for row in retry}
    if len(retry_by_key) != len(retry):
        raise ValueError("duplicate key in retry")
    if not set(retry_by_key) <= {key(row) for row in base}:
        raise ValueError("retry contains a key absent from base")
    output_rows = [retry_by_key.get(key(row), row) for row in base]
    if any(row.get("errors") or row.get("protocol_errors") for row in output_rows):
        bad = [key(row) for row in output_rows
               if row.get("errors") or row.get("protocol_errors")]
        raise ValueError(f"merged output still contains invalid rows: {bad}")
    output = Path(args.output)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    manifest = {
        "base": {"path": str(Path(args.base).resolve()), "sha256": sha256(args.base)},
        "retry": [{"path": str(Path(path).resolve()), "sha256": sha256(path)}
                  for path in args.retry],
        "replaced_keys": [list(item) for item in sorted(retry_by_key)],
        "n": len(output_rows),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
