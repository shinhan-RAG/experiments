#!/usr/bin/env python3
"""Build the 0821 semantic-tag BM25 index (no tag embeddings)."""
import argparse
import json
from pathlib import Path

from tag_hybrid import build_index, default_paths


def main():
    elements, tags, jo, out_dir = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", type=Path, default=elements)
    ap.add_argument("--tags", type=Path, default=tags)
    ap.add_argument("--jo", type=Path, default=jo)
    ap.add_argument("--out-dir", type=Path, default=out_dir)
    a = ap.parse_args()
    for p in (a.elements, a.tags, a.jo):
        if not p.exists():
            raise SystemExit(f"필수 입력 없음: {p}")
    meta = build_index(a.elements, a.tags, a.jo, a.out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
