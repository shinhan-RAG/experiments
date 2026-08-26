#!/usr/bin/env python3
"""Build the 0826 semantic-tag BM25 index — LLM v3 태그 지원."""
import argparse
import json
from pathlib import Path

from tag_hybrid import build_index, default_paths, HERE

FS = HERE.parents[2] / "filesearch"


def main():
    ap = argparse.ArgumentParser(description="태그 인덱스 빌더 (0826)")
    ap.add_argument("--elements", type=Path, default=FS / "out" / "elements_u3.jsonl")
    ap.add_argument("--tags", type=Path, default=None,
                    help="태그 파일. 기본값: filesearch/out/tags_u4_fact_rules.jsonl")
    ap.add_argument("--jo", type=Path, default=FS / "out" / "elements_u3jo.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="인덱스 출력 디렉토리. 태그 파일명에서 자동 생성")
    a = ap.parse_args()

    if a.tags is None:
        a.tags = FS / "out" / "tags_u4_fact_rules.jsonl"
    if a.out_dir is None:
        stem = a.tags.stem.replace("tags_", "idx_")
        a.out_dir = HERE / "out" / stem

    for p in (a.elements, a.tags, a.jo):
        if not p.exists():
            raise SystemExit(f"필수 입력 없음: {p}")

    print(f"Building index: tags={a.tags.name} → {a.out_dir}", flush=True)
    meta = build_index(a.elements, a.tags, a.jo, a.out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
