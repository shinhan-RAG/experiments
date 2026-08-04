"""shinhan-tos 청크 메타데이터 생성 (규칙 기반, LLM 0회).

회의(0804) §26.1의 청크 메타데이터 후보 중 규칙으로 추출 가능한 필드를 만든다:
문서유형·상품명·작성일·조항명(section)·element_type·문서크기 bucket.
tos-skeleton frontmatter 트랙의 "구조가 파일명보다 정확" 원칙에 따라
문서유형은 docs_selected.json(수작업 검증된 선별 목록)을 정본으로 쓴다.

출력: data/metadata/shinhan-tos_chunk.json  — {chunk_id: {field: value}}
      (run_experiment.py part6가 인덱스 직렬화에 사용)
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "shinhan-tos"
OUT = BASE / "data" / "metadata" / "shinhan-tos_chunk.json"

DOCTYPE_PREFIX = re.compile(
    r"^\[?\d*\]?\[?[\d~]*\]?\s*(판매약관|공시약관|약관|사업방법서|사업방법|상품요약서)_?"
)
DATE_TAIL = re.compile(r"_(\d{6,8})(?:_\d+)?$")


def product_of(name: str) -> str:
    """문서 파일명에서 상품명 추정 — 유형 접두·날짜 접미·괄호 주석 제거."""
    s = DOCTYPE_PREFIX.sub("", name)
    s = DATE_TAIL.sub("", s)
    s = re.sub(r"_(판매약관|공시약관|약관|사업방법서|예사비약관)(\(.*?\))?", "", s)
    s = re.sub(r"\((무배당[^)]*|해약환급금[^)]*|갱신형[^)]*)\)", "", s)
    s = s.replace("_", " ").strip(" -—")
    return s or name


def date_of(name: str) -> str:
    m = DATE_TAIL.search(name)
    return m.group(1) if m else ""


def main():
    doc_info = {}
    with open(RAW / "docs_selected.json", encoding="utf-8") as f:
        for d in json.load(f)["docs"]:
            doc_info[d["name"]] = {"type": d["type"], "bucket": d["bucket"]}

    meta = {}
    with open(RAW / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            doc = c["doc"]
            info = doc_info.get(doc, {})
            meta[c["_id"]] = {
                "문서유형": info.get("type", ""),
                "상품명": product_of(doc),
                "작성일": date_of(doc),
                "조항": c.get("section", ""),
                "요소": c.get("element_type", ""),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"청크 {len(meta)}건 → {OUT}")
    sample_id = next(iter(meta))
    print("샘플:", sample_id, json.dumps(meta[sample_id], ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
