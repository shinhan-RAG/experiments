"""longdoc 데이터셋에 질의/qrels를 부여해 다른 데이터셋과 같은 계약으로 맞춘다.

data/raw/longdoc 은 corpus.jsonl만 있어서 run_experiment.load_queries()가 요구하는
queries.jsonl / qrels.jsonl이 없었다. 그래서 config에 등록해도 파트 실행이 불가능했다.

규칙은 ruling-anon과 동일하게 맞춘다.
  - 합본 판례(synth_case)의 【판시사항】을 질의로 사용한다.
  - 그 판례 본문에서는 【판시사항】을 제거해 질의 문장이 corpus에 남지 않게 한다.
    (ruling-anon: "판시사항 section is excluded from corpus text")
  - gold는 해당 판례 parent 문서 → qrels는 parent 기준(청크형 corpus 공통 계약).
  - 【판결요지】는 ruling-anon과 동일하게 corpus에 남긴다.
법령(law) parent는 판시사항이 없으므로 질의 없이 corpus에만 남는다(디스트랙터).

청킹/정규화/PII/계약검증은 데이터 빌드 스크립트(_build_scripts/src/data)와 동일한
구현을 사용해 다른 데이터셋과 chunk 경계가 어긋나지 않게 한다.

실행:
    python scripts/build_longdoc_queries.py
    python scripts/build_longdoc_queries.py --source-root <law_longdoc 상위 폴더>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LONGDOC_DIR = DATA_DIR / "raw" / "longdoc"

SYNTH_CASE_HEADING = re.compile(r"^제\d+조\s+(?P<title>.+?)\s*\((?P<cite>[^()]+)\)\s*$")
SYNTH_PART_HEADING = re.compile(r"^제\d+편\b")
CASE_NO_PATTERN = re.compile(r"\d{2,4}[가-힣]{1,4}\d{1,7}")
JDGMN_MARKER = "【판시사항】"

PII_PATTERNS = {
    "resident_id": re.compile(r"(?<!\d)\d{6}\s*-\s*[1-4]\d{6}(?!\d)"),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:01[016789]|0[2-6][1-5]?)[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)"),
}


# ---------------------------------------------------------------------------
# 빌드 스크립트와 동일한 정규화/청킹/검증 (src.data.aihub 미러)
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """NFC 정규화 + 공백 정리, 단락 경계는 유지."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif out and not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def detect_pii(text: str) -> list[str]:
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]


def extract_case_numbers(value: object) -> list[str]:
    return list(dict.fromkeys(CASE_NO_PATTERN.findall(normalize_text(value))))


def _choose_boundary(text: str, start: int, hard_end: int, minimum: int) -> int:
    if hard_end >= len(text):
        return len(text)
    region = text[start:hard_end]
    candidates = []
    for separator in ("\n\n", "\n", ". ", "다. "):
        position = region.rfind(separator)
        if position >= 0:
            candidates.append(position + len(separator))
    relative_end = max(candidates, default=-1)
    if relative_end >= minimum:
        return min(hard_end, start + relative_end)
    return hard_end


def chunk_document(doc: dict, max_chars: int = 3000, overlap: int = 300) -> list[dict]:
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("require max_chars > overlap >= 0")
    text = normalize_text(doc.get("text"))
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = _choose_boundary(text, start, min(len(text), start + max_chars), max_chars // 3)
        part = text[start:end].strip()
        if part:
            chunk = {
                "_id": f"{doc['_id']}#chunk-{index:04d}",
                "parent_id": doc["_id"],
                "chunk_index": index,
                "title": doc.get("title", ""),
                "text": part,
            }
            for key in ("metadata", "taxonomy", "tags", "prefix"):
                if key in doc:
                    chunk[key] = doc[key]
            chunks.append(chunk)
            index += 1
        if end >= len(text):
            break
        next_start = end - overlap
        while next_start > start and next_start < len(text) and not text[next_start - 1].isspace():
            next_start -= 1
        start = next_start if next_start > start else end
    return chunks


def validate_outputs(corpus: list[dict], queries: list[dict], qrels: list[dict], max_chars: int) -> dict:
    chunk_ids = [row["_id"] for row in corpus]
    query_ids = [row["_id"] for row in queries]
    parent_ids = {row["parent_id"] for row in corpus}
    errors = []
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("duplicate chunk _id")
    if len(query_ids) != len(set(query_ids)):
        errors.append("duplicate query _id")
    if any(len(row["text"]) > max_chars for row in corpus):
        errors.append("chunk exceeds max_chars")
    if any(row["corpus-id"] not in parent_ids for row in qrels if row.get("score", 0) > 0):
        errors.append("positive qrel parent missing from corpus")
    known_queries = set(query_ids)
    if any(row["query-id"] not in known_queries for row in qrels):
        errors.append("qrel query missing")
    return {
        "valid": not errors,
        "errors": errors,
        "chunk_count": len(corpus),
        "parent_count": len(parent_ids),
        "query_count": len(queries),
        "qrel_count": len(qrels),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# longdoc parent 파싱 (판시사항 분리판)
# ---------------------------------------------------------------------------

def _finish_synth_case(case: dict, stem: str) -> dict:
    """판례 1건 → parent 문서. 【판시사항】 단락은 본문에서 떼어 질의로 돌린다."""
    question = ""
    body: list[str] = []
    for part in case["parts"]:
        if not question and part.startswith(JDGMN_MARKER):
            question = normalize_text(part[len(JDGMN_MARKER):])
            continue
        body.append(part)
    case_no = (extract_case_numbers(case["cite"]) or [""])[0]
    return {
        "_id": f"synthcase:{stem}:{case['index']:04d}",
        "title": f"{case['title']} ({case['cite']})",
        "text": "\n\n".join(body),
        "question": question,
        "metadata": {"source_type": "synth_case", "file": stem, "case_no": case_no},
    }


def iter_longdoc_parents(path: Path) -> Iterator[dict]:
    """블록 구조 longdoc JSON 1개 → parent 문서들.

    법령 파일은 parent 1건, 판례 합본은 ``제N조 <사건명> (<법원 날짜 사건번호>)``
    heading마다 parent 1건으로 쪼갠다. (_build_scripts 원본과 동일 규칙)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = data.get("blocks") or []
    doc_title = next(
        (normalize_text(b.get("markdown")) for b in blocks if b.get("label") == "doc_title"),
        path.stem,
    )
    case_headings = [
        b for b in blocks
        if b.get("label") == "paragraph_title" and SYNTH_CASE_HEADING.match(normalize_text(b.get("markdown")))
    ]
    if len(case_headings) < 2:
        text = "\n\n".join(
            normalize_text(b.get("markdown"))
            for b in blocks
            if b.get("label") != "doc_title" and normalize_text(b.get("markdown"))
        )
        yield {
            "_id": f"law:{path.stem}",
            "title": doc_title,
            "text": text,
            "question": "",
            "metadata": {"source_type": "law", "file": path.name},
        }
        return

    current: dict | None = None
    index = 0
    for block in blocks:
        label = block.get("label")
        markdown = normalize_text(block.get("markdown"))
        if label == "doc_title" or not markdown:
            continue
        match = SYNTH_CASE_HEADING.match(markdown) if label == "paragraph_title" else None
        if match:
            if current and current["parts"]:
                yield _finish_synth_case(current, path.stem)
            current = {
                "index": index,
                "title": match.group("title"),
                "cite": match.group("cite"),
                "parts": [],
            }
            index += 1
            continue
        if label == "paragraph_title" and SYNTH_PART_HEADING.match(markdown):
            continue
        if current is not None:
            current["parts"].append(markdown)
    if current and current["parts"]:
        yield _finish_synth_case(current, path.stem)


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------

def resolve_source_roots(explicit: list[str] | None) -> list[Path]:
    """--source-root 우선, 없으면 기존 manifest.json의 source_roots를 재사용."""
    if explicit:
        roots = []
        for raw in explicit:
            root = Path(raw)
            # law_longdoc/synth_longdoc의 상위 폴더를 준 경우도 받아준다
            if not root.name.endswith("_longdoc") and (root / "law_longdoc").exists():
                roots.extend([root / "law_longdoc", root / "synth_longdoc"])
            else:
                roots.append(root)
        return roots
    manifest_path = LONGDOC_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"source root를 알 수 없다: {manifest_path} 부재. --source-root 를 지정하라."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = [Path(p) for p in manifest.get("source_roots") or []]
    if not roots:
        raise SystemExit("manifest.json에 source_roots가 없다. --source-root 를 지정하라.")
    return roots


def build(roots: list[Path], chunk_size: int, overlap: int) -> dict:
    started = time.perf_counter()
    documents: list[dict] = []
    queries: list[dict] = []
    audit: list[dict] = []
    counts: Counter = Counter()
    seen_hashes: dict[str, str] = {}

    for root in roots:
        if not root.exists():
            raise SystemExit(f"source root 없음: {root}")
        for path in sorted(root.glob("*.json")):
            for doc in iter_longdoc_parents(path):
                if not doc["text"]:
                    audit.append({"reason": "empty_text", "id": doc["_id"], "path": str(path)})
                    counts["empty_text"] += 1
                    continue
                pii = detect_pii(doc["text"])
                if pii:
                    audit.append({"reason": "pii", "types": pii, "id": doc["_id"], "path": str(path)})
                    counts["pii"] += 1
                    continue
                digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    audit.append({
                        "reason": "duplicate_content",
                        "id": doc["_id"],
                        "duplicate_of": seen_hashes[digest],
                    })
                    counts["duplicate_content"] += 1
                    continue
                seen_hashes[digest] = doc["_id"]

                question = doc.pop("question", "")
                documents.append(doc)
                counts[doc["metadata"]["source_type"]] += 1

                if not question:
                    if doc["metadata"]["source_type"] == "synth_case":
                        audit.append({"reason": "no_jdgmn", "id": doc["_id"]})
                        counts["no_jdgmn"] += 1
                    continue
                if detect_pii(question):
                    audit.append({"reason": "pii_query", "id": doc["_id"]})
                    counts["pii_query"] += 1
                    continue
                queries.append({
                    "_id": f"lq:{doc['_id'].split(':', 1)[1]}",
                    "text": question,
                    "parent_id": doc["_id"],
                })

    qrels = [{"query-id": q["_id"], "corpus-id": q["parent_id"], "score": 1} for q in queries]
    public_queries = [{"_id": q["_id"], "text": q["text"]} for q in queries]
    corpus = [chunk for doc in documents for chunk in chunk_document(doc, chunk_size, overlap)]
    report = validate_outputs(corpus, public_queries, qrels, chunk_size)
    if not report["valid"]:
        raise SystemExit("longdoc output contract failed: " + "; ".join(report["errors"]))

    LONGDOC_DIR.mkdir(parents=True, exist_ok=True)
    # 기존 corpus는 판시사항을 포함하고 있어 질의와 겹친다 → 교체 전 1회 백업
    old_corpus = LONGDOC_DIR / "corpus.jsonl"
    backup = LONGDOC_DIR / "corpus.jsonl.pre-queries.bak"
    if old_corpus.exists() and not backup.exists():
        shutil.copy2(old_corpus, backup)
        print(f"  기존 corpus 백업: {backup.name}")

    _write_jsonl(old_corpus, corpus)
    _write_jsonl(LONGDOC_DIR / "queries.jsonl", public_queries)
    _write_jsonl(LONGDOC_DIR / "qrels.jsonl", qrels)
    _write_jsonl(LONGDOC_DIR / "audit.jsonl", audit)

    manifest = {
        "dataset": "longdoc",
        "source_roots": [str(r.resolve()) for r in roots],
        "chunk_size": chunk_size,
        "overlap": overlap,
        "parent_count": len(documents),
        "chunk_count": len(corpus),
        "query_count": len(public_queries),
        "parent_counts_by_type": {
            k: v for k, v in sorted(counts.items()) if k in ("law", "synth_case")
        },
        "quality_counts": {
            k: v for k, v in sorted(counts.items()) if k not in ("law", "synth_case")
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": report,
        "notes": (
            "판시사항은 corpus 본문에서 제외하고 질의로 사용한다(ruling-anon과 동일 규칙). "
            "gold는 해당 판례 parent. 법령(law) parent는 질의 없이 디스트랙터로 남는다."
        ),
    }
    (LONGDOC_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", action="append",
        help="law_longdoc/synth_longdoc 폴더(또는 그 상위). 미지정 시 manifest.json의 source_roots 사용",
    )
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--overlap", type=int, default=300)
    args = parser.parse_args()

    roots = resolve_source_roots(args.source_root)
    print("longdoc 질의/qrels 빌드")
    for root in roots:
        print(f"  source: {root}")

    manifest = build(roots, args.chunk_size, args.overlap)
    print(
        f"  parents={manifest['parent_count']} chunks={manifest['chunk_count']} "
        f"queries={manifest['query_count']} qrels={manifest['contract']['qrel_count']}"
    )
    print(f"  by_type={manifest['parent_counts_by_type']} quality={manifest['quality_counts']}")
    print(f"  contract.valid={manifest['contract']['valid']}")
    print(f"  -> {LONGDOC_DIR}")


if __name__ == "__main__":
    main()
