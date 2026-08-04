"""
docs_selected.json의 선정 문서(parsed_md) → shinhan-tos BEIR corpus.jsonl 청킹.

build_shinhan_corpus.py의 검증된 로직 유지:
- 헤더(#..######) 기준 섹션 분할, 표 블록 통째 보존, ~1,400자 패킹
- 청크별 element_type(table/formula/text) 태깅
- 청크 id: 문서해시8_섹션idx_청크idx
출력: data/raw/shinhan-tos/corpus.jsonl, corpus_stats.json
"""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "raw" / "shinhan-tos"
SRC_ROOT = BASE.parents[0] / "parsed_md"

CHUNK_CHARS = 1400
MIN_CHARS = 120
HEADER_RE = re.compile(r'(?m)^(#{1,6})\s+(.*)$')
TABLE_ROW_RE = re.compile(r'(?m)^\s*\|.*\|\s*$')
FORMULA_RE = re.compile(r'\$\$|\\\(|\\\[')


def winpath(p) -> str:
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + os.path.abspath(s)
    return s


def resolve_path(rel: str) -> str:
    """NFC/NFD 정규화가 뒤섞인 파일명 대응: 원본→NFC→NFD 순으로 실존 경로 탐색."""
    for cand in (rel, unicodedata.normalize("NFC", rel), unicodedata.normalize("NFD", rel)):
        full = winpath(SRC_ROOT / cand)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(rel)


def doc_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:8]


def element_type(text: str) -> str:
    if len(TABLE_ROW_RE.findall(text)) >= 3:
        return "table"
    if FORMULA_RE.search(text):
        return "formula"
    return "text"


def split_sections(md: str):
    matches = list(HEADER_RE.finditer(md))
    if not matches:
        return [("", md)]
    sections = []
    if matches[0].start() > 0:
        pre = md[:matches[0].start()].strip()
        if pre:
            sections.append(("(preamble)", pre))
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def pack_chunks(body: str):
    lines = body.split("\n")
    blocks, cur, cur_is_table = [], [], False
    for ln in lines:
        is_tbl = bool(re.match(r'\s*\|.*\|\s*$', ln))
        if is_tbl != cur_is_table and cur:
            blocks.append("\n".join(cur))
            cur, cur_is_table = [], is_tbl
        cur.append(ln)
        cur_is_table = is_tbl if not cur[:-1] else cur_is_table or is_tbl
    if cur:
        blocks.append("\n".join(cur))

    chunks, buf = [], ""
    for blk in blocks:
        blk = blk.strip("\n")
        if not blk:
            continue
        if len(blk) > CHUNK_CHARS and TABLE_ROW_RE.search(blk):
            if buf.strip():
                chunks.append(buf.strip()); buf = ""
            chunks.append(blk)
            continue
        if len(buf) + len(blk) + 1 > CHUNK_CHARS and buf.strip():
            chunks.append(buf.strip()); buf = ""
        buf += ("\n" if buf else "") + blk
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= MIN_CHARS]


def main():
    with open(DATA_DIR / "docs_selected.json", encoding="utf-8") as f:
        sel = json.load(f)

    corpus, stats = [], {"docs": 0, "chunks": 0, "by_element": {}, "per_doc": []}
    for d in sel["docs"]:
        with open(resolve_path(d["path"]), encoding="utf-8", errors="replace") as f:
            md = unicodedata.normalize("NFC", f.read())
        doc = d["name"]
        did = doc_id_from_name(d["path"])
        n_doc_chunks = 0
        for si, (sec_title, body) in enumerate(split_sections(md)):
            for ci, chunk in enumerate(pack_chunks(body)):
                et = element_type(chunk)
                cid = f"{did}_{si:03d}_{ci:02d}"
                corpus.append({
                    "_id": cid,
                    "title": (doc[:80] + (" — " + sec_title[:60] if sec_title else "")),
                    "text": chunk,
                    "doc": doc,
                    "section": sec_title,
                    "element_type": et,
                })
                stats["by_element"][et] = stats["by_element"].get(et, 0) + 1
                n_doc_chunks += 1
        stats["docs"] += 1
        stats["per_doc"].append({"doc": doc, "doc_id": did, "type": d["type"],
                                 "bucket": d["bucket"], "chunks": n_doc_chunks})
    stats["chunks"] = len(corpus)

    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(DATA_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"docs={stats['docs']} chunks={stats['chunks']}")
    print("element 분포:", stats["by_element"])
    for pd in sorted(stats["per_doc"], key=lambda x: -x["chunks"]):
        print(f"  {pd['chunks']:>5} chunks  [{pd['type']}/{pd['bucket']}] {pd['doc'][:55]}")


if __name__ == "__main__":
    main()
