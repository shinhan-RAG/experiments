"""
신한라이프 파싱본(md) → dr-dci BEIR corpus.jsonl 청킹.

- 헤더(#..######) 기준으로 섹션 분할, 표 블록은 통째로 보존
- 너무 긴 섹션은 문단/표 경계에서 ~CHUNK_CHARS 단위로 재분할
- 각 청크에 element_type(table/formula/text) 태깅 → 제안1(pre-filter) 실험용
출력: data/raw/shinhan/corpus.jsonl  ({_id, title, text, doc, section, element_type})
      data/raw/shinhan/corpus_stats.json
"""
import json, re, hashlib
from pathlib import Path

SRC_DIR = Path("/Users/seyoung/workspace/braincrew/Data-LAB/corpus_md")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "shinhan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_CHARS = 1400        # 목표 청크 길이(글자)
MIN_CHARS = 120           # 이보다 짧은 조각은 앞 청크에 흡수/버림
HEADER_RE = re.compile(r'(?m)^(#{1,6})\s+(.*)$')
TABLE_ROW_RE = re.compile(r'(?m)^\s*\|.*\|\s*$')
FORMULA_RE = re.compile(r'\$\$|\\\(|\\\[')


def doc_id_from_name(name: str) -> str:
    """파일명 → 짧은 안정적 doc prefix"""
    return hashlib.sha1(name.encode()).hexdigest()[:8]


def element_type(text: str) -> str:
    table_rows = len(TABLE_ROW_RE.findall(text))
    if table_rows >= 3:
        return "table"
    if FORMULA_RE.search(text):
        return "formula"
    return "text"


def split_sections(md: str):
    """헤더 기준 (section_title, body) 리스트로 분할."""
    matches = list(HEADER_RE.finditer(md))
    if not matches:
        return [("", md)]
    sections = []
    # 첫 헤더 이전 프리앰블
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
    """섹션 본문을 표 블록 보존하며 ~CHUNK_CHARS 청크로 패킹."""
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
        # 표 블록이 통째로 너무 크면 단독 청크로
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
    files = sorted(SRC_DIR.glob("*.md"))
    corpus, stats = [], {"docs": 0, "chunks": 0, "by_element": {}, "per_doc": []}
    for f in files:
        md = f.read_text()
        doc = f.stem
        did = doc_id_from_name(f.name)
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
        stats["per_doc"].append({"doc": doc, "doc_id": did, "chunks": n_doc_chunks})
    stats["chunks"] = len(corpus)

    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "corpus_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"docs={stats['docs']} chunks={stats['chunks']}")
    print("element 분포:", stats["by_element"])
    print("문서당 청크 top5:",
          sorted(stats["per_doc"], key=lambda x: -x["chunks"])[:5])


if __name__ == "__main__":
    main()
