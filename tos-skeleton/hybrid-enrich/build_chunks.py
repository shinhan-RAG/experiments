#!/usr/bin/env python3
"""Recursive 600/100 청킹. line 범위 보존. 구조 컨텍스트(section_path, contract_scope)는
청크 위치 기준으로 헤딩 스택에서 계산해 함께 저장한다 (enrichment 원료, BASE 색인에는 미포함)."""
import json, re, unicodedata, hashlib, os

DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CHUNK_SIZE = 600
OVERLAP = 100
SEPS = ["\n\n", "\n", ". ", " ", ""]


def recursive_split(text, size, seps):
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep == "":
            return [text[j:j + size] for j in range(0, len(text), size)]
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        pieces, buf = [], ""
        for p in parts:
            cand = buf + sep + p if buf else p
            if len(cand) <= size:
                buf = cand
            else:
                if buf:
                    pieces.append(buf)
                if len(p) > size:
                    pieces.extend(recursive_split(p, size, seps[i + 1:]))
                    buf = ""
                else:
                    buf = p
        if buf:
            pieces.append(buf)
        return [x for x in pieces if x.strip()]
    return [text]


def add_overlap(pieces, overlap):
    out = []
    for i, p in enumerate(pieces):
        if i == 0:
            out.append(p)
        else:
            tail = pieces[i - 1][-overlap:]
            out.append(tail + p)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = open(DOC, encoding="utf-8").read()
    raw = unicodedata.normalize("NFC", raw.replace("\r\n", "\n"))
    lines = raw.split("\n")
    # char offset -> line 매핑
    line_starts = []
    pos = 0
    for l in lines:
        line_starts.append(pos)
        pos += len(l) + 1

    import bisect
    def line_of(off):
        return bisect.bisect_right(line_starts, off)  # 1-based

    # 헤딩 스택: (line_no, level, title)
    headings = []
    for i, l in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)", l)
        if m:
            headings.append((i + 1, len(m.group(1)), m.group(2).strip()))

    def section_path(line_no):
        stack = []
        for hl, lv, t in headings:
            if hl > line_no:
                break
            while stack and stack[-1][0] >= lv:
                stack.pop()
            stack.append((lv, t))
        return [t for _, t in stack]

    RIDER = re.compile(r"(특약|주계약|보통약관)")
    def contract_scope(path):
        for t in reversed(path):
            if RIDER.search(t):
                return t
        return ""

    # 문단(빈 줄) 단위로 먼저 모으고 recursive
    chunks = []
    para_re = re.compile(r"\n{2,}")
    # 전체 텍스트를 그대로 recursive (문서 전체가 하나의 텍스트)
    # 위치 추적을 위해 순차 탐색으로 offset 복원
    pieces = recursive_split(raw, CHUNK_SIZE, SEPS)
    pieces = add_overlap(pieces, OVERLAP)
    cursor = 0
    for idx, p in enumerate(pieces):
        core = p if idx == 0 else p[OVERLAP:] if len(p) > OVERLAP else p
        off = raw.find(core, cursor)
        if off < 0:
            off = raw.find(core[:200], cursor)
        if off < 0:
            off = cursor
        start = max(0, off - (0 if idx == 0 else OVERLAP))
        end = off + len(core)
        cursor = off + max(1, len(core) - 5)
        ls, le = line_of(start), line_of(max(start, end - 1))
        path = section_path(ls)
        has_table = "|" in p and re.search(r"^\s*\|", p, re.M) is not None
        has_formula = "$$" in p or re.search(r"\\frac|\\times|\\sum", p) is not None
        etype = "table" if has_table else ("formula" if has_formula else "text")
        chunks.append({
            "chunk_id": f"c{idx:05d}",
            "text": p,
            "char_start": start, "char_end": end,
            "line_start": ls, "line_end": le,
            "section_path": path[-4:],
            "contract_scope": contract_scope(path),
            "element_type": etype,
        })

    with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    import collections
    stats = {
        "n_chunks": len(chunks),
        "doc_chars": len(raw), "doc_lines": len(lines),
        "chunk_size": CHUNK_SIZE, "overlap": OVERLAP,
        "element_type": dict(collections.Counter(c["element_type"] for c in chunks)),
        "avg_len": sum(len(c["text"]) for c in chunks) // len(chunks),
        "doc_sha256": hashlib.sha256(raw.encode()).hexdigest()[:16],
    }
    json.dump(stats, open(os.path.join(OUT, "chunk_stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
