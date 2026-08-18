#!/usr/bin/env python3
"""구조 엘리먼트 추출 (파일서치/grep 단위). 결정론적: 표 블록 / 산식 블록 / 빈 줄 구분 단락.
elements.jsonl: element_id, element_type, text, char_start, char_end, line_start, line_end, contract_scope"""
import json, re, unicodedata, os, bisect

_DEFAULT_DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
DOC = os.environ.get("DOC_PATH", _DEFAULT_DOC)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MAX_ELEM_CHARS = 4000  # 초대형 표는 4000자 단위로 분절(part) — 전 Arm 동일 규칙


def main():
    raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    lines = raw.split("\n")
    line_offs = []
    pos = 0
    for l in lines:
        line_offs.append(pos)
        pos += len(l) + 1

    # 특약 경계 (update_scope와 동일 패턴)
    RIDER = re.compile(r"^\(간편\).{0,60}특약\(무배당[^)]*\)\s*$")
    MAIN = re.compile(r"^(?:\(간편\))?신한(?:\(간편가입\))?통합건강보[장험]+ 원\(ONE\)\(무배당[^)]*\)\s*$")
    bounds = []
    for i, l in enumerate(lines):
        s = l.strip()
        if "|" in s:
            continue
        if RIDER.match(s):
            bounds.append((i + 1, s))
        elif MAIN.match(s) and i + 1 > 200:
            bounds.append((i + 1, "주계약(" + s + ")"))
    bl = [b[0] for b in bounds]

    def scope_of(line_no):
        j = bisect.bisect_right(bl, line_no) - 1
        return bounds[j][1] if j >= 0 else ""

    def flush(blocks, i0, i1, etype):
        text = "\n".join(lines[i0:i1])
        if not text.strip():
            return
        blocks.append((i0, i1, etype, text))

    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].lstrip()
        if s.startswith("|"):
            j = i
            while j < n and (lines[j].lstrip().startswith("|") or not lines[j].strip()):
                if not lines[j].strip() and (j + 1 >= n or not lines[j + 1].lstrip().startswith("|")):
                    break
                j += 1
            flush(blocks, i, j, "table")
            i = j
        elif s.startswith("$$"):
            j = i + 1
            while j < n and not lines[j].lstrip().startswith("$$"):
                j += 1
            flush(blocks, i, min(j + 1, n), "formula")
            i = min(j + 1, n)
        elif not s:
            i += 1
        else:
            j = i
            while j < n and lines[j].strip() and not lines[j].lstrip().startswith(("|", "$$")):
                j += 1
            etype = "heading" if re.match(r"^#{1,6}\s", lines[i]) and j - i == 1 else "paragraph"
            flush(blocks, i, j, etype)
            i = j

    elems = []
    for i0, i1, etype, text in blocks:
        c0 = line_offs[i0]
        c1 = line_offs[i1 - 1] + len(lines[i1 - 1]) if i1 - 1 < n else len(raw)
        # 초대형 엘리먼트 분절
        if len(text) > MAX_ELEM_CHARS:
            tls = text.split("\n")
            part, plen, p0 = [], 0, i0
            parts = []
            for k, tl in enumerate(tls):
                part.append(tl)
                plen += len(tl) + 1
                if plen >= MAX_ELEM_CHARS:
                    parts.append((p0, i0 + k + 1, "\n".join(part)))
                    part, plen, p0 = [], 0, i0 + k + 1
            if part:
                parts.append((p0, i1, "\n".join(part)))
            for (pp0, pp1, ptext) in parts:
                pc0 = line_offs[pp0]
                pc1 = line_offs[pp1 - 1] + len(lines[pp1 - 1])
                elems.append((pp0, pp1, etype, ptext, pc0, pc1))
        else:
            elems.append((i0, i1, etype, text, c0, c1))

    out = []
    for k, (i0, i1, etype, text, c0, c1) in enumerate(elems):
        out.append({
            "element_id": f"e{k:05d}", "element_type": etype, "text": text,
            "char_start": c0, "char_end": c1, "line_start": i0 + 1, "line_end": i1,
            "contract_scope": scope_of(i0 + 1),
        })
    with open(os.path.join(OUT, "elements.jsonl"), "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    import collections
    stats = {"n_elements": len(out),
             "by_type": dict(collections.Counter(e["element_type"] for e in out)),
             "avg_len": sum(len(e["text"]) for e in out) // len(out),
             "split_parts": sum(1 for b in blocks if len(b[3]) > MAX_ELEM_CHARS)}
    print(json.dumps(stats, ensure_ascii=False))
    json.dump(stats, open(os.path.join(OUT, "element_stats.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
