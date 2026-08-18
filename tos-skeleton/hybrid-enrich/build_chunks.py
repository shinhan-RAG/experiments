#!/usr/bin/env python3
"""Source-faithful recursive 600/100 chunking with hard contract boundaries."""
import json, re, unicodedata, hashlib, os

_DEFAULT_DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
DOC = os.environ.get("DOC_PATH", _DEFAULT_DOC)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CHUNK_SIZE = 600
OVERLAP = 100
SEPS = ["\n\n", "\n", ". ", " "]
MIN_CORE = CHUNK_SIZE // 2


def recursive_ranges(text, start, end):
    """Return contiguous, gap-free core ranges while preferring semantic separators."""
    ranges = []
    cursor = start
    while cursor < end:
        hard_end = min(end, cursor + CHUNK_SIZE)
        if hard_end == end:
            ranges.append((cursor, end))
            break
        cut = None
        low = cursor + MIN_CORE
        window = text[cursor:hard_end]
        for sep in SEPS:
            relative = window.rfind(sep, MIN_CORE)
            if relative >= 0:
                candidate = cursor + relative + len(sep)
                if candidate >= low:
                    cut = candidate
                    break
        if cut is None or cut <= cursor:
            cut = hard_end
        ranges.append((cursor, cut))
        cursor = cut
    # Avoid a tiny tail created only by a separator near a hard boundary.
    if len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < MIN_CORE:
        left, right = ranges[-2][0], ranges[-1][1]
        total = right - left
        if total <= CHUNK_SIZE:
            ranges[-2:] = [(left, right)]
        else:
            target = left + total // 2
            lower, upper = right - CHUNK_SIZE, left + CHUNK_SIZE
            cut = None
            for sep in SEPS:
                candidates = []
                before = text.rfind(sep, lower, target + 1)
                after = text.find(sep, target, upper)
                if before >= 0:
                    candidates.append(before + len(sep))
                if after >= 0:
                    candidates.append(after + len(sep))
                if candidates:
                    cut = min(candidates, key=lambda point: abs(point - target))
                    break
            cut = cut or target
            ranges[-2:] = [(left, cut), (cut, right)]
    return ranges


def limit_article_mix(text, ranges):
    """Keep at most two article headings in a core range."""
    article = re.compile(r"(?m)(?=^#{0,6}\s*제\s?\d+(?:-\d+)?조(?:의\s?\d+)?(?:\s|$))")
    output = []
    for left, right in ranges:
        cursor = left
        while cursor < right:
            starts = [cursor + match.start() for match in article.finditer(text[cursor:right])]
            if len(starts) <= 2:
                output.append((cursor, right))
                break
            cut = starts[2]
            if cut <= cursor:
                cut = min(right, cursor + CHUNK_SIZE)
            output.append((cursor, cut))
            cursor = cut
    return output


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

    # Contract headers are hard boundaries. Recursive ranges are contiguous inside each scope.
    contract_line_re = re.compile(
        r"^(?:[^|\n]{2,180}특약[^()\n]{0,40}\(무배당[^)\n]*\)|"
        r"(?:\(간편\))?신한(?:\(간편가입\))?통합건강보[장험]+ 원\(ONE\)\(무배당[^\n]*\))\s*$"
    )
    hard_bounds = [(0, "")]
    for line_no, line in enumerate(lines):
        value = line.strip()
        if "|" not in line and contract_line_re.match(value) and line_no + 1 > 200:
            label = "주계약(" + value + ")" if value.startswith("신한(간편가입)") else value
            hard_bounds.append((line_starts[line_no], label))
    hard_bounds = sorted(dict(hard_bounds).items())
    hard_starts = [point for point, _ in hard_bounds] + [len(raw)]
    core_ranges = []
    for left, right in zip(hard_starts, hard_starts[1:]):
        core_ranges.extend(recursive_ranges(raw, left, right))

    chunks = []
    for idx, (core_start, end) in enumerate(core_ranges):
        # Overlap never crosses a contract hard boundary.
        boundary_index = bisect.bisect_right(hard_starts, core_start) - 1
        scope_start = hard_starts[boundary_index]
        scope_label = hard_bounds[boundary_index][1]
        start = max(scope_start, core_start - OVERLAP)
        p = raw[start:end]
        ls, le = line_of(start), line_of(max(start, end - 1))
        core_ls = line_of(core_start)
        path = section_path(core_ls)
        has_table = "|" in p and re.search(r"^\s*\|", p, re.M) is not None
        has_formula = "$$" in p or re.search(r"\\frac|\\times|\\sum", p) is not None
        etype = "table" if has_table else ("formula" if has_formula else "text")
        chunks.append({
            "chunk_id": f"c{idx:05d}",
            "text": p,
            "char_start": start, "char_end": end,
            "core_char_start": core_start,
            "line_start": ls, "line_end": le, "core_line_start": core_ls,
            "section_path": path[-4:],
            "contract_scope": scope_label or contract_scope(path),
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
