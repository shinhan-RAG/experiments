#!/usr/bin/env python3
"""우주(element universe) 생성 — hybrid-enrich/build_elements.py 의 분할 규칙을 그대로 이식.

규칙(결정론): 표 블록(`|` 시작 연속 줄) / 산식 블록(`$$` ... `$$`) / 빈 줄 구분 단락 /
단독 `#` 줄 = heading. 4,000자 초과 블록은 줄 단위로 4,000자 분절(part).
contract_scope 는 특약·주계약 헤더 줄로 결정(페이지 헤더 반복은 무해 — 같은 값 재배정).

원본 대비 변경점: 문서 경로 CLI 인자화, 주계약 헤더 정규식을 250212판 상품명
"(간편)신한통합건강보장보험 원(ONE)(무배당, …)" 까지 허용, 산출물 SHA 기록.
QA·gold 파일은 열지 않는다.
"""
import argparse, bisect, collections, hashlib, json, os, re, unicodedata

MAX_ELEM_CHARS = 4000
RULE_VERSION = "hybrid-enrich/build_elements.py@dev-fe43198 + main-regex-250212 + formula-singleline-fix + rider-prefix-neutral"

# Product families do not consistently prefix rider headings with ``(간편)``.
# The prior rule silently attached valid non-prefixed riders to the preceding
# contract.  The semantic boundary is the rider suffix plus the parenthesised
# product attributes; a leading product-family marker is optional.  Table rows
# are excluded by the caller before this expression is evaluated.
RIDER = re.compile(r"^(?!.*\|).{1,100}특약\s*\(\s*무배당[^)]*\)\s*$")
MAINS = [
    re.compile(r"^신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^)]*\)\s*$"),          # 260507판 표기
    re.compile(r"^\(간편\)신한통합건강보장보험\s*원\(ONE\)\(무배당[^)]*\)\s*$"),        # 250212판 표기
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NOISE = [
    re.compile(r"^_{3,}$"), re.compile(r"^-{3}$"), re.compile(r"^\d{10,}$"), re.compile(r"^\d{1,4}$"),
    re.compile(r"^#?\s*SHINHAN LIFE$"), re.compile(r"^S$"), re.compile(r"^S 신한라\s?이프$"),
    re.compile(r"^\(\d+,\d+\),\(\d+,\d+\)$"),
]
STARTS_MARKER = re.compile(r"^(?:[①-⑳]|제\d|\d+[.)]|[가-힣][.)]\s|[·•▪■□○●\-\[\(<「『#|]|[a-z][.)]?\s)")
ARTICLE_TITLE = re.compile(r"^제\d+(?:-\d+)?(?:조|관|편|장|절)(?:의\d+)?\s*[\(\[]?[^.。]{0,60}$")
ENDS_SENT = re.compile(r"(?:다\.|[.)\]」』>:]|니다|습니다|음|함)\s*$")


def is_noise(s):
    return any(p.match(s) for p in NOISE)


def build(raw, clean=False):
    lines = raw.split("\n")
    line_offs, pos = [], 0
    for l in lines:
        line_offs.append(pos)
        pos += len(l) + 1

    bounds = []
    for i, l in enumerate(lines):
        s = l.strip()
        if "|" in s:
            continue
        if RIDER.match(s):
            bounds.append((i + 1, s))
        elif any(m.match(s) for m in MAINS) and i + 1 > 200:
            bounds.append((i + 1, "주계약(" + s + ")"))
    bl = [b[0] for b in bounds]

    def scope_of(line_no):
        j = bisect.bisect_right(bl, line_no) - 1
        return bounds[j][1] if j >= 0 else ""

    blocks = []

    def flush(i0, i1, etype):
        text = "\n".join(lines[i0:i1])
        if not text.strip():
            return
        st = text.strip()
        if clean and etype in ("paragraph", "heading"):
            if is_noise(st) or RIDER.match(st) or any(m.match(st) for m in MAINS):
                return  # 페이지 헤더·페이지 번호·구분선 등 노이즈 블록 제거(scope 판정에는 이미 사용됨)
            # 페이지 경계로 끊긴 문장 이어붙이기: 직전 단락이 문장 종결이 아니고 현 단락이 마커로 시작하지 않으면 병합
            if blocks and blocks[-1][2] == "paragraph" and etype == "paragraph" \
                    and not ENDS_SENT.search(blocks[-1][3].strip()) and not STARTS_MARKER.match(st) \
                    and len(blocks[-1][3]) + len(text) <= MAX_ELEM_CHARS:
                p0, p1, pt, ptext = blocks[-1]
                blocks[-1] = (p0, i1, "paragraph", ptext + "\n" + text)
                return
        blocks.append((i0, i1, etype, text))

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].lstrip()
        if s.startswith("|"):
            j = i
            while j < n and (lines[j].lstrip().startswith("|") or not lines[j].strip()):
                if not lines[j].strip() and (j + 1 >= n or not lines[j + 1].lstrip().startswith("|")):
                    break
                j += 1
            flush(i, j, "table")
            i = j
        elif s.startswith("$$"):
            # 250212 md 의 산식은 한 줄 자기완결(`$$ … $$`, 160/170). 원 규칙(다음 `$$` 줄까지 확장)은
            # 이 문서에서 최대 수만 자를 산식으로 오포획하므로 한 줄 블록으로 고정. 홀로 선 `$$`(10줄)는 파서 잔재.
            if s.strip() == "$$":
                if not clean:
                    flush(i, i + 1, "formula")
            else:
                flush(i, i + 1, "formula")
            i += 1
        elif s.startswith("```"):
            # 펜스 블록(QR코드 json 잔재). 여는 펜스는 언어 태그가 있는 줄(```json 등)만 인정하고
            # 닫는 펜스는 다음 ``` 줄. 홀로 선 ``` 은 잔재로 보고 그 줄만 소비 (짝 오류로 대량 삭제 방지)
            if s.strip() == "```":
                if not clean:
                    flush(i, i + 1, "paragraph")
                i += 1
            else:
                j = i + 1
                while j < n and lines[j].strip() != "```" and j - i < 40:
                    j += 1
                if not clean:
                    flush(i, min(j + 1, n), "paragraph")
                i = min(j + 1, n)
        elif not s:
            i += 1
        else:
            j = i
            while j < n and lines[j].strip() and not lines[j].lstrip().startswith(("|", "$$")):
                j += 1
            etype = "heading" if re.match(r"^#{1,6}\s", lines[i]) and j - i == 1 else "paragraph"
            if clean and etype == "paragraph" and j - i == 1 and ARTICLE_TITLE.match(lines[i].strip()) and not ENDS_SENT.search(lines[i].strip()):
                etype = "heading"  # 조·관·편 제목 줄(이 md 는 # 마크 없음)
            flush(i, j, etype)
            i = j

    elems = []
    for i0, i1, etype, text in blocks:
        c0 = line_offs[i0]
        c1 = line_offs[i1 - 1] + len(lines[i1 - 1]) if i1 - 1 < n else len(raw)
        if len(text) > MAX_ELEM_CHARS:
            tls = text.split("\n")
            part, plen, p0, parts = [], 0, i0, []
            for k, tl in enumerate(tls):
                part.append(tl)
                plen += len(tl) + 1
                if plen >= MAX_ELEM_CHARS:
                    parts.append((p0, i0 + k + 1, "\n".join(part)))
                    part, plen, p0 = [], 0, i0 + k + 1
            if part:
                parts.append((p0, i1, "\n".join(part)))
            for pp0, pp1, ptext in parts:
                elems.append((pp0, pp1, etype, ptext, line_offs[pp0], line_offs[pp1 - 1] + len(lines[pp1 - 1])))
        else:
            elems.append((i0, i1, etype, text, c0, c1))

    out = []
    for k, (i0, i1, etype, text, c0, c1) in enumerate(elems):
        out.append({
            "element_id": f"e{k:05d}", "element_type": etype, "text": text,
            "char_start": c0, "char_end": c1, "line_start": i0 + 1, "line_end": i1,
            "contract_scope": scope_of(i0 + 1),
        })
    n_split = sum(1 for b in blocks if len(b[3]) > MAX_ELEM_CHARS)
    return out, n_split, len(bounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--name", default="elements_u1.jsonl")
    ap.add_argument("--clean", action="store_true", help="노이즈 블록 제거 + 페이지 경계 문장 병합 (u2 규칙)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = unicodedata.normalize("NFC", open(a.doc, encoding="utf-8").read().replace("\r\n", "\n"))
    out, n_split, n_bounds = build(raw, clean=a.clean)
    path = os.path.join(a.out, a.name)
    with open(path, "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    lens = sorted(len(e["text"]) for e in out)
    stats = {
        "universe": a.name, "rule_version": RULE_VERSION + (" + clean(noise-drop, pagebreak-merge)" if a.clean else ""), "max_elem_chars": MAX_ELEM_CHARS,
        "doc": os.path.basename(a.doc), "doc_sha256": sha256_file(a.doc),
        "doc_lines": raw.count("\n") + 1, "doc_chars": len(raw),
        "n_elements": len(out),
        "by_type": dict(collections.Counter(e["element_type"] for e in out)),
        "avg_len": sum(lens) // len(lens), "median_len": lens[len(lens) // 2],
        "p90_len": lens[int(len(lens) * 0.9)], "split_parts": n_split,
        "scope_bounds": n_bounds,
        "n_scopes": len({e["contract_scope"] for e in out}),
        "elements_sha256": sha256_file(path),
    }
    json.dump(stats, open(path.replace(".jsonl", "_stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
