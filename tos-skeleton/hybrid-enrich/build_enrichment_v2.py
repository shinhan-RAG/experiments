#!/usr/bin/env python3
"""Enrichment v2: 전체 스키마 + running 조항 컨텍스트 + alias 선별 + 표 태그 재설계.
QA 접근 가드 포함. 출력: chunk_metadata_v2.jsonl / element_tags_v2.jsonl"""
import json, re, os, hashlib, bisect, unicodedata
import builtins

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"

_real_open = open
def _guarded_open(path, *a, **kw):
    if "QA_set/정답셋" in str(path) or "qa100_gold" in str(path):
        raise PermissionError(f"QA 접근 금지: {path}")
    return _real_open(path, *a, **kw)
builtins.open = _guarded_open

# ---------- running context: line -> (조항 라벨+제목) ----------
ART_HEAD = re.compile(r"^#{0,6}\s*(제\s?\d+(?:-\d+)?조(?:의\s?\d+)?)\s*[\(（]?([^)）\n]{0,40})[\)）]?\s*$")
BUPYO = re.compile(r"^#{0,6}\s*\[?(부표\s?[\d-]+(?:-\d+)?)\]?\s*(.{0,40})$")

ROLE_MAP = [
    (re.compile(r"지급하지 않|면제하지 않|면책"), "면책사유(보장 제외)"),
    (re.compile(r"지급사유"), "보험금 지급사유(보장 내용)"),
    (re.compile(r"지급기준|지급기준표"), "보험금 지급기준·지급한도"),
    (re.compile(r"세부규정"), "지급 세부규정"),
    (re.compile(r"정의|용어"), "용어 정의"),
    (re.compile(r"갱신"), "갱신 조건"),
    (re.compile(r"납입면제|납입 면제"), "보험료 납입면제"),
    (re.compile(r"보험기간|보장개시|책임개시"), "보험기간·보장개시"),
    (re.compile(r"해지|해약|환급"), "해지·해약환급금"),
    (re.compile(r"청약|철회"), "청약·철회"),
    (re.compile(r"수익자"), "보험수익자"),
    (re.compile(r"계약의 성립|무효|취소"), "계약 성립·무효"),
    (re.compile(r"알릴 의무|고지"), "고지·통지의무"),
    (re.compile(r"보험료의 납입|납입최고|부활"), "보험료 납입·부활"),
]

def role_of(title):
    for rx, role in ROLE_MAP:
        if rx.search(title):
            return role
    return ""

VAL_RX = re.compile(r"(?:보험가입금액|가입금액)의?\s?\d+(?:\.\d+)?%|\d+(?:\.\d+)?%|최초\s?1회한?|\d+회한?|\d+일(?:\s?한도|분)?|\d+년|\d+(?:,\d{3})*(?:만)?원|\d+세")
COND_RX = re.compile(r"(보장개시일[^,.\n)]{2,25}|진단\s?확정[^,.\n)]{0,20}|장해지급률[^,.\n)]{0,20}|보험기간 중[^,.\n)]{0,20}|피보험자가[^,.\n)]{2,25})")

def main():
    raw = unicodedata.normalize("NFC", _real_open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    lines = raw.split("\n")
    # running article map
    art_bounds = []  # (line_no, label, title, role)
    for i, l in enumerate(lines):
        s = l.strip()
        if len(s) > 70 or "|" in s:
            continue
        m = ART_HEAD.match(s)
        if m:
            title = (m.group(2) or "").strip()
            art_bounds.append((i + 1, m.group(1).replace(" ", ""), title, role_of(title)))
            continue
        b = BUPYO.match(s)
        if b:
            art_bounds.append((i + 1, b.group(1).replace(" ", ""), b.group(2).strip()[:30], "보험금 지급기준·지급한도"))
    al = [a[0] for a in art_bounds]

    def art_of(line_no):
        j = bisect.bisect_right(al, line_no) - 1
        return art_bounds[j] if j >= 0 else None

    aliases = json.load(_real_open(os.path.join(BASE, "aliases.json"), encoding="utf-8"))
    aliases.pop("_comment", None)

    def sel_alias(text, is_toc):
        if is_toc:
            return []
        out = []
        for canon, alts in aliases.items():
            if canon in text:  # canonical 실제 등장 시에만
                out.extend(a for a in alts if a not in text)  # 본문에 없는 변형만 (간극 메우기)
        return list(dict.fromkeys(out))[:5]

    TOC_RX = re.compile(r"목\s?차|\|\s*\d+\s*\|\s*$")
    def looks_toc(text, etype):
        if etype == "heading":
            return True
        rows = text.split("\n")
        pagey = sum(1 for r in rows if re.search(r"\|\s*\d{1,4}\s*\|?\s*$", r))
        return ("목 차" in text) or (pagey >= max(3, len(rows) // 2))

    def core_scope(scope):
        s = re.sub(r"\(무배당[^)]*\)|\(간편\)|주계약\(|\)$", "", scope)
        return s.strip()

    def prev_para_title(elems, idx):
        for k in range(idx - 1, max(-1, idx - 4), -1):
            t = elems[k]["text"].strip()
            if elems[k]["element_type"] in ("paragraph", "heading") and 2 < len(t) < 80 and "|" not in t:
                return re.sub(r"^#+\s*", "", t)[:60]
        return ""

    # ---------- 엘리먼트 태그 v2 ----------
    elems = [json.loads(l) for l in _real_open(os.path.join(OUT, "elements.jsonl"), encoding="utf-8")]
    with _real_open(os.path.join(OUT, "element_tags_v2.jsonl"), "w", encoding="utf-8") as f:
        for idx, e in enumerate(elems):
            text = e["text"]
            a = art_of(e["line_start"])
            is_toc = looks_toc(text, e["element_type"])
            vals = list(dict.fromkeys(VAL_RX.findall(text)))[:8] if not is_toc else []
            conds = list(dict.fromkeys(m if isinstance(m, str) else m[0] for m in COND_RX.findall(text)))[:4] if not is_toc else []
            tag = {
                "element_id": e["element_id"], "element_type": e["element_type"],
                "contract_scope": e["contract_scope"], "topic": core_scope(e["contract_scope"]),
                "article": (a[1] + (f"({a[2]})" if a[2] else "")) if a else "",
                "semantic_role": a[3] if a else "",
                "values": vals, "conditions": conds,
                "aliases": sel_alias(text, is_toc), "is_toc": is_toc,
            }
            if e["element_type"] == "table":
                # 의미 있는 헤더만
                hdrs = []
                for l2 in text.split("\n"):
                    s2 = l2.strip()
                    if s2.startswith("|"):
                        cells = [c.strip() for c in s2.strip("|").split("|")]
                        cells = [c for c in cells if c and not set(c) <= set("-: ") and not c.replace(",", "").replace(".", "").isdigit() and len(c) > 2]
                        if cells:
                            hdrs = cells[:8]
                            break
                tag["table_headers"] = hdrs
                tag["table_title"] = prev_para_title(elems, idx)
            if e["element_type"] == "formula":
                tag["formula_subject"] = prev_para_title(elems, idx)
            f.write(json.dumps(tag, ensure_ascii=False) + "\n")

    # ---------- 청크 메타데이터 v2 (벡터용: 짧고 밀도 높게) ----------
    chunks = [json.loads(l) for l in _real_open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")]
    with _real_open(os.path.join(OUT, "chunk_metadata_v2.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            a = art_of(c["line_start"])
            text = c["text"]
            vals = list(dict.fromkeys(VAL_RX.findall(text)))[:6]
            meta = {
                "chunk_id": c["chunk_id"],
                "contract_scope": c["contract_scope"], "topic": core_scope(c["contract_scope"]),
                "article": (a[1] + (f"({a[2]})" if a[2] else "")) if a else "",
                "semantic_role": a[3] if a else "",
                "values": vals,
                "aliases": sel_alias(text, False)[:5],
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    stats = {"articles_detected": len(art_bounds),
             "toc_elements": sum(1 for l in _real_open(os.path.join(OUT, "element_tags_v2.jsonl"), encoding="utf-8") if json.loads(l)["is_toc"]),
             "version": "v2"}
    print(json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    main()
