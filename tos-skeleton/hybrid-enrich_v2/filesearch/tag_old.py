#!/usr/bin/env python3
"""OLD 태그 스키마(hybrid-enrich/build_enrichment_v2.py, element_tags_v2) 를 u2 우주에 이식 — 보고용 2×2 의 'OLD 스키마' arm.
키: element_id, element_type, contract_scope, topic, article, semantic_role(한국어 역할), values, conditions, aliases, is_toc, table_title, table_headers, formula_subject
규칙·정규식·alias 사전(aliases.json)은 원본 그대로. 조 제목 인식만 250212 md 형식(마크 없는 제목 줄)에 맞춰 heading element 기준으로 수행. QA 미접근.
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
HE = HERE
sys.path.insert(0, str(HE))
ROLE_MAP = [
    (re.compile(r"지급하지 않|면제하지 않|면책"), "면책사유(보장 제외)"), (re.compile(r"지급사유"), "보험금 지급사유(보장 내용)"),
    (re.compile(r"지급기준|지급기준표"), "보험금 지급기준·지급한도"), (re.compile(r"세부규정"), "지급 세부규정"),
    (re.compile(r"정의|용어"), "용어 정의"), (re.compile(r"갱신"), "갱신 조건"), (re.compile(r"납입면제|납입 면제"), "보험료 납입면제"),
    (re.compile(r"보험기간|보장개시|책임개시"), "보험기간·보장개시"), (re.compile(r"해지|해약|환급"), "해지·해약환급금"),
    (re.compile(r"청약|철회"), "청약·철회"), (re.compile(r"수익자"), "보험수익자"), (re.compile(r"계약의 성립|무효|취소"), "계약 성립·무효"),
    (re.compile(r"알릴 의무|고지"), "고지·통지의무"), (re.compile(r"보험료의 납입|납입최고|부활"), "보험료 납입·부활"),
]
VAL_RX = re.compile(r"(?:보험가입금액|가입금액)의?\s?\d+(?:\.\d+)?%|\d+(?:\.\d+)?%|최초\s?1회한?|\d+회한?|\d+일(?:\s?한도|분)?|\d+년|\d+(?:,\d{3})*(?:만)?원|\d+세")
COND_RX = re.compile(r"(보장개시일[^,.\n)]{2,25}|진단\s?확정[^,.\n)]{0,20}|장해지급률[^,.\n)]{0,20}|보험기간 중[^,.\n)]{0,20}|피보험자가[^,.\n)]{2,25})")
JO_RE = re.compile(r"^(제\d+(?:-\d+)?조(?:의\d+)?)\s*(.*)$")


def role_of(title):
    for rx, role in ROLE_MAP:
        if rx.search(title):
            return role
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(HERE / "out/elements_u2.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out/tags_u2_old.jsonl"))
    a = ap.parse_args()
    aliases = json.load(open(HERE / "aliases.json", encoding="utf-8")); aliases.pop("_comment", None)
    E = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    cur_scope, art_label, art_title = None, "", ""
    out = []
    for idx, e in enumerate(E):
        if e["contract_scope"] != cur_scope:
            cur_scope, art_label, art_title = e["contract_scope"], "", ""
        text = e["text"]; st = text.strip()
        if e["element_type"] == "heading":
            m = JO_RE.match(st)
            if m:
                art_label, art_title = m.group(1), m.group(2).strip("() ")
        is_toc = e["element_type"] == "heading" or ("목 차" in text) or (
            sum(1 for r in text.split("\n") if re.search(r"\|\s*\d{1,4}\s*\|?\s*$", r)) >= max(3, len(text.split("\n")) // 2))
        vals = list(dict.fromkeys(VAL_RX.findall(text)))[:8] if not is_toc else []
        conds = list(dict.fromkeys(m if isinstance(m, str) else m[0] for m in COND_RX.findall(text)))[:4] if not is_toc else []
        al = []
        if not is_toc:
            for canon, alts in aliases.items():
                if canon in text:
                    al.extend(x for x in alts if x not in text)
        tag = {"element_id": e["element_id"], "element_type": e["element_type"], "contract_scope": e["contract_scope"],
               "topic": re.sub(r"\(무배당[^)]*\)|\(간편\)|주계약\(|\)$", "", e["contract_scope"]).strip(),
               "article": (art_label + (f"({art_title})" if art_title else "")) if art_label else "",
               "semantic_role": role_of(art_title) if art_label else "", "values": vals, "conditions": conds,
               "aliases": list(dict.fromkeys(al))[:5], "is_toc": is_toc}
        if e["element_type"] == "table":
            hdrs = []
            for l2 in text.split("\n"):
                s2 = l2.strip()
                if s2.startswith("|"):
                    cells = [c.strip() for c in s2.strip("|").split("|")]
                    if cells and not all(re.fullmatch(r"[-: ]*", c) for c in cells):
                        hdrs = [c for c in cells if c][:8]; break
            tag["table_headers"] = hdrs
            prev = next((E[k]["text"].strip()[:60] for k in range(idx - 1, max(-1, idx - 4), -1)
                         if E[k]["element_type"] in ("paragraph", "heading") and 2 < len(E[k]["text"].strip()) < 80), "")
            tag["table_title"] = prev
        if e["element_type"] == "formula":
            tag["formula_subject"] = next((E[k]["text"].strip()[:60] for k in range(idx - 1, max(-1, idx - 4), -1)
                                           if E[k]["element_type"] == "paragraph" and len(E[k]["text"].strip()) < 80), "")
        out.append(tag)
    with open(a.out, "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n = len(out)
    st = {"schema": "OLD(element_tags_v2 이식)", "n": n,
          "coverage": {k: round(sum(1 for t in out if t.get(k)) / n, 3) for k in ("article", "semantic_role", "values", "conditions", "aliases")},
          "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(st, open(a.out.replace(".jsonl", "_stats.json"), "w"), ensure_ascii=False, indent=2); print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    main()
