#!/usr/bin/env python3
"""u2 element → 8슬롯 semantic tag (규칙 기반, LLM 0회). 04B 스키마 키를 그대로 쓴다.

슬롯: contract / subject / role / article / table / qualifier / reference / schema
 - 구조 규칙(결정론): contract(=contract_scope 정규화), article(직전 조 제목), section(관·편), schema(element_type),
   table(헤더·행키), reference(본문 내 조 참조)
 - 내용 규칙(정규식, hybrid-enrich/build_element_fields_v4.py 이식): subject / role / qualifier
LLM 슬롯 채움은 별도 스크립트로 덧씌운다(schema_version 로 구분). QA·gold 파일은 열지 않는다.
"""
import argparse, hashlib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from patterns import (QUOTED_RE, SUBJECT_RE, DEFINITION_RE, CONDITION_RE, VALUE_RE,  # noqa: E402
                                     ROLE_RULES, ROLE_KO, unique, clean_subjects, table_structure)

VERSION = "semtag-v2-rules-1.0"
JO_RE = re.compile(r"^(제\d+(?:-\d+)?조(?:의\d+)?)\s*(.*)$")
SEC_RE = re.compile(r"^(제\d+(?:관|편|장|절))\s*(.*)$")
REF_RE = re.compile(r"제\d+(?:-\d+)?조(?:의\d+)?(?:\([^)]{0,40}\))?")


def contract_key(scope):
    s = re.sub(r"^주계약\((.*)\)$", r"\1", scope or "")
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(HERE / "out" / "elements_u2.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out" / "tags_u2_rules.jsonl"))
    a = ap.parse_args()
    E = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    out = []
    cur_scope, article, article_title, section = None, "", "", ""
    prev_texts = []
    for i, e in enumerate(E):
        if e["contract_scope"] != cur_scope:
            cur_scope, article, article_title, section = e["contract_scope"], "", "", ""
        text = e["text"]; st = text.strip()
        if e["element_type"] == "heading":
            m = JO_RE.match(st)
            if m:
                article, article_title = m.group(1), m.group(2).strip("() ")
            else:
                m2 = SEC_RE.match(st)
                if m2:
                    section = st[:60]; article, article_title = "", ""
        headers, row_keys = table_structure(text) if e["element_type"] == "table" else ([], [])
        prev = "\n".join(t for t in prev_texts[-3:] if len(t) <= 160)
        subjects = (clean_subjects(QUOTED_RE.findall(text[:5000]) + DEFINITION_RE.findall(text[:3000]), 20, require_domain=True)
                    + clean_subjects(SUBJECT_RE.findall(text[:5000]), 25, require_domain=True)
                    + clean_subjects(row_keys, 30, require_domain=True))
        if article_title and re.search(r"보험금|급여금|지원비|치료비|진단비|수술비|질병|질환|암|수술|치료|검사|입원|통원", article_title):
            subjects.append(re.sub(r"보험금의?|지급(?:사유|금액|에 관한 세부규정)|정의|세부규정|기준", " ", article_title))
        if e["element_type"] == "formula":
            subjects += QUOTED_RE.findall(prev) + SUBJECT_RE.findall(prev)
        subjects = clean_subjects(subjects, 30)
        ctx = article_title + "\n" + section + "\n" + text[:4000]
        roles = unique([r for p, r in ROLE_RULES if re.search(p, ctx)], 8)
        qualifiers = unique(CONDITION_RE.findall(text[:6000]) + VALUE_RE.findall(text[:6000]), 20)
        refs = []
        for r in REF_RE.findall(text):
            r = re.sub(r"\s+", "", r)
            if r != article.replace(" ", "") and r not in refs:
                refs.append(r)
        refs = refs[:12]
        ck = contract_key(e["contract_scope"])
        out.append({
            "element_id": e["element_id"], "schema_version": VERSION, "schema_tag": e["element_type"],
            "contract_key": ck, "subject_key": subjects, "role": roles,
            "locator": {"article": article, "article_title": article_title, "table_headers": headers,
                        "row_keys": row_keys, "section": section},
            "qualifier": qualifiers, "reference": refs,
            "search_text": " | ".join(x for x in [
                f"[schema] {e['element_type']}", f"[contract] {ck}" if ck else "",
                f"[subject] {' | '.join(subjects)}" if subjects else "",
                f"[role] {' | '.join(roles)} | {' | '.join(ROLE_KO.get(r, r) for r in roles)}" if roles else "",
                f"[article] {article} {article_title}".strip() if article else "",
                f"[section] {section}" if section else "",
                f"[qualifier] {' | '.join(qualifiers)}" if qualifiers else "",
                f"[reference] {' | '.join(refs)}" if refs else ""] if x),
        })
        prev_texts.append(text)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out)
    cov = {k: sum(1 for r in out if r[k]) / n for k in ("contract_key", "subject_key", "role", "qualifier", "reference")}
    cov["article"] = sum(1 for r in out if r["locator"]["article"]) / n
    cov["table"] = sum(1 for r in out if r["locator"]["table_headers"] or r["locator"]["row_keys"]) / n
    stats = {"version": VERSION, "n": n, "coverage": {k: round(v, 3) for k, v in cov.items()},
             "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(stats, open(a.out.replace(".jsonl", "_stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
