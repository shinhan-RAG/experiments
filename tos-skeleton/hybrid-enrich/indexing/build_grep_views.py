#!/usr/bin/env python3
"""grep 뷰 재생성: 엘리먼트 단위. grep_base(원문) / grep_tag(태그+원문)."""
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")

def serialize_tag(t):
    parts = [f"[유형] {t['element_type']}"]
    if t["contract_scope"]:
        parts.append(f"[특약범위] {t['contract_scope']}")
    if t["article_label"]:
        parts.append("[조항] " + ", ".join(t["article_label"]))
    if t.get("table_title"):
        parts.append(f"[표제목] {t['table_title']}")
    if t.get("column_headers"):
        parts.append("[표헤더] " + " | ".join(t["column_headers"]))
    if t.get("row_labels"):
        parts.append("[행라벨] " + " | ".join(t["row_labels"]))
    if t["structures"]:
        parts.append("[구조] " + ", ".join(t["structures"]))
    if t["aliases"]:
        parts.append("[관련어] " + ", ".join(t["aliases"]))
    return " ".join(parts)

def main():
    elems = [json.loads(l) for l in open(os.path.join(OUT, "elements.jsonl"))]
    tags = {t["element_id"]: t for t in (json.loads(l) for l in open(os.path.join(OUT, "element_tags.jsonl")))}
    with open(os.path.join(OUT, "grep_base.jsonl"), "w", encoding="utf-8") as gb, \
         open(os.path.join(OUT, "grep_tag.jsonl"), "w", encoding="utf-8") as gt:
        for e in elems:
            flat = e["text"].replace("\n", " ")
            gb.write(json.dumps({"element_id": e["element_id"], "g": flat}, ensure_ascii=False) + "\n")
            gt.write(json.dumps({"element_id": e["element_id"],
                                 "g": serialize_tag(tags[e["element_id"]]) + " ||| " + flat}, ensure_ascii=False) + "\n")
    print("grep views:", len(elems))

if __name__ == "__main__":
    main()
