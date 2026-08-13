#!/usr/bin/env python3
"""v2 뷰: vec_meta_v2_texts.jsonl(벡터), grep_tag_v2.jsonl(grep). grep_base/vec_base는 불변."""
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")

def ser_meta(m):
    p = []
    if m["topic"]:
        p.append(f"[특약] {m['topic']}")
    if m["article"]:
        p.append(f"[조항] {m['article']}")
    if m["semantic_role"]:
        p.append(f"[역할] {m['semantic_role']}")
    if m["values"]:
        p.append("[값] " + ", ".join(m["values"]))
    if m["aliases"]:
        p.append("[관련어] " + ", ".join(m["aliases"]))
    return "\n".join(p)

def ser_tag(t):
    if t["is_toc"]:
        return f"[유형] 목차"
    p = [f"[유형] {t['element_type']}"]
    if t["topic"]:
        p.append(f"[특약] {t['topic']}")
    if t["article"]:
        p.append(f"[조항] {t['article']}")
    if t["semantic_role"]:
        p.append(f"[역할] {t['semantic_role']}")
    if t.get("table_title"):
        p.append(f"[표제목] {t['table_title']}")
    if t.get("table_headers"):
        p.append("[표헤더] " + " | ".join(t["table_headers"]))
    if t.get("formula_subject"):
        p.append(f"[산식대상] {t['formula_subject']}")
    if t["values"]:
        p.append("[값] " + ", ".join(t["values"]))
    if t["conditions"]:
        p.append("[조건] " + " / ".join(t["conditions"]))
    if t["aliases"]:
        p.append("[관련어] " + ", ".join(t["aliases"]))
    return " ".join(p)

def main():
    chunks = {c["chunk_id"]: c for c in (json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl")))}
    metas = [json.loads(l) for l in open(os.path.join(OUT, "chunk_metadata_v2.jsonl"))]
    with open(os.path.join(OUT, "vec_meta_v2_texts.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps({"chunk_id": m["chunk_id"],
                                "text": ser_meta(m) + "\n" + chunks[m["chunk_id"]]["text"]}, ensure_ascii=False) + "\n")
    elements_input = os.environ.get("ELEMENTS_INPUT", os.path.join(OUT, "elements.jsonl"))
    tags_input = os.environ.get("ELEMENT_TAGS_V2_INPUT", os.path.join(OUT, "element_tags_v2.jsonl"))
    grep_output = os.environ.get("GREP_TAG_V2_OUTPUT", os.path.join(OUT, "grep_tag_v2.jsonl"))
    elems = {e["element_id"]: e for e in (json.loads(l) for l in open(elements_input))}
    tags = [json.loads(l) for l in open(tags_input)]
    with open(grep_output, "w", encoding="utf-8") as f:
        for t in tags:
            flat = elems[t["element_id"]]["text"].replace("\n", " ")
            f.write(json.dumps({"element_id": t["element_id"],
                                "g": ser_tag(t) + " ||| " + flat}, ensure_ascii=False) + "\n")
    print("v2 views ok")

if __name__ == "__main__":
    main()
