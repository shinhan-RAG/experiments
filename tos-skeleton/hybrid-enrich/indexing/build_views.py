#!/usr/bin/env python3
"""Arm별 뷰 생성.
- 벡터 색인 텍스트 2종: vec_base (BASE, TAG), vec_meta (META, BOTH)
- grep 뷰 2종: grep_base.jsonl (BASE, META), grep_tag.jsonl (TAG, BOTH)
이후 embed.py가 vec_*를 임베딩한다."""
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")

def serialize_meta(m):
    parts = []
    if m["contract_scope"]:
        parts.append(f"[특약범위] {m['contract_scope']}")
    if m["section_path"]:
        parts.append("[목차] " + " > ".join(m["section_path"]))
    if m["articles"]:
        parts.append("[조항] " + ", ".join(m["articles"]))
    if m["aliases"]:
        parts.append("[관련어] " + ", ".join(m["aliases"]))
    return "\n".join(parts)

def serialize_tag(t):
    parts = [f"[유형] {t['element_type']}"]
    if t["contract_scope"]:
        parts.append(f"[특약범위] {t['contract_scope']}")
    if t["section_path"]:
        parts.append("[목차] " + " > ".join(t["section_path"]))
    if t["article_label"]:
        parts.append("[조항] " + ", ".join(t["article_label"]))
    if t["table_headers"]:
        parts.append("[표헤더] " + " | ".join(t["table_headers"]))
    if t["structures"]:
        parts.append("[구조] " + ", ".join(t["structures"]))
    if t["aliases"]:
        parts.append("[관련어] " + ", ".join(t["aliases"]))
    return "\n".join(parts)

def main():
    chunks = {c["chunk_id"]: c for c in (json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl")))}
    metas = {m["chunk_id"]: m for m in (json.loads(l) for l in open(os.path.join(OUT, "chunk_metadata.jsonl")))}
    tags = {t["chunk_id"]: t for t in (json.loads(l) for l in open(os.path.join(OUT, "semantic_tags.jsonl")))}

    with open(os.path.join(OUT, "vec_base_texts.jsonl"), "w", encoding="utf-8") as fb, \
         open(os.path.join(OUT, "vec_meta_texts.jsonl"), "w", encoding="utf-8") as fm, \
         open(os.path.join(OUT, "grep_base.jsonl"), "w", encoding="utf-8") as gb, \
         open(os.path.join(OUT, "grep_tag.jsonl"), "w", encoding="utf-8") as gt:
        for cid, c in chunks.items():
            fb.write(json.dumps({"chunk_id": cid, "text": c["text"]}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"chunk_id": cid, "text": serialize_meta(metas[cid]) + "\n" + c["text"]}, ensure_ascii=False) + "\n")
            flat = c["text"].replace("\n", " ")
            gb.write(json.dumps({"chunk_id": cid, "g": flat}, ensure_ascii=False) + "\n")
            gt.write(json.dumps({"chunk_id": cid, "g": serialize_tag(tags[cid]).replace("\n", " ") + " ||| " + flat}, ensure_ascii=False) + "\n")
    print("views built:", len(chunks))

if __name__ == "__main__":
    main()
