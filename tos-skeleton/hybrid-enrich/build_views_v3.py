#!/usr/bin/env python3
"""Create v3 vector and lexical views without depending on Docurator."""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"


def load(name, key):
    return {row[key]: row for row in (json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines())}


def write_jsonl(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    chunks = load("chunks.jsonl", "chunk_id")
    metas = load("chunk_metadata_v3.jsonl", "chunk_id")
    chunk_only = os.environ.get("CHUNK_METADATA_ONLY") == "1"
    elements = {} if chunk_only else load("elements.jsonl", "element_id")
    tags = {} if chunk_only else load("element_tags_v3.jsonl", "element_id")

    write_jsonl("vec_base_texts.jsonl", (
        {"chunk_id": cid, "text": chunk["text"]}
        for cid, chunk in chunks.items()
    ))

    write_jsonl("vec_meta_v3_concat_texts.jsonl", (
        {"chunk_id": cid, "text": metas[cid]["embedding_text"] + "\n[원문]\n" + chunk["text"]}
        for cid, chunk in chunks.items()
    ))
    write_jsonl("vec_meta_v3_only_texts.jsonl", (
        {"chunk_id": cid, "text": metas[cid]["embedding_text"]}
        for cid in chunks
    ))
    if chunk_only:
        print(json.dumps({"chunks": len(chunks), "elements": "unchanged"}, ensure_ascii=False))
        return

    write_jsonl("lex_tag_v3.jsonl", (
        {"element_id": eid, "fields": {
            "schema": tag["schema_tag"], "contract": tag["contract_key_tag"],
            "subject": " ".join(tag["subject_key_tag"]),
            "role": " ".join(tag["role_tag"]), "locator": " ".join(tag["locator_tag"]),
            "qualifier": " ".join(tag["qualifier_tag"]),
            "reference": " ".join(tag["reference_tag"]),
            "tag": tag["search_text"], "raw": elements[eid]["text"],
        }} for eid, tag in tags.items()
    ))
    # Same one-line JSONL contract consumed by yesterday's rg implementation.
    write_jsonl("grep_tag_v3.jsonl", (
        {"element_id": eid, "g": tag["search_text"] + " ||| " + elements[eid]["text"].replace("\n", " ")}
        for eid, tag in tags.items()
    ))
    print(json.dumps({"chunks": len(chunks), "elements": len(elements)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
