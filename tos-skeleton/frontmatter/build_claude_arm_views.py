#!/usr/bin/env python3
"""Arm별 artifact view 생성 — 물리적 격리 (핸드오프 명세).

views/A: files.jsonl                           (파일명·경로·document_id)
views/B: files.jsonl + index.jsonl             (Index 필드 전체, FM 없음)
views/C: files.jsonl + frontmatter.jsonl       (FM 전체, Index 없음)
views/D: files.jsonl + index.jsonl + fm/<document_id>.json (배치 단위 FM 읽기용)
"""
import json
import shutil
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
VIEWS = HERE / "views"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def main():
    ci = json.loads((OUT / "collection_index.json").read_text())
    index_rows = []
    path2id = {}
    for p in ci["products"]:
        for dd in p["documents"]:
            row = {"document_id": dd["document_id"],
                   "product_name": p["product_name"],
                   "title": dd.get("title", ""),
                   "file_name": dd["file_name"],
                   "doc_type": dd["doc_type"],
                   "effective_date": dd.get("effective_date"),
                   "version": dd.get("version"),
                   "is_representative": bool(dd.get("is_representative")),
                   "flags": p["flags"], "kind_hints": p["kind_hints"]}
            index_rows.append(row)
            path2id[nfc(p["product_name"] + "/" + dd["file_name"])] = dd["document_id"]

    fm_rows = []
    for l in open(OUT / "doc_frontmatter.jsonl"):
        d = json.loads(l)
        key = nfc(d["product"] + "/" + Path(d["file"]).name)
        did = path2id.get(key)
        if not did:
            continue
        fm_rows.append({"document_id": did, "product_name": d["product"],
                        "file_name": Path(d["file"]).name, "ftype": d["ftype"],
                        "fm": d["fm"]})

    files_rows = [{"document_id": r["document_id"], "product_name": r["product_name"],
                   "file_name": r["file_name"]} for r in index_rows]

    if VIEWS.exists():
        shutil.rmtree(VIEWS)
    for arm in "ABCD":
        (VIEWS / arm).mkdir(parents=True)
        with open(VIEWS / arm / "files.jsonl", "w") as f:
            for r in files_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for arm in ("B", "D"):
        with open(VIEWS / arm / "index.jsonl", "w") as f:
            for r in index_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(VIEWS / "C" / "frontmatter.jsonl", "w") as f:
        for r in fm_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fmdir = VIEWS / "D" / "fm"
    fmdir.mkdir()
    for r in fm_rows:
        (fmdir / f"{r['document_id']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")

    print(f"views 생성: index {len(index_rows):,} | fm {len(fm_rows):,} | "
          f"D/fm 파일 {len(list(fmdir.iterdir())):,}")


if __name__ == "__main__":
    main()
