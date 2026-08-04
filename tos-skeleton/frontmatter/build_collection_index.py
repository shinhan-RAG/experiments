#!/usr/bin/env python3
"""Build the canonical collection_index.json used for document selection."""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collection_fm_v2 import FLAGS, KINDS, doc_type, parse_date_v2

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path(__file__).parent / "out" / "collection_index.json"


def nfc(value):
    return unicodedata.normalize("NFC", value)


def stable_id(prefix, value):
    digest = hashlib.sha256(nfc(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalized_title(filename):
    title = Path(filename).stem
    title = re.sub(r"(?<!\d)(?:19|20)\d{6}(?!\d)", " ", title)
    title = re.sub(r"(?<!\d)\d{6}(?!\d)", " ", title)
    title = re.sub(r"(?i)(?:^|[_\s-])v\d+(?:\.\d+)*$", " ", title)
    title = re.sub(r"[_]+", " ", title)
    return re.sub(r"\s+", " ", title).strip(" -")


def iso_date(parsed):
    if not parsed:
        return None
    value = parsed["date"]
    if parsed["granularity"] == "month":
        return f"{value[:4]}-{value[4:6]}"
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def main():
    products = []
    document_count = 0
    for product_dir in sorted(ROOT.iterdir()):
        if not product_dir.is_dir():
            continue
        product_name = nfc(product_dir.name)
        product_id = stable_id("prd", product_name)
        flags = [name for name, patterns in FLAGS.items()
                 if any(pattern in product_name for pattern in patterns)]
        kind_hints = [name for name in KINDS if name in product_name]

        raw_docs = []
        for path in sorted(product_dir.rglob("*.md")):
            relative = nfc(str(path.relative_to(ROOT)))
            filename = nfc(path.name)
            parsed = parse_date_v2(filename)
            raw_docs.append({
                "document_id": stable_id("doc", relative),
                "title": normalized_title(filename),
                "file_name": filename,
                "file_path": relative,
                "doc_type": doc_type(filename),
                "version": parsed["raw"] if parsed else None,
                "effective_date": iso_date(parsed),
                "date_granularity": parsed["granularity"] if parsed else None,
                "date_confidence": parsed["confidence"] if parsed else None,
                "is_representative": False,
            })

        for dtype in ("판매약관", "사업방법서", "공시약관", "상품요약서"):
            candidates = [doc for doc in raw_docs if doc["doc_type"] == dtype
                          and doc["effective_date"] and doc["date_confidence"] == "high"]
            if candidates:
                latest = max(doc["effective_date"] for doc in candidates)
                for doc in candidates:
                    if doc["effective_date"] == latest:
                        doc["is_representative"] = True

        products.append({
            "product_id": product_id,
            "product_name": product_name,
            "flags": flags,
            "kind_hints": kind_hints,
            "documents": raw_docs,
        })
        document_count += len(raw_docs)

    output = {
        "schema_version": "1.0",
        "collection_id": "shinhan-product-documents",
        "collection_name": "신한 상품문서 컬렉션",
        "n_products": len(products),
        "n_documents": document_count,
        "products": products,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"products={len(products):,} documents={document_count:,} bytes={OUT.stat().st_size:,}")


if __name__ == "__main__":
    main()
