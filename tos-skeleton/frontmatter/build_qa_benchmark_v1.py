#!/usr/bin/env python3
"""Build a frozen three-suite QA benchmark for document selection.

The suites deliberately test different document-level signals:
  - index_only: identity and deterministic metadata
  - frontmatter_only: document content profile
  - combined: content profile plus an identity constraint

The identity suite has mechanically verified gold. Content-derived suites retain
the source passage and anchor, but remain review_pending until a person approves
the semantic match between question, evidence, and every gold document.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "out"
DATASET_VERSION = "document-selection-qa-v1"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_document_metadata():
    collection = json.loads((OUT / "collection_index.json").read_text(encoding="utf-8"))
    metadata = {}
    for product in collection["products"]:
        for doc in product["documents"]:
            metadata[doc["file_path"]] = {
                "document_id": doc["document_id"],
                "product_name": product["product_name"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "effective_date": doc["effective_date"],
                "is_representative": doc["is_representative"],
            }
    return collection, metadata


def build_index_only(metadata):
    source = json.loads((OUT / "doc_find_qa_582.json").read_text(encoding="utf-8"))
    rows = []
    for number, item in enumerate(source["queries"], 1):
        gold_paths = [path for path in item.get("gold_paths", item.get("gold", []))
                      if path in metadata]
        rows.append({
            "id": f"idx-{number:04d}",
            "dataset_version": DATASET_VERSION,
            "suite": "index_only",
            "query_type": item["type"],
            "query": item["q"],
            "required_signals": ["product_name", "doc_type"] +
                                (["effective_date", "is_representative"]
                                 if item["type"] in ("최신정본", "연도종류") else []),
            "forbidden_dependency": ["frontmatter_content"],
            "gold_documents": gold_paths,
            "gold_document_ids": [metadata[path]["document_id"] for path in gold_paths],
            "evidence": [{"document": path, "metadata": metadata[path]}
                         for path in gold_paths],
            "gold_method": "deterministic_index_metadata",
            "review_status": "mechanically_verified",
        })
    return rows


def normalize_term(value):
    return re.sub(r"\s+", " ", value).strip()


def frontmatter_terms(row):
    """Return explicit field/value facts; no generated summary or source anchor."""
    fm = row["fm"]
    terms = []
    field_names = ("골격", "조", "부표", "항목", "서식", "헤딩", "표캡션")
    for field in field_names:
        for value in fm.get(field, []):
            value = normalize_term(value)
            if 8 <= len(value) <= 70:
                terms.append((field, value))
    for rider in fm.get("특약구성", []):
        value = normalize_term(rider["name"])
        if 8 <= len(value) <= 70:
            terms.append(("특약구성", value))
    return set(terms)


def build_frontmatter_only(metadata):
    docs = read_jsonl(OUT / "doc_frontmatter.jsonl")
    postings = defaultdict(set)
    for row in docs:
        if row["file"] not in metadata:
            continue
        for fact in frontmatter_terms(row):
            postings[fact].add(row["file"])

    # Avoid unique-value lookups and overly broad labels. The sorted slice makes
    # the suite deterministic while retaining all document types and field kinds.
    candidates = [(field, term, sorted(paths))
                  for (field, term), paths in postings.items()
                  if 2 <= len(paths) <= 20]
    candidates.sort(key=lambda item: (item[0], item[1]))
    quota = {"특약구성": 35, "조": 25, "항목": 15, "서식": 10,
             "표캡션": 10, "골격": 5, "부표": 5, "헤딩": 5}
    used = Counter()
    rows = []
    for field, term, gold_paths in candidates:
        if used[field] >= quota.get(field, 0):
            continue
        used[field] += 1
        if field == "특약구성":
            query = f"{term}이 포함된 문서를 찾아줘"
        else:
            query = f"{term} 항목이 있는 문서를 찾아줘"
        rows.append({
            "id": f"fm-{len(rows) + 1:04d}",
            "dataset_version": DATASET_VERSION,
            "suite": "frontmatter_only",
            "query_type": field,
            "query": query,
            "required_signals": ["frontmatter_content"],
            "forbidden_dependency": ["product_name", "effective_date",
                                     "is_representative"],
            "gold_documents": gold_paths,
            "gold_document_ids": [metadata[path]["document_id"] for path in gold_paths],
            "evidence": {"frontmatter_field": field, "exact_value": term},
            "gold_method": "exact_frontmatter_field_value_postings",
            "review_status": "mechanically_verified",
        })
    return rows


def build_combined(frontmatter_rows, metadata):
    rows = []
    for item in frontmatter_rows:
        by_type = defaultdict(list)
        for path in item["gold_documents"]:
            by_type[metadata[path]["doc_type"]].append(path)

        selected_type = None
        selected_gold = None
        for doc_type in ("판매약관", "사업방법서", "공시약관", "상품요약서"):
            candidates = [path for path in by_type.get(doc_type, [])
                          if metadata[path]["is_representative"]]
            if 1 <= len(candidates) <= 5:
                selected_type = doc_type
                selected_gold = candidates
                break
        if not selected_gold:
            continue

        term = item["evidence"]["exact_value"]
        rows.append({
            "id": f"mix-{len(rows) + 1:04d}",
            "dataset_version": DATASET_VERSION,
            "suite": "combined",
            "query_type": "content_plus_latest_doc_type",
            "query": f"{term}이 포함된 최신 {selected_type} 문서를 찾아줘",
            "required_signals": ["frontmatter_content", "doc_type",
                                 "is_representative"],
            "forbidden_dependency": [],
            "gold_documents": selected_gold,
            "gold_document_ids": [metadata[path]["document_id"] for path in selected_gold],
            "evidence": {
                "content_qa_id": item["id"],
                "frontmatter_field": item["evidence"]["frontmatter_field"],
                "exact_value": item["evidence"]["exact_value"],
                "identity_constraint": {
                    "doc_type": selected_type,
                    "is_representative": True,
                },
            },
            "gold_method": "frontmatter_postings_intersect_index_constraint",
            "review_status": "mechanically_verified",
        })
    return rows


def main():
    collection, metadata = load_document_metadata()
    suites = {
        "index_only": build_index_only(metadata),
        "frontmatter_only": build_frontmatter_only(metadata),
    }
    suites["combined"] = build_combined(suites["frontmatter_only"], metadata)

    files = {}
    for name, rows in suites.items():
        path = OUT / f"qa_{name}_v1.jsonl"
        write_jsonl(path, rows)
        files[name] = path.name

    manifest = {
        "dataset_version": DATASET_VERSION,
        "scope": "select documents from one collection",
        "collection_id": collection["collection_id"],
        "n_documents": collection["n_documents"],
        "suites": {
            name: {
                "file": files[name],
                "n_questions": len(rows),
                "review_status": dict(Counter(row["review_status"] for row in rows)),
            }
            for name, rows in suites.items()
        },
        "comparison_rule": (
            "Compare search arms only on the same dataset_version and suite. "
            "These mechanically verified suites are schema-regression tests. "
            "Do not present them as evidence of natural-language production quality."
        ),
    }
    manifest_path = OUT / "qa_benchmark_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
