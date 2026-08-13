#!/usr/bin/env python3
"""Quality audit for chunk/vector metadata and element/semantic fields."""
from __future__ import annotations

import collections
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
DOC = Path("/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
ARTICLE_RE = re.compile(r"(?:^|\n)#{0,6}\s*제\s?\d+(?:-\d+)?조(?:의\s?\d+)?", re.M)
CONTRACT_RE = re.compile(r"(?:^|\n)[^\n]{2,100}특약(?:\([^\n]{0,100}\))?\s*(?:\n|$)")
BAD_SUBJECT_RE = re.compile(r"제\d+.*(?:조|관)|부표|별표|한국표준질병|보험금 지급|간편심사형|직접적인 치료|제\s?\d+항|경우|동안|에도 불구하고")


def load(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def pct(value, total):
    return round(value / total * 100, 2) if total else 0.0


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    def q(p):
        return values[min(len(values) - 1, round((len(values) - 1) * p))]
    return {"min": values[0], "p50": q(.5), "p90": q(.9), "p95": q(.95), "p99": q(.99), "max": values[-1], "mean": round(statistics.mean(values), 2)}


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text))).lower()


def embedding_audit(stem, expected_ids):
    matrix = np.load(OUT / f"{stem}.npy", mmap_mode="r")
    info = json.loads((OUT / f"{stem}_ids.json").read_text())
    ids = info["ids"]
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "shape": list(matrix.shape), "ids": len(ids), "id_alignment": ids == expected_ids,
        "zero_vectors": int(np.sum(norms == 0)), "nan_values": int(np.isnan(matrix).sum()),
        "norm_min": round(float(norms.min()), 6), "norm_max": round(float(norms.max()), 6),
        "norm_mean": round(float(norms.mean()), 6), "truncated": info.get("truncated"),
        "model": info.get("model"), "max_chars": info.get("max_chars"),
    }


def main():
    raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    chunks = load("chunks.jsonl")
    metadata = load("chunk_metadata_v3.jsonl")
    meta_by = {row["chunk_id"]: row for row in metadata}
    elements = load("elements.jsonl")
    tags = load("element_semantic_fields_v4.jsonl")
    tag_by = {row["element_id"]: row for row in tags}
    gold = load("train350_gold.jsonl")
    gold_chunks = {uid for row in gold for uid in row["gold_chunk_ids"]}
    gold_elements = {uid for row in gold for uid in row["gold_element_ids"]}

    # Chunk boundary and metadata audit.
    chunk_lengths = [len(row["text"]) for row in chunks]
    chunk_mixed_contract_headers = [row["chunk_id"] for row in chunks if len(CONTRACT_RE.findall(row["text"])) > 1]
    chunk_many_articles = [row["chunk_id"] for row in chunks if len(ARTICLE_RE.findall(row["text"])) > 2]
    chunk_span_mismatch = [row["chunk_id"] for row in chunks if not (0 <= row["char_start"] < row["char_end"] <= len(raw))]
    chunk_text_mismatch = []
    for row in chunks:
        # Overlap prefix means exact equality is not expected; require the non-overlap tail to occur in the source span vicinity.
        source = raw[max(0, row["char_start"] - 5):min(len(raw), row["char_end"] + 5)]
        probe = row["text"][-min(180, len(row["text"])):]
        if probe and probe not in source:
            chunk_text_mismatch.append(row["chunk_id"])

    meta_fields = ["retrieval_summary", "scope_context", "topic_context", "canonical_entities", "event_relations", "condition_context", "exception_context", "temporal_numeric_context", "reference_context", "role_context", "embedding_text"]
    meta_coverage = {}
    meta_gold_coverage = {}
    for field in meta_fields:
        meta_coverage[field] = {"n": sum(bool(row.get(field)) for row in metadata), "pct": pct(sum(bool(row.get(field)) for row in metadata), len(metadata))}
        selected = [meta_by[uid] for uid in gold_chunks if uid in meta_by]
        meta_gold_coverage[field] = {"n": sum(bool(row.get(field)) for row in selected), "pct": pct(sum(bool(row.get(field)) for row in selected), len(selected))}

    unsupported = collections.Counter()
    checked = collections.Counter()
    unsupported_examples = []
    for chunk in chunks:
        meta = meta_by[chunk["chunk_id"]]
        vicinity = norm(raw[max(0, chunk["char_start"] - 1800):min(len(raw), chunk["char_end"] + 1800)] + " " + chunk.get("contract_scope", "") + " " + " ".join(chunk.get("section_path", [])))
        for field in ("canonical_entities", "condition_context", "exception_context", "temporal_numeric_context", "reference_context"):
            for value in meta.get(field) or []:
                checked[field] += 1
                compact = norm(value)
                # Very short numerics are non-discriminative but still evidence-backed if local.
                if compact and compact not in vicinity:
                    unsupported[field] += 1
                    if len(unsupported_examples) < 30:
                        unsupported_examples.append({"chunk_id": chunk["chunk_id"], "field": field, "value": value})
    support = {field: {"checked": checked[field], "unsupported": unsupported[field], "unsupported_pct": pct(unsupported[field], checked[field])} for field in checked}
    metadata_exact_duplicates = collections.Counter(row["embedding_text"] for row in metadata if row.get("embedding_text"))
    metadata_duplicate_groups = sum(value > 1 for value in metadata_exact_duplicates.values())

    vector = {
        stem: embedding_audit(stem, [row["chunk_id"] for row in chunks])
        for stem in ("vec_base", "vec_meta_v2", "vec_meta_v3_concat")
    }

    # Element boundary/type audit.
    element_lengths = [len(row["text"]) for row in elements]
    line_spans = [row["line_end"] - row["line_start"] + 1 for row in elements]
    invalid_spans = [row["element_id"] for row in elements if not (0 <= row["char_start"] < row["char_end"] <= len(raw))]
    text_mismatch = [row["element_id"] for row in elements if raw[row["char_start"]:row["char_end"]] != row["text"]]
    giant = [row for row in elements if len(row["text"]) > 4000 or row["line_end"] - row["line_start"] + 1 > 80]
    mixed_articles = [row for row in elements if len(ARTICLE_RE.findall(row["text"])) >= 3]
    mixed_contracts = [row for row in elements if len(CONTRACT_RE.findall(row["text"])) >= 2]
    suspicious_formula = [row for row in elements if row["element_type"] == "formula" and (len(row["text"]) > 4000 or len(ARTICLE_RE.findall(row["text"])) >= 2 or row["text"].count("|") > 30)]
    giant_gold = sorted({row["element_id"] for row in giant if row["element_id"] in gold_elements})
    suspicious_formula_gold = sorted({row["element_id"] for row in suspicious_formula if row["element_id"] in gold_elements})

    # Tag field coverage, evidence support, collision and repetition.
    gold_tag_rows = [tag_by[uid] for uid in gold_elements if uid in tag_by]
    field_names = ["contract_key", "subject_key", "role", "qualifier", "reference"]
    tag_coverage = {field: {"n": sum(bool(row[field]) for row in tags), "pct": pct(sum(bool(row[field]) for row in tags), len(tags))} for field in field_names}
    tag_gold_coverage = {field: {"n": sum(bool(row[field]) for row in gold_tag_rows), "pct": pct(sum(bool(row[field]) for row in gold_tag_rows), len(gold_tag_rows))} for field in field_names}

    def address(row):
        locator = row["locator"]
        return (row["schema_tag"], row["contract_key"], tuple(row["subject_key"]), tuple(row["role"]), locator.get("article", ""), tuple(locator.get("table_headers", [])), tuple(locator.get("row_keys", [])), locator.get("section", ""), tuple(row["qualifier"]))
    address_counts = collections.Counter(address(row) for row in tags if row["contract_key"] and (row["subject_key"] or row["role"]))
    gold_address_counts = collections.Counter(address(row) for row in gold_tag_rows if row["contract_key"] and (row["subject_key"] or row["role"]))
    repetitions = {}
    for field in ("subject_key", "role", "qualifier", "reference"):
        values = collections.Counter(value for row in tags for value in row[field])
        repetitions[field] = {"distinct": len(values), "top": values.most_common(15)}
    bad_subjects = [{"element_id": row["element_id"], "value": value} for row in tags for value in row["subject_key"] if BAD_SUBJECT_RE.search(value)]
    tag_support_checked = 0
    tag_support_unsupported = []
    element_by = {row["element_id"]: row for row in elements}
    for row in tags:
        element = element_by[row["element_id"]]
        vicinity = norm(raw[max(0, element["char_start"] - 1200):min(len(raw), element["char_end"] + 1200)] + " " + element.get("contract_scope", ""))
        for field in ("subject_key", "qualifier", "reference"):
            for value in row[field]:
                tag_support_checked += 1
                if norm(value) and norm(value) not in vicinity:
                    if len(tag_support_unsupported) < 50:
                        tag_support_unsupported.append({"element_id": row["element_id"], "field": field, "value": value})

    result = {
        "scope": {"document_count": 1, "train_questions": len(gold), "chunks": len(chunks), "elements": len(elements), "gold_chunks": len(gold_chunks), "gold_elements": len(gold_elements)},
        "chunk": {
            "lengths": quantiles(chunk_lengths), "invalid_spans": len(chunk_span_mismatch), "source_tail_mismatches": len(chunk_text_mismatch),
            "multiple_contract_headers": len(chunk_mixed_contract_headers), "more_than_two_article_headers": len(chunk_many_articles),
            "examples": {"multiple_contract_headers": chunk_mixed_contract_headers[:20], "many_articles": chunk_many_articles[:20], "source_mismatch": chunk_text_mismatch[:20]},
        },
        "metadata": {
            "coverage_all": meta_coverage, "coverage_train_gold": meta_gold_coverage, "local_evidence_support": support,
            "unsupported_examples": unsupported_examples, "embedding_text_duplicate_groups": metadata_duplicate_groups,
        },
        "vector_indices": vector,
        "element": {
            "lengths": quantiles(element_lengths), "line_spans": quantiles(line_spans), "invalid_spans": len(invalid_spans), "source_text_mismatches": len(text_mismatch),
            "giant_or_80plus_lines": len(giant), "three_plus_article_headers": len(mixed_articles), "multiple_contract_headers": len(mixed_contracts),
            "suspicious_formula": len(suspicious_formula), "gold_in_giant": len(giant_gold), "gold_in_suspicious_formula": len(suspicious_formula_gold),
            "examples": {"giant": [{"id": row["element_id"], "type": row["element_type"], "chars": len(row["text"]), "lines": row["line_end"]-row["line_start"]+1} for row in giant[:30]], "mixed_articles": [row["element_id"] for row in mixed_articles[:30]], "suspicious_formula_gold": suspicious_formula_gold[:30]},
        },
        "semantic_fields": {
            "coverage_all": tag_coverage, "coverage_train_gold": tag_gold_coverage,
            "address_unique_pct_all": pct(sum(value == 1 for value in address_counts.values()), len(address_counts)),
            "address_collision_groups_all": sum(value > 1 for value in address_counts.values()),
            "address_unique_pct_gold": pct(sum(value == 1 for value in gold_address_counts.values()), len(gold_address_counts)),
            "address_collision_groups_gold": sum(value > 1 for value in gold_address_counts.values()),
            "repetition": repetitions, "obvious_bad_subjects": len(bad_subjects), "bad_subject_examples": bad_subjects[:30],
            "support_checked": tag_support_checked, "unsupported_example_count_capped": len(tag_support_unsupported), "unsupported_examples": tag_support_unsupported,
        },
    }

    # Conservative verdicts: quality gates are about input integrity, not retrieval performance.
    result["verdict"] = {
        "chunk_structure": "pass" if not chunk_span_mismatch and pct(len(chunk_text_mismatch), len(chunks)) < 1 else "conditional",
        "metadata_fields": "conditional" if support.get("reference_context", {}).get("unsupported_pct", 0) <= 10 else "fail",
        "vector_index": "pass" if all(v["id_alignment"] and not v["zero_vectors"] and not v["nan_values"] for v in vector.values()) else "fail",
        "element_structure": "fail" if suspicious_formula_gold or mixed_articles else "conditional",
        "semantic_fields": "blocked_by_element_structure" if suspicious_formula_gold or mixed_articles else "conditional",
    }
    (OUT / "retrieval_input_quality_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Retrieval Input Quality Audit", "", "## 판정", "",
        "| 대상 | 판정 |", "|---|---|",
        *[f"| {key} | **{value}** |" for key, value in result["verdict"].items()], "",
        "## 핵심 수치", "",
        f"- Chunk: {len(chunks):,}개, span 오류 {len(chunk_span_mismatch)}, source tail 불일치 {len(chunk_text_mismatch)}",
        f"- Metadata reference local unsupported: {support.get('reference_context', {}).get('unsupported_pct', 0)}%",
        f"- Vector index: " + ", ".join(f"{key} {value['shape']} zero={value['zero_vectors']} nan={value['nan_values']}" for key, value in vector.items()),
        f"- Element: {len(elements):,}개, giant/80+line {len(giant)}, 3+ article 혼합 {len(mixed_articles)}, suspicious formula {len(suspicious_formula)}",
        f"- Train gold element 중 giant {len(giant_gold)}, suspicious formula {len(suspicious_formula_gold)}",
        f"- Semantic address unique(all/gold): {result['semantic_fields']['address_unique_pct_all']}% / {result['semantic_fields']['address_unique_pct_gold']}%",
        "", "상세 사례와 전체 통계는 `out/retrieval_input_quality_audit.json` 참조.",
    ]
    (OUT / "RETRIEVAL_INPUT_QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "chunk": result["chunk"], "element": result["element"], "vector_indices": vector, "metadata_support": support, "semantic_address": {"all_unique_pct": result["semantic_fields"]["address_unique_pct_all"], "gold_unique_pct": result["semantic_fields"]["address_unique_pct_gold"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
