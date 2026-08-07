#!/usr/bin/env python3
"""First-25 FileSearch-only A/B: fixed vector, metadata and semantic tags."""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import run_local_agent_v3 as legacy

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
ARMS = {
    "OLD_RET": ("vec_meta_v3_concat", "grep_tag_new_repaired_v1"),
    "NEW_RET": ("vec_meta_v3_concat", "grep_tag_new_repaired_v1"),
}
STOP = {"무엇", "어떻게", "인가요", "있나요", "되나요", "경우", "대한", "관련",
        "해당", "알려", "주세요", "에서는", "으로", "하는", "되는", "정한", "보험"}
FIELD_WEIGHTS = {
    "contract": 4.0,
    "subject": 3.2,
    "locator": 2.6,
    "qualifier": 2.2,
    "role": 1.8,
    "reference": 1.0,
    "schema": 0.4,
    "raw": 0.55,
}
ROLE_KO = {
    "exclusion_exception": "면책 제외 예외 부지급 지급하지 않는 사유",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tokens(text: str):
    output = set()
    for word in re.findall(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?", str(text).lower()):
        if len(word) < 2 or word in STOP:
            continue
        output.add(word)
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 4:
            output.update(word[index:index + 3] for index in range(len(word) - 2))
    return output


def normkey(text):
    return re.sub(r"[^가-힣a-z0-9]", "", str(text).lower())


def leaf_fields(value, prefix=""):
    """Flatten arbitrary JSON while retaining paths; no schema names are assumed."""
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"chunk_id", "element_id", "schema_version", "field_sources", "search_text", "embedding_text"}:
                continue
            out.update(leaf_fields(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        out[prefix] = " ".join(str(item) for item in value if not isinstance(item, (dict, list)))
        for index, child in enumerate(item for item in value if isinstance(item, (dict, list))):
            out.update(leaf_fields(child, f"{prefix}[]"))
    elif value not in (None, ""):
        out[prefix] = str(value)
    return out


def structural_key(query, row, raw=""):
    """Generic structure evidence: phrase, field breadth, and token overlap."""
    qtokens = tokens(query)
    qnorm = normkey(query)
    fields = leaf_fields(row)
    overlaps = [len(qtokens & tokens(value)) for value in fields.values()]
    phrase_fields = sum(bool(qnorm and qnorm in normkey(value)) for value in fields.values())
    matched_fields = sum(value > 0 for value in overlaps)
    return (phrase_fields, matched_fields, max(overlaps, default=0), sum(overlaps),
            len(qtokens & tokens(raw)))


def flatten_locator(value):
    if not isinstance(value, dict):
        return str(value or "")
    parts = []
    for item in value.values():
        parts.extend(item if isinstance(item, list) else [item])
    return " ".join(str(item) for item in parts if item)


class FieldedFileIndex:
    """Lexical/regex FileSearch that understands the semantic-tag schema."""
    def __init__(self, structured, elements, grep_rows):
        self.rows = {row["element_id"]: row for row in structured}
        self.elements = elements
        self.order = [row["element_id"] for row in structured]
        self.grep_rows = grep_rows
        self.fields = {}
        self.postings = collections.defaultdict(lambda: collections.defaultdict(dict))
        for row in structured:
            eid = row["element_id"]
            role_text = " ".join((row.get("role") or []) + [ROLE_KO.get(role, role) for role in row.get("role") or []])
            fields = {
                "contract": row.get("contract_key", ""),
                "subject": " ".join(row.get("subject_key") or []),
                "locator": flatten_locator(row.get("locator") or {}),
                "qualifier": " ".join(row.get("qualifier") or []),
                "role": role_text,
                "reference": " ".join(row.get("reference") or []),
                "schema": row.get("schema_tag", ""),
                "raw": elements.get(eid, ""),
            }
            self.fields[eid] = fields
            for field, value in fields.items():
                for token in tokens(value):
                    self.postings[token][field][eid] = 1
        self.idf = {}
        total = len(structured)
        for token, by_field in self.postings.items():
            docs = set()
            for postings in by_field.values():
                docs.update(postings)
            self.idf[token] = math.log(1 + total / max(1, len(docs)))

    def search(self, question, pattern, top_k=20):
        try:
            regex = re.compile(pattern, re.I)
        except re.error:
            regex = re.compile(re.escape(pattern), re.I)
        regex_ids = {row["element_id"] for row in self.grep_rows if regex.search(row["g"])}
        query = (question + " " + pattern).strip()
        query_tokens = tokens(query)
        candidates = set(regex_ids)
        for token in query_tokens:
            for postings in self.postings.get(token, {}).values():
                candidates.update(postings)

        contract_mentions = re.findall(r"[가-힣A-Za-z0-9·()\[\]\-]{3,120}?특약", query)
        mention_keys = [normkey(value) for value in contract_mentions if normkey(value)]
        ranked = []
        for eid in candidates:
            field_matches = {}
            score = 0.0
            for field, value in self.fields[eid].items():
                matched = query_tokens & tokens(value)
                if not matched:
                    continue
                contribution = sum(self.idf.get(token, 0.0) for token in matched) * FIELD_WEIGHTS[field]
                field_matches[field] = {"tokens": sorted(matched), "score": round(contribution, 4)}
                score += contribution
            # Reward agreement across independent fields, not repetition inside one field.
            score += 2.0 * max(0, len(field_matches) - 1)
            contract = normkey(self.fields[eid]["contract"])
            contract_exact = bool(mention_keys and any(key in contract or contract in key for key in mention_keys))
            if contract_exact:
                score += 12.0
            if eid in regex_ids:
                score += 2.0
            coverage = len(set().union(*(set(item["tokens"]) for item in field_matches.values()))) / max(1, len(query_tokens)) if field_matches else 0.0
            score += 3.0 * coverage
            ranked.append((score, contract_exact, len(field_matches), eid, field_matches, coverage, eid in regex_ids))
        ranked.sort(key=lambda item: (-item[0], -int(item[1]), -item[2], item[3]))
        results = []
        for score, contract_exact, _, eid, matches, coverage, regex_match in ranked[:top_k]:
            results.append({
                "id": eid, "score": round(score, 4),
                "matched_fields": matches, "contract_exact": contract_exact,
                "query_coverage": round(coverage, 4), "regex_match": regex_match,
                "match": re.sub(r"\s+", " ", self.elements.get(eid, ""))[:180],
            })
        return {
            "total_candidates": len(candidates), "regex_total_matches": len(regex_ids),
            "results": results, "ranking": "fielded_idf_regex_v2",
        }


class NewSearch(legacy.Search):
    def __init__(self):
        super().__init__()
        self.structured = load_jsonl(OUT / "element_tags_new_repaired_full_v1.jsonl")
        self.chunk_meta = {
            row["chunk_id"]: row for row in load_jsonl(OUT / "chunk_metadata_v3.jsonl")
        }
        self.element_text = {
            row["element_id"]: row["text"]
            for row in load_jsonl(OUT / "elements_repaired_v1.jsonl")
        }
        self.elements = self.element_text
        self.fielded_file = FieldedFileIndex(
            self.structured, self.element_text, self.grep["grep_tag_new_repaired_v1"]
        )

    def vector(self, stem, query):
        # A/B invariant: both arms return the exact same bge-m3 top-10.
        return legacy.Search.vector(self, stem, query)

    def file_structured(self, stem, question, pattern):
        return self.fielded_file.search(question, pattern)

    def file(self, stem, pattern):
        return self.file_structured(stem, pattern, pattern)


class BoundNewSearch:
    """Bind the full user question so file-search does not lose query structure."""
    def __init__(self, search, question):
        self.search = search
        self.question = question

    def vector(self, stem, query):
        return self.search.vector(stem, query)

    def file(self, stem, pattern):
        return self.search.file_structured(stem, self.question, pattern)

    def read(self, uid):
        return self.search.read(uid)


def ensure_batch(gold):
    path = OUT / "retriever_ab25_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        assert len(manifest["qids"]) == 25
        return manifest
    qids = [row["qid"] for row in gold[:25]]
    manifest = {
        "version": "filesearch-ab25-v2",
        "source": "train350_gold_repaired_v1.jsonl",
        "selection": "first 25 in frozen source order",
        "qids": qids,
        "arms": ARMS,
        "new_retriever": "filesearch_fielded_idf_regex_v2",
        "vector_invariant": "identical bge-m3 top10 in both arms",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    gold_all = load_jsonl(OUT / "train350_gold_repaired_v1.jsonl")
    manifest = ensure_batch(gold_all)
    by_qid = {row["qid"]: row for row in gold_all}
    selected = [by_qid[qid] for qid in manifest["qids"]]
    if args.limit:
        selected = selected[:args.limit]

    legacy.ARMS = ARMS
    old_search = legacy.Search()
    old_search.elements = {
        row["element_id"]: row["text"]
        for row in load_jsonl(OUT / "elements_repaired_v1.jsonl")
    }
    new_search = NewSearch()
    output_dir = OUT / "retriever_ab25_sessions"
    output_dir.mkdir(exist_ok=True)
    jobs = [(row, arm) for row in selected for arm in ARMS]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(legacy.run_one, old_search if arm == "OLD_RET" else BoundNewSearch(new_search, row["question"]),
                        row["qid"], row["question"], arm, output_dir)
            for row, arm in jobs
        ]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 5 == 0 or index == len(futures):
                errors = sum(row["status"] == "error" for row in results)
                print(f"{index}/{len(futures)} errors={errors}", flush=True)
    with (OUT / "retrieval_retriever_ab25.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: (item["qid"], item["arm"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
