"""Deterministic EDA for BEIR-style corpus, query, and qrels files."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
KOREAN_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")
EDA_VERSION = 1

DATASET_SOURCE_IDS = {
    "trec-covid": {
        "corpus_queries": "BeIR/trec-covid",
        "qrels": "BeIR/trec-covid-qrels",
    },
    "fiqa": {
        "corpus_queries": "BeIR/fiqa",
        "qrels": "BeIR/fiqa-qrels",
    },
    "ko-strategyqa": {
        "preferred": "taeminlee/Ko-StrategyQA",
        "alternate_candidates_in_code": [
            "KETI-AIR/ko-strategy-qa",
            "wics/strategy-qa",
        ],
    },
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": round(float(array.min()), 4),
        "mean": round(float(array.mean()), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "max": round(float(array.max()), 4),
    }


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _query_text(row: dict) -> str:
    return row.get("title") or row.get("text", "")


def analyze_dataset(dataset: str, raw_dir: Path) -> dict:
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels.jsonl"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(path)

    queries = {str(row["_id"]): _query_text(row) for row in read_jsonl(queries_path)}
    query_gold = defaultdict(set)
    qrels_total = 0
    positive_qrels = 0
    relevance_scores = Counter()
    for row in read_jsonl(qrels_path):
        qrels_total += 1
        score = float(row.get("score", 0))
        relevance_scores[str(row.get("score", 0))] += 1
        if score >= 1:
            positive_qrels += 1
            query_gold[str(row["query-id"])].add(str(row["corpus-id"]))

    query_token_sets = {qid: set(tokenize(text)) for qid, text in queries.items()}
    all_query_terms = set().union(*query_token_sets.values()) if query_token_sets else set()
    gold_ids = set().union(*query_gold.values()) if query_gold else set()

    doc_chars = []
    doc_tokens = []
    paragraph_counts = []
    title_present = 0
    empty_docs = 0
    korean_docs = 0
    latin_docs = 0
    corpus_ids = set()
    gold_space_ids = set()  # qrels가 가리키는 단위(청크형이면 parent, BEIR면 doc)
    gold_token_sets = {}
    gold_normalized_texts = {}
    query_term_df = Counter()

    for row in read_jsonl(corpus_path):
        doc_id = str(row["_id"])
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        combined = f"{title} {text}".strip()
        tokens = tokenize(combined)
        token_set = set(tokens)
        corpus_ids.add(doc_id)
        doc_chars.append(len(combined))
        doc_tokens.append(len(tokens))
        paragraph_counts.append(len([p for p in re.split(r"\n\s*\n", text) if p.strip()]))
        title_present += bool(title.strip())
        empty_docs += not bool(combined)
        korean_docs += bool(KOREAN_RE.search(combined))
        latin_docs += bool(LATIN_RE.search(combined))
        for term in token_set & all_query_terms:
            query_term_df[term] += 1
        # 청크형 corpus(longdoc·ruling-anon·aihub)는 qrels가 parent를 가리키므로
        # gold는 parent 단위로 청크를 합쳐서 본다. BEIR corpus면 parent_id가 없어
        # doc_id 그대로이며 기존 동작과 같다.
        gold_key = str(row.get("parent_id") or doc_id)
        gold_space_ids.add(gold_key)
        if gold_key in gold_ids:
            if gold_key in gold_token_sets:
                gold_token_sets[gold_key] |= token_set
                gold_normalized_texts[gold_key] += " " + " ".join(tokens)
            else:
                gold_token_sets[gold_key] = set(token_set)
                gold_normalized_texts[gold_key] = " ".join(tokens)

    query_chars = [len(text) for text in queries.values()]
    query_tokens = [len(tokenize(text)) for text in queries.values()]
    evaluated_query_ids = [qid for qid in queries if query_gold.get(qid)]
    gold_counts = [len(query_gold[qid]) for qid in evaluated_query_ids]
    lexical_coverages = []
    max_gold_jaccards = []
    rare_term_ratios = []
    zero_overlap_queries = 0
    exact_query_in_gold = 0
    rare_df_limit = max(5, int(len(corpus_ids) * 0.001))

    for qid in evaluated_query_ids:
        query_text = queries[qid]
        q_tokens = query_token_sets[qid]
        gold_doc_ids = query_gold.get(qid, set())
        gold_sets = [gold_token_sets[doc_id] for doc_id in gold_doc_ids if doc_id in gold_token_sets]
        gold_union = set().union(*gold_sets) if gold_sets else set()
        overlap = q_tokens & gold_union
        lexical_coverages.append(len(overlap) / len(q_tokens) if q_tokens else 0.0)
        zero_overlap_queries += bool(q_tokens) and not bool(overlap)
        jaccards = [
            len(q_tokens & tokens) / len(q_tokens | tokens)
            for tokens in gold_sets
            if q_tokens | tokens
        ]
        max_gold_jaccards.append(max(jaccards, default=0.0))
        rare_terms = [term for term in q_tokens if query_term_df.get(term, 0) <= rare_df_limit]
        rare_term_ratios.append(len(rare_terms) / len(q_tokens) if q_tokens else 0.0)
        normalized_query = " ".join(tokenize(query_text))
        if normalized_query and any(
            normalized_query in gold_normalized_texts[doc_id]
            for doc_id in gold_doc_ids
            if doc_id in gold_normalized_texts
        ):
            exact_query_in_gold += 1

    doc_count = len(corpus_ids)
    query_count = len(queries)
    evaluated_query_count = len(evaluated_query_ids)
    return {
        "eda_version": EDA_VERSION,
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "directory": str(raw_dir),
            "configured_huggingface_ids": DATASET_SOURCE_IDS.get(dataset, {}),
            "files": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in (corpus_path, queries_path, qrels_path)
            },
        },
        "corpus": {
            "document_count": doc_count,
            "empty_document_count": empty_docs,
            "title_present_rate": round(title_present / doc_count, 6) if doc_count else 0.0,
            "contains_korean_rate": round(korean_docs / doc_count, 6) if doc_count else 0.0,
            "contains_latin_rate": round(latin_docs / doc_count, 6) if doc_count else 0.0,
            "characters": distribution(doc_chars),
            "tokens_regex": distribution(doc_tokens),
            "paragraphs": distribution(paragraph_counts),
            "short_document_rates": {
                f"tokens_le_{limit}": round(sum(v <= limit for v in doc_tokens) / doc_count, 6)
                if doc_count else 0.0
                for limit in (64, 128, 256, 512, 1024)
            },
            "chars_le_4000_rate": round(sum(v <= 4000 for v in doc_chars) / doc_count, 6)
            if doc_count else 0.0,
        },
        "queries": {
            "query_count": query_count,
            "characters": distribution(query_chars),
            "tokens_regex": distribution(query_tokens),
        },
        "relevance": {
            "qrels_total": qrels_total,
            "positive_qrels": positive_qrels,
            "relevance_score_counts": dict(sorted(relevance_scores.items())),
            "queries_with_positive_gold": evaluated_query_count,
            "queries_without_positive_gold": query_count - evaluated_query_count,
            "unique_gold_documents": len(gold_ids),
            "missing_gold_documents": len(gold_ids - gold_space_ids),
            "gold_documents_per_query": distribution(gold_counts),
            "multi_gold_query_rate": round(
                sum(v > 1 for v in gold_counts) / evaluated_query_count, 6
            ) if evaluated_query_count else 0.0,
            "metric_population": "queries with at least one positive qrel",
        },
        "lexical_alignment": {
            "evaluated_query_count": evaluated_query_count,
            "query_token_coverage_in_gold": distribution(lexical_coverages),
            "max_query_gold_jaccard": distribution(max_gold_jaccards),
            "zero_token_overlap_query_rate": round(
                zero_overlap_queries / evaluated_query_count, 6
            ) if evaluated_query_count else 0.0,
            "exact_normalized_query_in_gold_rate": round(
                exact_query_in_gold / evaluated_query_count, 6
            ) if evaluated_query_count else 0.0,
            "rare_query_term_ratio": distribution(rare_term_ratios),
            "rare_document_frequency_limit": rare_df_limit,
        },
    }


def analyze_augmentations(dataset: str, data_dir: Path, size_key: str) -> dict:
    result = {}
    taxonomy_path = data_dir / "taxonomy" / f"{dataset}_{size_key}.json"
    if taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text())
        fields = {}
        for field in ("L1", "L2", "L3"):
            counts = Counter(str(value.get(field, "")) for value in taxonomy.values())
            fields[field] = counts.most_common(30)
        result["taxonomy"] = {"document_count": len(taxonomy), "field_counts": fields}

    prefix_path = data_dir / "prefix" / f"{dataset}_{size_key}.json"
    if prefix_path.exists():
        prefixes = json.loads(prefix_path.read_text())
        result["prefix"] = {
            "document_count": len(prefixes),
            "characters": distribution([len(value or "") for value in prefixes.values()]),
        }

    metadata_path = data_dir / "metadata" / f"{dataset}_{size_key}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        field_presence = Counter()
        for value in metadata.values():
            field_presence.update(key for key, item in value.items() if item not in (None, "", [], {}))
        result["metadata"] = {
            "document_count": len(metadata),
            "field_presence": dict(field_presence.most_common()),
        }

    tag_root = data_dir / "tags" / dataset
    tag_result = {}
    for approach in ("a", "b", "c"):
        tag_path = tag_root / f"approach_{approach}" / f"{size_key}.json"
        if not tag_path.exists():
            continue
        rows = json.loads(tag_path.read_text())
        by_doc = Counter(str(row.get("doc_id", "")) for row in rows)
        tag_result[approach.upper()] = {
            "element_count": len(rows),
            "document_count": len(by_doc),
            "elements_per_document": distribution(list(by_doc.values())),
            "tag_counts": dict(Counter(str(row.get("tag", "")) for row in rows).most_common()),
        }
    if tag_result:
        result["tags"] = tag_result
    return result


def render_markdown(report: dict) -> str:
    corpus = report["corpus"]
    queries = report["queries"]
    relevance = report["relevance"]
    lexical = report["lexical_alignment"]
    token_stats = corpus["tokens_regex"]
    return "\n".join([
        f"# {report['dataset']} EDA",
        "",
        "## Dataset Size",
        "",
        f"- Documents: {corpus['document_count']:,}",
        f"- Queries: {queries['query_count']:,}",
        f"- Queries with positive gold: {relevance['queries_with_positive_gold']:,}",
        f"- Positive qrels: {relevance['positive_qrels']:,}",
        f"- Unique gold documents: {relevance['unique_gold_documents']:,}",
        "",
        "## Document Shape",
        "",
        f"- Regex-token p50/p95/p99/max: {token_stats.get('p50', 0):,.1f} / {token_stats.get('p95', 0):,.1f} / {token_stats.get('p99', 0):,.1f} / {token_stats.get('max', 0):,.1f}",
        f"- Documents <=256 tokens: {corpus['short_document_rates']['tokens_le_256']:.2%}",
        f"- Documents <=4,000 characters: {corpus['chars_le_4000_rate']:.2%}",
        f"- Contains Korean: {corpus['contains_korean_rate']:.2%}",
        f"- Contains Latin: {corpus['contains_latin_rate']:.2%}",
        "",
        "## Relevance Structure",
        "",
        f"- Multi-gold query rate: {relevance['multi_gold_query_rate']:.2%}",
        f"- Gold documents/query p50/p95/max: {relevance['gold_documents_per_query'].get('p50', 0):,.1f} / {relevance['gold_documents_per_query'].get('p95', 0):,.1f} / {relevance['gold_documents_per_query'].get('max', 0):,.1f}",
        f"- Missing gold documents: {relevance['missing_gold_documents']:,}",
        "",
        "## Lexical Alignment",
        "",
        f"- Evaluated queries: {lexical['evaluated_query_count']:,} (positive-gold queries only)",
        f"- Query-token coverage in gold p50: {lexical['query_token_coverage_in_gold'].get('p50', 0):.2%}",
        f"- Zero token-overlap query rate: {lexical['zero_token_overlap_query_rate']:.2%}",
        f"- Rare query-term ratio p50: {lexical['rare_query_term_ratio'].get('p50', 0):.2%}",
        "",
        "Token counts use a deterministic Unicode regex and are diagnostic values, not model-tokenizer lengths.",
    ]) + "\n"
