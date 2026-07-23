"""Immutable, passage-level preparation helpers for MIRACL Korean.

This module intentionally does not import the DR-DCI agent, retriever, or
focused Part 1/2 runner.  It acquires, normalizes, validates, and scales the
public MIRACL Korean artifacts without creating taxonomy or embeddings.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Iterator
import unicodedata


PREPARATION_VERSION = "miracl-ko-preparation-v1"
NORMALIZATION_VERSION = "miracl-ko-nfc-v1"
SCALE_VERSION = "miracl-ko-scale-v1"
SCALE_SIZES = (20_000, 50_000, 110_000)
RETRIEVAL_UNIT = "passage"
NORMALIZED_SCHEMA = {
    "corpus": {
        "corpus_id": "article_id#passage_index (preserved MIRACL docid)",
        "article_id": "derived from corpus_id before #",
        "passage_index": "integer derived from corpus_id after #",
        "title": "Unicode NFC source title",
        "text": "Unicode NFC source passage text",
    },
    "query": {
        "qid": "preserved MIRACL topic ID",
        "query": "Unicode NFC topic text",
        "split": "train or dev",
    },
    "qrel": {
        "qid": "preserved MIRACL topic ID",
        "corpus_id": "preserved MIRACL passage ID",
        "relevance": "preserved numeric relevance",
        "split": "train or dev",
    },
}

TOPICS_QRELS_REPO = "miracl/miracl"
CORPUS_REPO = "miracl/miracl-corpus"
TOPICS_QRELS_URL = f"https://huggingface.co/datasets/{TOPICS_QRELS_REPO}"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}"
WIKIPEDIA_LICENSE_URL = "https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content"

# These are reference counts published by the official MIRACL repository and
# dataset card.  They are deliberately not acquisition revisions or file
# checksums: the downloaded, API-resolved artifacts remain the source of truth.
OFFICIAL_REFERENCE_COUNTS = {
    "corpus": {"passage_count": 1_486_752, "article_count": 437_373},
    "queries": {"train": 868, "dev": 213},
    "qrels": {"train": 12_767, "dev": 3_057},
}

TOPICS_QRELS_FILES = (
    "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-train.tsv",
    "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
    "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-train.tsv",
    "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
)
CORPUS_FILES = tuple(
    f"miracl-corpus-v1.0-ko/docs-{index}.jsonl.gz" for index in range(3)
)

FORBIDDEN_LEAKAGE_TERMS = {
    "qrel", "qrels", "relevance", "gold", "answer", "rubric", "explanation",
    "positive_passages", "negative_passages",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalized_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("MIRACL text fields must be strings")
    return unicodedata.normalize("NFC", value)


def normalized_for_diagnostic(value: str) -> str:
    return " ".join(normalized_text(value).casefold().split())


def derive_passage_fields(corpus_id: str) -> dict[str, Any]:
    if not isinstance(corpus_id, str) or corpus_id.count("#") != 1:
        raise ValueError(f"MIRACL corpus_id must use article#passage: {corpus_id!r}")
    article_id, passage_index = corpus_id.rsplit("#", 1)
    if not article_id or not passage_index.isdigit():
        raise ValueError(f"MIRACL corpus_id must use article#passage: {corpus_id!r}")
    return {
        "corpus_id": corpus_id,
        "article_id": article_id,
        "passage_index": int(passage_index),
    }


def positive_passage_ids(qrels: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(row["corpus_id"])
        for row in qrels
        if float(row["relevance"]) > 0
    }


def scale_rank(corpus_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        SCALE_VERSION.encode("utf-8") + b"\0" + corpus_id.encode("utf-8")
    ).digest()
    return digest, corpus_id


def build_nested_subset_ids(
    corpus_ids: set[str], positive_ids: set[str], *, sizes: tuple[int, ...] = SCALE_SIZES
) -> dict[int, list[str]]:
    if tuple(sorted(sizes)) != tuple(sizes) or not sizes:
        raise ValueError("scale sizes must be a non-empty ascending tuple")
    if len(positive_ids) > sizes[0]:
        raise ValueError(f"M+ has {len(positive_ids):,} passages and cannot fit in {sizes[0]:,}")
    missing = sorted(positive_ids - corpus_ids)
    if missing:
        raise ValueError(f"positive qrel references {len(missing)} orphan passage IDs")
    if len(corpus_ids) < sizes[-1]:
        raise ValueError(
            f"corpus has {len(corpus_ids):,} passages; {sizes[-1]:,} passages are required"
        )

    mandatory = sorted(positive_ids, key=scale_rank)
    distractors = sorted(corpus_ids - positive_ids, key=scale_rank)
    canonical = mandatory + distractors[:sizes[-1] - len(mandatory)]
    if len(canonical) != sizes[-1]:
        raise ValueError("unable to construct canonical 110K passage subset")
    return {size: canonical[:size] for size in sizes}


def passage_content_hash(row: dict[str, Any]) -> str:
    value = {
        "corpus_id": str(row["corpus_id"]),
        "title": str(row.get("title", "")),
        "text": str(row.get("text", "")),
        "article_id": str(row["article_id"]),
        "passage_index": int(row["passage_index"]),
    }
    return sha256_json(value)


def validate_nested_subset_payloads(
    payloads: dict[int, list[dict[str, Any]]], *, positive_ids: set[str]
) -> None:
    expected_sizes = sorted(payloads)
    previous_rows: list[dict[str, Any]] | None = None
    previous_by_id: dict[str, str] = {}
    for size in expected_sizes:
        rows = payloads[size]
        ids = [str(row["corpus_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"subset {size} has duplicate passage IDs")
        if not positive_ids.issubset(ids):
            raise ValueError(f"subset {size} does not preserve all positive passages")
        if previous_rows is not None and ids[:len(previous_rows)] != [
            str(row["corpus_id"]) for row in previous_rows
        ]:
            raise ValueError("nested subsets do not preserve deterministic passage order")
        for row in rows:
            corpus_id = str(row["corpus_id"])
            content_hash = passage_content_hash(row)
            if corpus_id in previous_by_id and previous_by_id[corpus_id] != content_hash:
                raise ValueError(f"shared passage content changed across scales: {corpus_id}")
            previous_by_id[corpus_id] = content_hash
        previous_rows = rows


def validate_nested_subset_manifest(manifest: dict[str, Any]) -> None:
    """Guard scale-invariant query/qrel provenance in the persisted fixture manifest."""
    if manifest.get("retrieval_unit") != RETRIEVAL_UNIT:
        raise ValueError("MIRACL nested subset manifest must be passage-level")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict) or set(subsets) != {str(size) for size in SCALE_SIZES}:
        raise ValueError("MIRACL nested subset manifest must contain exactly 20K/50K/110K")
    query_qrel_inputs = [subsets[str(size)].get("query_qrel_inputs") for size in SCALE_SIZES]
    if not all(isinstance(value, dict) and value for value in query_qrel_inputs):
        raise ValueError("MIRACL nested subset manifest is missing query/qrel provenance")
    if any(value != query_qrel_inputs[0] for value in query_qrel_inputs[1:]):
        raise ValueError("MIRACL query/qrel inputs must be invariant across scales")
    for size in SCALE_SIZES:
        record = subsets[str(size)]
        if record.get("passage_count") != size:
            raise ValueError(f"MIRACL nested subset has incorrect passage count for {size}")
        if record.get("positive_passages_preserved") is not True:
            raise ValueError(f"MIRACL nested subset does not preserve positive passages for {size}")


def validate_acquisition_manifest(manifest: dict[str, Any]) -> None:
    required = (
        "dataset", "language", "source_urls", "resolved_revision", "downloaded_at",
        "license", "underlying_content_license", "acquisition_script_version",
        "acquisition_script_sha256", "python_version", "files",
    )
    for key in required:
        if not manifest.get(key):
            raise ValueError(f"acquisition manifest missing {key}")
    revision = manifest["resolved_revision"]
    if not isinstance(revision, dict) or not revision.get("topics_qrels") or not revision.get("corpus"):
        raise ValueError("acquisition manifest missing resolved_revision for topics_qrels or corpus")
    for record in manifest["files"]:
        for key in ("source_url", "relative_path", "byte_size", "sha256"):
            if key not in record:
                raise ValueError(f"acquisition file record missing {key}")
        if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64:
            raise ValueError("acquisition file record has invalid sha256")
        if int(record["byte_size"]) < 0:
            raise ValueError("acquisition file record has invalid byte_size")


def validate_leakage_inputs(input_fields: dict[str, list[str]]) -> None:
    for stage, fields in input_fields.items():
        lowered = {str(field).casefold() for field in fields}
        forbidden = sorted(
            term for term in FORBIDDEN_LEAKAGE_TERMS
            if any(term in field for field in lowered)
        )
        if forbidden:
            raise ValueError(f"{stage} input contains forbidden leakage field: {forbidden[0]}")
    allowed = {
        "taxonomy": {"corpus_id", "title", "text"},
        "retriever": {"corpus_id", "title", "text"},
        "query_rewrite": {"qid", "query"},
    }
    for stage, permitted in allowed.items():
        unexpected = set(input_fields.get(stage, [])) - permitted
        if unexpected:
            raise ValueError(f"{stage} input has unsupported fields: {sorted(unexpected)}")


def validate_miracl_result_unit(result: dict[str, Any]) -> None:
    if result.get("retrieval_unit") != RETRIEVAL_UNIT:
        raise ValueError("MIRACL retrieval results must declare passage retrieval_unit")
    for key in (result.get("metrics") or {}):
        if str(key).startswith("document_"):
            raise ValueError("document_ metric names are forbidden for MIRACL passage retrieval")


def validate_miracl_smoke_payload(result: dict[str, Any]) -> None:
    """Require provenance and per-query rows for any completed lexical smoke."""
    validate_miracl_result_unit(result)
    raw_rows = result.get("raw_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("MIRACL lexical smoke requires raw per-query rows")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("MIRACL lexical smoke requires provenance")
    for key in ("subset_sha256", "query_qrel_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"MIRACL lexical smoke provenance has invalid {key}")


def raw_paths(data_dir: Path) -> dict[str, Path]:
    raw_root = data_dir / "raw"
    return {
        "corpus_0": raw_root / CORPUS_FILES[0],
        "corpus_1": raw_root / CORPUS_FILES[1],
        "corpus_2": raw_root / CORPUS_FILES[2],
        "topics_train": raw_root / TOPICS_QRELS_FILES[0],
        "topics_dev": raw_root / TOPICS_QRELS_FILES[1],
        "qrels_train": raw_root / TOPICS_QRELS_FILES[2],
        "qrels_dev": raw_root / TOPICS_QRELS_FILES[3],
    }


def normalized_paths(data_dir: Path) -> dict[str, Path]:
    root = data_dir / "normalized"
    return {
        "corpus": root / "corpus.jsonl",
        "queries_train": root / "queries_train.jsonl",
        "queries_dev": root / "queries_dev.jsonl",
        "qrels_train": root / "qrels_train.jsonl",
        "qrels_dev": root / "qrels_dev.jsonl",
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield row


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as temporary:
        temporary_path = Path(temporary.name)
        row_count = 0
        for row in rows:
            temporary.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            row_count += 1
    os.replace(temporary_path, path)
    return {"rows": row_count, **file_record(path)}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
    os.replace(temporary_path, path)


def _normalization_counter() -> Counter[str]:
    return Counter({"title": 0, "text": 0, "query": 0})


def _normalize_field(value: Any, name: str, changes: Counter[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"MIRACL {name} must be a string")
    normalized = normalized_text(value)
    changes[name] += normalized != value
    return normalized


def normalize_miracl_ko(data_dir: Path) -> dict[str, Any]:
    raw = raw_paths(data_dir)
    missing = [name for name, path in raw.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MIRACL raw files: {', '.join(missing)}")
    output = normalized_paths(data_dir)
    changes = _normalization_counter()

    def corpus_rows() -> Iterator[dict[str, Any]]:
        for source_name in ("corpus_0", "corpus_1", "corpus_2"):
            with gzip.open(raw[source_name], "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        source = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid corpus JSON at {raw[source_name]}:{line_number}") from error
                    corpus_id = source.get("docid")
                    fields = derive_passage_fields(corpus_id)
                    yield {
                        **fields,
                        "title": _normalize_field(source.get("title"), "title", changes),
                        "text": _normalize_field(source.get("text"), "text", changes),
                    }

    def query_rows(split: str) -> Iterator[dict[str, Any]]:
        source = raw[f"topics_{split}"]
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    qid, query = line.split("\t", 1)
                except ValueError as error:
                    raise ValueError(f"invalid topic TSV at {source}:{line_number}") from error
                yield {"qid": qid, "query": _normalize_field(query, "query", changes), "split": split}

    def qrel_rows(split: str) -> Iterator[dict[str, Any]]:
        source = raw[f"qrels_{split}"]
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 4:
                    raise ValueError(f"invalid TREC qrel at {source}:{line_number}")
                qid, _iteration, corpus_id, relevance = fields
                derive_passage_fields(corpus_id)
                try:
                    numeric_relevance: int | float = int(relevance)
                except ValueError:
                    numeric_relevance = float(relevance)
                yield {
                    "qid": qid,
                    "corpus_id": corpus_id,
                    "relevance": numeric_relevance,
                    "split": split,
                }

    outputs = {
        "corpus": _atomic_write_jsonl(output["corpus"], corpus_rows()),
        "queries_train": _atomic_write_jsonl(output["queries_train"], query_rows("train")),
        "queries_dev": _atomic_write_jsonl(output["queries_dev"], query_rows("dev")),
        "qrels_train": _atomic_write_jsonl(output["qrels_train"], qrel_rows("train")),
        "qrels_dev": _atomic_write_jsonl(output["qrels_dev"], qrel_rows("dev")),
    }
    manifest = {
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "transformation_version": NORMALIZATION_VERSION,
        "generated_at": utc_now(),
        "inputs": {name: file_record(path, relative_to=data_dir) for name, path in raw.items()},
        "outputs": {name: {**record, "relative_path": str(output[name].relative_to(data_dir))}
                    for name, record in outputs.items()},
        "unicode_nfc_changed_fields": dict(changes),
    }
    _atomic_write_json(data_dir / "normalization_manifest.json", manifest)
    return manifest


def _distribution(values: array) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        index = (len(ordered) - 1) * percent / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered), "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 4),
        **{f"p{item}": round(percentile(item), 4) for item in (50, 90, 95, 99)},
        "max": ordered[-1],
    }


def _query_char_coverage(query: str, passage: str) -> float:
    query_chars = set(normalized_for_diagnostic(query).replace(" ", ""))
    passage_chars = set(normalized_for_diagnostic(passage).replace(" ", ""))
    return len(query_chars & passage_chars) / len(query_chars) if query_chars else 0.0


def _has_exact_ngram(query: str, passage: str, *, ngram_size: int = 20) -> bool:
    query = normalized_for_diagnostic(query)
    passage = normalized_for_diagnostic(passage)
    if len(query) < ngram_size:
        return False
    return any(query[index:index + ngram_size] in passage for index in range(len(query) - ngram_size + 1))


def _char_shingles(value: str, *, size: int = 3) -> set[str]:
    value = normalized_for_diagnostic(value).replace(" ", "")
    return {value[index:index + size] for index in range(max(0, len(value) - size + 1))} or {value}


def _near_duplicate_query_pairs(train: dict[str, str], dev: dict[str, str]) -> list[dict[str, Any]]:
    candidates = []
    train_shingles = {qid: _char_shingles(query) for qid, query in train.items()}
    for dev_qid, dev_query in dev.items():
        dev_shingles = _char_shingles(dev_query)
        for train_qid, train_set in train_shingles.items():
            union = train_set | dev_shingles
            score = len(train_set & dev_shingles) / len(union) if union else 1.0
            if score >= 0.8:
                candidates.append({"train_qid": train_qid, "dev_qid": dev_qid, "char_trigram_jaccard": round(score, 6)})
    return candidates


def _load_query_splits(
    paths: dict[str, Path]
) -> tuple[dict[str, dict[str, str]], dict[str, int], dict[str, int]]:
    splits: dict[str, dict[str, str]] = {}
    duplicates: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for split in ("train", "dev"):
        by_qid: dict[str, str] = {}
        duplicate_count = 0
        for row in iter_jsonl(paths[f"queries_{split}"]):
            qid = str(row["qid"])
            if qid in by_qid:
                duplicate_count += 1
            by_qid[qid] = str(row["query"])
        splits[split] = by_qid
        duplicates[split] = duplicate_count
        row_counts[split] = len(by_qid) + duplicate_count
    return splits, duplicates, row_counts


def _load_qrel_splits(paths: dict[str, Path]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    label_counts: dict[str, Counter[str]] = {}
    duplicate_counts: dict[str, int] = {}
    conflicts: dict[str, int] = {}
    positives_by_query: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "dev"):
        rows = list(iter_jsonl(paths[f"qrels_{split}"]))
        by_split[split] = rows
        all_rows.extend(rows)
        labels = Counter()
        seen: dict[tuple[str, str], float] = {}
        duplicates = 0
        conflicting = 0
        positives: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            qid = str(row["qid"])
            corpus_id = str(row["corpus_id"])
            relevance = float(row["relevance"])
            labels[str(row["relevance"])] += 1
            key = (qid, corpus_id)
            if key in seen:
                duplicates += 1
                conflicting += seen[key] != relevance
            seen[key] = relevance
            if relevance > 0:
                positives[qid].add(corpus_id)
        label_counts[split] = labels
        duplicate_counts[split] = duplicates
        conflicts[split] = conflicting
        positives_by_query[split] = positives
    return {
        "rows": all_rows,
        "by_split": by_split,
        "label_counts": label_counts,
        "duplicate_counts": duplicate_counts,
        "conflicts": conflicts,
        "positives_by_query": positives_by_query,
    }


def validate_miracl_ko(data_dir: Path) -> dict[str, Any]:
    """Compute passage-level EDA and fail-loud integrity findings from normalized data."""
    paths = normalized_paths(data_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MIRACL normalized files: {', '.join(missing)}")

    query_splits, duplicate_qids, query_row_counts = _load_query_splits(paths)
    qrel_data = _load_qrel_splits(paths)
    qrel_rows = qrel_data["rows"]
    positive_ids = positive_passage_ids(qrel_rows)
    all_qrel_corpus_ids = {str(row["corpus_id"]) for row in qrel_rows}
    positive_by_query = qrel_data["positives_by_query"]

    orphan_qids = {
        split: sorted({str(row["qid"]) for row in qrel_data["by_split"][split]} - set(query_splits[split]))
        for split in ("train", "dev")
    }
    title_lengths = array("I")
    text_lengths = array("I")
    article_counts: Counter[str] = Counter()
    corpus_hashes: dict[str, str] = {}
    duplicate_corpus_ids = 0
    conflicting_corpus_content = 0
    empty_title = 0
    empty_text = 0
    nfc_changed_on_recheck = 0
    positive_passages: dict[str, tuple[str, str]] = {}

    for row in iter_jsonl(paths["corpus"]):
        corpus_id = str(row["corpus_id"])
        derived = derive_passage_fields(corpus_id)
        if row.get("article_id") != derived["article_id"] or row.get("passage_index") != derived["passage_index"]:
            raise ValueError(f"normalized corpus has inconsistent passage derivation: {corpus_id}")
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        nfc_changed_on_recheck += normalized_text(title) != title
        nfc_changed_on_recheck += normalized_text(text) != text
        empty_title += not bool(title)
        empty_text += not bool(text)
        title_lengths.append(len(title))
        text_lengths.append(len(text))
        article_counts[derived["article_id"]] += 1
        content_hash = passage_content_hash(row)
        if corpus_id in corpus_hashes:
            duplicate_corpus_ids += 1
            conflicting_corpus_content += corpus_hashes[corpus_id] != content_hash
        corpus_hashes[corpus_id] = content_hash
        if corpus_id in positive_ids:
            positive_passages[corpus_id] = (title, text)

    corpus_ids = set(corpus_hashes)
    orphan_corpus_ids = sorted(all_qrel_corpus_ids - corpus_ids)
    query_positive_counts: dict[str, array] = {}
    multi_positive: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        counts = array("I", [len(positive_by_query[split].get(qid, set())) for qid in query_splits[split]])
        query_positive_counts[split] = counts
        nonzero = [value for value in counts if value > 0]
        multi_positive[split] = {
            "queries_with_positive": len(nonzero),
            "multi_positive_query_count": sum(value > 1 for value in nonzero),
            "multi_positive_query_rate": round(sum(value > 1 for value in nonzero) / len(nonzero), 6)
            if nonzero else 0.0,
            "positive_passages_per_query": _distribution(array("I", nonzero)),
        }

    exact_query_overlap = set(
        normalized_for_diagnostic(query) for query in query_splits["train"].values()
    ) & set(normalized_for_diagnostic(query) for query in query_splits["dev"].values())
    near_duplicates = _near_duplicate_query_pairs(query_splits["train"], query_splits["dev"])
    train_positive_articles = {
        derive_passage_fields(corpus_id)["article_id"]
        for passages in positive_by_query["train"].values() for corpus_id in passages
    }
    dev_positive_articles = {
        derive_passage_fields(corpus_id)["article_id"]
        for passages in positive_by_query["dev"].values() for corpus_id in passages
    }

    diagnostic_coverages = array("I")
    exact_ngram_queries = 0
    title_exact_match_queries = 0
    for split in ("train", "dev"):
        for qid, query in query_splits[split].items():
            passages = [positive_passages[corpus_id] for corpus_id in positive_by_query[split].get(qid, set())
                        if corpus_id in positive_passages]
            if not passages:
                continue
            diagnostic_coverages.append(round(max(
                _query_char_coverage(query, f"{title} {text}") for title, text in passages
            ) * 10_000))
            exact_ngram_queries += any(
                _has_exact_ngram(query, f"{title} {text}") for title, text in passages
            )
            normalized_query = normalized_for_diagnostic(query)
            title_exact_match_queries += any(
                normalized_query == normalized_for_diagnostic(title) for title, _text in passages
            )

    normalization_manifest_path = data_dir / "normalization_manifest.json"
    normalization_manifest = json.loads(normalization_manifest_path.read_text()) if normalization_manifest_path.exists() else {}
    violations = []
    if duplicate_corpus_ids:
        violations.append(f"duplicate corpus_id rows: {duplicate_corpus_ids}")
    if conflicting_corpus_content:
        violations.append(f"conflicting corpus_id content: {conflicting_corpus_content}")
    if any(orphan_qids.values()):
        violations.append("orphan qids in qrels")
    if orphan_corpus_ids:
        violations.append(f"orphan corpus_ids in qrels: {len(orphan_corpus_ids)}")
    if any(qrel_data["conflicts"].values()):
        violations.append("conflicting duplicate qrels")
    if nfc_changed_on_recheck:
        violations.append(f"normalized output is not NFC: {nfc_changed_on_recheck} fields")

    positive_id_hash = sha256_json(sorted(positive_ids))
    evaluated_queries = sum(item["queries_with_positive"] for item in multi_positive.values())
    coverage_distribution = _distribution(diagnostic_coverages)
    for key in ("min", "mean", "p50", "p90", "p95", "p99", "max"):
        if key in coverage_distribution:
            coverage_distribution[key] = round(coverage_distribution[key] / 10_000, 6)
    official_actual = {
        "corpus": {
            "passage_count": len(corpus_hashes),
            "article_count": len(article_counts),
        },
        "queries": {split: query_row_counts[split] for split in ("train", "dev")},
        "qrels": {split: len(qrel_data["by_split"][split]) for split in ("train", "dev")},
    }
    official_mismatches = {
        f"{section}.{key}": {"official": expected, "actual": official_actual[section][key]}
        for section, values in OFFICIAL_REFERENCE_COUNTS.items()
        for key, expected in values.items()
        if official_actual[section][key] != expected
    }
    report = {
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "generated_at": utc_now(),
        "normalization_manifest_sha256": sha256_file(normalization_manifest_path)
        if normalization_manifest_path.exists() else None,
        "corpus": {
            "passage_count": len(corpus_hashes),
            "unique_passage_id_count": len(corpus_ids),
            "article_count": len(article_counts),
            "duplicate_passage_id_rows": duplicate_corpus_ids,
            "conflicting_passage_content": conflicting_corpus_content,
            "empty_title_count": empty_title,
            "empty_text_count": empty_text,
            "title_characters": _distribution(title_lengths),
            "text_characters": _distribution(text_lengths),
            "passages_per_article": _distribution(array("I", article_counts.values())),
            "unicode_nfc_changed_on_recheck": nfc_changed_on_recheck,
        },
        "queries": {
            "train_query_row_count": query_row_counts["train"],
            "dev_query_row_count": query_row_counts["dev"],
            "train_query_count": len(query_splits["train"]),
            "dev_query_count": len(query_splits["dev"]),
            "empty_query_count": {
                split: sum(not query for query in query_splits[split].values())
                for split in ("train", "dev")
            },
            "query_characters": {
                split: _distribution(array("I", (len(query) for query in query_splits[split].values())))
                for split in ("train", "dev")
            },
            "duplicate_qid_rows": duplicate_qids,
            "train_dev_shared_qids": len(set(query_splits["train"]) & set(query_splits["dev"])),
            "train_dev_exact_normalized_query_count": len(exact_query_overlap),
            "train_dev_near_duplicate_query_candidates": near_duplicates,
        },
        "qrels": {
            "train_count": len(qrel_data["by_split"]["train"]),
            "dev_count": len(qrel_data["by_split"]["dev"]),
            "label_distribution": {
                split: dict(sorted(counter.items())) for split, counter in qrel_data["label_counts"].items()
            },
            "duplicate_qrel_rows": qrel_data["duplicate_counts"],
            "conflicting_duplicate_qrels": qrel_data["conflicts"],
            "orphan_qids": {split: len(items) for split, items in orphan_qids.items()},
            "orphan_corpus_ids": len(orphan_corpus_ids),
            "positive_passage_count": len(positive_ids),
            "positive_passage_id_sha256": positive_id_hash,
            "per_split_positive_structure": multi_positive,
        },
        "split_overlap": {
            "train_positive_article_count": len(train_positive_articles),
            "dev_positive_article_count": len(dev_positive_articles),
            "shared_positive_article_count": len(train_positive_articles & dev_positive_articles),
        },
        "leakage_diagnostics": {
            "scope": "diagnostic only; not a discard threshold",
            "evaluated_positive_query_count": evaluated_queries,
            "query_positive_max_normalized_character_coverage": {
                **coverage_distribution,
                "unit": "fraction_of_unique_query_characters",
            },
            "query_with_positive_passage_exact_20_char_ngram_count": exact_ngram_queries,
            "query_with_positive_title_exact_match_count": title_exact_match_queries,
        },
        "normalization": normalization_manifest.get("unicode_nfc_changed_fields", {}),
        "official_reference_count_comparison": {
            "source": "MIRACL official repository and dataset card",
            "expected": OFFICIAL_REFERENCE_COUNTS,
            "actual": official_actual,
            "status": "matches" if not official_mismatches else "release_drift_or_transform_difference",
            "mismatches": official_mismatches,
        },
        "integrity_status": "passed" if not violations else "blocked",
        "violations": violations,
    }
    return report


def build_miracl_ko_subsets(data_dir: Path, eda: dict[str, Any]) -> dict[str, Any]:
    """Create 110K first, then its ordered 50K and 20K passage prefixes."""
    if eda.get("retrieval_unit") != RETRIEVAL_UNIT:
        raise ValueError("MIRACL subset builder requires passage-level EDA")
    if eda.get("integrity_status") != "passed":
        raise ValueError("MIRACL subset builder requires passing integrity EDA")
    paths = normalized_paths(data_dir)
    acquisition_path = data_dir / "acquisition_manifest.json"
    if not acquisition_path.is_file():
        raise FileNotFoundError("MIRACL subset builder requires acquisition_manifest.json")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    validate_acquisition_manifest(acquisition)
    qrels = list(iter_jsonl(paths["qrels_train"])) + list(iter_jsonl(paths["qrels_dev"]))
    positive_ids = positive_passage_ids(qrels)
    if len(positive_ids) != eda["qrels"]["positive_passage_count"]:
        raise ValueError("EDA positive-passage count does not match normalized qrels")
    corpus_ids = {str(row["corpus_id"]) for row in iter_jsonl(paths["corpus"])}
    ordered_ids = build_nested_subset_ids(corpus_ids, positive_ids)
    canonical_ids = ordered_ids[SCALE_SIZES[-1]]
    canonical_set = set(canonical_ids)
    selected_rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(paths["corpus"]):
        corpus_id = str(row["corpus_id"])
        if corpus_id in canonical_set:
            selected_rows[corpus_id] = row
    if len(selected_rows) != SCALE_SIZES[-1]:
        raise ValueError("canonical 110K subset is missing selected passages")
    payloads = {size: [selected_rows[corpus_id] for corpus_id in ids]
                for size, ids in ordered_ids.items()}
    validate_nested_subset_payloads(payloads, positive_ids=positive_ids)

    root = data_dir / "subsets"
    subset_records: dict[str, Any] = {}
    shared_inputs = {
        name: file_record(path, relative_to=data_dir)
        for name, path in paths.items() if name != "corpus"
    }
    for size in reversed(SCALE_SIZES):
        destination = root / f"{size // 1000}k" / "corpus.jsonl"
        record = _atomic_write_jsonl(destination, payloads[size])
        content_hashes = [passage_content_hash(row) for row in payloads[size]]
        subset_records[str(size)] = {
            "passage_count": len(payloads[size]),
            "corpus": {**record, "relative_path": str(destination.relative_to(data_dir))},
            "ordered_passage_id_sha256": sha256_json(ordered_ids[size]),
            "passage_content_aggregate_sha256": sha256_json(content_hashes),
            "positive_passage_count": len(positive_ids),
            "positive_passages_preserved": positive_ids.issubset(ordered_ids[size]),
            "query_qrel_inputs": shared_inputs,
        }
    manifest = {
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "design": "controlled distractor scaling; positive passages fixed, distractors increase",
        "scale_version": SCALE_VERSION,
        "acquisition_manifest_sha256": sha256_file(acquisition_path),
        "source_revisions": acquisition["resolved_revision"],
        "positive_set_definition": "relevance > 0 across normalized train and dev qrels only",
        "positive_passage_count": len(positive_ids),
        "positive_passage_id_sha256": sha256_json(sorted(positive_ids)),
        "negative_qrels_forced_into_mandatory_set": False,
        "subsets": subset_records,
    }
    validate_nested_subset_manifest(manifest)
    _atomic_write_json(root / "manifest.json", manifest)
    return manifest


def _repository_files(info: Any) -> dict[str, Any]:
    return {item.rfilename: item for item in (info.siblings or [])}


def acquire_miracl_ko(data_dir: Path, *, acquisition_script: Path) -> dict[str, Any]:
    """Download only official artifacts at their API-resolved immutable commits."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    topics_info = api.dataset_info(TOPICS_QRELS_REPO, files_metadata=True)
    corpus_info = api.dataset_info(CORPUS_REPO, files_metadata=True)
    repositories = {
        "topics_qrels": (TOPICS_QRELS_REPO, TOPICS_QRELS_URL, topics_info, TOPICS_QRELS_FILES),
        "corpus": (CORPUS_REPO, CORPUS_URL, corpus_info, CORPUS_FILES),
    }
    raw_root = data_dir / "raw"
    files = []
    for source_key, (repo_id, source_url, info, required_files) in repositories.items():
        listing = _repository_files(info)
        for filename in required_files:
            if filename not in listing:
                raise ValueError(f"official repository revision {info.sha} is missing {filename}")
            remote = listing[filename]
            local_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=info.sha,
                local_dir=raw_root,
            ))
            record = {
                "dataset": "MIRACL",
                "language": "ko",
                "source_key": source_key,
                "source_url": source_url,
                "resolved_revision": info.sha,
                **file_record(local_path, relative_to=data_dir),
            }
            expected_size = getattr(remote, "size", None)
            if expected_size is not None and local_path.stat().st_size != expected_size:
                raise ValueError(f"downloaded byte size does not match official metadata: {filename}")
            lfs = getattr(remote, "lfs", None)
            expected_sha = getattr(lfs, "sha256", None) if lfs else None
            if expected_sha and record["sha256"] != expected_sha:
                raise ValueError(f"downloaded SHA-256 does not match official LFS metadata: {filename}")
            record["official_lfs_sha256"] = expected_sha
            files.append(record)
    manifest = {
        "dataset": "MIRACL",
        "language": "ko",
        "source_urls": {"topics_qrels": TOPICS_QRELS_URL, "corpus": CORPUS_URL},
        "resolved_revision": {"topics_qrels": topics_info.sha, "corpus": corpus_info.sha},
        "downloaded_at": utc_now(),
        "license": "Apache-2.0 (MIRACL repository and dataset cards)",
        "underlying_content_license": {
            "content": "Wikipedia text corpus",
            "terms": "CC BY-SA reuse terms: attribution, share-alike, change indication, and license notice apply",
            "source_url": WIKIPEDIA_LICENSE_URL,
        },
        "acquisition_script_version": PREPARATION_VERSION,
        "acquisition_script_sha256": sha256_file(acquisition_script),
        "python_version": os.sys.version,
        "transformation_version": NORMALIZATION_VERSION,
        "files": files,
    }
    validate_acquisition_manifest(manifest)
    _atomic_write_json(data_dir / "acquisition_manifest.json", manifest)
    return manifest


def raw_data_paths_are_ignored(repo_root: Path, data_dir: Path) -> bool:
    try:
        relative = data_dir.relative_to(repo_root)
    except ValueError:
        return False
    if relative.parts[:1] != ("data",):
        return False
    probe = relative / "raw" / ".miracl-ko-ignore-probe"
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", "--no-index", "--", str(probe)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def write_miracl_pretest_artifacts(
    docs_dir: Path,
    *,
    acquisition: dict[str, Any],
    normalization: dict[str, Any],
    eda: dict[str, Any],
    subsets: dict[str, Any],
    smoke: dict[str, Any],
    stamp: str,
) -> tuple[Path, Path, Path]:
    summary = {
        "schema_version": "dr-dci.miracl-ko-pretest.v1",
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "normalized_schema": NORMALIZED_SCHEMA,
        "acquisition": acquisition,
        "normalization": normalization,
        "eda": eda,
        "nested_subsets": subsets,
        "lexical_plumbing_smoke": smoke,
        "part12_status": {
            "taxonomy_artifact": "not_generated",
            "agent_or_llm": "not_executed",
            "focused_part1_part2": "not_approved",
            "minimum_practical_effect_status": "unapproved",
            "minimum_practical_effect_version": "pending-miracl-passage",
        },
        "gate_decision": {
            "status": "suitable" if eda.get("integrity_status") == "passed" else "unsuitable",
            "scope": "Korean passage-retrieval algorithm screening only",
            "not_evidence_for": [
                "taxonomy effect", "Agentic RAG improvement", "insurance-domain transfer",
                "110K physical-document operations",
            ],
        },
    }
    hash_manifest = {
        "schema_version": "dr-dci.miracl-ko-hash-manifest.v1",
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "normalized_schema": NORMALIZED_SCHEMA,
        "acquisition": acquisition,
        "normalization": normalization,
        "nested_subsets": subsets,
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = docs_dir / f"MIRACL_KO_PRETEST_{stamp}.json"
    hash_path = docs_dir / f"MIRACL_KO_HASH_MANIFEST_{stamp}.json"
    markdown_path = docs_dir / f"MIRACL_KO_PRETEST_{stamp}.md"
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(hash_path, hash_manifest)
    corpus = eda["corpus"]
    qrels = eda["qrels"]
    subset_rows = subsets["subsets"]
    markdown = "\n".join([
        f"# MIRACL Korean pretest — {stamp}", "",
        "## Gate decision", "",
        "**suitable for Korean passage-retrieval algorithm screening**, subject to the recorded data revisions and passage-level contract.",
        "This is not a taxonomy result, Agentic RAG result, insurance-domain validation, or 110K physical-document claim.", "",
        "## Acquisition and unit", "",
        f"- Topics/qrels revision: `{acquisition['resolved_revision']['topics_qrels']}`",
        f"- Corpus revision: `{acquisition['resolved_revision']['corpus']}`",
        f"- MIRACL artifact license: {acquisition['license']}; underlying Wikipedia terms are recorded separately in the hash manifest.",
        "- Retrieval unit: **passage** (`article_id#passage_index`); article aggregation is not used.",
        f"- Raw artifacts are under ignored `data/`; no raw or normalized passage text is committed.", "",
        "## Measured integrity", "",
        f"- Corpus: {corpus['passage_count']:,} passages from {corpus['article_count']:,} articles.",
        f"- Queries: train {eda['queries']['train_query_count']:,}; dev {eda['queries']['dev_query_count']:,}.",
        f"- Judgments: train {qrels['train_count']:,}; dev {qrels['dev_count']:,}; positive passage union M+ {qrels['positive_passage_count']:,}.",
        f"- Orphan qids/passage IDs: {sum(qrels['orphan_qids'].values())} / {qrels['orphan_corpus_ids']}; integrity status `{eda['integrity_status']}`.",
        f"- Official reference-count comparison: `{eda['official_reference_count_comparison']['status']}`; NFC changed title/text/query fields: `{eda['normalization']}`.",
        "",
        "## Normalized schema",
        "- Corpus: `corpus_id`, `article_id`, `passage_index`, `title`, `text`.",
        "- Query: `qid`, `query`, `split`; qrel: `qid`, `corpus_id`, `relevance`, `split`.",
        "- IDs and numeric relevance are preserved; passage IDs are never collapsed to article IDs.",
        "",
        "## Controlled distractor fixtures", "",
        "Positive passages (`relevance > 0`) are fixed across 20K/50K/110K; only SHA-256-ranked distractor passages increase.",
        "The deterministic rank is `SHA256(\"miracl-ko-scale-v1\\0\" + corpus_id)`; the 110K fixture is built first, then ordered 50K/20K prefixes are derived.",
        *[f"- {size}: {subset_rows[str(size)]['passage_count']:,} passages; positive preservation `{subset_rows[str(size)]['positive_passages_preserved']}`; corpus SHA-256 `{subset_rows[str(size)]['corpus']['sha256']}`" for size in SCALE_SIZES],
        "",
        "## Leakage and smoke", "",
        "- No taxonomy artifact was generated. The preparation contract allows taxonomy only from 110K corpus `title`/`text`; qrels and queries are forbidden inputs.",
        "- Train qrels are not used for dev scoring by this preparation layer.",
        f"- Lexical plumbing smoke: **{smoke['status']}** — {smoke['reason']}",
        "",
        "## Next boundary", "",
        "A later, separately instructed MIRACL experiment may create a taxonomy artifact and approve a passage-specific practical-effect rule. It must not reuse the TREC document-level 0.01 screening rule automatically.",
        "",
    ])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=markdown_path.parent, delete=False, prefix=f".{markdown_path.name}."
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(markdown)
    os.replace(temporary_path, markdown_path)
    return markdown_path, summary_path, hash_path
