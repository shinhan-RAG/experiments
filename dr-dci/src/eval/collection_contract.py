"""Fail-loud contracts for collection-scoped retrieval evaluation bundles."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


MANIFEST_SCHEMA_VERSION = "shinhan.collection-eval-manifest.v1"
DOCUMENT_SCHEMA_VERSION = "shinhan.collection-document.v1"
CHUNK_SCHEMA_VERSION = "shinhan.collection-chunk.v1"
ELEMENT_SCHEMA_VERSION = "shinhan.collection-element.v1"
TAG_SCHEMA_VERSION = "shinhan.collection-semantic-tag.v1"
QA_SCHEMA_VERSION = "shinhan.collection-qa.v1"

REQUIRED_SCOPES = {"per_collection", "all_collections"}
REQUIRED_RETRIEVAL_MODES = {"chunk", "semantic_tag", "combined"}
REQUIRED_METRICS = {
    "evidence_ndcg@10",
    "evidence_recall@20",
    "parent_hit@10",
    "evidence_coverage@20",
    "latency_p95_ms",
}
ALLOWED_TAG_INPUT_FIELDS = {
    "element_text",
    "element_type",
    "document_title",
    "document_metadata",
}
FORBIDDEN_TAG_INPUT_TOKENS = {
    "query",
    "qrel",
    "gold",
    "answer",
    "evidence",
    "qa",
    "relevance",
    "ranking",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COLLECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class ContractError(ValueError):
    """Raised when an evaluation artifact violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be non-empty")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    _require(isinstance(value, list) and bool(value), f"{name} must be a non-empty list")
    _require(
        all(isinstance(item, str) and bool(item.strip()) for item in value),
        f"{name} must contain non-empty strings",
    )
    _require(len(value) == len(set(value)), f"{name} contains duplicates")
    return value


def _namespaced(value: Any, collection_id: str, name: str) -> str:
    identifier = _nonempty_string(value, name)
    _require(
        identifier.startswith(f"{collection_id}::"),
        f"{name} must start with '{collection_id}::'",
    )
    return identifier


def _sha256(value: Any, name: str) -> str:
    digest = _nonempty_string(value, name)
    _require(bool(SHA256_RE.fullmatch(digest)), f"{name} must be lowercase SHA-256")
    return digest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {error}") from error
            _require(isinstance(value, dict), f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _artifact_path(root: Path, spec: Any, name: str) -> tuple[Path, int]:
    artifact = _mapping(spec, name)
    relative = Path(_nonempty_string(artifact.get("path"), f"{name}.path"))
    _require(not relative.is_absolute(), f"{name}.path must be relative")
    _require(".." not in relative.parts, f"{name}.path cannot traverse parents")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{name}.path escapes bundle root") from error
    _require(path.is_file(), f"{name}.path does not exist: {relative}")
    expected_hash = _sha256(artifact.get("sha256"), f"{name}.sha256")
    _require(sha256_file(path) == expected_hash, f"{name}.sha256 mismatch")
    records = artifact.get("records")
    _require(isinstance(records, int) and records >= 1, f"{name}.records must be positive")
    return path, records


def _validate_model_lock(model: Any, name: str) -> None:
    lock = _mapping(model, name)
    _nonempty_string(lock.get("model_id"), f"{name}.model_id")
    revision = _nonempty_string(lock.get("revision"), f"{name}.revision")
    _require(
        revision.lower() not in {"main", "master", "latest", "unknown"},
        f"{name}.revision must be immutable",
    )


def validate_manifest(manifest: Any) -> dict[str, Any]:
    value = _mapping(manifest, "manifest")
    _require(
        value.get("schema_version") == MANIFEST_SCHEMA_VERSION,
        f"schema_version must be {MANIFEST_SCHEMA_VERSION}",
    )
    _nonempty_string(value.get("experiment_id"), "experiment_id")
    collections = value.get("collections")
    _require(isinstance(collections, list) and collections, "collections must be non-empty")
    collection_ids = []
    for index, collection in enumerate(collections):
        item = _mapping(collection, f"collections[{index}]")
        collection_id = _nonempty_string(
            item.get("collection_id"), f"collections[{index}].collection_id"
        )
        _require(
            bool(COLLECTION_ID_RE.fullmatch(collection_id)),
            f"invalid collection_id: {collection_id}",
        )
        collection_ids.append(collection_id)
        _nonempty_string(item.get("source_revision"), f"{collection_id}.source_revision")
        artifacts = _mapping(item.get("artifacts"), f"{collection_id}.artifacts")
        _require(
            set(artifacts) == {"documents", "chunks", "elements", "semantic_tags", "qa"},
            f"{collection_id}.artifacts must contain exactly documents/chunks/elements/"
            "semantic_tags/qa",
        )
    _require(len(collection_ids) == len(set(collection_ids)), "duplicate collection_id")

    evaluation = _mapping(value.get("evaluation"), "evaluation")
    _require(
        set(_string_list(evaluation.get("scopes"), "evaluation.scopes")) == REQUIRED_SCOPES,
        f"evaluation.scopes must be {sorted(REQUIRED_SCOPES)}",
    )
    _require(
        set(_string_list(evaluation.get("retrieval_modes"), "evaluation.retrieval_modes"))
        == REQUIRED_RETRIEVAL_MODES,
        f"evaluation.retrieval_modes must be {sorted(REQUIRED_RETRIEVAL_MODES)}",
    )
    _require(
        evaluation.get("shared_qa_across_modes") is True,
        "evaluation.shared_qa_across_modes must be true",
    )
    _require(
        evaluation.get("all_collections_qa") == "union_of_per_collection_qa",
        "evaluation.all_collections_qa must be union_of_per_collection_qa",
    )

    budgets = _mapping(evaluation.get("candidate_budget"), "evaluation.candidate_budget")
    expected_budget_keys = {
        "chunk_pool",
        "semantic_tag_pool",
        "combined_chunk_pool",
        "combined_semantic_tag_pool",
        "final_top_k",
    }
    _require(
        set(budgets) == expected_budget_keys,
        f"candidate_budget must contain exactly {sorted(expected_budget_keys)}",
    )
    _require(
        all(isinstance(budgets[key], int) and budgets[key] > 0 for key in budgets),
        "candidate budgets must be positive integers",
    )
    _require(
        budgets["chunk_pool"] == budgets["semantic_tag_pool"],
        "single-source candidate pools must be equal",
    )
    _require(
        budgets["combined_chunk_pool"] + budgets["combined_semantic_tag_pool"]
        == budgets["chunk_pool"],
        "combined source pools must sum to the single-source pool",
    )
    _require(
        budgets["final_top_k"] <= min(
            budgets["chunk_pool"], budgets["semantic_tag_pool"]
        ),
        "final_top_k exceeds a source pool",
    )

    fusion = _mapping(evaluation.get("fusion"), "evaluation.fusion")
    _require(fusion.get("method") == "rrf", "evaluation.fusion.method must be rrf")
    _require(
        isinstance(fusion.get("rrf_k"), int) and fusion["rrf_k"] > 0,
        "evaluation.fusion.rrf_k must be positive",
    )
    models = _mapping(evaluation.get("models"), "evaluation.models")
    _require(set(models) == {"embedding", "reranker"}, "models must lock embedding/reranker")
    _validate_model_lock(models["embedding"], "evaluation.models.embedding")
    _validate_model_lock(models["reranker"], "evaluation.models.reranker")

    metrics = set(_string_list(evaluation.get("metrics"), "evaluation.metrics"))
    _require(REQUIRED_METRICS <= metrics, f"metrics must include {sorted(REQUIRED_METRICS)}")
    _require(
        evaluation.get("report_per_collection") is True
        and evaluation.get("report_macro_average") is True
        and evaluation.get("report_micro_average") is True,
        "collection, macro, and micro reporting must all be enabled",
    )

    tag_policy = _mapping(value.get("semantic_tag_generation"), "semantic_tag_generation")
    _require(tag_policy.get("qa_access") is False, "semantic tags cannot access QA")
    _require(tag_policy.get("qrel_access") is False, "semantic tags cannot access qrels")
    _require(tag_policy.get("gold_access") is False, "semantic tags cannot access gold")
    declared_inputs = set(
        _string_list(tag_policy.get("allowed_input_fields"), "allowed_input_fields")
    )
    _require(
        declared_inputs <= ALLOWED_TAG_INPUT_FIELDS,
        "semantic tag input fields exceed the allowlist",
    )

    duplicate_policy = _mapping(
        value.get("cross_collection_duplicates"), "cross_collection_duplicates"
    )
    _require(
        duplicate_policy.get("policy") == "fail_on_duplicate",
        "v1 requires cross_collection_duplicates.policy=fail_on_duplicate",
    )
    return value


def _validate_document(record: dict[str, Any], collection_id: str) -> str:
    _require(record.get("schema_version") == DOCUMENT_SCHEMA_VERSION, "document schema")
    _require(record.get("collection_id") == collection_id, "document collection mismatch")
    document_id = _namespaced(record.get("document_id"), collection_id, "document_id")
    text = _nonempty_string(record.get("text"), f"{document_id}.text")
    content_sha256 = _sha256(
        record.get("content_sha256"), f"{document_id}.content_sha256"
    )
    _require(
        hashlib.sha256(text.encode("utf-8")).hexdigest() == content_sha256,
        f"{document_id}.content_sha256 does not match text",
    )
    return document_id


def _validate_chunk(
    record: dict[str, Any], collection_id: str
) -> tuple[str, str]:
    _require(record.get("schema_version") == CHUNK_SCHEMA_VERSION, "chunk schema")
    _require(record.get("collection_id") == collection_id, "chunk collection mismatch")
    chunk_id = _namespaced(record.get("chunk_id"), collection_id, "chunk_id")
    document_id = _namespaced(record.get("document_id"), collection_id, "chunk.document_id")
    _nonempty_string(record.get("text"), f"{chunk_id}.text")
    _validate_location(record.get("source_location"), f"{chunk_id}.source_location")
    return chunk_id, document_id


def _validate_location(value: Any, name: str) -> None:
    location = _mapping(value, name)
    status = location.get("status")
    _require(status in {"verified", "unavailable"}, f"{name}.status is invalid")
    _require(
        location.get("basis") in {"source_text", "element_text", "unavailable"},
        f"{name}.basis is invalid",
    )
    start, end = location.get("char_start"), location.get("char_end")
    if status == "verified":
        _require(
            isinstance(start, int) and isinstance(end, int) and 0 <= start < end,
            f"{name} verified offsets are invalid",
        )
        _require(location.get("basis") != "unavailable", f"{name}.basis unavailable")
    else:
        _require(start is None and end is None, f"{name} unavailable offsets must be null")
        _require(location.get("basis") == "unavailable", f"{name}.basis must be unavailable")


def _validate_element(
    record: dict[str, Any], collection_id: str
) -> tuple[str, str]:
    _require(record.get("schema_version") == ELEMENT_SCHEMA_VERSION, "element schema")
    _require(record.get("collection_id") == collection_id, "element collection mismatch")
    element_id = _namespaced(record.get("element_id"), collection_id, "element_id")
    document_id = _namespaced(
        record.get("document_id"), collection_id, "element.document_id"
    )
    _nonempty_string(record.get("text"), f"{element_id}.text")
    _nonempty_string(record.get("element_type"), f"{element_id}.element_type")
    _require(
        isinstance(record.get("ordinal"), int) and record["ordinal"] >= 0,
        f"{element_id}.ordinal must be non-negative",
    )
    _validate_location(record.get("source_location"), f"{element_id}.source_location")
    return element_id, document_id


def _validate_tag(
    record: dict[str, Any], collection_id: str
) -> tuple[str, str, str]:
    _require(record.get("schema_version") == TAG_SCHEMA_VERSION, "semantic tag schema")
    _require(record.get("collection_id") == collection_id, "tag collection mismatch")
    tag_id = _namespaced(record.get("tag_id"), collection_id, "tag_id")
    element_id = _namespaced(record.get("element_id"), collection_id, "tag.element_id")
    document_id = _namespaced(record.get("document_id"), collection_id, "tag.document_id")
    _string_list(record.get("tags"), f"{tag_id}.tags")
    generation = _mapping(record.get("generation"), f"{tag_id}.generation")
    input_fields = set(_string_list(generation.get("input_fields"), "tag input_fields"))
    _require(input_fields <= ALLOWED_TAG_INPUT_FIELDS, "tag input field is not allowed")
    lowered = {field.lower() for field in input_fields}
    _require(
        not any(token in field for field in lowered for token in FORBIDDEN_TAG_INPUT_TOKENS),
        "tag generation input leaks QA/qrel/gold information",
    )
    _require(generation.get("qa_access") is False, "tag generation accessed QA")
    _require(generation.get("qrel_access") is False, "tag generation accessed qrels")
    _require(generation.get("gold_access") is False, "tag generation accessed gold")
    _nonempty_string(generation.get("generator_id"), "tag generator_id")
    _nonempty_string(generation.get("revision"), "tag generator revision")
    _sha256(generation.get("prompt_sha256"), "tag prompt_sha256")
    return tag_id, element_id, document_id


def _validate_qa(
    record: dict[str, Any], collection_id: str
) -> tuple[str, set[str], set[str], set[str]]:
    _require(record.get("schema_version") == QA_SCHEMA_VERSION, "QA schema")
    _require(record.get("collection_id") == collection_id, "QA collection mismatch")
    qa_id = _namespaced(record.get("qa_id"), collection_id, "qa_id")
    _nonempty_string(record.get("query"), f"{qa_id}.query")
    _require(record.get("split") in {"dev", "test"}, f"{qa_id}.split must be dev/test")
    documents = set(_string_list(record.get("gold_document_ids"), "gold_document_ids"))
    elements = set(_string_list(record.get("gold_element_ids"), "gold_element_ids"))
    chunks = set(_string_list(record.get("gold_chunk_ids"), "gold_chunk_ids"))
    for name, identifiers in (
        ("gold_document_ids", documents),
        ("gold_element_ids", elements),
        ("gold_chunk_ids", chunks),
    ):
        for identifier in identifiers:
            _namespaced(identifier, collection_id, name)

    evidence = record.get("evidence")
    _require(isinstance(evidence, list) and evidence, f"{qa_id}.evidence must be non-empty")
    evidence_elements: set[str] = set()
    evidence_chunks: set[str] = set()
    evidence_documents: set[str] = set()
    for index, item in enumerate(evidence):
        value = _mapping(item, f"{qa_id}.evidence[{index}]")
        evidence_documents.add(
            _namespaced(value.get("document_id"), collection_id, "evidence.document_id")
        )
        evidence_elements.add(
            _namespaced(value.get("element_id"), collection_id, "evidence.element_id")
        )
        evidence_chunks.update(
            _string_list(value.get("chunk_ids"), "evidence.chunk_ids")
        )
        quote = _nonempty_string(value.get("quote"), "evidence.quote")
        _require(len(quote.strip()) >= 8, "evidence.quote must have at least 8 characters")
        _validate_location(value.get("source_location"), "evidence.source_location")
        alignment = _mapping(value.get("alignment"), "evidence.alignment")
        _require(
            alignment.get("status") == "verified",
            "evidence alignment must be verified",
        )
        _require(
            alignment.get("method")
            in {"exact_offset", "exact_text", "normalized_text", "parser_derived"},
            "unsupported evidence alignment method",
        )
    _require(evidence_documents <= documents, "evidence document is not gold")
    _require(evidence_elements <= elements, "evidence element is not gold")
    _require(evidence_chunks <= chunks, "evidence chunk is not gold")

    provenance = _mapping(record.get("generation"), f"{qa_id}.generation")
    _nonempty_string(provenance.get("generator_id"), "QA generator_id")
    _nonempty_string(provenance.get("revision"), "QA generator revision")
    _sha256(provenance.get("prompt_sha256"), "QA prompt_sha256")
    anchor = _namespaced(
        provenance.get("anchor_element_id"), collection_id, "anchor_element_id"
    )
    _require(anchor in elements, "QA anchor element is not gold")
    return qa_id, documents, elements, chunks


def _unique(records: Iterable[str], name: str) -> set[str]:
    values = list(records)
    _require(len(values) == len(set(values)), f"duplicate {name}")
    return set(values)


def validate_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = validate_manifest(yaml.safe_load(stream))
    root = manifest_path.parent
    summary: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "collections": {},
    }
    content_hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for collection in manifest["collections"]:
        collection_id = collection["collection_id"]
        artifacts = collection["artifacts"]
        loaded: dict[str, list[dict[str, Any]]] = {}
        for artifact_name, spec in artifacts.items():
            path, expected_records = _artifact_path(
                root, spec, f"{collection_id}.{artifact_name}"
            )
            records = _load_jsonl(path)
            _require(
                len(records) == expected_records,
                f"{collection_id}.{artifact_name}.records mismatch",
            )
            loaded[artifact_name] = records

        document_pairs = [
            (_validate_document(record, collection_id), record["content_sha256"])
            for record in loaded["documents"]
        ]
        document_ids = _unique((pair[0] for pair in document_pairs), "document_id")
        for document_id, content_hash in document_pairs:
            content_hashes[content_hash].append((collection_id, document_id))

        chunk_pairs = [
            _validate_chunk(record, collection_id) for record in loaded["chunks"]
        ]
        chunk_ids = _unique((pair[0] for pair in chunk_pairs), "chunk_id")
        _require(
            all(document_id in document_ids for _, document_id in chunk_pairs),
            "chunk references an absent document",
        )

        element_pairs = [
            _validate_element(record, collection_id) for record in loaded["elements"]
        ]
        element_ids = _unique((pair[0] for pair in element_pairs), "element_id")
        _require(
            all(document_id in document_ids for _, document_id in element_pairs),
            "element references an absent document",
        )

        tag_triples = [
            _validate_tag(record, collection_id) for record in loaded["semantic_tags"]
        ]
        _unique((triple[0] for triple in tag_triples), "tag_id")
        _require(
            all(
                element_id in element_ids and document_id in document_ids
                for _, element_id, document_id in tag_triples
            ),
            "semantic tag references an absent element/document",
        )
        tagged_elements = {triple[1] for triple in tag_triples}
        _require(tagged_elements == element_ids, "semantic tag coverage is not 100%")

        qa_ids = []
        for record in loaded["qa"]:
            qa_id, gold_documents, gold_elements, gold_chunks = _validate_qa(
                record, collection_id
            )
            qa_ids.append(qa_id)
            _require(gold_documents <= document_ids, f"{qa_id} has absent gold document")
            _require(gold_elements <= element_ids, f"{qa_id} has absent gold element")
            _require(gold_chunks <= chunk_ids, f"{qa_id} has absent gold chunk")
        _unique(qa_ids, "qa_id")
        summary["collections"][collection_id] = {
            "documents": len(document_ids),
            "chunks": len(chunk_ids),
            "elements": len(element_ids),
            "semantic_tags": len(tag_triples),
            "qa": len(qa_ids),
        }

    duplicates = {
        digest: records
        for digest, records in content_hashes.items()
        if len({collection_id for collection_id, _ in records}) > 1
    }
    if duplicates:
        raise ContractError("cross-collection duplicate document content detected")
    summary["cross_collection_duplicate_groups"] = len(duplicates)
    summary["status"] = "ok"
    return summary
