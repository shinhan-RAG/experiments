"""Generic academic-paper collection adapter for the accepted alignment_layer EDA.

Converts read-only AIHub academic-paper label archives into the frozen
``config/collection_eval`` document/chunk/element schemas plus a separate
element-alignment artifact. Scope is conversion and alignment only: no QA,
no semantic tags, no model calls, and no Peter Part 1-4 wiring.

Source character offsets are not recoverable (accepted EDA), so every
``source_location`` stays explicitly unavailable. Ranges computed against the
constructed document text are recorded in the alignment artifact under the
``constructed_document_text`` basis and are never presented as PDF/PPTX
source offsets.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Iterator
import zipfile

import yaml

from src.data.collection_publication import (
    LockConflictError,
    PublicationError,
    TargetLock,
    canonical_json_sha256,
    discard_staging,
    prepare_staging,
    publish_staging,
    target_state,
)
from src.eval.collection_contract import (
    CHUNK_SCHEMA_VERSION,
    COLLECTION_ID_RE,
    ContractError,
    DOCUMENT_SCHEMA_VERSION,
    ELEMENT_SCHEMA_VERSION,
    SHA256_RE,
    _validate_chunk,
    _validate_document,
    _validate_element,
    sha256_file,
)


SOURCE_CONFIG_SCHEMA_VERSION = "academic.collection-source-config.v1"
CHUNK_POLICY_SCHEMA_VERSION = "academic.chunking-policy-config.v1"
ACCEPTANCE_CONTRACT_SCHEMA_VERSION = "academic.accepted-full-run-contract.v1"
COMPAT_CONTRACT_SCHEMA_VERSION = "academic.chunk-model-compat-contract.v1"
RUN_IDENTITY_SCHEMA_VERSION = "academic.conversion-run-identity.v1"
RUN_METRICS_SCHEMA_VERSION = "academic.conversion-run-metrics.v1"
ALIGNMENT_SCHEMA_VERSION = "shinhan.collection-element-alignment.v1"
CONVERSION_MANIFEST_SCHEMA_VERSION = "shinhan.collection-conversion.v1"
MANIFEST_FILE_NAME = "conversion_manifest.json"
METRICS_FILE_NAME = "run_metrics.json"
CHUNK_POLICY_ID = "academic.element-packed-chunk.v1"
CONSTRUCTED_TEXT_POLICY_ID = "academic.constructed-document-text.v1"
IMPLEMENTATION_STAGE = "element_alignment"
IMPLEMENTATION_DECISION = "alignment_layer"
DOCUMENT_RANGE_BASIS = "constructed_document_text"
DOCUMENT_RANGE_METHOD = "ordered_concatenation"
ELEMENT_TYPE_SECTION = "section"
REQUIRED_SPLITS = ("Training", "Validation")
ARTIFACT_NAMES = ("documents", "chunks", "elements", "alignment")

UNAVAILABLE_LOCATION = {
    "status": "unavailable",
    "basis": "unavailable",
    "char_start": None,
    "char_end": None,
}

# Accepted-EDA field policy: only section original_text is document content.
CONTENT_FIELD = "training_data_info.section_info[].original_text"
FORBIDDEN_CONTENT_FIELDS = (
    "summary_text",
    "summary_cnt",
    "original_cnt",
    "procede",
    "image_caption",
    "image_category",
    "location",
    "page",
)

SAFE_ID_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CHUNK_ID_MARKER = "::chunk::"


class ConversionError(ValueError):
    """Raised when source data or converted artifacts violate the stage contract."""


class AmbiguousAlignmentError(ConversionError):
    """Raised when a text-search alignment does not resolve to exactly one range."""


def _fail(message: str) -> None:
    raise ConversionError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _safe_id_part(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    _require(
        bool(SAFE_ID_PART_RE.fullmatch(value)),
        f"{name} contains unsafe identifier characters: {value!r}",
    )
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _archive_spec(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be a mapping")
    path = value.get("path")
    _require(isinstance(path, str) and bool(path.strip()), f"{name}.path must be non-empty")
    relative = Path(path)
    _require(not relative.is_absolute(), f"{name}.path must be data-root relative")
    _require(".." not in relative.parts, f"{name}.path cannot traverse parents")
    _require(
        isinstance(value.get("bytes"), int) and value["bytes"] > 0,
        f"{name}.bytes must be a positive integer",
    )
    digest = value.get("sha256")
    _require(
        isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest)),
        f"{name}.sha256 must be lowercase SHA-256",
    )
    return value


def load_source_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    _require(isinstance(config, dict), "source config must be a mapping")
    _require(
        config.get("schema_version") == SOURCE_CONFIG_SCHEMA_VERSION,
        f"source config schema_version must be {SOURCE_CONFIG_SCHEMA_VERSION}",
    )
    collections = config.get("collections")
    _require(
        isinstance(collections, dict) and bool(collections),
        "source config collections must be a non-empty mapping",
    )
    for collection_id, spec in collections.items():
        _require(
            isinstance(collection_id, str)
            and bool(COLLECTION_ID_RE.fullmatch(collection_id)),
            f"invalid collection_id: {collection_id!r}",
        )
        _require(isinstance(spec, dict), f"{collection_id} spec must be a mapping")
        _safe_id_part(spec.get("domain"), f"{collection_id}.domain")
        splits = spec.get("splits")
        _require(isinstance(splits, dict), f"{collection_id}.splits must be a mapping")
        _require(
            tuple(sorted(splits)) == tuple(sorted(REQUIRED_SPLITS)),
            f"{collection_id}.splits must define exactly {list(REQUIRED_SPLITS)}; "
            "Training/Validation are splits, not collections",
        )
        for split, cell in splits.items():
            _require(isinstance(cell, dict), f"{collection_id}.{split} must be a mapping")
            _archive_spec(cell.get("label_archive"), f"{collection_id}.{split}.label_archive")
            _archive_spec(cell.get("source_archive"), f"{collection_id}.{split}.source_archive")
    eda = config.get("eda")
    _require(isinstance(eda, dict), "source config eda must be a mapping")
    _require(
        eda.get("decision") == IMPLEMENTATION_DECISION,
        f"source config eda.decision must be {IMPLEMENTATION_DECISION}",
    )
    digest = eda.get("handoff_manifest_sha256")
    _require(
        isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest)),
        "source config eda.handoff_manifest_sha256 must be lowercase SHA-256",
    )
    return config


def load_chunking_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    _require(isinstance(config, dict), "chunking config must be a mapping")
    _require(
        config.get("schema_version") == CHUNK_POLICY_SCHEMA_VERSION,
        f"chunking config schema_version must be {CHUNK_POLICY_SCHEMA_VERSION}",
    )
    _require(
        config.get("policy_id") == CHUNK_POLICY_ID,
        f"chunking policy_id must be {CHUNK_POLICY_ID}",
    )
    separator = config.get("separator")
    _require(
        isinstance(separator, str) and separator != "",
        "chunking separator must be a non-empty string",
    )
    _require(
        isinstance(config.get("max_chars"), int) and config["max_chars"] > 0,
        "chunking max_chars must be a positive integer",
    )
    _require(
        config.get("element_atomic") is True,
        "chunking element_atomic must be true for this policy",
    )
    return config


def load_acceptance_contract(path: Path) -> dict[str, Any]:
    """Load the reviewed accepted-full-run contract (EDA denominators)."""
    with Path(path).open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    _require(isinstance(contract, dict), "acceptance contract must be a mapping")
    _require(
        contract.get("schema_version") == ACCEPTANCE_CONTRACT_SCHEMA_VERSION,
        f"acceptance contract schema_version must be {ACCEPTANCE_CONTRACT_SCHEMA_VERSION}",
    )
    for key in ("collections", "splits"):
        value = contract.get(key)
        _require(
            isinstance(value, list) and value and all(isinstance(v, str) for v in value),
            f"acceptance contract {key} must be a non-empty string list",
        )
    totals = contract.get("totals")
    _require(isinstance(totals, dict), "acceptance contract totals must be a mapping")
    cells = contract.get("cells")
    _require(isinstance(cells, dict) and cells, "acceptance contract cells must be a mapping")
    expected_cells = {
        f"{collection}/{split}"
        for collection in contract["collections"]
        for split in contract["splits"]
    }
    _require(
        set(cells) == expected_cells,
        "acceptance contract cells must cover exactly collections x splits",
    )
    for cell_name, cell in cells.items():
        _require(isinstance(cell, dict), f"acceptance cell {cell_name} must be a mapping")
        for key in ("documents", "elements", "images_excluded"):
            _require(
                isinstance(cell.get(key), int) and cell[key] >= 0,
                f"acceptance cell {cell_name}.{key} must be a non-negative integer",
            )
    _require(
        isinstance(contract.get("archive_count"), int) and contract["archive_count"] > 0,
        "acceptance contract archive_count must be positive",
    )
    return contract


# ---------------------------------------------------------------------------
# Run identity (input identity vs runtime identity)
# ---------------------------------------------------------------------------
#
# The INPUT identity hashes everything that determines output bytes: source
# and chunking configs (which pin archive bytes/SHA-256), selection, options,
# and the behavior-affecting code/schema files. It decides whether an existing
# published target may be reused as-is.
#
# The RUNTIME identity records where and how a run happened (git commit,
# dirty flag, interpreter, dependencies). It never affects output bytes, but
# acceptance eligibility requires a clean, known runtime identity.

_MODULE_BASE = Path(__file__).resolve().parents[2]
# Every behavior-affecting local module, schema, and entry point. The
# import-coverage test in tests/test_academic_collection_gates.py fails if a
# src import of this module (transitively) is missing from this map.
_CODE_IDENTITY_FILES = {
    "adapter": Path(__file__).resolve(),
    "publication": Path(__file__).resolve().parent / "collection_publication.py",
    "collection_contract": _MODULE_BASE / "src" / "eval" / "collection_contract.py",
    "build_cli": _MODULE_BASE / "scripts" / "build_academic_collections.py",
    "compat_cli": _MODULE_BASE / "scripts" / "validate_chunk_model_compatibility.py",
    "alignment_schema": _MODULE_BASE
    / "config"
    / "collection_academic"
    / "element_alignment.schema.json",
}


def code_identity_hashes() -> dict[str, str]:
    return {
        name: sha256_file(path) for name, path in sorted(_CODE_IDENTITY_FILES.items())
    }


def compute_run_identity(
    *,
    source_config: dict[str, Any],
    chunking_config: dict[str, Any],
    selection: dict[str, Any],
    options: dict[str, Any],
    acceptance_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "source_config_sha256": canonical_json_sha256(source_config),
        "chunking_config_sha256": canonical_json_sha256(chunking_config),
        "acceptance_contract_sha256": (
            canonical_json_sha256(acceptance_contract) if acceptance_contract else None
        ),
        "code": code_identity_hashes(),
        "selection": selection,
        "options": options,
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def collect_runtime_identity(base: Path | None = None) -> dict[str, Any]:
    """Best-effort runtime identity; unknown/dirty states are recorded, not hidden."""
    base = Path(base) if base else _MODULE_BASE
    commit = "unknown"
    dirty = True
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
        dirty = True
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pyyaml_version": getattr(yaml, "__version__", "unknown"),
    }


# ---------------------------------------------------------------------------
# Read-only archive access
# ---------------------------------------------------------------------------

def normalize_member_name(name: str) -> str:
    """Normalize one archive member name to a flat, safe file name.

    The academic archives store every member with a single leading slash
    (accepted EDA: absolute_members == member_count). Only that documented
    form and the already-relative form are accepted.
    """
    _require(isinstance(name, str) and bool(name), "archive member name must be non-empty")
    _require("\\" not in name, f"unsupported backslash in member name: {name!r}")
    normalized = name[1:] if name.startswith("/") else name
    _require(bool(normalized), f"member name empty after normalization: {name!r}")
    _require(
        not normalized.startswith("/"),
        f"member name is not safely normalizable: {name!r}",
    )
    _require(
        re.match(r"^[A-Za-z]:", normalized) is None,
        f"unsupported drive-letter member name: {name!r}",
    )
    parts = normalized.split("/")
    _require(
        all(part not in ("", ".", "..") for part in parts),
        f"member name traverses or repeats separators: {name!r}",
    )
    _require(
        len(parts) == 1,
        f"unsupported nested member layout (EDA observed flat archives): {name!r}",
    )
    _require(
        not any(ord(ch) < 32 for ch in normalized),
        f"control character in member name: {name!r}",
    )
    return normalized


def verify_archive_file(
    archive_path: Path,
    spec: dict[str, Any],
    *,
    verify_sha256: bool,
) -> dict[str, Any]:
    _require(archive_path.is_file(), f"archive does not exist: {archive_path}")
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != spec["bytes"]:
        _fail(
            f"archive bytes mismatch: {archive_path} "
            f"expected={spec['bytes']} actual={actual_bytes}"
        )
    record = {
        "path": spec["path"],
        "bytes": actual_bytes,
        "sha256": spec["sha256"],
        "sha256_verified": False,
    }
    if verify_sha256:
        actual_sha = sha256_file(archive_path)
        if actual_sha != spec["sha256"]:
            _fail(
                f"archive sha256 mismatch: {archive_path} "
                f"expected={spec['sha256']} actual={actual_sha}"
            )
        record["sha256_verified"] = True
    return record


def _member_map(archive: zipfile.ZipFile, archive_name: str) -> dict[str, str]:
    members: dict[str, str] = {}
    for raw_name in archive.namelist():
        if raw_name.endswith("/"):
            continue
        normalized = normalize_member_name(raw_name)
        _require(
            normalized not in members,
            f"duplicate normalized member name in {archive_name}: {normalized!r}",
        )
        members[normalized] = raw_name
    _require(bool(members), f"archive has no file members: {archive_name}")
    return members


def _read_member(archive: zipfile.ZipFile, raw_name: str) -> bytes:
    with archive.open(raw_name) as stream:
        return stream.read()


def _hash_member(archive: zipfile.ZipFile, raw_name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(raw_name) as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


# ---------------------------------------------------------------------------
# Label parsing (accepted-EDA schema; fail loud on anything else)
# ---------------------------------------------------------------------------

def _nonempty_str(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{name} must be a non-empty string",
    )
    return value


def parse_label_document(data: bytes, member: str) -> dict[str, Any]:
    """Parse one label JSON member into the fields this stage may use."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"invalid label JSON {member!r}: {error}") from error
    _require(isinstance(payload, dict), f"label root must be an object: {member!r}")
    raw_meta = payload.get("raw_data_meta_info")
    _require(isinstance(raw_meta, dict), f"raw_data_meta_info missing: {member!r}")
    source_meta = payload.get("source_data_meta_info")
    _require(isinstance(source_meta, dict), f"source_data_meta_info missing: {member!r}")
    training = payload.get("training_data_info")
    _require(isinstance(training, dict), f"training_data_info missing: {member!r}")
    sections = training.get("section_info")
    _require(
        isinstance(sections, list) and bool(sections),
        f"section_info must be a non-empty list: {member!r}",
    )
    parsed_sections = []
    seen_paragraph_ids: set[str] = set()
    for index, section in enumerate(sections):
        _require(
            isinstance(section, dict),
            f"section_info[{index}] must be an object: {member!r}",
        )
        paragraph_id = _safe_id_part(
            section.get("paragraph_id"), f"{member}#section_info[{index}].paragraph_id"
        )
        _require(
            paragraph_id not in seen_paragraph_ids,
            f"duplicate paragraph_id {paragraph_id!r} in {member!r}",
        )
        seen_paragraph_ids.add(paragraph_id)
        text = section.get("original_text")
        _require(
            isinstance(text, str) and bool(text.strip()),
            f"{member}#section_info[{index}].original_text must be non-empty "
            "(accepted EDA reports zero empty sections)",
        )
        parsed_sections.append({"paragraph_id": paragraph_id, "original_text": text})
    images = training.get("image_info")
    if images is None:
        images = []
    _require(isinstance(images, list), f"image_info must be a list: {member!r}")
    for index, image in enumerate(images):
        _require(
            isinstance(image, dict),
            f"image_info[{index}] must be an object: {member!r}",
        )
    return {
        "raw_doc_id": _nonempty_str(raw_meta.get("doc_id"), f"{member}#doc_id"),
        "source_data_id": _nonempty_str(
            source_meta.get("source_data_id"), f"{member}#source_data_id"
        ),
        "sections": parsed_sections,
        "image_record_count": len(images),
    }


# ---------------------------------------------------------------------------
# Constructed text, chunk packing, and alignment
# ---------------------------------------------------------------------------

def pack_elements_into_chunks(
    element_lengths: list[int],
    *,
    separator_length: int,
    max_chars: int,
) -> list[list[int]]:
    """Greedy deterministic packing of consecutive whole elements.

    Elements are atomic: an element longer than ``max_chars`` becomes its own
    chunk rather than being split, so every element maps to exactly one chunk.
    """
    _require(max_chars > 0, "max_chars must be positive")
    _require(all(length > 0 for length in element_lengths), "element length must be positive")
    groups: list[list[int]] = []
    current: list[int] = []
    current_length = 0
    for index, length in enumerate(element_lengths):
        extended = current_length + separator_length + length if current else length
        if current and extended > max_chars:
            groups.append(current)
            current = [index]
            current_length = length
        else:
            current.append(index)
            current_length = extended
    if current:
        groups.append(current)
    return groups


def _percentile(sorted_values: list[int], fraction: float) -> int:
    """Deterministic nearest-rank percentile over a non-empty sorted list."""
    index = min(len(sorted_values) - 1, max(0, int(fraction * len(sorted_values))))
    return sorted_values[index]


def chunk_length_distribution(
    lengths: list[int], *, max_chars: int, oversized_element_ids: list[str]
) -> dict[str, Any]:
    """Aggregate chunk-length statistics; element IDs only as SHA-256 samples."""
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)
    oversized = sum(1 for length in ordered if length > max_chars)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "max_chars_policy": max_chars,
        "oversized_count": oversized,
        "oversized_rate": round(oversized / len(ordered), 6),
        "oversized_element_id_sha256_samples": [
            sha256_text(element_id) for element_id in oversized_element_ids[:5]
        ],
    }


def _normalize_with_index_map(value: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space, keeping original index per char.

    ``index_map[i]`` is the original index of the character that produced
    normalized character ``i`` (for a collapsed run, the run's first index).
    Leading whitespace produces no normalized character.
    """
    normalized: list[str] = []
    index_map: list[int] = []
    position = 0
    length = len(value)
    while position < length:
        if value[position].isspace():
            run_start = position
            while position < length and value[position].isspace():
                position += 1
            if normalized:
                normalized.append(" ")
                index_map.append(run_start)
        else:
            normalized.append(value[position])
            index_map.append(position)
            position += 1
    return "".join(normalized), index_map


def locate_unique_normalized(document_text: str, element_text: str) -> tuple[int, int]:
    """Whitespace-normalized search returning ORIGINAL document offsets.

    Fails loud unless exactly one normalized match exists. The returned
    ``[start, end)`` range indexes ``document_text`` itself (never the
    normalized string), so the original slice normalizes back to the needle.
    The builder never uses this (ranges are exact by construction); it exists
    so any search-based alignment path is forced through an ambiguity check.
    """
    needle, _ = _normalize_with_index_map(element_text)
    needle = needle.strip()
    _require(bool(needle), "cannot align an empty element text")
    haystack, index_map = _normalize_with_index_map(document_text)
    matches = [
        match.start() for match in re.finditer(re.escape(needle), haystack)
    ]
    if len(matches) != 1:
        raise AmbiguousAlignmentError(
            f"normalized-text alignment matched {len(matches)} ranges; "
            "exactly one is required"
        )
    normalized_start = matches[0]
    normalized_last = normalized_start + len(needle) - 1
    start = index_map[normalized_start]
    end = index_map[normalized_last] + 1
    check, _ = _normalize_with_index_map(document_text[start:end])
    _require(
        check.strip() == needle,
        "internal error: normalized offset mapping does not round-trip",
    )
    return start, end


def build_document_bundle(
    *,
    collection_id: str,
    split: str,
    json_stem: str,
    label: dict[str, Any],
    separator: str,
    max_chars: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build document/element/chunk/alignment records for one label document."""
    _safe_id_part(json_stem, "json_stem")
    _require(split in REQUIRED_SPLITS, f"unknown split: {split!r}")
    document_id = f"{collection_id}::{json_stem}"
    sections = label["sections"]

    texts = [section["original_text"] for section in sections]
    document_text = separator.join(texts)

    element_records = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for ordinal, section in enumerate(sections):
        if ordinal:
            cursor += len(separator)
        start = cursor
        end = start + len(section["original_text"])
        ranges.append((start, end))
        cursor = end
        element_records.append(
            {
                "schema_version": ELEMENT_SCHEMA_VERSION,
                "collection_id": collection_id,
                "element_id": f"{document_id}::{section['paragraph_id']}",
                "document_id": document_id,
                "element_type": ELEMENT_TYPE_SECTION,
                "ordinal": ordinal,
                "text": section["original_text"],
                "source_location": dict(UNAVAILABLE_LOCATION),
            }
        )
    _require(cursor == len(document_text), "constructed text cursor mismatch")

    groups = pack_elements_into_chunks(
        [len(text) for text in texts],
        separator_length=len(separator),
        max_chars=max_chars,
    )
    chunk_records = []
    chunk_meta = []
    membership: dict[int, tuple[str, int]] = {}
    for chunk_index, group in enumerate(groups):
        chunk_id = f"{document_id}{_CHUNK_ID_MARKER}{chunk_index:04d}"
        chunk_start = ranges[group[0]][0]
        chunk_end = ranges[group[-1]][1]
        chunk_records.append(
            {
                "schema_version": CHUNK_SCHEMA_VERSION,
                "collection_id": collection_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": document_text[chunk_start:chunk_end],
                "source_location": dict(UNAVAILABLE_LOCATION),
            }
        )
        chunk_meta.append(
            {
                "chunk_id": chunk_id,
                "length": chunk_end - chunk_start,
                "element_ids": [
                    element_records[index]["element_id"] for index in group
                ],
            }
        )
        for element_index in group:
            _require(
                element_index not in membership,
                f"element assigned to multiple chunks: {element_records[element_index]['element_id']}",
            )
            membership[element_index] = (chunk_id, chunk_start)
    _require(len(membership) == len(element_records), "element without a chunk assignment")

    alignment_records = []
    for ordinal, element in enumerate(element_records):
        chunk_id, chunk_start = membership[ordinal]
        start, end = ranges[ordinal]
        alignment_records.append(
            {
                "schema_version": ALIGNMENT_SCHEMA_VERSION,
                "collection_id": collection_id,
                "document_id": document_id,
                "element_id": element["element_id"],
                "ordinal": ordinal,
                "document_range": {
                    "status": "verified",
                    "basis": DOCUMENT_RANGE_BASIS,
                    "method": DOCUMENT_RANGE_METHOD,
                    "char_start": start,
                    "char_end": end,
                },
                "chunk_memberships": [
                    {
                        "chunk_id": chunk_id,
                        "char_start": start - chunk_start,
                        "char_end": end - chunk_start,
                    }
                ],
                "source_location": dict(UNAVAILABLE_LOCATION),
            }
        )

    document_record = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "collection_id": collection_id,
        "document_id": document_id,
        "text": document_text,
        "content_sha256": sha256_text(document_text),
        "metadata": {
            "split": split,
            "json_stem": json_stem,
            "raw_doc_id": label["raw_doc_id"],
            "source_data_id": label["source_data_id"],
            "constructed_text_policy": CONSTRUCTED_TEXT_POLICY_ID,
            "element_count": len(element_records),
            "excluded_image_records": label["image_record_count"],
            "provenance": provenance,
        },
    }
    return {
        "document": document_record,
        "elements": element_records,
        "chunks": chunk_records,
        "chunk_meta": chunk_meta,
        "alignment": alignment_records,
        "image_records_excluded": label["image_record_count"],
    }


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def _repo_root_of(base: Path) -> Path | None:
    for candidate in (base, *base.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def ensure_output_dir_outside_git(output_dir: Path, *, code_base: Path) -> None:
    """Outputs hold private text; only ignored data/ paths may live in the repo."""
    resolved = output_dir.resolve()
    repo_root = _repo_root_of(code_base.resolve())
    if repo_root is None:
        return
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return
    allowed = (code_base.resolve() / "data").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError:
        _fail(
            f"output dir {resolved} is inside the repository but not under the "
            f"ignored {allowed} directory"
        )


def _write_jsonl_row(stream, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _artifact_entry(path: Path, records: int, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _normalize_selection(
    source_config: dict[str, Any],
    collections: list[str] | None,
    splits: list[str] | None,
    limit_documents: int,
) -> tuple[list[str], list[str]]:
    selected_collections = sorted(collections or source_config["collections"])
    for collection_id in selected_collections:
        _require(
            collection_id in source_config["collections"],
            f"unknown collection: {collection_id}",
        )
    selected_splits = sorted(splits or REQUIRED_SPLITS)
    for split in selected_splits:
        _require(split in REQUIRED_SPLITS, f"unknown split: {split!r}")
    _require(limit_documents >= 0, "limit_documents must be >= 0")
    return selected_collections, selected_splits


def _build_collections(
    *,
    data_root: Path,
    source_config: dict[str, Any],
    chunking_config: dict[str, Any],
    output_dir: Path,
    collections: list[str] | None = None,
    splits: list[str] | None = None,
    limit_documents: int = 0,
    verify_archive_sha256: bool = True,
    hash_source_members: bool = True,
    counters: dict[str, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write all conversion artifacts into ``output_dir`` (a staging directory).

    Complexity: time O(A + T + S log S) where A is archive bytes read
    (verification and member decompression), T is output text bytes, and
    S log S is the explicit per-cell sort of JSON stems (S = documents per
    cell). Peak memory is bounded by one document bundle plus per-cell member
    listings and chunk-length lists — never by total corpus text.
    """
    data_root = Path(data_root)
    _require(data_root.is_dir(), f"data root does not exist: {data_root}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = counters if counters is not None else {}
    counters.setdefault("bytes_read", 0)
    counters.setdefault("bytes_written", 0)

    selected_collections, selected_splits = _normalize_selection(
        source_config, collections, splits, limit_documents
    )

    separator = chunking_config["separator"]
    max_chars = chunking_config["max_chars"]

    archives_manifest: list[dict[str, Any]] = []
    cells_manifest: dict[str, dict[str, Any]] = {}
    collections_manifest: dict[str, dict[str, Any]] = {}
    global_content_hashes: dict[str, str] = {}
    global_stems: dict[str, str] = {}

    for collection_id in selected_collections:
        spec = source_config["collections"][collection_id]
        collection_dir = output_dir / collection_id
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Cell discovery: open archives, normalize members, pair labels/sources.
        cell_plans = []
        open_archives: list[zipfile.ZipFile] = []
        streams: dict[str, Any] = {}
        try:
            for split in selected_splits:
                cell = spec["splits"][split]
                label_path = data_root / cell["label_archive"]["path"]
                source_path = data_root / cell["source_archive"]["path"]
                label_info = verify_archive_file(
                    label_path,
                    cell["label_archive"],
                    verify_sha256=verify_archive_sha256,
                )
                source_info = verify_archive_file(
                    source_path,
                    cell["source_archive"],
                    verify_sha256=verify_archive_sha256,
                )
                archives_manifest.extend([label_info, source_info])
                if verify_archive_sha256:
                    counters["bytes_read"] += label_info["bytes"] + source_info["bytes"]

                label_zip = zipfile.ZipFile(label_path)
                open_archives.append(label_zip)
                source_zip = zipfile.ZipFile(source_path)
                open_archives.append(source_zip)
                label_members = _member_map(label_zip, cell["label_archive"]["path"])
                source_members = _member_map(
                    source_zip, cell["source_archive"]["path"]
                )

                json_stems: dict[str, str] = {}
                png_members = 0
                for normalized in sorted(label_members):
                    lowered = normalized.lower()
                    if lowered.endswith(".json"):
                        stem = normalized[: -len(".json")]
                        _safe_id_part(
                            stem, f"label stem in {cell['label_archive']['path']}"
                        )
                        json_stems[stem] = label_members[normalized]
                    elif lowered.endswith(".png"):
                        png_members += 1
                    else:
                        _fail(
                            "unsupported label member type "
                            f"in {cell['label_archive']['path']}: {normalized!r}"
                        )

                source_by_stem: dict[str, dict[str, str]] = defaultdict(dict)
                for normalized in sorted(source_members):
                    lowered = normalized.lower()
                    _require(
                        lowered.endswith(".pdf") or lowered.endswith(".pptx"),
                        "unsupported source member type "
                        f"in {cell['source_archive']['path']}: {normalized!r}",
                    )
                    stem, extension = normalized.rsplit(".", 1)
                    _require(
                        extension.lower() not in source_by_stem[stem],
                        f"duplicate source member for stem {stem!r}",
                    )
                    source_by_stem[stem][extension.lower()] = source_members[normalized]

                label_stem_set = set(json_stems)
                source_stem_set = set(source_by_stem)
                _require(
                    label_stem_set == source_stem_set,
                    f"label/source stem pairing mismatch in {collection_id}/{split}: "
                    f"label_only={sorted(label_stem_set - source_stem_set)[:3]} "
                    f"source_only={sorted(source_stem_set - label_stem_set)[:3]}",
                )
                for stem in sorted(source_by_stem):
                    _require(
                        set(source_by_stem[stem]) == {"pdf", "pptx"},
                        f"stem {stem!r} lacks a full pdf/pptx pair "
                        f"in {cell['source_archive']['path']}",
                    )

                selected_stems = sorted(json_stems)
                if limit_documents:
                    selected_stems = selected_stems[:limit_documents]
                cell_plans.append(
                    {
                        "split": split,
                        "cell": cell,
                        "label_zip": label_zip,
                        "source_zip": source_zip,
                        "json_stems": json_stems,
                        "source_by_stem": source_by_stem,
                        "selected_stems": selected_stems,
                        "png_members": png_members,
                    }
                )

            stem_to_plan: dict[str, dict[str, Any]] = {}
            for plan in cell_plans:
                for stem in plan["selected_stems"]:
                    _require(
                        stem not in stem_to_plan,
                        f"duplicate json stem across cells in {collection_id}: {stem!r}",
                    )
                    _require(
                        stem not in global_stems,
                        f"json stem {stem!r} already used by {global_stems.get(stem)}",
                    )
                    stem_to_plan[stem] = plan

            artifact_paths = {
                name: collection_dir / f"{name}.jsonl" for name in ARTIFACT_NAMES
            }
            counts = {
                "documents": 0,
                "chunks": 0,
                "elements": 0,
                "alignment": 0,
                "image_records_excluded": 0,
            }
            cell_counts = {
                plan["split"]: {
                    "json_documents_in_cell": len(plan["json_stems"]),
                    "selected_documents": len(plan["selected_stems"]),
                    "converted_documents": 0,
                    "excluded_documents": 0,
                    "failed_documents": 0,
                    "elements_retained": 0,
                    "chunks": 0,
                    "image_records_excluded": 0,
                    "label_png_members_ignored": plan["png_members"],
                }
                for plan in cell_plans
            }
            cell_chunk_lengths: dict[str, list[int]] = {
                plan["split"]: [] for plan in cell_plans
            }
            cell_oversized_elements: dict[str, list[str]] = {
                plan["split"]: [] for plan in cell_plans
            }

            streams = {
                name: path.open("w", encoding="utf-8")
                for name, path in artifact_paths.items()
            }
            for stem in sorted(stem_to_plan):
                plan = stem_to_plan[stem]
                split = plan["split"]
                cell = plan["cell"]
                label_raw_name = plan["json_stems"][stem]
                label_bytes = _read_member(plan["label_zip"], label_raw_name)
                counters["bytes_read"] += len(label_bytes)
                label = parse_label_document(label_bytes, f"{stem}.json")

                source_members_provenance = []
                for extension in ("pdf", "pptx"):
                    raw_name = plan["source_by_stem"][stem][extension]
                    if hash_source_members:
                        size, digest = _hash_member(plan["source_zip"], raw_name)
                        counters["bytes_read"] += size
                    else:
                        size = plan["source_zip"].getinfo(raw_name).file_size
                        digest = None
                    source_members_provenance.append(
                        {
                            "archive": cell["source_archive"]["path"],
                            "member": f"{stem}.{extension}",
                            "bytes": size,
                            "sha256": digest,
                        }
                    )
                provenance = {
                    "label_archive": cell["label_archive"]["path"],
                    "label_member": f"{stem}.json",
                    "label_member_bytes": len(label_bytes),
                    "label_member_sha256": sha256_bytes(label_bytes),
                    "source_members": source_members_provenance,
                }

                bundle = build_document_bundle(
                    collection_id=collection_id,
                    split=split,
                    json_stem=stem,
                    label=label,
                    separator=separator,
                    max_chars=max_chars,
                    provenance=provenance,
                )
                content_hash = bundle["document"]["content_sha256"]
                _require(
                    content_hash not in global_content_hashes,
                    "duplicate document content: "
                    f"{bundle['document']['document_id']} matches "
                    f"{global_content_hashes.get(content_hash)}",
                )
                global_content_hashes[content_hash] = bundle["document"]["document_id"]
                global_stems[stem] = collection_id

                _write_jsonl_row(streams["documents"], bundle["document"])
                for element in bundle["elements"]:
                    _write_jsonl_row(streams["elements"], element)
                for chunk in bundle["chunks"]:
                    _write_jsonl_row(streams["chunks"], chunk)
                for record in bundle["alignment"]:
                    _write_jsonl_row(streams["alignment"], record)

                counts["documents"] += 1
                counts["elements"] += len(bundle["elements"])
                counts["chunks"] += len(bundle["chunks"])
                counts["alignment"] += len(bundle["alignment"])
                counts["image_records_excluded"] += bundle["image_records_excluded"]
                for meta in bundle["chunk_meta"]:
                    cell_chunk_lengths[split].append(meta["length"])
                    if meta["length"] > max_chars:
                        cell_oversized_elements[split].extend(meta["element_ids"])
                cell_counts[split]["converted_documents"] += 1
                cell_counts[split]["elements_retained"] += len(bundle["elements"])
                cell_counts[split]["chunks"] += len(bundle["chunks"])
                cell_counts[split]["image_records_excluded"] += (
                    bundle["image_records_excluded"]
                )
                if progress and counts["documents"] % 500 == 0:
                    progress(
                        f"{collection_id}: {counts['documents']} documents converted"
                    )
        finally:
            for stream in streams.values():
                stream.close()
            for archive in open_archives:
                archive.close()

        artifact_entries = {
            name: _artifact_entry(path, counts[name], output_dir)
            for name, path in artifact_paths.items()
        }
        counters["bytes_written"] += sum(
            entry["bytes"] for entry in artifact_entries.values()
        )
        all_lengths = [
            length
            for lengths in cell_chunk_lengths.values()
            for length in lengths
        ]
        all_oversized = [
            element_id
            for elements_over in cell_oversized_elements.values()
            for element_id in elements_over
        ]
        collections_manifest[collection_id] = {
            "artifacts": artifact_entries,
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "elements": counts["elements"],
            "alignment_records": counts["alignment"],
            "image_records_excluded": counts["image_records_excluded"],
            "source_location": {
                "verified": 0,
                "unavailable": counts["elements"] + counts["chunks"],
            },
            "chunk_length_stats": chunk_length_distribution(
                all_lengths,
                max_chars=max_chars,
                oversized_element_ids=sorted(all_oversized),
            ),
        }
        for split, values in cell_counts.items():
            values["chunk_length_stats"] = chunk_length_distribution(
                cell_chunk_lengths[split],
                max_chars=max_chars,
                oversized_element_ids=sorted(cell_oversized_elements[split]),
            )
            cells_manifest[f"{collection_id}/{split}"] = values

    exclusions = {
        "image_record_excluded_pending_provenance": sum(
            cell["image_records_excluded"] for cell in cells_manifest.values()
        ),
        "reason": (
            "image_info records (image-only and caption-backed) are excluded "
            "until source provenance is verified; captions/categories are "
            "never used as document content"
        ),
    }
    manifest = {
        "schema_version": CONVERSION_MANIFEST_SCHEMA_VERSION,
        "stage": IMPLEMENTATION_STAGE,
        "implementation": IMPLEMENTATION_DECISION,
        "dataset_name": source_config.get("dataset_name"),
        "eda": {
            "handoff_manifest_sha256": source_config["eda"]["handoff_manifest_sha256"],
            "decision": source_config["eda"]["decision"],
        },
        "content_field": CONTENT_FIELD,
        "forbidden_content_fields": list(FORBIDDEN_CONTENT_FIELDS),
        "constructed_text_policy": {
            "policy_id": CONSTRUCTED_TEXT_POLICY_ID,
            "separator": separator,
            "order": "section_info array order",
        },
        "chunking_policy": {
            "policy_id": chunking_config["policy_id"],
            "separator": separator,
            "max_chars": max_chars,
            "element_atomic": chunking_config["element_atomic"],
        },
        "source_location_policy": dict(UNAVAILABLE_LOCATION),
        "selection": {
            "collections": selected_collections,
            "splits": selected_splits,
            "limit_documents_per_cell": limit_documents,
        },
        "options": {
            "verify_archive_sha256": verify_archive_sha256,
            "hash_source_members": hash_source_members,
        },
        "retrieval_compatibility": {
            "status": "retrieval_unapproved",
            "reason": (
                "embedding tokenizer/revision and input-token budget are not "
                "frozen; the atomic-element policy keeps oversized elements as "
                "single oversized chunks"
            ),
            "note": (
                "exact one-to-one element-chunk mapping does not prove "
                "retrieval compatibility"
            ),
            "oversized_chunks_total": sum(
                collection["chunk_length_stats"].get("oversized_count", 0)
                for collection in collections_manifest.values()
            ),
            "validator": "scripts/validate_chunk_model_compatibility.py",
        },
        "archives": archives_manifest,
        "cells": cells_manifest,
        "collections": collections_manifest,
        "exclusions": exclusions,
        "totals": {
            "documents": sum(c["documents"] for c in collections_manifest.values()),
            "chunks": sum(c["chunks"] for c in collections_manifest.values()),
            "elements": sum(c["elements"] for c in collections_manifest.values()),
            "alignment_records": sum(
                c["alignment_records"] for c in collections_manifest.values()
            ),
            "image_records_excluded": sum(
                c["image_records_excluded"] for c in collections_manifest.values()
            ),
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Output verification (independent re-read of converted artifacts)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    _require(path.is_file(), f"missing artifact: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConversionError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            _require(
                isinstance(value, dict), f"{path}:{line_number}: record must be an object"
            )
            yield value


class _Peekable:
    """One-record lookahead over a JSONL stream (bounded-memory merge join)."""

    def __init__(self, iterator: Iterator[dict[str, Any]]):
        self._iterator = iterator
        self._buffer: dict[str, Any] | None = None
        self._exhausted = False

    def peek(self) -> dict[str, Any] | None:
        if self._buffer is None and not self._exhausted:
            self._buffer = next(self._iterator, None)
            if self._buffer is None:
                self._exhausted = True
        return self._buffer

    def take(self) -> dict[str, Any]:
        record = self.peek()
        if record is None:
            raise StopIteration
        self._buffer = None
        return record


def _validate_range(value: Any, name: str) -> tuple[int, int]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(value.get("status") == "verified", f"{name}.status must be verified")
    _require(
        value.get("basis") == DOCUMENT_RANGE_BASIS,
        f"{name}.basis must be {DOCUMENT_RANGE_BASIS}",
    )
    _require(
        value.get("method") == DOCUMENT_RANGE_METHOD,
        f"{name}.method must be {DOCUMENT_RANGE_METHOD}",
    )
    start, end = value.get("char_start"), value.get("char_end")
    _require(
        isinstance(start, int) and isinstance(end, int) and 0 <= start < end,
        f"{name} offsets are invalid",
    )
    return start, end


def _validate_alignment_record(record: dict[str, Any], collection_id: str) -> None:
    _require(
        record.get("schema_version") == ALIGNMENT_SCHEMA_VERSION,
        "alignment schema_version mismatch",
    )
    _require(
        record.get("collection_id") == collection_id,
        "alignment collection mismatch",
    )
    for field in ("element_id", "document_id"):
        value = record.get(field)
        _require(
            isinstance(value, str) and value.startswith(f"{collection_id}::"),
            f"alignment {field} must start with '{collection_id}::'",
        )
    _require(
        isinstance(record.get("ordinal"), int) and record["ordinal"] >= 0,
        "alignment ordinal must be non-negative",
    )
    _validate_range(record.get("document_range"), "alignment.document_range")
    _require(
        record.get("source_location") == UNAVAILABLE_LOCATION,
        "alignment source_location must be explicitly unavailable",
    )
    memberships = record.get("chunk_memberships")
    _require(
        isinstance(memberships, list) and bool(memberships),
        "alignment.chunk_memberships must be non-empty",
    )
    for index, membership in enumerate(memberships):
        _require(
            isinstance(membership, dict),
            f"alignment.chunk_memberships[{index}] must be an object",
        )
        chunk_id = membership.get("chunk_id")
        _require(
            isinstance(chunk_id, str) and chunk_id.startswith(f"{collection_id}::"),
            f"alignment.chunk_memberships[{index}].chunk_id must be namespaced",
        )
        start, end = membership.get("char_start"), membership.get("char_end")
        _require(
            isinstance(start, int) and isinstance(end, int) and 0 <= start < end,
            f"alignment.chunk_memberships[{index}] offsets are invalid",
        )


def _take_document_rows(
    stream: _Peekable, document_id: str, kind: str
) -> list[dict[str, Any]]:
    """Consume the current document's rows from a doc-sorted stream."""
    rows: list[dict[str, Any]] = []
    while True:
        record = stream.peek()
        if record is None:
            break
        row_document = record.get("document_id")
        _require(
            isinstance(row_document, str) and bool(row_document),
            f"{kind} record lacks document_id",
        )
        if row_document == document_id:
            rows.append(stream.take())
        elif row_document < document_id:
            _fail(
                f"{kind} references an absent document or violates "
                f"deterministic order: {row_document}"
            )
        else:
            break
    return rows


def _verify_document_group(
    *,
    collection_id: str,
    document: dict[str, Any],
    elements: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    separator: str,
    element_atomic: bool,
) -> None:
    document_id = document["document_id"]
    text = document["text"]

    element_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(elements):
        element_id, _ = _validate_element(record, collection_id)
        _require(
            element_id.startswith(f"{document_id}::"),
            f"element {element_id} is not namespaced under {document_id}",
        )
        suffix = element_id[len(document_id) + 2 :]
        _require(
            "::" not in suffix,
            f"element suffix collides with chunk namespace: {element_id}",
        )
        _require(
            element_id not in element_by_id,
            f"duplicate element_id: {element_id}",
        )
        _require(
            record["ordinal"] == index,
            f"element ordinals are not contiguous for {document_id}",
        )
        _require(
            record.get("source_location") == UNAVAILABLE_LOCATION,
            "source_location must be the explicit unavailable object",
        )
        element_by_id[element_id] = record

    chunk_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(chunks):
        chunk_id, _ = _validate_chunk(record, collection_id)
        _require(
            chunk_id == f"{document_id}{_CHUNK_ID_MARKER}{index:04d}",
            f"chunks are not in deterministic order for {document_id}: {chunk_id}",
        )
        _require(
            record.get("source_location") == UNAVAILABLE_LOCATION,
            "source_location must be the explicit unavailable object",
        )
        chunk_by_id[chunk_id] = record

    alignment_by_element: dict[str, dict[str, Any]] = {}
    chunk_member_elements: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in alignment:
        _validate_alignment_record(record, collection_id)
        element_id = record["element_id"]
        _require(
            element_id in element_by_id,
            f"alignment references absent element: {element_id}",
        )
        _require(
            element_id not in alignment_by_element,
            f"duplicate alignment record for element: {element_id}",
        )
        element = element_by_id[element_id]
        _require(
            record["ordinal"] == element["ordinal"],
            f"alignment ordinal mismatch for {element_id}",
        )
        _require(
            len(record["chunk_memberships"]) == 1 or not element_atomic,
            f"element unexpectedly assigned to multiple chunks: {element_id}",
        )
        start, end = _validate_range(
            record["document_range"], f"{element_id}.document_range"
        )
        _require(end <= len(text), f"document_range exceeds text: {element_id}")
        _require(
            text[start:end] == element["text"],
            f"document_range slice does not equal element text: {element_id}",
        )
        for membership in record["chunk_memberships"]:
            chunk_id = membership["chunk_id"]
            chunk = chunk_by_id.get(chunk_id)
            _require(
                chunk is not None,
                f"alignment references absent chunk: {chunk_id}",
            )
            piece = chunk["text"][membership["char_start"] : membership["char_end"]]
            _require(
                piece == element["text"],
                f"chunk membership slice does not equal element text: {element_id}",
            )
            chunk_member_elements[chunk_id].append(
                {
                    "element_id": element_id,
                    "ordinal": element["ordinal"],
                    "char_start": membership["char_start"],
                    "char_end": membership["char_end"],
                }
            )
        alignment_by_element[element_id] = record

    missing_alignment = set(element_by_id) - set(alignment_by_element)
    _require(
        not missing_alignment,
        "elements without any chunk/document alignment: "
        f"{sorted(missing_alignment)[:3]}",
    )

    cursor = 0
    for index, record in enumerate(elements):
        aligned = alignment_by_element[record["element_id"]]
        start = aligned["document_range"]["char_start"]
        end = aligned["document_range"]["char_end"]
        expected_start = cursor if index == 0 else cursor + len(separator)
        _require(
            start == expected_start,
            f"dropped/duplicated/reordered text before {record['element_id']}",
        )
        _require(
            text[start:end] == record["text"],
            f"document_range slice does not equal element text: {record['element_id']}",
        )
        cursor = end
    _require(cursor == len(text), f"constructed text tail lost for {document_id}")

    _require(
        set(chunk_member_elements) == set(chunk_by_id),
        f"chunk without any element in {document_id}",
    )
    for chunk_id, members in chunk_member_elements.items():
        ordered = sorted(members, key=lambda member: member["ordinal"])
        chunk_text = chunk_by_id[chunk_id]["text"]
        chunk_cursor = 0
        for index, member in enumerate(ordered):
            expected_start = chunk_cursor if index == 0 else chunk_cursor + len(separator)
            _require(
                member["char_start"] == expected_start,
                f"chunk {chunk_id} has a gap/overlap before {member['element_id']}",
            )
            chunk_cursor = member["char_end"]
        _require(
            chunk_cursor == len(chunk_text),
            f"chunk {chunk_id} tail is not covered by its elements",
        )


def verify_collection_outputs(
    output_dir: Path,
    collection_id: str,
    *,
    separator: str,
    element_atomic: bool = True,
) -> dict[str, Any]:
    """Streaming re-read of one collection's artifacts; fails loud on any defect.

    Bounded-memory strategy: all four artifacts are written in the same
    deterministic per-document order, so a four-way merge join verifies every
    invariant one document at a time. Time is O(T) over output bytes; peak
    memory is one document's records plus one content-hash entry per document
    (O(D)), never total corpus text. No disk-backed temporary index is
    required, so there is nothing to clean up on failure. Invariants checked:
    missing/duplicate IDs, reference errors, offset errors, dropped/reordered
    text, uncovered chunks/elements, content duplication, and order drift.
    """
    collection_dir = Path(output_dir) / collection_id
    documents = _Peekable(_iter_jsonl(collection_dir / "documents.jsonl"))
    elements = _Peekable(_iter_jsonl(collection_dir / "elements.jsonl"))
    chunks = _Peekable(_iter_jsonl(collection_dir / "chunks.jsonl"))
    alignment = _Peekable(_iter_jsonl(collection_dir / "alignment.jsonl"))

    content_hash_owner: dict[str, str] = {}
    previous_document_id: str | None = None
    counts = {"documents": 0, "elements": 0, "chunks": 0, "alignment": 0}

    while documents.peek() is not None:
        record = documents.take()
        document_id = _validate_document(record, collection_id)
        if previous_document_id is not None:
            _require(
                document_id != previous_document_id, "duplicate document_id"
            )
            _require(
                document_id > previous_document_id,
                f"documents are not in deterministic order: {document_id}",
            )
        previous_document_id = document_id
        digest = record["content_sha256"]
        _require(
            digest not in content_hash_owner,
            f"duplicate document content: {document_id} matches "
            f"{content_hash_owner.get(digest)}",
        )
        content_hash_owner[digest] = document_id
        metadata = record.get("metadata")
        _require(isinstance(metadata, dict), "document metadata missing")
        _require(
            metadata.get("split") in REQUIRED_SPLITS,
            f"document split must be one of {list(REQUIRED_SPLITS)}",
        )
        _require(
            document_id == f"{collection_id}::{metadata.get('json_stem')}",
            "document_id must be collection::json_stem",
        )

        document_elements = _take_document_rows(elements, document_id, "element")
        document_chunks = _take_document_rows(chunks, document_id, "chunk")
        document_alignment = _take_document_rows(alignment, document_id, "alignment")
        _verify_document_group(
            collection_id=collection_id,
            document=record,
            elements=document_elements,
            chunks=document_chunks,
            alignment=document_alignment,
            separator=separator,
            element_atomic=element_atomic,
        )
        counts["documents"] += 1
        counts["elements"] += len(document_elements)
        counts["chunks"] += len(document_chunks)
        counts["alignment"] += len(document_alignment)

    for stream, kind in ((elements, "element"), (chunks, "chunk"), (alignment, "alignment")):
        leftover = stream.peek()
        _require(
            leftover is None,
            f"{kind} references an absent document: {leftover.get('document_id') if leftover else ''}",
        )

    return {
        "status": "ok",
        "collection_id": collection_id,
        "documents": counts["documents"],
        "chunks": counts["chunks"],
        "elements": counts["elements"],
        "alignment_records": counts["alignment"],
        "element_to_document_coverage": 1.0 if counts["elements"] else 0.0,
        "element_to_chunk_coverage": 1.0 if counts["elements"] else 0.0,
        "source_location": {
            "verified": 0,
            "unavailable": counts["elements"] + counts["chunks"],
        },
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    path = Path(output_dir) / MANIFEST_FILE_NAME
    persisted = {
        key: value for key, value in manifest.items() if key != "publication_status"
    }
    path.write_text(
        json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Acceptance gate (machine-enforced; smoke/partial runs stay ineligible)
# ---------------------------------------------------------------------------

def evaluate_acceptance(
    manifest: dict[str, Any], acceptance_contract: dict[str, Any] | None
) -> dict[str, Any]:
    """Machine-readable acceptance decision against the reviewed EDA contract."""
    if acceptance_contract is None:
        return {
            "eligible": False,
            "reasons": ["acceptance contract not provided"],
            "contract_sha256": None,
        }
    reasons: list[str] = []
    selection = manifest["selection"]
    expected_collections = sorted(acceptance_contract["collections"])
    expected_splits = sorted(acceptance_contract["splits"])
    if selection["collections"] != expected_collections:
        reasons.append(
            f"collections {selection['collections']} != contract {expected_collections}"
        )
    if selection["splits"] != expected_splits:
        reasons.append(f"splits {selection['splits']} != contract {expected_splits}")
    if selection["limit_documents_per_cell"] != 0:
        reasons.append(
            f"limit_documents_per_cell={selection['limit_documents_per_cell']} (must be 0)"
        )
    options = manifest["options"]
    if not options.get("verify_archive_sha256"):
        reasons.append("archive SHA-256 verification was disabled")
    if not options.get("hash_source_members"):
        reasons.append("source pdf/pptx member hashing was disabled")
    archives = manifest.get("archives", [])
    if len(archives) != acceptance_contract["archive_count"]:
        reasons.append(
            f"archive count {len(archives)} != {acceptance_contract['archive_count']}"
        )
    unverified = sorted(
        archive["path"] for archive in archives if not archive.get("sha256_verified")
    )
    if unverified:
        reasons.append(f"archives without verified SHA-256: {unverified[:3]}")
    verification = manifest.get("verification") or {}
    for collection_id in expected_collections:
        if verification.get(collection_id, {}).get("status") != "ok":
            reasons.append(f"verification missing or not ok: {collection_id}")
    for cell_name in sorted(acceptance_contract["cells"]):
        expected = acceptance_contract["cells"][cell_name]
        actual = manifest.get("cells", {}).get(cell_name)
        if actual is None:
            reasons.append(f"cell missing: {cell_name}")
            continue
        observed = {
            "documents": actual.get("converted_documents"),
            "elements": actual.get("elements_retained"),
            "images_excluded": actual.get("image_records_excluded"),
        }
        for key, value in observed.items():
            if value != expected[key]:
                reasons.append(f"{cell_name}.{key} {value} != {expected[key]}")
        if actual.get("failed_documents") or actual.get("excluded_documents"):
            reasons.append(f"{cell_name} has failed/excluded documents")
    totals = manifest.get("totals", {})
    contract_totals = acceptance_contract["totals"]
    if totals.get("documents") != contract_totals["documents"]:
        reasons.append(
            f"total documents {totals.get('documents')} != {contract_totals['documents']}"
        )
    if (
        totals.get("elements") != contract_totals["elements"]
        or totals.get("alignment_records") != contract_totals["elements"]
    ):
        reasons.append(
            f"total elements/alignment {totals.get('elements')}/"
            f"{totals.get('alignment_records')} != {contract_totals['elements']}"
        )
    if totals.get("image_records_excluded") != contract_totals["images_excluded"]:
        reasons.append(
            f"total excluded image records {totals.get('image_records_excluded')} "
            f"!= {contract_totals['images_excluded']}"
        )
    run_identity = manifest.get("run_identity") or {}
    if not run_identity.get("identity_sha256"):
        reasons.append("missing reproducible run identity")
    runtime = run_identity.get("runtime") or {}
    if runtime.get("git_commit") in (None, "", "unknown"):
        reasons.append("unknown code identity (git commit unavailable)")
    if runtime.get("git_dirty") is not False:
        reasons.append("dirty or unknown git worktree state")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "contract_sha256": canonical_json_sha256(acceptance_contract),
    }


def _audit_published_manifest(
    target: Path,
    *,
    acceptance_contract: dict[str, Any] | None,
    expected_identity_sha256: str | None = None,
    chunking_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fully re-audit a published target; never trust persisted flags.

    Recomputes the run-identity self-hash, cross-checks manifest selection/
    options against the identity, re-binds the provided acceptance contract,
    re-runs ``evaluate_acceptance`` and requires it to equal the persisted
    decision exactly, then re-hashes every declared artifact and re-runs the
    streaming semantic verifier.
    """
    target = Path(target)
    manifest_path = target / MANIFEST_FILE_NAME
    _require(manifest_path.is_file(), f"conversion manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(
            f"target manifest is unreadable or corrupt: {manifest_path}: {error}"
        ) from error
    _require(isinstance(manifest, dict), f"manifest must be an object: {manifest_path}")
    _require(
        manifest.get("schema_version") == CONVERSION_MANIFEST_SCHEMA_VERSION,
        f"manifest schema_version must be {CONVERSION_MANIFEST_SCHEMA_VERSION}",
    )

    run_identity = manifest.get("run_identity")
    _require(isinstance(run_identity, dict), "manifest run_identity missing")
    identity_core = {
        key: value
        for key, value in run_identity.items()
        if key not in ("identity_sha256", "runtime")
    }
    recomputed_identity = canonical_json_sha256(identity_core)
    _require(
        recomputed_identity == run_identity.get("identity_sha256"),
        "run identity self-hash mismatch: "
        f"recorded={run_identity.get('identity_sha256')} "
        f"recomputed={recomputed_identity}",
    )
    if expected_identity_sha256 is not None and (
        run_identity["identity_sha256"] != expected_identity_sha256
    ):
        raise ConversionError(
            "target holds a different input identity: "
            f"recorded={run_identity['identity_sha256']} "
            f"requested={expected_identity_sha256}; "
            f"refusing to merge or overwrite {target}"
        )
    _require(
        manifest.get("selection") == run_identity.get("selection"),
        "manifest selection does not match run identity",
    )
    _require(
        manifest.get("options") == run_identity.get("options"),
        "manifest options do not match run identity",
    )
    if acceptance_contract is not None:
        _require(
            canonical_json_sha256(acceptance_contract)
            == run_identity.get("acceptance_contract_sha256"),
            "acceptance contract does not match the contract bound at publication",
        )
    recomputed_acceptance = evaluate_acceptance(manifest, acceptance_contract)
    _require(
        recomputed_acceptance == manifest.get("acceptance"),
        "persisted acceptance decision does not match recomputation: "
        f"persisted={manifest.get('acceptance')} recomputed={recomputed_acceptance}",
    )

    policy = manifest.get("chunking_policy") or {}
    if chunking_config is not None:
        _require(
            policy.get("separator") == chunking_config["separator"]
            and policy.get("max_chars") == chunking_config["max_chars"]
            and policy.get("element_atomic") == chunking_config["element_atomic"],
            "manifest chunking policy does not match the provided config",
        )
    separator = policy.get("separator")
    element_atomic = policy.get("element_atomic")
    _require(
        isinstance(separator, str) and separator != "",
        "manifest chunking separator missing",
    )
    _require(isinstance(element_atomic, bool), "manifest element_atomic missing")

    for collection_id in sorted(manifest.get("collections", {})):
        collection = manifest["collections"][collection_id]
        for name in sorted(collection["artifacts"]):
            entry = collection["artifacts"][name]
            path = target / entry["path"]
            _require(path.is_file(), f"declared artifact missing: {path}")
            _require(
                path.stat().st_size == entry["bytes"]
                and sha256_file(path) == entry["sha256"],
                f"declared artifact does not match manifest hash: {path}",
            )
        summary = verify_collection_outputs(
            target,
            collection_id,
            separator=separator,
            element_atomic=element_atomic,
        )
        _require(
            summary["documents"] == collection["artifacts"]["documents"]["records"]
            and summary["elements"] == collection["artifacts"]["elements"]["records"]
            and summary["chunks"] == collection["artifacts"]["chunks"]["records"]
            and summary["alignment_records"]
            == collection["artifacts"]["alignment"]["records"],
            f"published record counts do not match manifest for {collection_id}",
        )
    return manifest


def require_accepted_conversion(
    manifest_path: Path,
    *,
    acceptance_contract: dict[str, Any],
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Gate for the collection_eval stage: reject non-accepted conversions.

    Never trusts the persisted flag: the reviewed acceptance contract is a
    required input, the run identity self-hash and manifest consistency are
    recomputed, every declared artifact is re-hashed, the semantic verifier
    is re-run, and ``evaluate_acceptance`` is recomputed and must equal the
    persisted decision exactly. Callers should additionally pin
    ``expected_manifest_sha256`` so a consistently regenerated forgery is
    also rejected.
    """
    _require(
        isinstance(acceptance_contract, dict) and bool(acceptance_contract),
        "a reviewed acceptance contract is required",
    )
    manifest_path = Path(manifest_path)
    _require(manifest_path.is_file(), f"conversion manifest missing: {manifest_path}")
    if expected_manifest_sha256 is not None:
        actual = sha256_file(manifest_path)
        _require(
            actual == expected_manifest_sha256,
            "conversion manifest does not match the pinned SHA-256: "
            f"expected={expected_manifest_sha256} actual={actual}",
        )
    manifest = _audit_published_manifest(
        manifest_path.parent, acceptance_contract=acceptance_contract
    )
    acceptance = manifest.get("acceptance") or {}
    _require(
        acceptance.get("eligible") is True,
        "conversion is not acceptance-eligible: "
        f"{acceptance.get('reasons', ['no acceptance record'])}",
    )
    return manifest


# ---------------------------------------------------------------------------
# Atomic conversion orchestration (lock -> stage -> verify -> publish)
# ---------------------------------------------------------------------------

def _reuse_verified_target(
    target: Path,
    identity: dict[str, Any],
    chunking_config: dict[str, Any],
    acceptance_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Re-audit a complete same-identity target with the full gate rigor.

    Applies the same checks as ``require_accepted_conversion`` (identity
    self-hash, selection/options consistency, contract re-binding,
    acceptance recomputation vs the persisted decision, artifact re-hash,
    semantic re-verification) plus the requested-identity equality; only the
    eligibility requirement is omitted because smoke targets may be reused.
    """
    return _audit_published_manifest(
        target,
        acceptance_contract=acceptance_contract,
        expected_identity_sha256=identity["identity_sha256"],
        chunking_config=chunking_config,
    )


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def convert_collections(
    *,
    data_root: Path,
    source_config: dict[str, Any],
    chunking_config: dict[str, Any],
    output_dir: Path,
    collections: list[str] | None = None,
    splits: list[str] | None = None,
    limit_documents: int = 0,
    verify_archive_sha256: bool = True,
    hash_source_members: bool = True,
    runtime_identity: dict[str, Any] | None = None,
    acceptance_contract: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Convert with lock-protected staging and atomic publication.

    Deterministic output vs operational idempotency: the artifacts are
    byte-deterministic for a given input identity, and this wrapper adds the
    operational guarantees — an exclusive target lock, staging on the same
    filesystem, verification before an atomic rename, no partial targets on
    failure, in-place reuse of an identical published target, and a loud
    failure for different-identity, partial, or corrupt targets.

    The returned manifest carries a non-persisted ``publication_status`` key
    (``published`` or ``reused``). ``run_metrics.json`` (timings, peak RSS,
    byte counters) is operational telemetry and is excluded from the
    deterministic-output guarantee.
    """
    target = Path(output_dir)
    ensure_output_dir_outside_git(target, code_base=_MODULE_BASE)
    selected_collections, selected_splits = _normalize_selection(
        source_config, collections, splits, limit_documents
    )
    selection = {
        "collections": selected_collections,
        "splits": selected_splits,
        "limit_documents_per_cell": limit_documents,
    }
    options = {
        "verify_archive_sha256": verify_archive_sha256,
        "hash_source_members": hash_source_members,
    }
    identity = compute_run_identity(
        source_config=source_config,
        chunking_config=chunking_config,
        selection=selection,
        options=options,
        acceptance_contract=acceptance_contract,
    )
    runtime = runtime_identity if runtime_identity is not None else collect_runtime_identity()

    lock = TargetLock(target).acquire()
    try:
        state = target_state(target, MANIFEST_FILE_NAME)
        if state == "partial":
            raise ConversionError(
                f"target is incomplete or corrupt (missing {MANIFEST_FILE_NAME}); "
                f"refusing to merge, overwrite, or repair: {target}"
            )
        if state == "complete":
            manifest = _reuse_verified_target(
                target, identity, chunking_config, acceptance_contract
            )
            manifest["publication_status"] = "reused"
            return manifest
        if state == "empty":
            target.rmdir()

        staging = prepare_staging(target)
        timings: dict[str, float] = {}
        counters: dict[str, int] = {}
        try:
            started = time.monotonic()
            manifest = _build_collections(
                data_root=data_root,
                source_config=source_config,
                chunking_config=chunking_config,
                output_dir=staging,
                collections=selected_collections,
                splits=selected_splits,
                limit_documents=limit_documents,
                verify_archive_sha256=verify_archive_sha256,
                hash_source_members=hash_source_members,
                counters=counters,
                progress=progress,
            )
            timings["convert_seconds"] = round(time.monotonic() - started, 3)
            manifest["run_identity"] = {**identity, "runtime": runtime}

            started = time.monotonic()
            verification = {}
            for collection_id in selected_collections:
                summary = verify_collection_outputs(
                    staging,
                    collection_id,
                    separator=chunking_config["separator"],
                    element_atomic=chunking_config["element_atomic"],
                )
                artifacts = manifest["collections"][collection_id]["artifacts"]
                _require(
                    summary["documents"] == artifacts["documents"]["records"]
                    and summary["elements"] == artifacts["elements"]["records"]
                    and summary["chunks"] == artifacts["chunks"]["records"]
                    and summary["alignment_records"]
                    == artifacts["alignment"]["records"],
                    f"verification counts do not match artifact records for "
                    f"{collection_id}",
                )
                verification[collection_id] = summary
            timings["verify_seconds"] = round(time.monotonic() - started, 3)

            manifest["verification"] = verification
            manifest["acceptance"] = evaluate_acceptance(manifest, acceptance_contract)
            manifest_path = write_manifest(manifest, staging)
            counters["bytes_written"] += manifest_path.stat().st_size
            metrics = {
                "schema_version": RUN_METRICS_SCHEMA_VERSION,
                "note": (
                    "operational telemetry; excluded from the deterministic "
                    "output guarantee"
                ),
                "timings_seconds": timings,
                "peak_rss_bytes": _peak_rss_bytes(),
                "bytes_read": counters["bytes_read"],
                "bytes_written": counters["bytes_written"],
            }
            (staging / METRICS_FILE_NAME).write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish_staging(staging, target)
        except BaseException:
            discard_staging(target)
            raise
        manifest["publication_status"] = "published"
        return manifest
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Chunk / model-input compatibility gate (no model call; contract-driven)
# ---------------------------------------------------------------------------

def load_compat_contract(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    _require(isinstance(contract, dict), "compat contract must be a mapping")
    _require(
        contract.get("schema_version") == COMPAT_CONTRACT_SCHEMA_VERSION,
        f"compat contract schema_version must be {COMPAT_CONTRACT_SCHEMA_VERSION}",
    )
    _nonempty_str(contract.get("model_id"), "compat contract model_id")
    revision = _nonempty_str(contract.get("revision"), "compat contract revision")
    _require(
        revision.lower() not in {"main", "master", "latest", "unknown"},
        "compat contract revision must be immutable",
    )
    _require(
        isinstance(contract.get("max_input_chars"), int)
        and contract["max_input_chars"] > 0,
        "compat contract max_input_chars must be a positive integer",
    )
    policy = contract.get("approved_long_element_split_policy")
    if policy is not None:
        _require(isinstance(policy, dict), "approved split policy must be a mapping")
        _nonempty_str(policy.get("policy_id"), "approved split policy policy_id")
        _nonempty_str(policy.get("approved_by"), "approved split policy approved_by")
        _nonempty_str(policy.get("path"), "approved split policy path")
        _require(
            isinstance(policy.get("bytes"), int) and policy["bytes"] > 0,
            "approved split policy bytes must be a positive integer",
        )
        digest = policy.get("sha256")
        _require(
            isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest)),
            "approved split policy sha256 must be lowercase SHA-256",
        )
    return contract


def _verify_split_policy_file(
    policy: dict[str, Any], contract_dir: Path | None
) -> Path:
    policy_path = Path(policy["path"])
    if not policy_path.is_absolute():
        _require(
            contract_dir is not None,
            "relative split policy path requires the contract directory",
        )
        policy_path = Path(contract_dir) / policy_path
    _require(policy_path.is_file(), f"approved split policy file missing: {policy_path}")
    actual_bytes = policy_path.stat().st_size
    _require(
        actual_bytes == policy["bytes"],
        f"split policy bytes mismatch: {policy_path} "
        f"expected={policy['bytes']} actual={actual_bytes}",
    )
    actual_sha = sha256_file(policy_path)
    _require(
        actual_sha == policy["sha256"],
        f"split policy sha256 mismatch: {policy_path} "
        f"expected={policy['sha256']} actual={actual_sha}",
    )
    return policy_path


def validate_chunk_model_compatibility(
    chunks_path: Path,
    contract: dict[str, Any],
    *,
    contract_dir: Path | None = None,
) -> dict[str, Any]:
    """Prove every chunk fits the frozen model input contract, or fail loud.

    ``compatible`` requires zero violations. An owner-approved deterministic
    long-element split policy (whose file bytes/SHA-256 are verified) does
    NOT make an oversized corpus usable: the result is ``requires_rebuild``
    — the split corpus must be generated, its alignment updated, and this
    validation re-run before any retrieval use.
    """
    budget = contract["max_input_chars"]
    total = 0
    violations = 0
    max_length = 0
    for record in _iter_jsonl(Path(chunks_path)):
        length = len(str(record.get("text", "")))
        total += 1
        max_length = max(max_length, length)
        if length > budget:
            violations += 1
    result = {
        "chunks": total,
        "max_chunk_chars": max_length,
        "max_input_chars": budget,
        "violations": violations,
        "model_id": contract["model_id"],
        "revision": contract["revision"],
    }
    if violations == 0:
        result["status"] = "compatible"
        return result
    policy = contract.get("approved_long_element_split_policy")
    if policy:
        _verify_split_policy_file(policy, contract_dir)
        result["status"] = "requires_rebuild"
        result["policy_id"] = policy["policy_id"]
        result["approved_by"] = policy["approved_by"]
        result["policy_sha256"] = policy["sha256"]
        result["note"] = (
            "approved split policy recorded; corpus is NOT usable until the "
            "split corpus is rebuilt, alignment is regenerated, and this "
            "validation passes with zero violations"
        )
        return result
    raise ConversionError(
        f"{violations} of {total} chunks exceed the frozen model input budget "
        f"({budget} chars, max observed {max_length}); freeze a compatible "
        "contract or record a separately approved deterministic long-element "
        "split policy"
    )


# ---------------------------------------------------------------------------
# Retrieval approval attestation (frozen tokenizer; never a model/API call)
# ---------------------------------------------------------------------------

RETRIEVAL_APPROVAL_SCHEMA_VERSION = "academic.retrieval-approval-attestation.v1"


def _token_count(tokenizer: Callable[[str], Any], text: str) -> int:
    """Count tokens with a locally loaded frozen tokenizer callable."""
    value = tokenizer(text)
    if isinstance(value, int):
        count = value
    else:
        try:
            count = len(value)
        except TypeError as error:
            raise ConversionError(
                "tokenizer must return an int or a sized token sequence"
            ) from error
    _require(count >= 0, "tokenizer returned a negative count")
    return count


def _render_input_template(template: str, text: str) -> str:
    _require(
        isinstance(template, str) and "{text}" in template,
        "input_template must contain the literal {text} placeholder",
    )
    return template.replace("{text}", text)


def _scan_chunk_tokens(
    target: Path,
    manifest: dict[str, Any],
    *,
    tokenizer: Callable[[str], Any],
    input_template: str,
    token_budget: int,
) -> dict[str, Any]:
    total = 0
    violations = 0
    max_tokens = 0
    for collection_id in sorted(manifest["collections"]):
        entry = manifest["collections"][collection_id]["artifacts"]["chunks"]
        for record in _iter_jsonl(target / entry["path"]):
            rendered = _render_input_template(input_template, str(record.get("text", "")))
            count = _token_count(tokenizer, rendered)
            total += 1
            max_tokens = max(max_tokens, count)
            if count > token_budget:
                violations += 1
    return {
        "chunks_total": total,
        "max_tokens_observed": max_tokens,
        "token_violations": violations,
    }


def build_retrieval_approval_attestation(
    *,
    target_dir: Path,
    acceptance_contract: dict[str, Any],
    model_id: str,
    revision: str,
    token_budget: int,
    input_template: str,
    tokenizer: Callable[[str], Any],
    approved_by: str,
) -> dict[str, Any]:
    """Build the attestation binding corpus, contract, tokenizer, and result.

    The corpus must first pass ``require_accepted_conversion``. Token counts
    come from the caller-supplied frozen tokenizer callable (loaded locally
    from the immutable revision); no model or API is called. The result is
    ``approved`` only when every rendered chunk fits the token budget.
    """
    target = Path(target_dir)
    _nonempty_str(model_id, "model_id")
    revision = _nonempty_str(revision, "revision")
    _require(
        revision.lower() not in {"main", "master", "latest", "unknown"},
        "revision must be immutable",
    )
    _require(
        isinstance(token_budget, int) and token_budget > 0,
        "token_budget must be a positive integer",
    )
    _nonempty_str(approved_by, "approved_by")
    manifest = require_accepted_conversion(
        target / MANIFEST_FILE_NAME, acceptance_contract=acceptance_contract
    )
    scan = _scan_chunk_tokens(
        target,
        manifest,
        tokenizer=tokenizer,
        input_template=input_template,
        token_budget=token_budget,
    )
    chunks_entries = {
        collection_id: dict(manifest["collections"][collection_id]["artifacts"]["chunks"])
        for collection_id in sorted(manifest["collections"])
    }
    attestation = {
        "schema_version": RETRIEVAL_APPROVAL_SCHEMA_VERSION,
        "approved_by": approved_by,
        "conversion_manifest_sha256": sha256_file(target / MANIFEST_FILE_NAME),
        "run_identity_sha256": manifest["run_identity"]["identity_sha256"],
        "acceptance_contract_sha256": canonical_json_sha256(acceptance_contract),
        "chunks": chunks_entries,
        "model_id": model_id,
        "revision": revision,
        "token_budget": token_budget,
        "input_template": input_template,
        "input_template_sha256": sha256_text(input_template),
        "code_identity": code_identity_hashes(),
        **scan,
        "result": "approved" if scan["token_violations"] == 0 else "rejected",
    }
    return attestation


def require_retrieval_approved(
    attestation: dict[str, Any] | Path,
    *,
    target_dir: Path,
    acceptance_contract: dict[str, Any],
    tokenizer: Callable[[str], Any],
) -> dict[str, Any]:
    """Fail-loud gate before any retrieval use of the chunk corpus.

    Re-verifies the whole attestation binding: manifest SHA-256, run
    identity, acceptance contract hash, chunk artifact hashes, code identity,
    and — with the caller-loaded frozen tokenizer — the actual token counts
    of every rendered chunk. Approval flags are never trusted on their own.
    """
    if isinstance(attestation, (str, Path)):
        attestation = json.loads(Path(attestation).read_text(encoding="utf-8"))
    _require(isinstance(attestation, dict), "attestation must be an object")
    _require(
        attestation.get("schema_version") == RETRIEVAL_APPROVAL_SCHEMA_VERSION,
        f"attestation schema_version must be {RETRIEVAL_APPROVAL_SCHEMA_VERSION}",
    )
    _require(
        attestation.get("result") == "approved"
        and attestation.get("token_violations") == 0,
        "attestation does not record an approved zero-violation result",
    )
    revision = _nonempty_str(attestation.get("revision"), "attestation revision")
    _require(
        revision.lower() not in {"main", "master", "latest", "unknown"},
        "attestation revision must be immutable",
    )
    token_budget = attestation.get("token_budget")
    _require(
        isinstance(token_budget, int) and token_budget > 0,
        "attestation token_budget must be a positive integer",
    )
    input_template = attestation.get("input_template")
    _require(
        isinstance(input_template, str) and "{text}" in input_template,
        "attestation input_template must contain {text}",
    )
    _require(
        attestation.get("input_template_sha256") == sha256_text(input_template),
        "attestation input_template hash mismatch",
    )
    _require(
        attestation.get("code_identity") == code_identity_hashes(),
        "attestation code identity does not match the current validator code",
    )

    target = Path(target_dir)
    manifest_path = target / MANIFEST_FILE_NAME
    _require(
        sha256_file(manifest_path) == attestation.get("conversion_manifest_sha256"),
        "conversion manifest does not match the attested SHA-256",
    )
    manifest = require_accepted_conversion(
        manifest_path,
        acceptance_contract=acceptance_contract,
        expected_manifest_sha256=attestation["conversion_manifest_sha256"],
    )
    _require(
        manifest["run_identity"]["identity_sha256"]
        == attestation.get("run_identity_sha256"),
        "attested run identity does not match the manifest",
    )
    _require(
        canonical_json_sha256(acceptance_contract)
        == attestation.get("acceptance_contract_sha256"),
        "acceptance contract does not match the attested hash",
    )
    attested_chunks = attestation.get("chunks")
    _require(isinstance(attested_chunks, dict), "attestation chunks missing")
    _require(
        sorted(attested_chunks) == sorted(manifest["collections"]),
        "attested chunk collections do not match the manifest",
    )
    for collection_id in sorted(attested_chunks):
        manifest_entry = manifest["collections"][collection_id]["artifacts"]["chunks"]
        _require(
            attested_chunks[collection_id] == manifest_entry,
            f"attested chunks entry does not match the manifest: {collection_id}",
        )
        path = target / manifest_entry["path"]
        _require(
            sha256_file(path) == manifest_entry["sha256"],
            f"chunk artifact does not match the attested hash: {path}",
        )

    scan = _scan_chunk_tokens(
        target,
        manifest,
        tokenizer=tokenizer,
        input_template=input_template,
        token_budget=token_budget,
    )
    _require(
        scan["token_violations"] == 0,
        f"{scan['token_violations']} chunks exceed the attested token budget "
        f"({token_budget}) under the frozen tokenizer",
    )
    _require(
        scan["chunks_total"] == attestation.get("chunks_total")
        and scan["max_tokens_observed"] == attestation.get("max_tokens_observed"),
        "recomputed token scan does not match the attestation: "
        f"recomputed={scan} attested_total={attestation.get('chunks_total')} "
        f"attested_max={attestation.get('max_tokens_observed')}",
    )
    return attestation
