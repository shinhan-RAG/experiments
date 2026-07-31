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
import re
from typing import Any, Callable
import zipfile

import yaml

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
ALIGNMENT_SCHEMA_VERSION = "shinhan.collection-element-alignment.v1"
CONVERSION_MANIFEST_SCHEMA_VERSION = "shinhan.collection-conversion.v1"
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


def locate_unique_normalized(document_text: str, element_text: str) -> tuple[int, int]:
    """Search-based fallback alignment; fails loud unless exactly one match.

    The builder never uses this (ranges are exact by construction); it exists
    so any search-based alignment path is forced through an ambiguity check.
    """

    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value)

    needle = normalize(element_text).strip()
    _require(bool(needle), "cannot align an empty element text")
    haystack = normalize(document_text)
    matches = [
        match.start() for match in re.finditer(re.escape(needle), haystack)
    ]
    if len(matches) != 1:
        raise AmbiguousAlignmentError(
            f"normalized-text alignment matched {len(matches)} ranges; "
            "exactly one is required"
        )
    return matches[0], matches[0] + len(needle)


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
    code_fingerprint: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Convert selected collections/splits and write artifacts plus a manifest."""
    data_root = Path(data_root)
    _require(data_root.is_dir(), f"data root does not exist: {data_root}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                label = parse_label_document(label_bytes, f"{stem}.json")

                source_members_provenance = []
                for extension in ("pdf", "pptx"):
                    raw_name = plan["source_by_stem"][stem][extension]
                    if hash_source_members:
                        size, digest = _hash_member(plan["source_zip"], raw_name)
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

        collections_manifest[collection_id] = {
            "artifacts": {
                name: _artifact_entry(path, counts[name], output_dir)
                for name, path in artifact_paths.items()
            },
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "elements": counts["elements"],
            "alignment_records": counts["alignment"],
            "image_records_excluded": counts["image_records_excluded"],
            "source_location": {
                "verified": 0,
                "unavailable": counts["elements"] + counts["chunks"],
            },
        }
        for split, values in cell_counts.items():
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
        "code_fingerprint": code_fingerprint or {},
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

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict), f"{path}:{line_number}: record must be an object"
            )
            records.append(value)
    return records


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


def verify_collection_outputs(
    output_dir: Path,
    collection_id: str,
    *,
    separator: str,
    element_atomic: bool = True,
) -> dict[str, Any]:
    """Re-read one collection's artifacts and fail loud on any inconsistency."""
    collection_dir = Path(output_dir) / collection_id
    documents = _load_jsonl(collection_dir / "documents.jsonl")
    elements = _load_jsonl(collection_dir / "elements.jsonl")
    chunks = _load_jsonl(collection_dir / "chunks.jsonl")
    alignment = _load_jsonl(collection_dir / "alignment.jsonl")

    document_ids = [
        _validate_document(record, collection_id) for record in documents
    ]
    chunk_pairs = [_validate_chunk(record, collection_id) for record in chunks]
    element_pairs = [
        _validate_element(record, collection_id) for record in elements
    ]

    _require(len(set(document_ids)) == len(document_ids), "duplicate document_id")
    doc_text = {record["document_id"]: record["text"] for record in documents}
    content_hash_owner: dict[str, str] = {}
    for record in documents:
        digest = record["content_sha256"]
        _require(
            digest not in content_hash_owner,
            f"duplicate document content: {record['document_id']} matches "
            f"{content_hash_owner.get(digest)}",
        )
        content_hash_owner[digest] = record["document_id"]
        metadata = record.get("metadata")
        _require(isinstance(metadata, dict), "document metadata missing")
        _require(
            metadata.get("split") in REQUIRED_SPLITS,
            f"document split must be one of {list(REQUIRED_SPLITS)}",
        )
        _require(
            record["document_id"]
            == f"{collection_id}::{metadata.get('json_stem')}",
            "document_id must be collection::json_stem",
        )

    chunk_ids = [pair[0] for pair in chunk_pairs]
    _require(len(set(chunk_ids)) == len(chunk_ids), "duplicate chunk_id")
    element_ids = [pair[0] for pair in element_pairs]
    _require(len(set(element_ids)) == len(element_ids), "duplicate element_id")
    id_universe = set(document_ids) | set(chunk_ids) | set(element_ids)
    _require(
        len(id_universe) == len(document_ids) + len(chunk_ids) + len(element_ids),
        "document/chunk/element ID namespaces overlap",
    )
    for chunk_id, chunk_document in chunk_pairs:
        _require(chunk_document in doc_text, f"chunk references absent document: {chunk_id}")
    for element_id, element_document in element_pairs:
        _require(
            element_document in doc_text,
            f"element references absent document: {element_id}",
        )

    for record in elements + chunks:
        _require(
            record.get("source_location") == UNAVAILABLE_LOCATION,
            "source_location must be the explicit unavailable object",
        )

    element_by_id = {record["element_id"]: record for record in elements}
    chunk_by_id = {record["chunk_id"]: record for record in chunks}
    elements_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in elements:
        elements_by_document[record["document_id"]].append(record)

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
            record["document_id"] == element["document_id"],
            f"alignment document mismatch for {element_id}",
        )
        _require(
            record["ordinal"] == element["ordinal"],
            f"alignment ordinal mismatch for {element_id}",
        )
        _require(
            len(record["chunk_memberships"]) == 1 or not element_atomic,
            f"element unexpectedly assigned to multiple chunks: {element_id}",
        )
        text = doc_text[element["document_id"]]
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
            _require(
                chunk["document_id"] == element["document_id"],
                f"chunk/element document mismatch: {chunk_id}",
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
        f"elements without any chunk/document alignment: {sorted(missing_alignment)[:3]}",
    )

    for document_id, document_elements in elements_by_document.items():
        ordered = sorted(document_elements, key=lambda record: record["ordinal"])
        _require(
            [record["ordinal"] for record in ordered]
            == list(range(len(ordered))),
            f"element ordinals are not contiguous for {document_id}",
        )
        text = doc_text[document_id]
        cursor = 0
        for index, element in enumerate(ordered):
            record = alignment_by_element[element["element_id"]]
            start = record["document_range"]["char_start"]
            end = record["document_range"]["char_end"]
            expected_start = cursor if index == 0 else cursor + len(separator)
            _require(
                start == expected_start,
                f"dropped/duplicated/reordered text before {element['element_id']}",
            )
            cursor = end
        _require(
            cursor == len(text),
            f"constructed text tail lost for {document_id}",
        )
        reconstructed = separator.join(record["text"] for record in ordered)
        _require(
            reconstructed == text,
            f"element concatenation does not reconstruct {document_id}",
        )

    _require(set(chunk_member_elements) == set(chunk_by_id), "chunk without any element")
    for chunk_id, members in chunk_member_elements.items():
        ordered = sorted(members, key=lambda member: member["ordinal"])
        chunk_text = chunk_by_id[chunk_id]["text"]
        cursor = 0
        for index, member in enumerate(ordered):
            expected_start = cursor if index == 0 else cursor + len(separator)
            _require(
                member["char_start"] == expected_start,
                f"chunk {chunk_id} has a gap/overlap before {member['element_id']}",
            )
            cursor = member["char_end"]
        _require(
            cursor == len(chunk_text),
            f"chunk {chunk_id} tail is not covered by its elements",
        )

    element_count = len(elements)
    return {
        "status": "ok",
        "collection_id": collection_id,
        "documents": len(documents),
        "chunks": len(chunks),
        "elements": element_count,
        "alignment_records": len(alignment),
        "element_to_document_coverage": 1.0 if element_count else 0.0,
        "element_to_chunk_coverage": 1.0 if element_count else 0.0,
        "source_location": {
            "verified": 0,
            "unavailable": element_count + len(chunks),
        },
    }


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    path = Path(output_dir) / "conversion_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
