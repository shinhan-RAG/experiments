"""Leakage-safe taxonomy artifact contracts for MIRACL Korean passages.

This is a preparation-only boundary.  It does not generate a real taxonomy,
call a model, invoke a retriever, or register MIRACL with the Part 1/2 runner.
It instead validates the immutable artifact that a separately approved
generator would produce from 110K passage title/text and projects it to the
smaller nested fixtures without changing any shared assignment.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from src.miracl_ko.preparation import RETRIEVAL_UNIT, SCALE_SIZES, sha256_file, sha256_json


TAXONOMY_ARTIFACT_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-artifact.v1"
TAXONOMY_MANIFEST_SCHEMA_VERSION = "dr-dci.miracl-ko-taxonomy-manifest.v1"
TAXONOMY_SOURCE_SCALE = 110_000
TAXONOMY_SEMANTIC_INPUT_FIELDS = ("title", "text")
TAXONOMY_MAPPING_KEY = "corpus_id"
TAXONOMY_NONSEMANTIC_CORPUS_ID_USES = (
    "mapping_join",
    "duplicate_detection",
    "deterministic_output_order",
)
TAXONOMY_FORBIDDEN_INPUT_FIELDS = frozenset({
    "query",
    "queries",
    "qid",
    "qrel",
    "qrels",
    "relevance",
    "answer",
    "answers",
    "gold",
    "gold_id",
    "gold_ids",
    "evidence",
    "evidence_id",
    "evidence_ids",
    "positive_passages",
    "negative_passages",
    "ranked_passage_ids",
    "evaluation_result",
})
SOURCE_IDENTITY_METHOD = "source_110k_identity_v1"
FILTER_PROJECTION_METHOD = "filter_110k_mapping_by_corpus_id_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_semantic_input_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise ValueError("taxonomy input_contract must be an object")
    if contract.get("mapping_key") != TAXONOMY_MAPPING_KEY:
        raise ValueError("taxonomy mapping_key must be corpus_id")
    if contract.get("semantic_generator_input_fields") != list(TAXONOMY_SEMANTIC_INPUT_FIELDS):
        raise ValueError("taxonomy semantic generator inputs must be title/text only")
    if contract.get("corpus_id_semantic_use") != "prohibited":
        raise ValueError("taxonomy corpus_id semantic use must be prohibited")
    if contract.get("corpus_id_nonsemantic_uses") != list(TAXONOMY_NONSEMANTIC_CORPUS_ID_USES):
        raise ValueError("taxonomy corpus_id non-semantic uses must be declared exactly")
    forbidden = contract.get("forbidden_input_fields")
    if not isinstance(forbidden, list) or not TAXONOMY_FORBIDDEN_INPUT_FIELDS.issubset(
        {str(value).casefold() for value in forbidden}
    ):
        raise ValueError("taxonomy input_contract is missing forbidden leakage fields")


def build_semantic_generator_inputs(corpus_records: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Create model-facing inputs without passage IDs or evaluation information.

    ``corpus_id`` is permitted only at the outer mapping boundary.  It is
    checked for duplicate transport records here, but never returned to a
    semantic generator, tokenizer, clusterer, or label-naming operation.
    """
    inputs: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    allowed_transport_fields = {TAXONOMY_MAPPING_KEY, *TAXONOMY_SEMANTIC_INPUT_FIELDS}
    for index, record in enumerate(corpus_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"taxonomy corpus record {index} must be an object")
        lowered_fields = {str(field).casefold() for field in record}
        forbidden = sorted(lowered_fields & TAXONOMY_FORBIDDEN_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"taxonomy generator input contains forbidden leakage field: {forbidden[0]}")
        unsupported = set(record) - allowed_transport_fields
        if unsupported:
            raise ValueError(f"taxonomy generator input has unsupported fields: {sorted(unsupported)}")
        corpus_id = _require_nonempty_string(record.get(TAXONOMY_MAPPING_KEY), label="taxonomy corpus_id")
        if corpus_id in seen_ids:
            raise ValueError(f"taxonomy generator input has duplicate corpus_id: {corpus_id}")
        seen_ids.add(corpus_id)
        title = record.get("title")
        text = record.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("taxonomy title/text inputs must be strings")
        inputs.append({"title": title, "text": text})
    if not inputs:
        raise ValueError("taxonomy generator input requires at least one passage")
    return inputs


def _validate_generator(generator: Any) -> None:
    if not isinstance(generator, dict):
        raise ValueError("taxonomy generator provenance must be an object")
    for key in (
        "generator_type",
        "generator_version",
        "model_or_algorithm",
        "model_or_tokenizer_version",
        "determinism_mode",
    ):
        _require_nonempty_string(generator.get(key), label=f"taxonomy generator {key}")
    _require_sha256(generator.get("generator_code_sha256"), label="taxonomy generator code")
    template_sha256 = generator.get("prompt_template_sha256")
    if template_sha256 is not None:
        _require_sha256(template_sha256, label="taxonomy generator prompt template")
    if not isinstance(generator.get("seed"), int):
        raise ValueError("taxonomy generator seed must be an integer")
    if not isinstance(generator.get("parameters"), dict) or not generator["parameters"]:
        raise ValueError("taxonomy generator parameters must be a non-empty object")
    if generator.get("determinism_mode") not in {"deterministic", "replay_required"}:
        raise ValueError("taxonomy generator determinism_mode is invalid")


def _validate_provenance(provenance: Any) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("taxonomy artifact provenance must be an object")
    for key in (
        "revision_lock_sha256",
        "preparation_contract_sha256",
        "input_110k_corpus_sha256",
        "input_subset_manifest_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"taxonomy provenance {key}")
    _validate_generator(provenance.get("generator"))


def _validate_source_revisions(source_revisions: Any) -> None:
    if not isinstance(source_revisions, dict) or set(source_revisions) != {"topics_qrels", "corpus"}:
        raise ValueError("taxonomy source_revisions must contain topics_qrels and corpus")
    for key in ("topics_qrels", "corpus"):
        value = source_revisions[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"taxonomy source revision is invalid for {key}")


def _label_catalog_by_id(catalog: Any) -> dict[str, dict[str, str]]:
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("taxonomy label_catalog must be a non-empty array")
    by_id: dict[str, dict[str, str]] = {}
    seen_labels: set[str] = set()
    for item in catalog:
        if not isinstance(item, dict):
            raise ValueError("taxonomy label_catalog entries must be objects")
        if set(item) != {"label_id", "label"}:
            raise ValueError("taxonomy label_catalog entries must contain label_id and label only")
        label_id = _require_nonempty_string(item.get("label_id"), label="taxonomy label_id")
        label = _require_nonempty_string(item.get("label"), label="taxonomy label")
        if label_id in by_id:
            raise ValueError(f"taxonomy label_catalog has duplicate label_id: {label_id}")
        if label in seen_labels:
            raise ValueError(f"taxonomy label_catalog has duplicate label: {label}")
        by_id[label_id] = {"label_id": label_id, "label": label}
        seen_labels.add(label)
    if "unknown" not in by_id:
        raise ValueError("taxonomy label_catalog must define the unknown label")
    return by_id


def _assignments_by_id(assignments: Any, *, catalog: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("taxonomy assignments must be a non-empty array")
    by_id: dict[str, dict[str, Any]] = {}
    previous_id: str | None = None
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("taxonomy assignments must contain objects")
        if set(assignment) != {"corpus_id", "label_id", "score", "status"}:
            raise ValueError("taxonomy assignment has unsupported fields")
        corpus_id = _require_nonempty_string(assignment.get("corpus_id"), label="taxonomy assignment corpus_id")
        if corpus_id in by_id:
            raise ValueError(f"taxonomy assignments have duplicate corpus_id: {corpus_id}")
        if previous_id is not None and corpus_id < previous_id:
            raise ValueError("taxonomy assignments must use deterministic corpus_id ordering")
        previous_id = corpus_id
        label_id = _require_nonempty_string(assignment.get("label_id"), label="taxonomy assignment label_id")
        if label_id not in catalog:
            raise ValueError(f"taxonomy assignment references unknown label_id: {label_id}")
        score = assignment.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("taxonomy assignment score must be finite and within [0, 1]")
        status = assignment.get("status")
        if status not in {"assigned", "unknown"}:
            raise ValueError("taxonomy assignment status must be assigned or unknown")
        if status == "unknown" and (label_id != "unknown" or float(score) != 0.0):
            raise ValueError("taxonomy unknown assignment must use unknown label_id and score 0")
        if status == "assigned" and label_id == "unknown":
            raise ValueError("taxonomy assigned passage cannot use unknown label_id")
        by_id[corpus_id] = {
            "corpus_id": corpus_id,
            "label_id": label_id,
            "score": float(score),
            "status": status,
        }
    return by_id


def taxonomy_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Return a canonical content hash; timestamps live in the outer manifest."""
    return sha256_json(artifact)


def validate_taxonomy_artifact(
    artifact: Mapping[str, Any], *, expected_corpus_ids: set[str] | None = None
) -> dict[str, Any]:
    """Validate one source or projected artifact and return non-performance QA."""
    if not isinstance(artifact, Mapping):
        raise ValueError("taxonomy artifact must be an object")
    required = {
        "schema_version",
        "taxonomy_artifact_id",
        "dataset",
        "language",
        "source_revisions",
        "retrieval_unit",
        "source_scale",
        "projection_scale",
        "input_contract",
        "provenance",
        "label_catalog",
        "assignments",
        "projection",
    }
    if set(artifact) != required:
        raise ValueError("taxonomy artifact has missing or unsupported top-level fields")
    if artifact.get("schema_version") != TAXONOMY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("taxonomy artifact schema_version is invalid")
    if artifact.get("dataset") != "MIRACL" or artifact.get("language") != "ko":
        raise ValueError("taxonomy artifact must identify MIRACL Korean")
    _validate_source_revisions(artifact.get("source_revisions"))
    if artifact.get("retrieval_unit") != RETRIEVAL_UNIT:
        raise ValueError("taxonomy artifact retrieval_unit must be passage")
    _require_nonempty_string(artifact.get("taxonomy_artifact_id"), label="taxonomy artifact id")
    if artifact.get("source_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy artifact source_scale must be 110K")
    if artifact.get("projection_scale") not in SCALE_SIZES:
        raise ValueError("taxonomy artifact projection_scale must be 20K, 50K, or 110K")
    _validate_semantic_input_contract(artifact.get("input_contract"))
    _validate_provenance(artifact.get("provenance"))
    catalog = _label_catalog_by_id(artifact.get("label_catalog"))
    assignments = _assignments_by_id(artifact.get("assignments"), catalog=catalog)
    projection = artifact.get("projection")
    if not isinstance(projection, dict) or set(projection) != {"method", "source_artifact_sha256"}:
        raise ValueError("taxonomy artifact projection metadata is invalid")
    if artifact["projection_scale"] == TAXONOMY_SOURCE_SCALE:
        if projection.get("method") != SOURCE_IDENTITY_METHOD or projection.get("source_artifact_sha256") is not None:
            raise ValueError("110K taxonomy artifact must use the identity projection declaration")
    else:
        if projection.get("method") != FILTER_PROJECTION_METHOD:
            raise ValueError("projected taxonomy artifact must use fixed 110K filter projection")
        _require_sha256(projection.get("source_artifact_sha256"), label="taxonomy projection source artifact")
    if expected_corpus_ids is not None:
        actual_ids = set(assignments)
        missing = expected_corpus_ids - actual_ids
        orphan = actual_ids - expected_corpus_ids
        if orphan:
            raise ValueError(f"taxonomy artifact has orphan passage mapping: {sorted(orphan)[0]}")
        if missing:
            raise ValueError(f"taxonomy artifact coverage is missing passage: {sorted(missing)[0]}")

    assigned_labels = Counter(
        assignment["label_id"] for assignment in assignments.values()
        if assignment["status"] == "assigned"
    )
    unknown_count = sum(assignment["status"] == "unknown" for assignment in assignments.values())
    assigned_count = len(assignments) - unknown_count
    largest_label_count = max(assigned_labels.values(), default=0)
    return {
        "taxonomy_artifact_id": artifact["taxonomy_artifact_id"],
        "artifact_sha256": taxonomy_artifact_sha256(artifact),
        "passage_count": len(assignments),
        "coverage_rate": 1.0 if expected_corpus_ids is not None else None,
        "unknown_assignment_count": unknown_count,
        "unknown_assignment_rate": unknown_count / len(assignments),
        "assigned_label_count": len(assigned_labels),
        "label_distribution": dict(sorted(assigned_labels.items())),
        "minimum_assigned_label_size": min(assigned_labels.values(), default=0),
        "maximum_assigned_label_size": largest_label_count,
        "maximum_assigned_label_share": (
            largest_label_count / assigned_count if assigned_count else None
        ),
    }


def project_taxonomy_artifact(
    source_artifact: Mapping[str, Any], target_corpus_ids: set[str], *, target_scale: int
) -> dict[str, Any]:
    """Filter the one 110K mapping without relabeling or reassigning passages."""
    validate_taxonomy_artifact(source_artifact)
    if source_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy projection source must be the 110K identity artifact")
    if target_scale not in SCALE_SIZES or target_scale == TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy filter projection target must be 20K or 50K")
    source_assignments = {row["corpus_id"]: row for row in source_artifact["assignments"]}
    unknown = target_corpus_ids - set(source_assignments)
    if unknown:
        raise ValueError(f"taxonomy projection target has passage outside 110K source: {sorted(unknown)[0]}")
    projected = {
        **dict(source_artifact),
        "projection_scale": target_scale,
        "assignments": [
            dict(row) for row in source_artifact["assignments"]
            if row["corpus_id"] in target_corpus_ids
        ],
        "projection": {
            "method": FILTER_PROJECTION_METHOD,
            "source_artifact_sha256": taxonomy_artifact_sha256(source_artifact),
        },
    }
    validate_taxonomy_projection(
        source_artifact,
        projected,
        expected_corpus_ids=target_corpus_ids,
        target_scale=target_scale,
    )
    return projected


def validate_taxonomy_projection(
    source_artifact: Mapping[str, Any], projected_artifact: Mapping[str, Any], *,
    expected_corpus_ids: set[str], target_scale: int,
) -> None:
    """Require a byte-independent, assignment-exact filter projection."""
    validate_taxonomy_artifact(source_artifact)
    validate_taxonomy_artifact(projected_artifact, expected_corpus_ids=expected_corpus_ids)
    if source_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy projection source must remain the 110K artifact")
    if target_scale not in {20_000, 50_000} or projected_artifact.get("projection_scale") != target_scale:
        raise ValueError("taxonomy projection target scale is invalid")
    if projected_artifact.get("taxonomy_artifact_id") != source_artifact.get("taxonomy_artifact_id"):
        raise ValueError("taxonomy projection must preserve taxonomy_artifact_id")
    if projected_artifact.get("label_catalog") != source_artifact.get("label_catalog"):
        raise ValueError("taxonomy projection must preserve label catalog")
    if projected_artifact.get("provenance") != source_artifact.get("provenance"):
        raise ValueError("taxonomy projection must preserve generator provenance")
    projection = projected_artifact.get("projection")
    if projection != {
        "method": FILTER_PROJECTION_METHOD,
        "source_artifact_sha256": taxonomy_artifact_sha256(source_artifact),
    }:
        raise ValueError("taxonomy projection source hash or method is invalid")
    expected_assignments = [
        row for row in source_artifact["assignments"]
        if row["corpus_id"] in expected_corpus_ids
    ]
    if projected_artifact.get("assignments") != expected_assignments:
        raise ValueError("taxonomy projection changes a shared passage assignment")


def _validate_file_record(record: Any, *, data_dir: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} file record must be an object")
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} file record has invalid relative_path")
    root = data_dir.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"{label} artifact file is missing: {relative_path}")
    if record.get("byte_size") != path.stat().st_size:
        raise ValueError(f"{label} artifact byte_size does not match manifest")
    _require_sha256(record.get("sha256"), label=f"{label} artifact")
    if record["sha256"] != sha256_file(path):
        raise ValueError(f"{label} artifact sha256 does not match manifest")
    return path


def _load_artifact(path: Path, *, label: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} taxonomy artifact JSON is invalid") from error
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} taxonomy artifact must be an object")
    return artifact


def build_taxonomy_artifact_manifest(
    *, full_artifact: Mapping[str, Any], artifact_records: Mapping[int, Mapping[str, Any]],
    source_git_commit: str,
) -> dict[str, Any]:
    """Build the tracked manifest; large artifact bodies remain outside Git."""
    full_audit = validate_taxonomy_artifact(full_artifact)
    if full_artifact.get("projection_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy manifest requires the 110K identity artifact")
    if set(artifact_records) != set(SCALE_SIZES):
        raise ValueError("taxonomy manifest must contain 20K/50K/110K artifact records")
    if not isinstance(source_git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_git_commit):
        raise ValueError("taxonomy manifest source_git_commit is invalid")
    normalized_records: dict[str, dict[str, Any]] = {}
    for scale in SCALE_SIZES:
        record = dict(artifact_records[scale])
        _require_sha256(record.get("sha256"), label=f"taxonomy manifest {scale} artifact")
        if not isinstance(record.get("relative_path"), str) or not record["relative_path"]:
            raise ValueError("taxonomy manifest artifact relative_path is invalid")
        if not isinstance(record.get("byte_size"), int) or record["byte_size"] < 0:
            raise ValueError("taxonomy manifest artifact byte_size is invalid")
        normalized_records[str(scale)] = record
    provenance = dict(full_artifact["provenance"])
    return {
        "schema_version": TAXONOMY_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "taxonomy_artifact_id": full_artifact["taxonomy_artifact_id"],
        "dataset": "MIRACL",
        "language": "ko",
        "source_revisions": dict(full_artifact["source_revisions"]),
        "retrieval_unit": RETRIEVAL_UNIT,
        "source_git_commit": source_git_commit,
        "source_scale": TAXONOMY_SOURCE_SCALE,
        "input_contract": dict(full_artifact["input_contract"]),
        "provenance": provenance,
        "source_artifact_content_sha256": full_audit["artifact_sha256"],
        "artifacts": {
            str(scale): {
                "role": "source" if scale == TAXONOMY_SOURCE_SCALE else "filter_projection",
                "artifact_file": normalized_records[str(scale)],
                "source_artifact_content_sha256": (
                    None if scale == TAXONOMY_SOURCE_SCALE else full_audit["artifact_sha256"]
                ),
                "projection_method": (
                    SOURCE_IDENTITY_METHOD if scale == TAXONOMY_SOURCE_SCALE else FILTER_PROJECTION_METHOD
                ),
            }
            for scale in SCALE_SIZES
        },
    }


def validate_taxonomy_artifact_manifest(
    manifest: Mapping[str, Any], *, data_dir: Path,
    expected_ids_by_scale: Mapping[int, set[str]],
    expected_provenance: Mapping[str, str] | None = None,
    expected_source_revisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recheck ignored bodies against the tracked manifest and nested fixtures."""
    if not isinstance(manifest, Mapping):
        raise ValueError("taxonomy artifact manifest must be an object")
    required = {
        "schema_version", "generated_at", "taxonomy_artifact_id", "dataset", "language", "source_revisions", "retrieval_unit",
        "source_git_commit", "source_scale", "input_contract", "provenance",
        "source_artifact_content_sha256", "artifacts",
    }
    if set(manifest) != required:
        raise ValueError("taxonomy artifact manifest has missing or unsupported fields")
    if manifest.get("schema_version") != TAXONOMY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("taxonomy artifact manifest schema_version is invalid")
    _require_nonempty_string(manifest.get("generated_at"), label="taxonomy artifact manifest generated_at")
    if manifest.get("dataset") != "MIRACL" or manifest.get("language") != "ko":
        raise ValueError("taxonomy artifact manifest must identify MIRACL Korean")
    _validate_source_revisions(manifest.get("source_revisions"))
    if expected_source_revisions is not None:
        _validate_source_revisions(dict(expected_source_revisions))
        if manifest["source_revisions"] != dict(expected_source_revisions):
            raise ValueError("taxonomy manifest source_revisions do not match verified fixture")
    if manifest.get("retrieval_unit") != RETRIEVAL_UNIT or manifest.get("source_scale") != TAXONOMY_SOURCE_SCALE:
        raise ValueError("taxonomy artifact manifest has invalid retrieval unit or source scale")
    if not isinstance(manifest.get("source_git_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["source_git_commit"]
    ):
        raise ValueError("taxonomy artifact manifest source_git_commit is invalid")
    _validate_semantic_input_contract(manifest.get("input_contract"))
    _validate_provenance(manifest.get("provenance"))
    if expected_provenance is not None:
        for key in (
            "revision_lock_sha256",
            "preparation_contract_sha256",
            "input_110k_corpus_sha256",
            "input_subset_manifest_sha256",
        ):
            expected = expected_provenance.get(key)
            _require_sha256(expected, label=f"expected taxonomy provenance {key}")
            if manifest["provenance"].get(key) != expected:
                raise ValueError(f"taxonomy manifest provenance {key} does not match verified input")
    _require_sha256(manifest.get("source_artifact_content_sha256"), label="taxonomy manifest source artifact")
    if set(expected_ids_by_scale) != set(SCALE_SIZES):
        raise ValueError("taxonomy preflight requires expected passage IDs for 20K/50K/110K")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {str(scale) for scale in SCALE_SIZES}:
        raise ValueError("taxonomy artifact manifest must contain all scale artifacts")

    loaded: dict[int, dict[str, Any]] = {}
    for scale in SCALE_SIZES:
        record = artifacts[str(scale)]
        if not isinstance(record, Mapping):
            raise ValueError(f"taxonomy manifest artifact record is invalid for {scale}")
        expected_role = "source" if scale == TAXONOMY_SOURCE_SCALE else "filter_projection"
        expected_method = SOURCE_IDENTITY_METHOD if scale == TAXONOMY_SOURCE_SCALE else FILTER_PROJECTION_METHOD
        if record.get("role") != expected_role or record.get("projection_method") != expected_method:
            raise ValueError(f"taxonomy manifest projection declaration is invalid for {scale}")
        if scale == TAXONOMY_SOURCE_SCALE:
            if record.get("source_artifact_content_sha256") is not None:
                raise ValueError("110K taxonomy manifest source artifact must not point to another source")
        elif record.get("source_artifact_content_sha256") != manifest["source_artifact_content_sha256"]:
            raise ValueError(f"taxonomy manifest projection source hash is invalid for {scale}")
        path = _validate_file_record(record.get("artifact_file"), data_dir=data_dir, label=f"taxonomy {scale}")
        artifact = _load_artifact(path, label=f"taxonomy {scale}")
        audit = validate_taxonomy_artifact(artifact, expected_corpus_ids=set(expected_ids_by_scale[scale]))
        if artifact.get("taxonomy_artifact_id") != manifest["taxonomy_artifact_id"]:
            raise ValueError(f"taxonomy artifact id does not match manifest for {scale}")
        if artifact.get("source_revisions") != manifest["source_revisions"]:
            raise ValueError(f"taxonomy source_revisions do not match manifest for {scale}")
        if artifact.get("input_contract") != manifest["input_contract"]:
            raise ValueError(f"taxonomy input contract does not match manifest for {scale}")
        if artifact.get("provenance") != manifest["provenance"]:
            raise ValueError(f"taxonomy provenance does not match manifest for {scale}")
        loaded[scale] = {"artifact": artifact, "audit": audit}

    source = loaded[TAXONOMY_SOURCE_SCALE]["artifact"]
    if taxonomy_artifact_sha256(source) != manifest["source_artifact_content_sha256"]:
        raise ValueError("taxonomy 110K artifact content sha256 does not match manifest")
    for scale in (20_000, 50_000):
        validate_taxonomy_projection(
            source,
            loaded[scale]["artifact"],
            expected_corpus_ids=set(expected_ids_by_scale[scale]),
            target_scale=scale,
        )
    return {
        "status": "ready",
        "taxonomy_artifact_id": manifest["taxonomy_artifact_id"],
        "artifact_sha256": manifest["source_artifact_content_sha256"],
        "scales": {str(scale): loaded[scale]["audit"] for scale in SCALE_SIZES},
    }


def load_verified_taxonomy_projection(
    manifest_path: Path, *, data_dir: Path, expected_ids_by_scale: Mapping[int, set[str]], scale: int,
    expected_source_revisions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail loudly before a future consumer receives a projected mapping."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"taxonomy artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("taxonomy artifact manifest JSON is invalid") from error
    if scale not in SCALE_SIZES:
        raise ValueError("taxonomy projection scale is invalid")
    validate_taxonomy_artifact_manifest(
        manifest,
        data_dir=data_dir,
        expected_ids_by_scale=expected_ids_by_scale,
        expected_source_revisions=expected_source_revisions,
    )
    record = manifest["artifacts"][str(scale)]["artifact_file"]
    return _load_artifact(_validate_file_record(record, data_dir=data_dir, label=f"taxonomy {scale}"), label=f"taxonomy {scale}")


def validate_deterministic_regeneration(
    reference_artifact: Mapping[str, Any], regenerated_artifact: Mapping[str, Any]
) -> None:
    """Require byte-independent identical content for a deterministic generator."""
    validate_taxonomy_artifact(reference_artifact)
    validate_taxonomy_artifact(regenerated_artifact)
    for artifact in (reference_artifact, regenerated_artifact):
        if artifact["provenance"]["generator"]["determinism_mode"] != "deterministic":
            raise ValueError("taxonomy generator does not claim deterministic regeneration")
    if taxonomy_artifact_sha256(reference_artifact) != taxonomy_artifact_sha256(regenerated_artifact):
        raise ValueError("taxonomy deterministic regeneration changed artifact content")


def validate_focused_taxonomy_treatment_pair(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> None:
    """Freeze the later A/B to one treatment difference: taxonomy score boost."""
    if control.get("taxonomy") is not False or treatment.get("taxonomy") is not True:
        raise ValueError("focused taxonomy pair must toggle taxonomy score boost only")
    for key in ("taxonomy_prompt_schema", "workspace_taxonomy"):
        if control.get(key) != treatment.get(key):
            raise ValueError("focused taxonomy pair changes more than taxonomy score boost")
    if control.get("taxonomy_prompt_schema") is not True or control.get("workspace_taxonomy") is not False:
        raise ValueError("focused taxonomy pair requires shared prompt and disabled workspace taxonomy")
    ignored = {"name", "description", "taxonomy"}
    changed = {
        key for key in set(control) | set(treatment)
        if key not in ignored and control.get(key) != treatment.get(key)
    }
    if changed:
        raise ValueError("focused taxonomy pair changes more than taxonomy score boost")
