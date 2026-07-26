"""Fail-loud preflight for the AIHub legal execution path.

The legal configuration is only execution wiring.  It becomes runnable only
after independently recorded Gate L0, immutable inputs, a legal metric/effect
criterion, and a real execution approval all agree.  This module performs no
query loading and never calls a model endpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LEGAL_DATASETS = {"aihub-full", "legal-qa", "ruling-anon", "longdoc"}
EXPECTED_L0_FINDINGS = {
    "query_leakage_candidates": 148,
    "parent_qrel_mismatches": 84,
    "content_duplicates": 315,
    "duplicate_ids": 5,
    "pii_possible_records": 89,
}


class LegalExecutionBlocked(RuntimeError):
    """Raised before corpus/query loading when the legal execution contract lacks proof."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_names(config: dict[str, Any]) -> set[str]:
    parts = config.get("parts", {})
    names = set()
    for key in ("part1_stacking", "part2_scaling", "part3_tags", "part5_pull_backend"):
        value = parts.get(key, {})
        if value.get("dataset"):
            names.add(str(value["dataset"]))
    for entry in parts.get("part4_generalization", {}).get("datasets", []):
        names.add(str(entry.get("name") if isinstance(entry, dict) else entry))
    return names


def is_legal_execution_config(config: dict[str, Any]) -> bool:
    return bool(_dataset_names(config) & LEGAL_DATASETS)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_ref(root: Path, reference: Any, name: str) -> tuple[dict[str, Any], str]:
    if not isinstance(reference, dict):
        raise LegalExecutionBlocked(f"missing {name} reference")
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise LegalExecutionBlocked(f"{name} path is required")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise LegalExecutionBlocked(f"{name} SHA-256 is required")
    path = _resolve(root, path_value)
    if not path.is_file():
        raise LegalExecutionBlocked(f"{name} is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise LegalExecutionBlocked(f"{name} SHA-256 mismatch")
    try:
        return json.loads(path.read_text(encoding="utf-8")), actual_hash
    except json.JSONDecodeError as exc:
        raise LegalExecutionBlocked(f"{name} is not valid JSON") from exc


def _require_actual_approval(record: dict[str, Any], name: str) -> None:
    if record.get("status") != "approved":
        raise LegalExecutionBlocked(f"{name} status is not approved")
    if not isinstance(record.get("approved_by"), str) or not record["approved_by"].strip():
        raise LegalExecutionBlocked(f"{name} requires approved_by")
    if not isinstance(record.get("approved_at"), str) or not record["approved_at"].strip():
        raise LegalExecutionBlocked(f"{name} requires approved_at")
    if not isinstance(record.get("basis"), str) or not record["basis"].strip():
        raise LegalExecutionBlocked(f"{name} requires approval basis")


def _require_immutable_input(root: Path, reference: Any, name: str) -> str:
    if not isinstance(reference, dict):
        raise LegalExecutionBlocked(f"missing immutable {name} reference")
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise LegalExecutionBlocked(f"immutable {name} requires path and SHA-256")
    path = _resolve(root, path_value)
    if not path.is_file():
        raise LegalExecutionBlocked(f"immutable {name} is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise LegalExecutionBlocked(f"immutable {name} SHA-256 mismatch")
    return actual_hash


def _require_tag_execution_contract(root: Path, gate: dict[str, Any], steps: list[dict]) -> None:
    if not any(step.get("tags") for step in steps):
        return
    payload, _ = _load_ref(root, gate.get("tag_execution_contract"), "tag execution contract")
    if payload.get("element_universe") != "parser_derived":
        raise LegalExecutionBlocked("legal tags require a parser-derived element universe")
    if payload.get("structural_type_predicate") != "exact_typed_predicate":
        raise LegalExecutionBlocked("legal tags require an exact structural_type predicate")
    if payload.get("semantic_role_predicate") != "exact_typed_predicate":
        raise LegalExecutionBlocked("legal tags require an exact semantic_role predicate")
    if payload.get("failure_policy") != "fail_loud":
        raise LegalExecutionBlocked("legal tags may not silently fall back on classification failure")


def require_legal_execution_gate(
    config: dict[str, Any],
    *,
    artifact_root: Path,
    part: str,
    steps: list[dict] | None = None,
) -> None:
    """Verify every legal execution prerequisite before corpus or query loading."""
    if not is_legal_execution_config(config):
        return
    gate = config.get("legal_execution_gate")
    if not isinstance(gate, dict):
        raise LegalExecutionBlocked("legal execution requires a Legal Gate L0 contract")
    if gate.get("status") != "approved":
        raise LegalExecutionBlocked("legal execution gate is not approved")

    l0, l0_hash = _load_ref(artifact_root, gate.get("gate_l0_approval_manifest"), "Legal Gate L0 manifest")
    _require_actual_approval(l0, "Legal Gate L0 manifest")
    if l0.get("referential_integrity") != "passed":
        raise LegalExecutionBlocked("Legal Gate L0 referential integrity is not passed")
    if l0.get("license_internal_use") != "approved":
        raise LegalExecutionBlocked("Legal Gate L0 license/internal-use scope is not approved")
    findings = l0.get("findings")
    if not isinstance(findings, dict) or any(
        findings.get(key) != expected for key, expected in EXPECTED_L0_FINDINGS.items()
    ):
        raise LegalExecutionBlocked("Legal Gate L0 findings do not match the pre-registered audit")

    immutable_inputs = gate.get("immutable_inputs")
    if not isinstance(immutable_inputs, dict):
        raise LegalExecutionBlocked("legal execution requires immutable corpus, subset, query, and exclusion inputs")
    actual_inputs = {
        name: _require_immutable_input(artifact_root, immutable_inputs.get(name), name)
        for name in ("corpus", "subset", "final_query", "exclusion_list")
    }
    if l0.get("immutable_inputs") != actual_inputs:
        raise LegalExecutionBlocked("Legal Gate L0 immutable input hashes do not match")

    metric, _ = _load_ref(artifact_root, gate.get("metric_effect_approval"), "legal metric/effect approval")
    _require_actual_approval(metric, "legal metric/effect approval")
    if metric.get("retrieval_unit") != "parent_document":
        raise LegalExecutionBlocked("legal metric/effect approval must declare parent_document")
    execution, _ = _load_ref(artifact_root, gate.get("execution_approval"), "legal execution approval")
    _require_actual_approval(execution, "legal execution approval")
    if execution.get("gate_l0_manifest_sha256") != l0_hash:
        raise LegalExecutionBlocked("execution approval is not bound to the supplied Legal Gate L0 manifest")
    if execution.get("part") not in {part, "all"}:
        raise LegalExecutionBlocked("execution approval does not cover the requested legal part")

    _require_tag_execution_contract(artifact_root, gate, steps or [])


def require_longdoc_diagnostic_contract(entry: dict[str, Any]) -> None:
    """Reject treating the longdoc diagnostic collection as confirmatory evidence."""
    if entry.get("name") != "longdoc":
        return
    if entry.get("evaluation_role") != "diagnostic_only":
        raise LegalExecutionBlocked("longdoc must be declared diagnostic_only")
    if entry.get("confirmatory_gate_status") != "not_advanced":
        raise LegalExecutionBlocked("longdoc may not advance confirmatory gates")
    if not isinstance(entry.get("source_revision_sha256"), str):
        raise LegalExecutionBlocked("longdoc requires a recorded source revision SHA-256")
    if not isinstance(entry.get("lexical_overlap_stratum"), str):
        raise LegalExecutionBlocked("longdoc requires a lexical-overlap stratum")
