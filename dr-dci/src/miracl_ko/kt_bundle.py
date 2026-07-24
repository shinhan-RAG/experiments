"""Git-free execution locks shared by the MIRACL taxonomy KT bundle.

The bundle is transferred as a tar archive and deliberately has no Git remote,
credentials, corpus body, or model weights.  These helpers replace neither the
tracked taxonomy contract nor its validators: they bind the copied generator
source, external control records, and ignored MIRACL fixture files before a
local vLLM process is allowed to receive a request.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from src.miracl_ko.preparation import SCALE_SIZES, sha256_file, sha256_json
from src.miracl_ko.taxonomy_artifact import (
    approval_record_sha256,
    generator_contract_sha256,
    validate_generator_code_contract,
    validate_taxonomy_approval_record,
    validate_taxonomy_generation_plan,
)


KT_BUNDLE_SOURCE_MANIFEST_SCHEMA = "dr-dci.miracl-ko-kt-clean-source.v1"
KT_OPERATION_RECEIPT_SCHEMA = "dr-dci.miracl-ko-kt-operation-receipt.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SENSITIVE_CONFIG_RE = re.compile(r"(?:key|token|secret|password|credential)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def require_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 40-hex commit")
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    """Load a deliberately small KEY=VALUE config without shell evaluation."""
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"config.env line {number} must use KEY=VALUE")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"config.env line {number} has an invalid key")
        if _SENSITIVE_CONFIG_RE.search(key):
            raise ValueError(f"config.env must not contain credentials: {key}")
        if key in values:
            raise ValueError(f"config.env has duplicate key: {key}")
        if not value or value != value.strip():
            raise ValueError(f"config.env line {number} has an empty or padded value")
        values[key] = value
    return values


def require_config(values: Mapping[str, str]) -> dict[str, str]:
    required = {
        "KT_DATA_DIR", "KT_CONTROL_DIR", "KT_OUTPUT_DIR", "KT_MODEL_CACHE_DIR",
        "KT_VLLM_IMAGE", "KT_CONTAINER_DIGEST", "KT_VLLM_HOST", "KT_VLLM_PORT",
    }
    if set(values) != required:
        missing = sorted(required - set(values))
        extra = sorted(set(values) - required)
        raise ValueError(f"config.env keys do not match contract; missing={missing}, extra={extra}")
    require_sha256(values["KT_CONTAINER_DIGEST"].removeprefix("sha256:"), label="KT_CONTAINER_DIGEST")
    if not re.fullmatch(r"[1-9][0-9]{0,4}", values["KT_VLLM_PORT"]):
        raise ValueError("KT_VLLM_PORT must be a valid decimal port")
    return dict(values)


def path_from_config(bundle_root: Path, value: str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (bundle_root / path).resolve()
    if not str(path):
        raise ValueError(f"{label} path is invalid")
    return path


def input_provenance(*, data_dir: Path, revision_lock_path: Path) -> dict[str, str]:
    subset_manifest_path = data_dir / "subsets" / "manifest.json"
    subset_manifest = validate_subset_fixture_files(data_dir)
    try:
        corpus = subset_manifest["subsets"][str(110_000)]["corpus"]
    except (KeyError, TypeError) as error:
        raise ValueError("MIRACL subset manifest lacks the 110K corpus record") from error
    if not isinstance(corpus, Mapping):
        raise ValueError("MIRACL 110K corpus record is invalid")
    declared_corpus_sha256 = require_sha256(corpus.get("sha256"), label="MIRACL 110K corpus sha256")
    relative_path = corpus.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("MIRACL 110K corpus relative path is invalid")
    corpus_path = (data_dir / relative_path).resolve()
    if not corpus_path.is_relative_to(data_dir.resolve()) or not corpus_path.is_file():
        raise FileNotFoundError("MIRACL 110K corpus file is missing")
    if sha256_file(corpus_path) != declared_corpus_sha256:
        raise ValueError("MIRACL 110K corpus bytes do not match the subset manifest")
    return {
        "revision_lock_sha256": sha256_file(revision_lock_path),
        "preparation_contract_sha256": require_sha256(
            subset_manifest.get("preparation_contract_sha256"), label="MIRACL preparation contract sha256",
        ),
        "input_110k_corpus_sha256": declared_corpus_sha256,
        "input_subset_manifest_sha256": sha256_file(subset_manifest_path),
    }


def validate_subset_fixture_files(data_dir: Path) -> dict[str, Any]:
    """Rehash all nested corpus files named by the approved subset manifest."""
    subset_manifest = load_json_object(data_dir / "subsets" / "manifest.json", label="MIRACL subset manifest")
    root = data_dir.resolve()
    for scale in SCALE_SIZES:
        try:
            record = subset_manifest["subsets"][str(scale)]["corpus"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"MIRACL subset manifest lacks the {scale} corpus record") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"MIRACL {scale} corpus record is invalid")
        relative_path = record.get("relative_path")
        expected_sha256 = require_sha256(record.get("sha256"), label=f"MIRACL {scale} corpus sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"MIRACL {scale} corpus relative path is invalid")
        path = (data_dir / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"MIRACL {scale} corpus file is missing")
        expected_bytes = record.get("byte_size")
        if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes != path.stat().st_size):
            raise ValueError(f"MIRACL {scale} corpus bytes do not match the subset manifest")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"MIRACL {scale} corpus bytes do not match the subset manifest")
    return subset_manifest


def source_artifact_path(data_dir: Path) -> Path:
    subset_manifest = load_json_object(data_dir / "subsets" / "manifest.json", label="MIRACL subset manifest")
    try:
        relative_path = subset_manifest["subsets"][str(110_000)]["corpus"]["relative_path"]
    except (KeyError, TypeError) as error:
        raise ValueError("MIRACL subset manifest lacks the 110K corpus path") from error
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("MIRACL 110K corpus relative path is invalid")
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
        raise FileNotFoundError("MIRACL 110K corpus file is missing")
    return path


def validate_clean_source_manifest(
    manifest_path: Path, *, generator_source_root: Path, expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_json_object(manifest_path, label="clean generator source manifest")
    required = {
        "schema_version", "source_git_commit", "generator_contract_sha256", "generator_code_sha256", "files",
    }
    if set(manifest) != required or manifest.get("schema_version") != KT_BUNDLE_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("clean generator source manifest schema is invalid")
    require_commit(manifest.get("source_git_commit"), label="clean generator source commit")
    contract_hash = generator_contract_sha256(expected_contract)
    if manifest.get("generator_contract_sha256") != contract_hash:
        raise ValueError("clean generator source manifest contract hash does not match bundled contract")
    if manifest.get("generator_code_sha256") != expected_contract.get("generator_code_sha256"):
        raise ValueError("clean generator source manifest code hash does not match bundled contract")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("clean generator source manifest files are invalid")
    expected_files = expected_contract.get("generator_files")
    if files != expected_files:
        raise ValueError("clean generator source manifest files do not match bundled contract")
    validate_generator_code_contract(expected_contract, repo_root=generator_source_root)
    return manifest


def load_bundle_controls(control_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json_object(control_dir / "generation_plan.json", label="approved generation plan"),
        load_json_object(control_dir / "approval_record.json", label="approval record"),
        load_json_object(control_dir / "generator_code_contract.json", label="generator code contract"),
    )


def validate_execution_lock(
    *, bundle_root: Path, config: Mapping[str, str], plan: Mapping[str, Any],
    approval: Mapping[str, Any], supplied_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every non-model input required before a local request is sent."""
    validate_taxonomy_generation_plan(plan)
    validate_taxonomy_approval_record(approval)
    generator_root = bundle_root / "generator"
    bundled_contract = load_json_object(bundle_root / "source" / "generator_code_contract.json", label="bundled generator code contract")
    validate_generator_code_contract(bundled_contract, repo_root=generator_root)
    if supplied_contract != bundled_contract:
        raise ValueError("supplied generator code contract does not exactly match bundled source")
    source_manifest = validate_clean_source_manifest(
        bundle_root / "source" / "clean_generator_source_manifest.json",
        generator_source_root=generator_root,
        expected_contract=bundled_contract,
    )
    contract_hash = generator_contract_sha256(bundled_contract)
    code_hash = bundled_contract["generator_code_sha256"]
    if plan["generator_source_commit"] != source_manifest["source_git_commit"]:
        raise ValueError("generation plan source commit does not match clean source manifest")
    if plan["generator_code_contract_sha256"] != contract_hash or plan["generator_code_sha256"] != code_hash:
        raise ValueError("generation plan code contract does not match bundled generator source")
    if plan["generator"]["generator_code_sha256"] != code_hash:
        raise ValueError("generation plan generator metadata code does not match bundled generator source")
    if approval["approved_generation_plan_sha256"] != sha256_json(plan):
        raise ValueError("approval record does not approve the exact generation plan")
    if approval["approved_generator_source_commit"] != source_manifest["source_git_commit"]:
        raise ValueError("approval record source commit does not match clean source manifest")
    if approval["approved_generator_contract_sha256"] != contract_hash or approval["approved_generator_code_sha256"] != code_hash:
        raise ValueError("approval record code identity does not match bundled generator source")
    expected_runtime = plan["generator"]["runtime"]
    if config["KT_CONTAINER_DIGEST"] != expected_runtime["container_digest"]:
        raise ValueError("KT container digest does not match approved generation plan")
    if config.get("KT_MODEL_REVISION") is not None or config.get("KT_TOKENIZER_REVISION") is not None:
        raise ValueError("model/tokenizer revisions are read only from the approved plan, not config.env")
    data_dir = path_from_config(bundle_root, config["KT_DATA_DIR"], label="KT_DATA_DIR")
    provenance = input_provenance(
        data_dir=data_dir,
        revision_lock_path=bundle_root / "source" / "miracl_ko_revision_lock.json",
    )
    if plan["input_provenance"] != provenance:
        raise ValueError("generation plan dataset/subset provenance does not match external MIRACL files")
    return {
        "generation_plan_sha256": sha256_json(plan),
        "approval_record_sha256": approval_record_sha256(approval),
        "generator_contract_sha256": contract_hash,
        "generator_code_sha256": code_hash,
        "generator_source_commit": source_manifest["source_git_commit"],
        "model_repository": plan["generator"]["model_repository"],
        "model_revision": plan["generator"]["model_revision"],
        "tokenizer_repository": plan["generator"]["tokenizer_repository"],
        "tokenizer_revision": plan["generator"]["tokenizer_revision"],
        "container_digest": expected_runtime["container_digest"],
        "input_provenance": provenance,
        "data_dir": str(data_dir),
    }


def operation_receipt_path(output_dir: Path) -> Path:
    return output_dir / "generation" / "operation_receipt.json"


def require_operation_start(output_dir: Path, *, plan_sha256: str, run_sha256: str) -> dict[str, Any]:
    """Block a completed duplicate run; permit only an exact partial-run resume."""
    path = operation_receipt_path(output_dir)
    if path.exists():
        previous = load_json_object(path, label="operation receipt")
        if previous.get("generation_plan_sha256") != plan_sha256 or previous.get("generation_run_sha256") != run_sha256:
            raise ValueError("operation receipt belongs to another plan or generation run")
        if previous.get("status") == "complete":
            raise ValueError("completed taxonomy generation operation cannot be run twice")
        if previous.get("status") != "partial":
            raise ValueError("operation receipt has an unsupported status")
        return previous
    receipt = {
        "schema_version": KT_OPERATION_RECEIPT_SCHEMA,
        "generation_plan_sha256": plan_sha256,
        "generation_run_sha256": run_sha256,
        "status": "partial",
        "started_at": utc_now(),
        "completed_at": None,
        "batch_attempts": {},
    }
    atomic_write_json(path, receipt)
    return receipt
