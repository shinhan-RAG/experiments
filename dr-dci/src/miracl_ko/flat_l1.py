"""Frozen MIRACL-ko closed-set Flat-L1 taxonomy plan specification.

This module only materializes and validates an artifact-free generation plan.
It never loads a model, calls an endpoint, or creates a taxonomy artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.miracl_ko.kt_bundle import input_provenance
from src.miracl_ko.preparation import sha256_file, sha256_json
from src.miracl_ko.taxonomy_artifact import (
    FILTER_PROJECTION_METHOD,
    TAXONOMY_FORBIDDEN_INPUT_FIELDS,
    TAXONOMY_MAPPING_KEY,
    TAXONOMY_NONSEMANTIC_CORPUS_ID_USES,
    TAXONOMY_SEMANTIC_INPUT_FIELDS,
    build_taxonomy_generation_plan,
    validate_taxonomy_generation_plan,
)


FLAT_L1_ASSIGNMENT_PROFILE = "miracl-ko-flat-l1-ko-strategyqa-v1"
FLAT_L1_ARTIFACT_ID = "miracl-ko-110k-flat-l1-ko-strategyqa-v1"
MODEL_REPOSITORY = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VLLM_IMAGE_DIGEST = "sha256:df2c55e5107afea09ea1a50f9dd96c99ebf97a795334c4d08f691f3d79b2ab12"
MODEL_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
MODEL_TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
EXPECTED_CONTAINER_ENTRYPOINT = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
FLAT_L1_GENERATOR_SOURCE_FILES = (
    "scripts/kt_bundle/taxonomy_vllm_generator.py",
    "scripts/kt_bundle/preflight_validator.py",
    "scripts/kt_bundle/start_vllm.sh",
    "scripts/kt_bundle/vllm_runtime_contract.py",
    "src/miracl_ko/__init__.py",
    "src/miracl_ko/flat_l1.py",
    "src/miracl_ko/kt_bundle.py",
    "src/miracl_ko/preparation.py",
    "src/miracl_ko/taxonomy_artifact.py",
    "config/miracl_ko_taxonomy/flat_l1_catalog.json",
    "config/miracl_ko_taxonomy/flat_l1_output_schema.json",
    "config/miracl_ko_taxonomy/flat_l1_system_prompt.txt",
    "config/miracl_ko_taxonomy/vllm_v0_9_0_runtime_identity.json",
    "config/taxonomy_schemas/ko-strategyqa.yaml",
)


def _config_root(repo_root: Path) -> Path:
    return repo_root / "config" / "miracl_ko_taxonomy"


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _semantic_input_contract() -> dict[str, Any]:
    return {
        "mapping_key": TAXONOMY_MAPPING_KEY,
        "semantic_generator_input_fields": list(TAXONOMY_SEMANTIC_INPUT_FIELDS),
        "corpus_id_semantic_use": "prohibited",
        "corpus_id_nonsemantic_uses": list(TAXONOMY_NONSEMANTIC_CORPUS_ID_USES),
        "forbidden_input_fields": sorted(TAXONOMY_FORBIDDEN_INPUT_FIELDS),
    }


def load_flat_l1_assets(repo_root: Path) -> dict[str, Any]:
    """Load frozen, tracked semantic sources and record every source hash."""
    root = _config_root(repo_root)
    catalog_path = root / "flat_l1_catalog.json"
    schema_path = root / "flat_l1_output_schema.json"
    prompt_path = root / "flat_l1_system_prompt.txt"
    runtime_path = root / "vllm_v0_9_0_runtime_identity.json"
    source_schema_path = repo_root / "config" / "taxonomy_schemas" / "ko-strategyqa.yaml"

    catalog = _load_json(catalog_path, label="Flat-L1 catalog")
    schema = _load_json(schema_path, label="Flat-L1 output schema")
    runtime = _load_json(runtime_path, label="vLLM runtime identity")
    prompt = prompt_path.read_text(encoding="utf-8")
    if not isinstance(catalog, list) or not isinstance(schema, dict) or not isinstance(runtime, dict) or not prompt:
        raise ValueError("Flat-L1 semantic sources are invalid")
    if runtime.get("container_entrypoint") != EXPECTED_CONTAINER_ENTRYPOINT:
        raise ValueError("Flat-L1 runtime identity has an unexpected container entrypoint")
    if runtime.get("container_digest") != VLLM_IMAGE_DIGEST:
        raise ValueError("Flat-L1 runtime identity has an unexpected container digest")
    return {
        "catalog": catalog,
        "schema": schema,
        "prompt": prompt,
        "runtime": runtime,
        "source_hashes": {
            "label_catalog_source_sha256": sha256_file(catalog_path),
            "label_catalog_canonical_sha256": sha256_json(catalog),
            "strict_output_schema_source_sha256": sha256_file(schema_path),
            "strict_output_schema_canonical_sha256": sha256_json(schema),
            "prompt_source_sha256": sha256_file(prompt_path),
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "runtime_identity_source_sha256": sha256_file(runtime_path),
            "taxonomy_schema_source_sha256": sha256_file(source_schema_path),
        },
    }


def _control(mode: str, value: Any = None) -> dict[str, Any]:
    return {"mode": mode, "value": value}


def build_flat_l1_generation_plan(
    *, repo_root: Path, generator_source_commit: str,
    generator_code_contract_sha256: str, generator_code_sha256: str,
) -> dict[str, Any]:
    """Build the sole candidate for the approved closed-set Flat-L1 run."""
    assets = load_flat_l1_assets(repo_root)
    hashes = assets["source_hashes"]
    runtime_identity = assets["runtime"]
    launch_arguments = [
        "--model", MODEL_REPOSITORY,
        "--revision", MODEL_REVISION,
        "--tokenizer", MODEL_REPOSITORY,
        "--tokenizer-revision", MODEL_REVISION,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--generation-config", "vllm",
    ]
    generator = {
        "generator_type": "vllm_closed_set_flat_l1_classifier",
        "generator_version": FLAT_L1_ASSIGNMENT_PROFILE,
        "generator_code_sha256": generator_code_sha256,
        "model_or_algorithm": "Qwen3 schema-driven closed-set Flat-L1 classification",
        "model_or_tokenizer_version": MODEL_REVISION,
        "prompt_template_sha256": hashes["prompt_utf8_sha256"],
        "seed": 0,
        "parameters": {
            "assignment_profile": FLAT_L1_ASSIGNMENT_PROFILE,
            "taxonomy_artifact_id": FLAT_L1_ARTIFACT_ID,
            "label_catalog": assets["catalog"],
            "label_catalog_source_sha256": hashes["label_catalog_source_sha256"],
            "label_catalog_canonical_sha256": hashes["label_catalog_canonical_sha256"],
            "taxonomy_schema_source_sha256": hashes["taxonomy_schema_source_sha256"],
            "strict_output_schema_source_sha256": hashes["strict_output_schema_source_sha256"],
            "strict_output_schema_canonical_sha256": hashes["strict_output_schema_canonical_sha256"],
            "runtime_identity_source_sha256": hashes["runtime_identity_source_sha256"],
            "semantic_batching": "one_passage_per_request_v1",
            "assignment_score_policy": "assigned_1_unknown_0_v1",
            "failure_policy": "fail_loud_retry_no_label_fallback_v1",
            "container_entrypoint": EXPECTED_CONTAINER_ENTRYPOINT,
        },
        "determinism_mode": "replay_required",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": MODEL_REPOSITORY,
        "tokenizer_revision": MODEL_REVISION,
        "pooling": "not_applicable",
        "normalization": "not_applicable",
        "clustering_or_classification_algorithm": "closed_set_flat_l1_schema_driven_classification",
        "clustering_library": "not_applicable",
        "clustering_library_version": "not_applicable",
        "cluster_selection_rule": "fixed_ko_strategyqa_flat_l1_catalog_v1",
        "prompt_template": assets["prompt"],
        "generation_controls": {"temperature": 0, "max_tokens": 100},
        "runtime": {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_repository": MODEL_REPOSITORY,
            "tokenizer_revision": MODEL_REVISION,
            "library_versions": runtime_identity["library_versions"],
            "dependency_lock_sha256": hashes["runtime_identity_source_sha256"],
            "container_digest": VLLM_IMAGE_DIGEST,
        },
        "label_id_rule": "fixed_flat_l1_label_ids_from_tracked_catalog_v1",
        "unknown_outlier_handling": "unknown_only_for_damaged_or_insufficient_title_text_v1",
        "display_label_rule": "tracked_ko_strategyqa_l1_display_labels_v1",
        "vllm_execution": {
            "server_launch": {
                "command": "container_default_openai_api_entrypoint_v1",
                "arguments": launch_arguments,
                "arguments_sha256": sha256_json(launch_arguments),
                "generation_config_mode": "request_controls_only",
                "server_generation_config": None,
                "server_generation_config_sha256": "not_applicable",
                "server_generation_config_launch_value": "not_applicable",
                "model_tokenizer_binding_kind": "actual_execution",
            },
            "chat_template": {
                "mode": "resolved_model_template",
                "sha256": MODEL_CHAT_TEMPLATE_SHA256,
                "content_format": "qwen3_tokenizer_config_chat_template_utf8",
                "launch_argument": "not_applicable",
            },
            "thinking": {
                "enable_thinking": "disabled",
                "reasoning_parser": "not_applicable",
                "response_reasoning_content": "not_applicable",
            },
            "structured_output": {
                "mode": "json_schema",
                "content_format": "json_schema_draft_2020_12_utf8",
                "json_schema": assets["schema"],
                "json_schema_sha256": hashes["strict_output_schema_canonical_sha256"],
                "json_schema_name": "miracl_ko_flat_l1_assignment_v1",
            },
            "sampling_request_controls": {
                "temperature": _control("value", 0),
                "max_tokens": _control("value", 100),
                "top_p": _control("not_applicable"),
                "top_k": _control("not_applicable"),
                "min_p": _control("not_applicable"),
                "stop": _control("not_applicable"),
                "stop_token_ids": _control("not_applicable"),
                "presence_penalty": _control("not_applicable"),
                "frequency_penalty": _control("not_applicable"),
                "repetition_penalty": _control("not_applicable"),
                "seed": _control("value", 0),
                "n": _control("value", 1),
                "logprobs": _control("not_applicable"),
            },
            "request_body_template": {},
            "request_body_template_sha256": "",
        },
    }
    # Let the existing contract derive, rather than hand-copy, the canonical
    # request body. The empty values below are filled by this narrow helper.
    from src.miracl_ko.taxonomy_artifact import _expected_vllm_request_body_template
    request_template = _expected_vllm_request_body_template(
        generator=generator,
        execution=generator["vllm_execution"],
        sampling_values={
            "temperature": 0, "max_tokens": 100, "top_p": None, "top_k": None,
            "min_p": None, "stop": None, "stop_token_ids": None,
            "presence_penalty": None, "frequency_penalty": None,
            "repetition_penalty": None, "seed": 0, "n": 1, "logprobs": None,
        },
        request_model=MODEL_REPOSITORY,
    )
    generator["vllm_execution"]["request_body_template"] = request_template
    generator["vllm_execution"]["request_body_template_sha256"] = sha256_json(request_template)

    plan = build_taxonomy_generation_plan(
        input_provenance=input_provenance(
            data_dir=repo_root / "data" / "miracl-ko",
            revision_lock_path=repo_root / "config" / "miracl_ko_revision_lock.json",
        ),
        generator_source_commit=generator_source_commit,
        generator_code_contract_sha256=generator_code_contract_sha256,
        generator_code_sha256=generator_code_sha256,
        generator=generator,
        input_contract=_semantic_input_contract(),
        run_controls={
            "batch_size": 1,
            "transport_max_concurrency": 32,
            "batch_grouping": "corpus_id_sorted_contiguous_v1",
            "batch_order": "batch_ordinal_ascending_v1",
            "timeout_seconds": 60,
            "max_retries": 2,
            "max_retries_semantics": "per_batch_successful_response_v1",
            "resume_policy": "reuse_verified_complete_batches_only_v1",
            "idempotency_mode": "generation_request_sha256_response_file_v1",
        },
    )
    validate_flat_l1_plan(plan)
    return plan


def validate_flat_l1_plan(plan: Mapping[str, Any]) -> None:
    """Validate the fixed execution controls in addition to generic schema."""
    validate_taxonomy_generation_plan(plan)
    generator = plan["generator"]
    parameters = generator["parameters"]
    if parameters.get("assignment_profile") != FLAT_L1_ASSIGNMENT_PROFILE:
        raise ValueError("Flat-L1 assignment profile is invalid")
    controls = plan["run_controls"]
    if controls.get("batch_size") != 1:
        raise ValueError("Flat-L1 batch_size must be 1 passage per request")
    if controls.get("transport_max_concurrency") != 32:
        raise ValueError("Flat-L1 transport_max_concurrency must be 32")
    if generator.get("generation_controls") != {"temperature": 0, "max_tokens": 100}:
        raise ValueError("Flat-L1 generation controls are invalid")
    if generator["vllm_execution"]["thinking"].get("enable_thinking") != "disabled":
        raise ValueError("Flat-L1 thinking must be disabled")
    if generator["vllm_execution"]["thinking"].get("reasoning_parser") != "not_applicable":
        raise ValueError("Flat-L1 reasoning parser must be not_applicable")
    if generator["vllm_execution"]["structured_output"].get("mode") != "json_schema":
        raise ValueError("Flat-L1 strict JSON schema is required")
    arguments = generator["vllm_execution"]["server_launch"]["arguments"]
    if not arguments or arguments[0] != "--model":
        raise ValueError("Flat-L1 official container arguments must begin with --model")
