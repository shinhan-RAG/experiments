import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_miracl_ko_taxonomy_artifact import require_external_control_path
from src.miracl_ko.taxonomy_artifact import (
    TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
    TAXONOMY_ARTIFACT_SCHEMA_VERSION,
    TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION,
    TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION,
    TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION,
    TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION,
    TAXONOMY_MANIFEST_SCHEMA_VERSION,
    TAXONOMY_VLLM_REQUEST_BODY_TEMPLATE_SCHEMA_VERSION,
    approval_record_sha256,
    build_flat_l1_consumer_adapter,
    build_generator_code_contract,
    build_semantic_generator_inputs,
    build_taxonomy_generator_batch,
    build_taxonomy_generator_run,
    build_taxonomy_generation_plan,
    build_taxonomy_artifact_manifest,
    generator_contract_sha256,
    generation_plan_sha256,
    load_authorized_taxonomy_projection,
    load_integrity_only_taxonomy_projection,
    project_taxonomy_artifact,
    rejoin_taxonomy_generator_outputs,
    rejoin_taxonomy_generator_run_outputs,
    taxonomy_artifact_sha256,
    validate_flat_l1_consumer_adapter,
    validate_deterministic_regeneration,
    validate_focused_taxonomy_treatment_pair,
    validate_taxonomy_approval_record,
    validate_taxonomy_artifact,
    validate_taxonomy_artifact_manifest,
    validate_taxonomy_artifact_authorization,
    validate_taxonomy_generation_preflight,
    validate_taxonomy_generation_plan,
    validate_taxonomy_generator_batch,
    validate_taxonomy_generation_receipt,
    validate_taxonomy_generator_run_against_plan,
    validate_taxonomy_projection,
)
from src.miracl_ko.preparation import file_record, sha256_file


FIXTURE_DATA_DIR = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "miracl_ko_taxonomy_artifact"
    / "data"
)


def source_artifact() -> dict:
    prompt_template = "Synthetic taxonomy test prompt: title and text only."
    server_arguments = [
        "--synthetic-taxonomy-server", "--generation-config", "vllm",
        "--chat-template", "synthetic-qwen3.jinja",
    ]
    response_schema = {
        "type": "object",
        "required": ["request_index", "label_id", "score", "status"],
    }
    canonical_sha256 = lambda value: hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    request_body_template = {
        "schema_version": TAXONOMY_VLLM_REQUEST_BODY_TEMPLATE_SCHEMA_VERSION,
        "endpoint": "/v1/chat/completions",
        "body": {
            "model": "synthetic/mock-model",
            "messages": [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": "{{taxonomy_semantic_payload_json}}"},
            ],
            "temperature": 0.0,
            "max_tokens": 16,
            "seed": 7,
            "n": 1,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "taxonomy_assignment", "schema": response_schema},
            },
        },
    }
    return {
        "schema_version": TAXONOMY_ARTIFACT_SCHEMA_VERSION,
        "taxonomy_artifact_id": "miracl-ko-mock-taxonomy-v1",
        "dataset": "MIRACL",
        "language": "ko",
        "source_revisions": {"topics_qrels": "1" * 40, "corpus": "2" * 40},
        "retrieval_unit": "passage",
        "source_scale": 110_000,
        "projection_scale": 110_000,
        "input_contract": {
            "mapping_key": "corpus_id",
            "semantic_generator_input_fields": ["title", "text"],
            "corpus_id_semantic_use": "prohibited",
            "corpus_id_nonsemantic_uses": [
                "mapping_join", "duplicate_detection", "deterministic_output_order",
            ],
            "forbidden_input_fields": [
                "query", "queries", "qid", "qrel", "qrels", "relevance", "answer", "answers",
                "gold", "gold_id", "gold_ids", "evidence", "evidence_id", "evidence_ids",
                "positive_passages", "negative_passages", "ranked_passage_ids", "evaluation_result",
            ],
        },
        "provenance": {
            "revision_lock_sha256": "a" * 64,
            "preparation_contract_sha256": "b" * 64,
            "input_110k_corpus_sha256": "c" * 64,
            "input_subset_manifest_sha256": "d" * 64,
            "generator": {
                "generator_type": "deterministic_mock",
                "generator_version": "test-v1",
                "generator_code_sha256": "e" * 64,
                "model_or_algorithm": "content-hash-mock",
                "model_or_tokenizer_version": "test-v1",
                "prompt_template_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
                "seed": 7,
                "parameters": {"label_count": 2},
                "determinism_mode": "deterministic",
                "model_repository": "synthetic/mock-model",
                "model_revision": "synthetic-model-revision",
                "tokenizer_repository": "synthetic/mock-tokenizer",
                "tokenizer_revision": "synthetic-tokenizer-revision",
                "pooling": "not_applicable",
                "normalization": "not_applicable",
                "clustering_or_classification_algorithm": "deterministic-content-hash-assignment",
                "clustering_library": "not_applicable",
                "clustering_library_version": "not_applicable",
                "cluster_selection_rule": "fixed-synthetic-label-catalog-v1",
                "prompt_template": prompt_template,
                "generation_controls": {"temperature": 0.0, "max_tokens": 16},
                "runtime": {
                    "model_repository": "synthetic/mock-model",
                    "model_revision": "synthetic-model-revision",
                    "tokenizer_repository": "synthetic/mock-tokenizer",
                    "tokenizer_revision": "synthetic-tokenizer-revision",
                    "library_versions": {"synthetic-runtime": "test-v1"},
                    "dependency_lock_sha256": "f" * 64,
                    "container_digest": "sha256:" + "a" * 64,
                },
                "label_id_rule": "fixed-synthetic-label-id-v1",
                "unknown_outlier_handling": "unknown-label-score-zero-v1",
                "display_label_rule": "fixed-synthetic-display-label-v1",
                "vllm_execution": {
                    "server_launch": {
                        "command": "synthetic-vllm-serve",
                        "arguments": server_arguments,
                        "arguments_sha256": canonical_sha256(server_arguments),
                        "generation_config_mode": "request_controls_only",
                        "server_generation_config": None,
                        "server_generation_config_sha256": "not_applicable",
                        "server_generation_config_launch_value": "not_applicable",
                    },
                    "chat_template": {
                        "mode": "explicit_template",
                        "sha256": "1" * 64,
                        "content_format": "openai_chat_messages_v1",
                        "launch_argument": "synthetic-qwen3.jinja",
                    },
                    "thinking": {
                        "enable_thinking": "disabled",
                        "reasoning_parser": "not_applicable",
                        "response_reasoning_content": "not_applicable",
                    },
                    "structured_output": {
                        "mode": "json_schema",
                        "content_format": "json_object_utf8_v1",
                        "json_schema": response_schema,
                        "json_schema_sha256": canonical_sha256(response_schema),
                        "json_schema_name": "taxonomy_assignment",
                    },
                    "sampling_request_controls": {
                        "temperature": {"mode": "value", "value": 0.0},
                        "max_tokens": {"mode": "value", "value": 16},
                        "top_p": {"mode": "not_applicable", "value": None},
                        "top_k": {"mode": "not_applicable", "value": None},
                        "min_p": {"mode": "not_applicable", "value": None},
                        "stop": {"mode": "not_applicable", "value": None},
                        "stop_token_ids": {"mode": "not_applicable", "value": None},
                        "presence_penalty": {"mode": "not_applicable", "value": None},
                        "frequency_penalty": {"mode": "not_applicable", "value": None},
                        "repetition_penalty": {"mode": "not_applicable", "value": None},
                        "seed": {"mode": "value", "value": 7},
                        "n": {"mode": "value", "value": 1},
                        "logprobs": {"mode": "not_applicable", "value": None},
                    },
                    "request_body_template": request_body_template,
                    "request_body_template_sha256": canonical_sha256(request_body_template),
                },
            },
        },
        "label_catalog": [
            {"label_id": "topic.a", "label": "Topic A"},
            {"label_id": "topic.b", "label": "Topic B"},
            {"label_id": "unknown", "label": "Unknown"},
        ],
        "assignments": [
            {"corpus_id": "article#0", "label_id": "topic.a", "score": 1.0, "status": "assigned"},
            {"corpus_id": "article#1", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
            {"corpus_id": "article#2", "label_id": "unknown", "score": 0.0, "status": "unknown"},
        ],
        "projection": {
            "method": "source_110k_identity_v1",
            "source_artifact_sha256": None,
        },
    }


def generation_plan_for(
    artifact: dict, *, generator_source_commit: str = "f" * 40,
    generator_contract_hash: str = "f" * 64,
    generator_code_hash: str | None = None,
) -> dict:
    generator = copy.deepcopy(artifact["provenance"]["generator"])
    generator["prompt_template_sha256"] = hashlib.sha256(
        generator["prompt_template"].encode("utf-8")
    ).hexdigest()
    if generator_code_hash is not None:
        generator["generator_code_sha256"] = generator_code_hash
    return build_taxonomy_generation_plan(
        input_provenance={
            key: artifact["provenance"][key]
            for key in (
                "revision_lock_sha256", "preparation_contract_sha256",
                "input_110k_corpus_sha256", "input_subset_manifest_sha256",
            )
        },
        generator_source_commit=generator_source_commit,
        generator_code_contract_sha256=generator_contract_hash,
        generator_code_sha256=generator["generator_code_sha256"],
        generator=generator,
        input_contract=artifact["input_contract"],
        run_controls={
            "batch_size": 2,
            "batch_grouping": "corpus_id_sorted_contiguous_v1",
            "batch_order": "batch_ordinal_ascending_v1",
            "timeout_seconds": 120,
            "max_retries": 0,
            "max_retries_semantics": "per_batch_successful_response_v1",
            "resume_policy": "reuse_verified_complete_batches_only_v1",
            "idempotency_mode": "generation_request_sha256_response_file_v1",
        },
    )


def manifest_receipt_for(
    artifact: dict, *, generation_plan_hash: str = "f" * 64,
    approval_hash: str = "f" * 64, contract_hash: str = "f" * 64,
    source_commit: str = "f" * 40,
) -> dict:
    return {
        "schema_version": TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "generation_plan_sha256": generation_plan_hash,
        "approval_record_sha256": approval_hash,
        "generator_contract_sha256": contract_hash,
        "generator_source_commit": source_commit,
        "generation_run_sha256": "e" * 64,
        "ordered_generation_request_sha256s": ["d" * 64],
        "raw_response_files": [{
            "batch_ordinal": 0,
            "generation_request_sha256": "d" * 64,
            "relative_path": "synthetic-response.json",
            "byte_size": 0,
            "sha256": "c" * 64,
            "attempt_count": 1,
            "retry_count": 0,
        }],
        "assignment_canonical_sha256": hashlib.sha256(
            json.dumps(artifact["assignments"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "batch_summary": {"total": 1, "succeeded": 1, "failed": 0, "retries": 0},
        "started_at": "2026-07-24T12:00:00+00:00",
        "ended_at": "2026-07-24T12:00:01+00:00",
        "runtime": copy.deepcopy(artifact["provenance"]["generator"]["runtime"]),
        "determinism_status": "deterministic",
        "replay_required": False,
    }


def integrity_manifest_for(artifact: dict, artifact_records: dict[int, dict]) -> dict:
    source_hash = taxonomy_artifact_sha256(artifact)
    return {
        "schema_version": TAXONOMY_MANIFEST_SCHEMA_VERSION,
        "generated_at": "2026-07-24T12:00:00+00:00",
        "taxonomy_artifact_id": artifact["taxonomy_artifact_id"],
        "dataset": "MIRACL",
        "language": "ko",
        "source_revisions": copy.deepcopy(artifact["source_revisions"]),
        "retrieval_unit": "passage",
        "generator_source_commit": "f" * 40,
        "generator_contract_sha256": "f" * 64,
        "generation_plan_sha256": "f" * 64,
        "approval_record_sha256": "f" * 64,
        "generation_receipt_sha256": "f" * 64,
        "source_scale": 110_000,
        "input_contract": copy.deepcopy(artifact["input_contract"]),
        "provenance": copy.deepcopy(artifact["provenance"]),
        "source_artifact_content_sha256": source_hash,
        "artifacts": {
            str(scale): {
                "role": "source" if scale == 110_000 else "filter_projection",
                "artifact_file": copy.deepcopy(artifact_records[scale]),
                "source_artifact_content_sha256": None if scale == 110_000 else source_hash,
                "projection_method": "source_110k_identity_v1" if scale == 110_000 else "filter_110k_mapping_by_corpus_id_v1",
            }
            for scale in (20_000, 50_000, 110_000)
        },
    }


def write_complete_receipt(
    root: Path, artifact: dict, plan: dict, approval: dict, contract: dict,
) -> tuple[dict, dict, Path]:
    records = [
        {"corpus_id": row["corpus_id"], "title": "Synthetic", "text": f"Synthetic {row['corpus_id']}"}
        for row in artifact["assignments"]
    ]
    run = build_taxonomy_generator_run(
        records,
        generation_plan_sha256=generation_plan_sha256(plan),
        batch_size=plan["run_controls"]["batch_size"],
        run_controls=plan["run_controls"],
    )
    response_dir = root / "responses"
    response_dir.mkdir(exist_ok=True)
    assignments = {row["corpus_id"]: row for row in artifact["assignments"]}
    response_records = []
    for batch in run["batches"]:
        response = {
            "generation_request_sha256": batch["generation_request_sha256"],
            "outputs": [
                {
                    "request_index": row["request_index"],
                    "label_id": assignments[row["corpus_id"]]["label_id"],
                    "score": assignments[row["corpus_id"]]["score"],
                    "status": assignments[row["corpus_id"]]["status"],
                }
                for row in reversed(batch["mapping_envelope"])
            ],
        }
        path = response_dir / f"batch-{batch['batch_ordinal']}.json"
        path.write_text(json.dumps(response, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        response_records.append({
            "batch_ordinal": batch["batch_ordinal"],
            "generation_request_sha256": batch["generation_request_sha256"],
            "relative_path": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "attempt_count": 1,
            "retry_count": 0,
        })
    ordered_assignments = [
        assignments[row["corpus_id"]]
        for batch in run["batches"] for row in batch["mapping_envelope"]
    ]
    receipt = {
        "schema_version": TAXONOMY_GENERATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "generation_plan_sha256": generation_plan_sha256(plan),
        "approval_record_sha256": approval_record_sha256(approval),
        "generator_contract_sha256": generator_contract_sha256(contract),
        "generator_source_commit": plan["generator_source_commit"],
        "generation_run_sha256": run["generation_run_sha256"],
        "ordered_generation_request_sha256s": [batch["generation_request_sha256"] for batch in run["batches"]],
        "raw_response_files": response_records,
        "assignment_canonical_sha256": hashlib.sha256(
            json.dumps(ordered_assignments, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "batch_summary": {"total": len(run["batches"]), "succeeded": len(run["batches"]), "failed": 0, "retries": 0},
        "started_at": "2026-07-24T12:00:00+00:00",
        "ended_at": "2026-07-24T12:00:01+00:00",
        "runtime": copy.deepcopy(plan["generator"]["runtime"]),
        "determinism_status": plan["determinism_mode"],
        "replay_required": plan["determinism_mode"] == "replay_required",
    }
    return run, receipt, response_dir


class MiraclKoTaxonomyArtifactTests(unittest.TestCase):
    def test_audit_requires_control_paths_outside_generator_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "generator-source"
            control_root = root / "control"
            source_root.mkdir()
            control_root.mkdir()
            require_external_control_path(
                control_root / "plan.json", generator_source_root=source_root, label="taxonomy generation plan",
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                require_external_control_path(
                    source_root / "config" / "plan.json",
                    generator_source_root=source_root,
                    label="taxonomy generation plan",
                )

    def test_generator_responses_are_bound_to_their_request_and_run(self):
        plan_hash = "f" * 64
        batch_a = build_taxonomy_generator_batch(
            [
                {"corpus_id": "a#1", "title": "A", "text": "first"},
                {"corpus_id": "a#2", "title": "A", "text": "second"},
            ],
            generation_plan_sha256=plan_hash,
            batch_ordinal=0,
            batch_start=0,
            run_total_records=2,
        )
        batch_b = build_taxonomy_generator_batch(
            [
                {"corpus_id": "b#1", "title": "B", "text": "first"},
                {"corpus_id": "b#2", "title": "B", "text": "second"},
            ],
            generation_plan_sha256=plan_hash,
            batch_ordinal=0,
            batch_start=0,
            run_total_records=2,
        )
        response_a = {
            "generation_request_sha256": batch_a["generation_request_sha256"],
            "outputs": [
                {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"},
                {"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            ],
        }
        response_b = {
            "generation_request_sha256": batch_b["generation_request_sha256"],
            "outputs": [
                {"request_index": 0, "label_id": "topic.b", "score": 0.9, "status": "assigned"},
                {"request_index": 1, "label_id": "topic.a", "score": 0.8, "status": "assigned"},
            ],
        }
        self.assertEqual(
            rejoin_taxonomy_generator_outputs(batch_a, response_a),
            [
                {"corpus_id": "a#1", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
                {"corpus_id": "a#2", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "generation request"):
            rejoin_taxonomy_generator_outputs(batch_a, response_b)
        unknown_as_assigned = copy.deepcopy(response_a)
        unknown_as_assigned["outputs"][0]["label_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "assigned"):
            rejoin_taxonomy_generator_outputs(batch_a, unknown_as_assigned)

        run = build_taxonomy_generator_run(
            [
                {"corpus_id": "run#2", "title": "R", "text": "two"},
                {"corpus_id": "run#1", "title": "R", "text": "one"},
                {"corpus_id": "run#3", "title": "R", "text": "three"},
            ],
            generation_plan_sha256=plan_hash,
            batch_size=2,
        )
        responses = []
        for batch in run["batches"]:
            responses.append({
                "generation_request_sha256": batch["generation_request_sha256"],
                "outputs": [
                    {"request_index": index, "label_id": "topic.a", "score": 1.0, "status": "assigned"}
                    for index in reversed(range(len(batch["semantic_payload"])))
                ],
            })
        self.assertEqual(
            [row["corpus_id"] for row in rejoin_taxonomy_generator_run_outputs(run, list(reversed(responses)))],
            ["run#1", "run#2", "run#3"],
        )
        with self.assertRaisesRegex(ValueError, "missing|count"):
            rejoin_taxonomy_generator_run_outputs(run, responses[:1])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            rejoin_taxonomy_generator_run_outputs(run, [responses[0], responses[0]])
        invalid_run = copy.deepcopy(run)
        invalid_run["batches"][1]["batch_ordinal"] = 0
        with self.assertRaisesRegex(ValueError, "hash|ordinal"):
            rejoin_taxonomy_generator_run_outputs(invalid_run, responses)
        invalid_range = copy.deepcopy(run)
        invalid_range["batches"][0]["batch_end"] -= 1
        with self.assertRaisesRegex(ValueError, "range"):
            rejoin_taxonomy_generator_run_outputs(invalid_range, responses)

    def test_generator_source_checkout_is_separate_from_control_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "generator-source"
            control_root = root / "control-records"
            source_root.mkdir()
            control_root.mkdir()

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", "-C", str(source_root), *args],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.email", "fixture@example.invalid")
            git("config", "user.name", "Fixture")
            (source_root / "generator.py").write_text("# generator H\n", encoding="utf-8")
            git("add", "generator.py")
            git("commit", "-q", "-m", "generator source H")
            source_h = git("rev-parse", "HEAD")
            contract = build_generator_code_contract(source_root, ["generator.py"])
            artifact = source_artifact()
            artifact["provenance"]["generator"]["generator_code_sha256"] = contract["generator_code_sha256"]
            plan = generation_plan_for(
                artifact,
                generator_source_commit=source_h,
                generator_contract_hash=generator_contract_sha256(contract),
                generator_code_hash=contract["generator_code_sha256"],
            )
            approval = {
                "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
                "approval_kind": "actual_execution",
                "status": "approved_for_generation",
                "approved_generation_plan_sha256": generation_plan_sha256(plan),
                "approved_generator_source_commit": source_h,
                "approved_generator_contract_sha256": generator_contract_sha256(contract),
                "approved_generator_code_sha256": contract["generator_code_sha256"],
                "approved_by": "named-approver",
                "approved_at": "2026-07-24T12:00:00+00:00",
                "approval_basis": "separate approved change request",
            }
            for name, value in (("plan.json", plan), ("approval.json", approval), ("contract.json", contract)):
                (control_root / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

            # Legacy same-worktree control files make H dirty. Committing them
            # advances HEAD to H2, which no longer matches the approved H.
            (source_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=source_root,
                )
            git("add", "plan.json")
            git("commit", "-q", "-m", "legacy control H2")
            with self.assertRaisesRegex(ValueError, "source commit"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=source_root,
                )

            git("checkout", "-q", source_h)
            external_plan = json.loads((control_root / "plan.json").read_text(encoding="utf-8"))
            external_approval = json.loads((control_root / "approval.json").read_text(encoding="utf-8"))
            external_contract = json.loads((control_root / "contract.json").read_text(encoding="utf-8"))
            wrong_approval = copy.deepcopy(external_approval)
            wrong_approval["approved_generation_plan_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "generation plan"):
                validate_taxonomy_generation_preflight(
                    external_plan,
                    approval_record=wrong_approval,
                    generator_code_contract=external_contract,
                    verified_input_provenance=external_plan["input_provenance"],
                    generator_source_root=source_root,
                )
            validate_taxonomy_generation_preflight(
                external_plan,
                approval_record=external_approval,
                generator_code_contract=external_contract,
                verified_input_provenance=external_plan["input_provenance"],
                generator_source_root=source_root,
            )

            body_path = control_root / "synthetic-body.json"
            body_path.write_text(json.dumps(artifact), encoding="utf-8")
            generation_run, generation_receipt, response_dir = write_complete_receipt(
                control_root, artifact, external_plan, external_approval, external_contract
            )
            verified_receipt = validate_taxonomy_generation_receipt(
                generation_receipt,
                generation_plan=external_plan,
                approval_record=external_approval,
                generator_code_contract=external_contract,
                generation_run=generation_run,
                raw_response_dir=response_dir,
            )
            manifest = build_taxonomy_artifact_manifest(
                full_artifact=artifact,
                artifact_records={
                    scale: file_record(body_path, relative_to=control_root)
                    for scale in (20_000, 50_000, 110_000)
                },
                generator_source_commit=source_h,
                generator_contract_sha256=generator_contract_sha256(contract),
                generation_plan_sha256=generation_plan_sha256(plan),
                approval_record_sha256=approval_record_sha256(approval),
                verified_receipt=verified_receipt,
            )
            validate_taxonomy_artifact_authorization(
                manifest,
                generation_plan=external_plan,
                approval_record=external_approval,
                generator_code_contract=external_contract,
                verified_input_provenance=external_plan["input_provenance"],
                generator_source_root=source_root,
                generation_run=generation_run,
                generation_receipt=generation_receipt,
                raw_response_dir=response_dir,
                source_artifact=artifact,
            )

    def test_generator_response_request_indexes_allow_reordering_but_not_positional_rejoin(self):
        records = [
            {"corpus_id": "article#2", "title": "두", "text": "둘"},
            {"corpus_id": "article#1", "title": "하나", "text": "첫"},
        ]
        batch = build_taxonomy_generator_batch(records, generation_plan_sha256="f" * 64)
        ordered = [
            {"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"},
        ]
        expected = [
            {"corpus_id": "article#1", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            {"corpus_id": "article#2", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
        ]
        response = {"generation_request_sha256": batch["generation_request_sha256"], "outputs": ordered}
        reversed_response = {"generation_request_sha256": batch["generation_request_sha256"], "outputs": list(reversed(ordered))}
        self.assertEqual(rejoin_taxonomy_generator_outputs(batch, response), expected)
        self.assertEqual(rejoin_taxonomy_generator_outputs(batch, reversed_response), expected)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            rejoin_taxonomy_generator_outputs(
                batch,
                {"generation_request_sha256": batch["generation_request_sha256"], "outputs": [
                    {"label_id": "topic.a", "score": 0.9, "status": "assigned"},
                    {"label_id": "topic.b", "score": 0.8, "status": "assigned"},
                ]},
            )
        for invalid in (
            [{"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
             {"request_index": 0, "label_id": "topic.b", "score": 0.8, "status": "assigned"}],
            [{"request_index": 2, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
             {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"}],
            [{"request_index": True, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
             {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"}],
            [{"request_index": "0", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
             {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"}],
            [{"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"}],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "request_index|count"):
                    rejoin_taxonomy_generator_outputs(
                        batch,
                        {"generation_request_sha256": batch["generation_request_sha256"], "outputs": invalid},
                    )

    def test_generator_batch_is_order_independent_and_rejoins_only_through_envelope(self):
        forward = [
            {"corpus_id": "article#2", "title": "두", "text": "둘"},
            {"corpus_id": "article#1", "title": "하나", "text": "첫"},
        ]
        reverse = list(reversed(forward))

        # Compatibility extraction is now deterministic, but it has no output
        # correlation and is never an approved rejoin interface.
        self.assertEqual(
            build_semantic_generator_inputs(forward),
            build_semantic_generator_inputs(reverse),
        )
        batch = build_taxonomy_generator_batch(forward, generation_plan_sha256="f" * 64)
        self.assertEqual(batch, build_taxonomy_generator_batch(reverse, generation_plan_sha256="f" * 64))
        self.assertEqual(batch["schema_version"], TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION)
        self.assertEqual(
            batch["mapping_envelope"],
            [
                {"request_index": 0, "corpus_id": "article#1"},
                {"request_index": 1, "corpus_id": "article#2"},
            ],
        )
        self.assertEqual(batch["semantic_payload"], [{"title": "하나", "text": "첫"}, {"title": "두", "text": "둘"}])
        self.assertTrue(all("corpus_id" not in row for row in batch["semantic_payload"]))

        outputs = [
            {"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"},
        ]
        self.assertEqual(
            rejoin_taxonomy_generator_outputs(
                batch, {"generation_request_sha256": batch["generation_request_sha256"], "outputs": outputs},
            ),
            [
                {"corpus_id": "article#1", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
                {"corpus_id": "article#2", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
            ],
        )
        for outputs_with_wrong_count in (outputs[:1], outputs + outputs[:1]):
            with self.subTest(output_count=len(outputs_with_wrong_count)):
                with self.assertRaisesRegex(ValueError, "count"):
                    rejoin_taxonomy_generator_outputs(
                        batch,
                        {
                            "generation_request_sha256": batch["generation_request_sha256"],
                            "outputs": outputs_with_wrong_count,
                        },
                    )
        with self.assertRaisesRegex(ValueError, "duplicate corpus_id"):
            build_taxonomy_generator_batch(forward + [dict(forward[0])], generation_plan_sha256="f" * 64)

        leaking = copy.deepcopy(batch)
        leaking["semantic_payload"][0]["corpus_id"] = "article#1"
        with self.assertRaisesRegex(ValueError, "semantic payload"):
            validate_taxonomy_generator_batch(leaking)
        mismatched = copy.deepcopy(batch)
        mismatched["mapping_envelope"] = list(reversed(mismatched["mapping_envelope"]))
        with self.assertRaisesRegex(ValueError, "mapping envelope"):
            rejoin_taxonomy_generator_outputs(
                mismatched,
                {"generation_request_sha256": batch["generation_request_sha256"], "outputs": outputs},
            )

    def test_flat_l1_adapter_preserves_display_labels_and_excludes_unknown(self):
        artifact = source_artifact()
        expected_ids = {"article#0", "article#1", "article#2"}
        adapter = build_flat_l1_consumer_adapter(artifact, expected_corpus_ids=expected_ids)
        self.assertEqual(adapter["schema_version"], TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION)
        self.assertEqual(adapter["document_taxonomy"], {
            "article#0": {"L1": "Topic A"},
            "article#1": {"L1": "Topic B"},
        })
        self.assertEqual(adapter["agent_taxonomy_schema"], {"L1": ["Topic A", "Topic B"], "L2": {}})
        self.assertEqual(adapter["score_handling"], "not_used_by_existing_soft_boost")
        validate_flat_l1_consumer_adapter(adapter, artifact, expected_corpus_ids=expected_ids)

        label_id_leak = copy.deepcopy(adapter)
        label_id_leak["document_taxonomy"]["article#0"] = {"L1": "topic.a"}
        with self.assertRaisesRegex(ValueError, "display label|exact match"):
            validate_flat_l1_consumer_adapter(label_id_leak, artifact, expected_corpus_ids=expected_ids)
        unknown_boost = copy.deepcopy(adapter)
        unknown_boost["document_taxonomy"]["article#2"] = {"L1": "Unknown"}
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_flat_l1_consumer_adapter(unknown_boost, artifact, expected_corpus_ids=expected_ids)
        weighted = copy.deepcopy(adapter)
        weighted["score_handling"] = "use_as_boost_weight"
        with self.assertRaisesRegex(ValueError, "score"):
            validate_flat_l1_consumer_adapter(weighted, artifact, expected_corpus_ids=expected_ids)
        omitted = copy.deepcopy(adapter)
        del omitted["document_taxonomy"]["article#1"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_flat_l1_consumer_adapter(omitted, artifact, expected_corpus_ids=expected_ids)
        added = copy.deepcopy(adapter)
        added["document_taxonomy"]["article#unexpected"] = {"L1": "Topic A"}
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_flat_l1_consumer_adapter(added, artifact, expected_corpus_ids=expected_ids)

    def test_generator_provenance_requires_separate_approval_and_clean_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator_path = root / "generator.py"
            generator_path.write_text("# deterministic mock\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "fixture@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Fixture"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "generator.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "generator source"], check=True)
            source_commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
            ).stdout.strip()
            code_contract = build_generator_code_contract(root, ["generator.py"])
            full = source_artifact()
            plan = generation_plan_for(
                full,
                generator_source_commit=source_commit,
                generator_contract_hash=generator_contract_sha256(code_contract),
                generator_code_hash=code_contract["generator_code_sha256"],
            )
            self.assertEqual(plan["schema_version"], TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION)
            self.assertNotIn("artifacts", plan)
            self.assertNotIn("artifact_sha256", plan)
            validate_taxonomy_generation_plan(plan)
            approval = {
                "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
                "approval_kind": "actual_execution",
                "status": "approved_for_generation",
                "approved_generation_plan_sha256": generation_plan_sha256(plan),
                "approved_generator_source_commit": source_commit,
                "approved_generator_contract_sha256": generator_contract_sha256(code_contract),
                "approved_generator_code_sha256": code_contract["generator_code_sha256"],
                "approved_by": "named-approver",
                "approved_at": "2026-07-24T12:00:00+00:00",
                "approval_basis": "separate approved change request",
            }
            validate_taxonomy_generation_preflight(
                plan,
                approval_record=approval,
                generator_code_contract=code_contract,
                verified_input_provenance=plan["input_provenance"],
                generator_source_root=root,
            )
            with self.assertRaisesRegex(FileNotFoundError, "approval"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=None,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            other_plan = copy.deepcopy(plan)
            other_plan["generator"]["parameters"]["label_count"] = 3
            with self.assertRaisesRegex(ValueError, "generation plan"):
                validate_taxonomy_generation_preflight(
                    other_plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            changed_input = dict(plan["input_provenance"])
            changed_input["input_110k_corpus_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "input provenance"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=changed_input,
                    generator_source_root=root,
                )
            subprocess.run(["git", "-C", str(root), "commit", "--allow-empty", "-q", "-m", "different source"], check=True)
            with self.assertRaisesRegex(ValueError, "source commit"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            subprocess.run(["git", "-C", str(root), "checkout", "-q", source_commit], check=True)
            dirty_marker = root / "control-record.json"
            dirty_marker.write_text("must live outside source root", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            dirty_marker.unlink()
            wrong_approval = copy.deepcopy(approval)
            wrong_approval["approved_generator_code_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "generator code"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=wrong_approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            generator_path.write_text("# modified after contract\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )
            subprocess.run(["git", "-C", str(root), "checkout", "-q", source_commit], check=True)

            synthetic_approval = {
                **approval,
                "approval_kind": "synthetic_test",
                "status": "synthetic_only",
                "approved_by": "synthetic-fixture",
            }
            validate_taxonomy_approval_record(synthetic_approval, allow_synthetic=True)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                validate_taxonomy_approval_record(synthetic_approval)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=synthetic_approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=root,
                )

    def test_generation_receipt_binds_plan_run_raw_bytes_and_assignments(self):
        """RED regressions: receipt data cannot be swapped into an artifact."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator_path = root / "generator.py"
            generator_path.write_text("# deterministic mock\n", encoding="utf-8")
            contract = build_generator_code_contract(root, ["generator.py"])
            artifact = source_artifact()
            artifact["provenance"]["generator"]["generator_code_sha256"] = contract["generator_code_sha256"]
            plan = generation_plan_for(
                artifact,
                generator_contract_hash=generator_contract_sha256(contract),
                generator_code_hash=contract["generator_code_sha256"],
            )
            approval = {
                "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
                "approval_kind": "actual_execution",
                "status": "approved_for_generation",
                "approved_generation_plan_sha256": generation_plan_sha256(plan),
                "approved_generator_source_commit": plan["generator_source_commit"],
                "approved_generator_contract_sha256": generator_contract_sha256(contract),
                "approved_generator_code_sha256": contract["generator_code_sha256"],
                "approved_by": "named-approver",
                "approved_at": "2026-07-24T12:00:00+00:00",
                "approval_basis": "synthetic receipt contract regression fixture",
            }
            run, receipt, response_dir = write_complete_receipt(root, artifact, plan, approval, contract)
            verified_receipt = validate_taxonomy_generation_receipt(
                receipt,
                generation_plan=plan,
                approval_record=approval,
                generator_code_contract=contract,
                generation_run=run,
                raw_response_dir=response_dir,
            )
            self.assertEqual(list(verified_receipt.assignments), artifact["assignments"])
            self.assertEqual(verified_receipt.generator_code_sha256, contract["generator_code_sha256"])
            self.assertEqual(verified_receipt.generator_spec_sha256, plan["generator_spec_sha256"])

            foreign_generator = root / "g2.py"
            foreign_generator.write_text("# different deterministic mock\n", encoding="utf-8")
            foreign_contract = build_generator_code_contract(root, ["g2.py"])
            foreign_approval = copy.deepcopy(approval)
            foreign_approval["approved_generator_contract_sha256"] = generator_contract_sha256(foreign_contract)
            # Keep the original g1 aggregate code hash: only the contract is swapped.
            foreign_receipt = copy.deepcopy(receipt)
            foreign_receipt["approval_record_sha256"] = approval_record_sha256(foreign_approval)
            foreign_receipt["generator_contract_sha256"] = generator_contract_sha256(foreign_contract)
            with self.assertRaisesRegex(ValueError, "plan code contract"):
                validate_taxonomy_generation_receipt(
                    foreign_receipt,
                    generation_plan=plan,
                    approval_record=foreign_approval,
                    generator_code_contract=foreign_contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )

            wrong_code_approval = copy.deepcopy(approval)
            wrong_code_approval["approved_generator_code_sha256"] = foreign_contract["generator_code_sha256"]
            wrong_code_receipt = copy.deepcopy(receipt)
            wrong_code_receipt["approval_record_sha256"] = approval_record_sha256(wrong_code_approval)
            with self.assertRaisesRegex(ValueError, "approval generator code"):
                validate_taxonomy_generation_receipt(
                    wrong_code_receipt,
                    generation_plan=plan,
                    approval_record=wrong_code_approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )

            provenance_swapped_artifact = copy.deepcopy(artifact)
            provenance_swapped_artifact["provenance"]["generator"]["model_or_algorithm"] = "other-generator"
            with self.assertRaisesRegex(ValueError, "generator specification"):
                build_taxonomy_artifact_manifest(
                    full_artifact=provenance_swapped_artifact,
                    artifact_records={
                        scale: {"relative_path": f"{scale}.json", "byte_size": 0, "sha256": "a" * 64}
                        for scale in (20_000, 50_000, 110_000)
                    },
                    generator_source_commit=plan["generator_source_commit"],
                    generator_contract_sha256=generator_contract_sha256(contract),
                    generation_plan_sha256=generation_plan_sha256(plan),
                    approval_record_sha256=approval_record_sha256(approval),
                    verified_receipt=verified_receipt,
                )

            retry_over_limit = copy.deepcopy(receipt)
            retry_over_limit["raw_response_files"][0]["attempt_count"] = 2
            retry_over_limit["raw_response_files"][0]["retry_count"] = 1
            retry_over_limit["batch_summary"]["retries"] = 1
            with self.assertRaisesRegex(ValueError, "per-batch max_retries"):
                validate_taxonomy_generation_receipt(
                    retry_over_limit,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            retry_summary_mismatch = copy.deepcopy(receipt)
            retry_summary_mismatch["batch_summary"]["retries"] = 1
            with self.assertRaisesRegex(ValueError, "retry count"):
                validate_taxonomy_generation_receipt(
                    retry_summary_mismatch,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )

            altered_artifact = source_artifact()
            altered_artifact["provenance"]["generator"]["generator_code_sha256"] = contract["generator_code_sha256"]
            altered_launch = altered_artifact["provenance"]["generator"]["vllm_execution"]["server_launch"]
            altered_launch["arguments"] = [
                "--synthetic-taxonomy-server", "--different-launch-control",
                "--generation-config", "vllm", "--chat-template", "synthetic-qwen3.jinja",
            ]
            altered_launch["arguments_sha256"] = hashlib.sha256(
                json.dumps(altered_launch["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            altered_plan = generation_plan_for(
                altered_artifact,
                generator_contract_hash=generator_contract_sha256(contract),
                generator_code_hash=contract["generator_code_sha256"],
            )
            self.assertEqual(
                altered_plan["generator"]["runtime"]["container_digest"],
                plan["generator"]["runtime"]["container_digest"],
            )
            self.assertNotEqual(generation_plan_sha256(altered_plan), generation_plan_sha256(plan))

            different_controls = copy.deepcopy(plan["run_controls"])
            different_controls["batch_size"] = 1
            wrong_controls = build_taxonomy_generator_run(
                [
                    {"corpus_id": row["corpus_id"], "title": "Synthetic", "text": "Synthetic"}
                    for row in artifact["assignments"]
                ],
                generation_plan_sha256=generation_plan_sha256(plan),
                batch_size=1,
                run_controls=different_controls,
            )
            with self.assertRaisesRegex(ValueError, "controls"):
                validate_taxonomy_generator_run_against_plan(wrong_controls, plan)
            invalid_grouping = copy.deepcopy(plan["run_controls"])
            invalid_grouping["batch_grouping"] = "caller_order_v1"
            with self.assertRaisesRegex(ValueError, "batch_grouping"):
                build_taxonomy_generator_run(
                    [{"corpus_id": "one#0", "title": "One", "text": "One"}],
                    generation_plan_sha256=generation_plan_sha256(plan),
                    batch_size=invalid_grouping["batch_size"],
                    run_controls=invalid_grouping,
                )
            invalid_retry_semantics = copy.deepcopy(plan["run_controls"])
            invalid_retry_semantics["max_retries_semantics"] = "run_total_v1"
            with self.assertRaisesRegex(ValueError, "max_retries semantics"):
                build_taxonomy_generator_run(
                    [{"corpus_id": "one#0", "title": "One", "text": "One"}],
                    generation_plan_sha256=generation_plan_sha256(plan),
                    batch_size=invalid_retry_semantics["batch_size"],
                    run_controls=invalid_retry_semantics,
                )

            first_response = response_dir / receipt["raw_response_files"][0]["relative_path"]
            original_bytes = first_response.read_bytes()
            first_response.write_bytes(b"X" + original_bytes[1:])
            with self.assertRaisesRegex(ValueError, "sha256"):
                validate_taxonomy_generation_receipt(
                    receipt,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            first_response.write_bytes(original_bytes)

            other_run = build_taxonomy_generator_run(
                [
                    {"corpus_id": "other#0", "title": "Other", "text": "Other"},
                    {"corpus_id": "other#1", "title": "Other", "text": "Other"},
                ],
                generation_plan_sha256=generation_plan_sha256(plan),
                batch_size=plan["run_controls"]["batch_size"],
                run_controls=plan["run_controls"],
            )
            injected = copy.deepcopy(receipt)
            injected["generation_run_sha256"] = other_run["generation_run_sha256"]
            with self.assertRaisesRegex(ValueError, "ordered request|another batch|run"):
                validate_taxonomy_generation_receipt(
                    injected,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=other_run,
                    raw_response_dir=response_dir,
                )

            duplicate_success = copy.deepcopy(receipt)
            duplicate_success["raw_response_files"][1]["batch_ordinal"] = 0
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_taxonomy_generation_receipt(
                    duplicate_success,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )

            partial = copy.deepcopy(receipt)
            partial["status"] = "partial"
            partial["assignment_canonical_sha256"] = None
            partial["raw_response_files"] = []
            partial["batch_summary"] = {
                "total": len(run["batches"]), "succeeded": 0,
                "failed": len(run["batches"]), "retries": 0,
            }
            with self.assertRaisesRegex(ValueError, "verified generation receipt"):
                build_taxonomy_artifact_manifest(
                    full_artifact=artifact,
                    artifact_records={
                        scale: {"relative_path": f"{scale}.json", "byte_size": 0, "sha256": "a" * 64}
                        for scale in (20_000, 50_000, 110_000)
                    },
                    generator_source_commit=plan["generator_source_commit"],
                    generator_contract_sha256=generator_contract_sha256(contract),
                    generation_plan_sha256=generation_plan_sha256(plan),
                    approval_record_sha256=approval_record_sha256(approval),
                    verified_receipt=partial,
                )
            with self.assertRaisesRegex(ValueError, "verified generation receipt"):
                build_taxonomy_artifact_manifest(
                    full_artifact=artifact,
                    artifact_records={
                        scale: {"relative_path": f"{scale}.json", "byte_size": 0, "sha256": "a" * 64}
                        for scale in (20_000, 50_000, 110_000)
                    },
                    generator_source_commit=plan["generator_source_commit"],
                    generator_contract_sha256=generator_contract_sha256(contract),
                    generation_plan_sha256=generation_plan_sha256(plan),
                    approval_record_sha256=approval_record_sha256(approval),
                    verified_receipt=manifest_receipt_for(
                        artifact,
                        generation_plan_hash=generation_plan_sha256(plan),
                        approval_hash=approval_record_sha256(approval),
                        contract_hash=generator_contract_sha256(contract),
                        source_commit=plan["generator_source_commit"],
                    ),
                )
            missing_raw = response_dir / receipt["raw_response_files"][0]["relative_path"]
            missing_raw.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "raw response"):
                validate_taxonomy_generation_receipt(
                    receipt,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            wrong_plan_approval = copy.deepcopy(approval)
            wrong_plan_approval["approved_generation_plan_sha256"] = generation_plan_sha256(altered_plan)
            wrong_plan_receipt = copy.deepcopy(receipt)
            wrong_plan_receipt["approval_record_sha256"] = approval_record_sha256(wrong_plan_approval)
            with self.assertRaisesRegex(ValueError, "generation plan"):
                validate_taxonomy_generation_receipt(
                    wrong_plan_receipt,
                    generation_plan=plan,
                    approval_record=wrong_plan_approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            synthetic_approval = copy.deepcopy(approval)
            synthetic_approval["approval_kind"] = "synthetic_test"
            synthetic_approval["status"] = "synthetic_only"
            synthetic_approval["approved_by"] = "synthetic-fixture"
            synthetic_receipt = copy.deepcopy(receipt)
            synthetic_receipt["approval_record_sha256"] = approval_record_sha256(synthetic_approval)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                validate_taxonomy_generation_receipt(
                    synthetic_receipt,
                    generation_plan=plan,
                    approval_record=synthetic_approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            with self.assertRaisesRegex(ValueError, "forbidden"):
                build_taxonomy_generator_run(
                    [{"corpus_id": "leak#0", "title": "T", "text": "X", "qrel": "forbidden"}],
                    generation_plan_sha256=generation_plan_sha256(plan),
                    batch_size=plan["run_controls"]["batch_size"],
                    run_controls=plan["run_controls"],
                )

    def test_vllm_plan_requires_effective_launch_and_request_contract(self):
        """RED regressions: serving declarations must equal launch and request bytes."""
        def canonical_sha256(value: object) -> str:
            return hashlib.sha256(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        artifact = source_artifact()
        plan = generation_plan_for(artifact)
        launch = artifact["provenance"]["generator"]["vllm_execution"]["server_launch"]
        self.assertEqual(plan["generator"]["runtime"]["container_digest"], "sha256:" + "a" * 64)

        missing_request_controls_override = copy.deepcopy(artifact)
        missing_launch = missing_request_controls_override["provenance"]["generator"]["vllm_execution"]["server_launch"]
        missing_launch["arguments"] = [
            "--synthetic-taxonomy-server", "--chat-template", "synthetic-qwen3.jinja",
        ]
        missing_launch["arguments_sha256"] = canonical_sha256(missing_launch["arguments"])
        with self.assertRaisesRegex(ValueError, "request_controls_only generation-config"):
            generation_plan_for(missing_request_controls_override)

        missing_thinking_control = copy.deepcopy(artifact)
        missing_thinking_template = missing_thinking_control["provenance"]["generator"]["vllm_execution"]
        del missing_thinking_template["request_body_template"]["body"]["extra_body"]["chat_template_kwargs"]
        missing_thinking_template["request_body_template_sha256"] = canonical_sha256(
            missing_thinking_template["request_body_template"]
        )
        with self.assertRaisesRegex(ValueError, "request body template"):
            generation_plan_for(missing_thinking_control)

        missing_structured_schema = copy.deepcopy(artifact)
        missing_schema_template = missing_structured_schema["provenance"]["generator"]["vllm_execution"]
        del missing_schema_template["request_body_template"]["body"]["response_format"]
        missing_schema_template["request_body_template_sha256"] = canonical_sha256(
            missing_schema_template["request_body_template"]
        )
        with self.assertRaisesRegex(ValueError, "request body template"):
            generation_plan_for(missing_structured_schema)

        enabled_reasoning = copy.deepcopy(artifact)
        enabled_execution = enabled_reasoning["provenance"]["generator"]["vllm_execution"]
        enabled_execution["thinking"] = {
            "enable_thinking": "enabled",
            "reasoning_parser": "qwen3",
            "response_reasoning_content": "included",
        }
        enabled_launch = enabled_execution["server_launch"]
        enabled_launch["arguments"] += ["--reasoning-parser", "qwen3"]
        enabled_launch["arguments_sha256"] = canonical_sha256(enabled_launch["arguments"])
        enabled_execution["request_body_template"]["body"]["extra_body"]["chat_template_kwargs"] = {
            "enable_thinking": True,
        }
        enabled_execution["request_body_template_sha256"] = canonical_sha256(
            enabled_execution["request_body_template"]
        )
        generation_plan_for(enabled_reasoning)
        enabled_launch["arguments"] = enabled_launch["arguments"][:-2]
        enabled_launch["arguments_sha256"] = canonical_sha256(enabled_launch["arguments"])
        with self.assertRaisesRegex(ValueError, "reasoning_parser"):
            generation_plan_for(enabled_reasoning)

        changed_request_contract = copy.deepcopy(artifact)
        changed_execution = changed_request_contract["provenance"]["generator"]["vllm_execution"]
        changed_execution["sampling_request_controls"]["top_p"] = {"mode": "value", "value": 0.9}
        changed_execution["request_body_template"]["body"]["top_p"] = 0.9
        changed_execution["request_body_template_sha256"] = canonical_sha256(
            changed_execution["request_body_template"]
        )
        changed_plan = generation_plan_for(changed_request_contract)
        self.assertEqual(
            changed_plan["generator"]["runtime"]["container_digest"],
            plan["generator"]["runtime"]["container_digest"],
        )
        self.assertNotEqual(generation_plan_sha256(changed_plan), generation_plan_sha256(plan))

        server_config_mode = copy.deepcopy(artifact)
        server_launch = server_config_mode["provenance"]["generator"]["vllm_execution"]["server_launch"]
        server_config = {"temperature": 0.3}
        server_launch["generation_config_mode"] = "server_generation_config"
        server_launch["server_generation_config"] = server_config
        server_launch["server_generation_config_sha256"] = canonical_sha256(server_config)
        server_launch["server_generation_config_launch_value"] = "/approved/generation-config"
        server_launch["arguments"] = [
            "--synthetic-taxonomy-server", "--generation-config", "/approved/generation-config",
            "--chat-template", "synthetic-qwen3.jinja",
        ]
        server_launch["arguments_sha256"] = canonical_sha256(server_launch["arguments"])
        generation_plan_for(server_config_mode)
        server_launch["arguments"][2] = "/other/generation-config"
        server_launch["arguments_sha256"] = canonical_sha256(server_launch["arguments"])
        with self.assertRaisesRegex(ValueError, "server generation-config"):
            generation_plan_for(server_config_mode)

    def test_strict_types_canonical_labels_and_manifest_timestamp(self):
        artifact = source_artifact()
        seed_bool = copy.deepcopy(artifact)
        seed_bool["provenance"]["generator"]["seed"] = True
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_taxonomy_artifact(seed_bool)
        score_bool = copy.deepcopy(artifact)
        score_bool["assignments"][0]["score"] = True
        with self.assertRaisesRegex(ValueError, "score"):
            validate_taxonomy_artifact(score_bool)
        nan_parameter = copy.deepcopy(artifact)
        nan_parameter["provenance"]["generator"]["parameters"] = {"temperature": float("nan")}
        with self.assertRaisesRegex(ValueError, "parameters"):
            validate_taxonomy_artifact(nan_parameter)
        non_nfc_label = copy.deepcopy(artifact)
        non_nfc_label["label_catalog"][0]["label"] = "Cafe\u0301"
        with self.assertRaisesRegex(ValueError, "NFC"):
            validate_taxonomy_artifact(non_nfc_label)
        whitespace_duplicate = copy.deepcopy(artifact)
        whitespace_duplicate["label_catalog"].append({"label_id": "topic.c", "label": "Topic  A"})
        with self.assertRaisesRegex(ValueError, "whitespace|duplicate"):
            validate_taxonomy_artifact(whitespace_duplicate)
        unsorted_catalog = copy.deepcopy(artifact)
        unsorted_catalog["label_catalog"][0], unsorted_catalog["label_catalog"][1] = (
            unsorted_catalog["label_catalog"][1], unsorted_catalog["label_catalog"][0]
        )
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_taxonomy_artifact(unsorted_catalog)

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            path = data_dir / "110000.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            manifest = integrity_manifest_for(
                artifact,
                {scale: file_record(path, relative_to=data_dir) for scale in (20_000, 50_000, 110_000)},
            )
            manifest["generated_at"] = "2026-07-24T12:00:00"
            with self.assertRaisesRegex(ValueError, "generated_at"):
                validate_taxonomy_artifact_manifest(
                    manifest,
                    data_dir=data_dir,
                    expected_ids_by_scale={scale: {"article#0", "article#1", "article#2"} for scale in (20_000, 50_000, 110_000)},
                )

    def test_semantic_generator_inputs_allow_only_title_and_text(self):
        records = [
            {"corpus_id": "article#0", "title": "제목", "text": "본문"},
            {"corpus_id": "other#99", "title": "제목", "text": "본문"},
        ]
        semantic_inputs = build_semantic_generator_inputs(records)
        self.assertEqual(semantic_inputs, [{"title": "제목", "text": "본문"}] * 2)
        self.assertTrue(all(set(row) == {"title", "text"} for row in semantic_inputs))

        for forbidden_field in ("query", "qrel", "relevance", "gold", "evidence_id"):
            with self.subTest(forbidden_field=forbidden_field):
                leaking = [{**records[0], forbidden_field: "forbidden"}]
                with self.assertRaisesRegex(ValueError, "forbidden|unsupported"):
                    build_semantic_generator_inputs(leaking)

    def test_artifact_requires_complete_unique_passage_mapping_and_provenance(self):
        artifact = source_artifact()
        audit = validate_taxonomy_artifact(
            artifact,
            expected_corpus_ids={"article#0", "article#1", "article#2"},
        )
        self.assertEqual(audit["coverage_rate"], 1.0)
        self.assertEqual(audit["unknown_assignment_count"], 1)

        duplicate = copy.deepcopy(artifact)
        duplicate["assignments"].append(copy.deepcopy(duplicate["assignments"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate corpus_id"):
            validate_taxonomy_artifact(duplicate)

        orphan = copy.deepcopy(artifact)
        orphan["assignments"][2]["corpus_id"] = "article#3"
        with self.assertRaisesRegex(ValueError, "orphan|coverage"):
            validate_taxonomy_artifact(orphan, expected_corpus_ids={"article#0", "article#1", "article#2"})

        missing_seed = copy.deepcopy(artifact)
        del missing_seed["provenance"]["generator"]["seed"]
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_taxonomy_artifact(missing_seed)

    def test_projection_is_exact_filter_and_shared_assignments_are_invariant(self):
        full = source_artifact()
        projected = project_taxonomy_artifact(full, {"article#0", "article#2"}, target_scale=20_000)
        validate_taxonomy_artifact(projected, expected_corpus_ids={"article#0", "article#2"})
        full_by_id = {row["corpus_id"]: row for row in full["assignments"]}
        projected_by_id = {row["corpus_id"]: row for row in projected["assignments"]}
        self.assertEqual(projected_by_id["article#0"], full_by_id["article#0"])

        changed = copy.deepcopy(projected)
        changed["assignments"][0]["label_id"] = "topic.b"
        with self.assertRaisesRegex(ValueError, "projection|shared"):
            validate_taxonomy_projection(
                full, changed, expected_corpus_ids={"article#0", "article#2"}, target_scale=20_000
            )
        changed_source_revision = copy.deepcopy(projected)
        changed_source_revision["source_revisions"]["corpus"] = "3" * 40
        with self.assertRaisesRegex(ValueError, "source_revisions"):
            validate_taxonomy_projection(
                full, changed_source_revision,
                expected_corpus_ids={"article#0", "article#2"}, target_scale=20_000,
            )

    def test_manifest_rechecks_artifact_hash_and_projection_files(self):
        full = source_artifact()
        projected_20k = project_taxonomy_artifact(full, {"article#0", "article#2"}, target_scale=20_000)
        projected_50k = project_taxonomy_artifact(full, {"article#0", "article#1", "article#2"}, target_scale=50_000)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            artifact_dir = data_dir / "taxonomy"
            artifact_dir.mkdir()
            paths = {}
            for scale, artifact in ((110_000, full), (50_000, projected_50k), (20_000, projected_20k)):
                path = artifact_dir / f"{scale}.json"
                path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                paths[scale] = path
            manifest = integrity_manifest_for(
                full,
                {scale: file_record(path, relative_to=data_dir) for scale, path in paths.items()},
            )
            manifest_path = data_dir / "taxonomy_manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            audit = validate_taxonomy_artifact_manifest(
                manifest,
                data_dir=data_dir,
                expected_ids_by_scale={
                    20_000: {"article#0", "article#2"},
                    50_000: {"article#0", "article#1", "article#2"},
                    110_000: {"article#0", "article#1", "article#2"},
                },
                expected_provenance={
                    key: full["provenance"][key]
                    for key in (
                        "revision_lock_sha256", "preparation_contract_sha256",
                        "input_110k_corpus_sha256", "input_subset_manifest_sha256",
                    )
                },
                expected_source_revisions=full["source_revisions"],
            )
            self.assertEqual(audit["status"], "ready")
            self.assertEqual(audit["artifact_sha256"], taxonomy_artifact_sha256(full))

            wrong_revision = copy.deepcopy(manifest)
            wrong_revision["source_revisions"]["corpus"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "source_revisions"):
                validate_taxonomy_artifact_manifest(
                    wrong_revision,
                    data_dir=data_dir,
                    expected_ids_by_scale={
                        20_000: {"article#0", "article#2"},
                        50_000: {"article#0", "article#1", "article#2"},
                        110_000: {"article#0", "article#1", "article#2"},
                    },
                    expected_source_revisions=full["source_revisions"],
                )
            loaded = load_integrity_only_taxonomy_projection(
                manifest_path,
                data_dir=data_dir,
                expected_ids_by_scale={
                    20_000: {"article#0", "article#2"},
                    50_000: {"article#0", "article#1", "article#2"},
                    110_000: {"article#0", "article#1", "article#2"},
                },
                scale=20_000,
                expected_source_revisions=full["source_revisions"],
            )
            self.assertEqual(len(loaded["assignments"]), 2)

            original = paths[20_000].read_text(encoding="utf-8")
            paths[20_000].write_text("X" + original[1:], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256"):
                validate_taxonomy_artifact_manifest(
                    manifest,
                    data_dir=data_dir,
                    expected_ids_by_scale={
                        20_000: {"article#0", "article#2"},
                        50_000: {"article#0", "article#1", "article#2"},
                        110_000: {"article#0", "article#1", "article#2"},
                    },
                )

            wrong_provenance = copy.deepcopy(manifest)
            wrong_provenance["provenance"]["input_110k_corpus_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "input_110k_corpus_sha256"):
                validate_taxonomy_artifact_manifest(
                    wrong_provenance,
                    data_dir=data_dir,
                    expected_ids_by_scale={
                        20_000: {"article#0", "article#2"},
                        50_000: {"article#0", "article#1", "article#2"},
                        110_000: {"article#0", "article#1", "article#2"},
                    },
                    expected_provenance={
                        key: full["provenance"][key]
                        for key in (
                            "revision_lock_sha256", "preparation_contract_sha256",
                            "input_110k_corpus_sha256", "input_subset_manifest_sha256",
                        )
                    },
                    expected_source_revisions=full["source_revisions"],
                )

    def test_consumer_preflight_fails_loudly_when_manifest_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "taxonomy artifact manifest"):
                load_integrity_only_taxonomy_projection(
                    Path(directory) / "missing.json",
                    data_dir=Path(directory),
                    expected_ids_by_scale={20_000: set(), 50_000: set(), 110_000: set()},
                    scale=20_000,
                )

    def test_integrity_only_projection_cannot_bypass_generation_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator_source_root = root / "generator-source"
            generator_source_root.mkdir()
            generator_path = generator_source_root / "generator.py"
            generator_path.write_text("# mock generator\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(generator_source_root), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(generator_source_root), "config", "user.email", "fixture@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(generator_source_root), "config", "user.name", "Fixture"], check=True)
            subprocess.run(["git", "-C", str(generator_source_root), "add", "generator.py"], check=True)
            subprocess.run(["git", "-C", str(generator_source_root), "commit", "-q", "-m", "generator source"], check=True)
            source_commit = subprocess.run(
                ["git", "-C", str(generator_source_root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
            ).stdout.strip()
            code_contract = build_generator_code_contract(generator_source_root, ["generator.py"])
            full = source_artifact()
            full["provenance"]["generator"]["generator_code_sha256"] = code_contract["generator_code_sha256"]
            plan = generation_plan_for(
                full,
                generator_source_commit=source_commit,
                generator_contract_hash=generator_contract_sha256(code_contract),
                generator_code_hash=code_contract["generator_code_sha256"],
            )
            projected_20k = project_taxonomy_artifact(full, {"article#0", "article#2"}, target_scale=20_000)
            projected_50k = project_taxonomy_artifact(full, {"article#0", "article#1", "article#2"}, target_scale=50_000)
            data_dir = root / "data"
            taxonomy_dir = data_dir / "taxonomy"
            taxonomy_dir.mkdir(parents=True)
            paths = {}
            for scale, artifact in ((110_000, full), (50_000, projected_50k), (20_000, projected_20k)):
                path = taxonomy_dir / f"{scale}.json"
                path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                paths[scale] = path
            approval = {
                "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
                "approval_kind": "actual_execution",
                "status": "approved_for_generation",
                "approved_generation_plan_sha256": generation_plan_sha256(plan),
                "approved_generator_source_commit": source_commit,
                "approved_generator_contract_sha256": generator_contract_sha256(code_contract),
                "approved_generator_code_sha256": code_contract["generator_code_sha256"],
                "approved_by": "named-approver",
                "approved_at": "2026-07-24T12:00:00+00:00",
                "approval_basis": "separate approved change request",
            }
            generation_run, generation_receipt, response_dir = write_complete_receipt(
                root, full, plan, approval, code_contract
            )
            verified_receipt = validate_taxonomy_generation_receipt(
                generation_receipt,
                generation_plan=plan,
                approval_record=approval,
                generator_code_contract=code_contract,
                generation_run=generation_run,
                raw_response_dir=response_dir,
            )
            manifest = build_taxonomy_artifact_manifest(
                full_artifact=full,
                artifact_records={scale: file_record(path, relative_to=data_dir) for scale, path in paths.items()},
                generator_source_commit=source_commit,
                generator_contract_sha256=generator_contract_sha256(code_contract),
                generation_plan_sha256=generation_plan_sha256(plan),
                approval_record_sha256=approval_record_sha256(approval),
                verified_receipt=verified_receipt,
            )
            manifest_path = root / "artifact_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expected_ids = {
                20_000: {"article#0", "article#2"},
                50_000: {"article#0", "article#1", "article#2"},
                110_000: {"article#0", "article#1", "article#2"},
            }
            self.assertEqual(
                len(load_integrity_only_taxonomy_projection(
                    manifest_path, data_dir=data_dir, expected_ids_by_scale=expected_ids, scale=20_000
                )["assignments"]),
                2,
            )
            with self.assertRaisesRegex(FileNotFoundError, "approval"):
                load_authorized_taxonomy_projection(
                    manifest_path,
                    data_dir=data_dir,
                    expected_ids_by_scale=expected_ids,
                    scale=20_000,
                    generation_plan=plan,
                    approval_record=None,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=generator_source_root,
                    generation_run=generation_run,
                    generation_receipt=generation_receipt,
                    raw_response_dir=response_dir,
                )
            wrong_approval = copy.deepcopy(approval)
            wrong_approval["approved_generator_code_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "approval|generator code"):
                load_authorized_taxonomy_projection(
                    manifest_path,
                    data_dir=data_dir,
                    expected_ids_by_scale=expected_ids,
                    scale=20_000,
                    generation_plan=plan,
                    approval_record=wrong_approval,
                    generator_code_contract=code_contract,
                    verified_input_provenance=plan["input_provenance"],
                    generator_source_root=generator_source_root,
                    generation_run=generation_run,
                    generation_receipt=generation_receipt,
                    raw_response_dir=response_dir,
                )
            authorized = load_authorized_taxonomy_projection(
                manifest_path,
                data_dir=data_dir,
                expected_ids_by_scale=expected_ids,
                scale=20_000,
                generation_plan=plan,
                approval_record=approval,
                generator_code_contract=code_contract,
                verified_input_provenance=plan["input_provenance"],
                generator_source_root=generator_source_root,
                generation_run=generation_run,
                generation_receipt=generation_receipt,
                raw_response_dir=response_dir,
            )
            self.assertEqual(len(authorized["assignments"]), 2)

    def test_tracked_synthetic_fixture_rechecks_its_file_hashes_and_projection(self):
        manifest_path = FIXTURE_DATA_DIR / "taxonomy_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = validate_taxonomy_artifact_manifest(
            manifest,
            data_dir=FIXTURE_DATA_DIR,
            expected_ids_by_scale={
                20_000: {"article#0", "article#2"},
                50_000: {"article#0", "article#1", "article#2"},
                110_000: {"article#0", "article#1", "article#2"},
            },
            expected_provenance={
                "revision_lock_sha256": "a" * 64,
                "preparation_contract_sha256": "b" * 64,
                "input_110k_corpus_sha256": "c" * 64,
                "input_subset_manifest_sha256": "d" * 64,
            },
            expected_source_revisions={"topics_qrels": "1" * 40, "corpus": "2" * 40},
        )
        self.assertEqual(audit["status"], "ready")
        self.assertEqual(audit["artifact_sha256"], "4a3038f448c2bdf06843f1c5b5685b857fd40b7147b23a64269de87d10d599ee")
        synthetic_approval = json.loads(
            (FIXTURE_DATA_DIR.parent / "synthetic_approval_record.json").read_text(encoding="utf-8")
        )
        validate_taxonomy_approval_record(synthetic_approval, allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, "synthetic"):
            validate_taxonomy_approval_record(synthetic_approval)

    def test_deterministic_regeneration_and_focused_treatment_pair_are_strict(self):
        artifact = source_artifact()
        validate_deterministic_regeneration(artifact, copy.deepcopy(artifact))
        regenerated = copy.deepcopy(artifact)
        regenerated["assignments"][0]["score"] = 0.7
        with self.assertRaisesRegex(ValueError, "deterministic"):
            validate_deterministic_regeneration(artifact, regenerated)

        control = {
            "name": "baseline", "taxonomy": False, "taxonomy_prompt_schema": True,
            "workspace_taxonomy": False, "tags": False, "prefix": False,
            "metadata": False, "pull_backend": "dense",
        }
        treatment = {**control, "name": "taxonomy_only", "taxonomy": True}
        validate_focused_taxonomy_treatment_pair(control, treatment)
        drifted = {**treatment, "prefix": True}
        with self.assertRaisesRegex(ValueError, "score boost"):
            validate_focused_taxonomy_treatment_pair(control, drifted)


if __name__ == "__main__":
    unittest.main()
