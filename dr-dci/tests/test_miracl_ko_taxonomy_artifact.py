import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.miracl_ko.taxonomy_artifact import (
    TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
    TAXONOMY_ARTIFACT_SCHEMA_VERSION,
    TAXONOMY_FLAT_L1_ADAPTER_SCHEMA_VERSION,
    TAXONOMY_GENERATOR_BATCH_SCHEMA_VERSION,
    TAXONOMY_GENERATION_PLAN_SCHEMA_VERSION,
    TAXONOMY_MANIFEST_SCHEMA_VERSION,
    build_flat_l1_consumer_adapter,
    build_generator_code_contract,
    build_semantic_generator_inputs,
    build_taxonomy_generator_batch,
    build_taxonomy_generation_plan,
    build_taxonomy_artifact_manifest,
    generator_contract_sha256,
    generation_plan_sha256,
    load_authorized_taxonomy_projection,
    load_integrity_only_taxonomy_projection,
    project_taxonomy_artifact,
    rejoin_taxonomy_generator_outputs,
    taxonomy_artifact_sha256,
    validate_flat_l1_consumer_adapter,
    validate_deterministic_regeneration,
    validate_focused_taxonomy_treatment_pair,
    validate_taxonomy_approval_record,
    validate_taxonomy_artifact,
    validate_taxonomy_artifact_manifest,
    validate_taxonomy_generation_preflight,
    validate_taxonomy_generation_plan,
    validate_taxonomy_generator_batch,
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
                "prompt_template_sha256": None,
                "seed": 7,
                "parameters": {"label_count": 2},
                "determinism_mode": "deterministic",
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
    artifact: dict, *, source_git_commit: str = "f" * 40,
    generator_contract_hash: str = "f" * 64,
    generator_code_hash: str | None = None,
) -> dict:
    generator = copy.deepcopy(artifact["provenance"]["generator"])
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
        source_git_commit=source_git_commit,
        generator_code_contract_sha256=generator_contract_hash,
        generator_code_sha256=generator["generator_code_sha256"],
        generator=generator,
        input_contract=artifact["input_contract"],
    )


class MiraclKoTaxonomyArtifactTests(unittest.TestCase):
    def test_generator_response_request_indexes_allow_reordering_but_not_positional_rejoin(self):
        records = [
            {"corpus_id": "article#2", "title": "두", "text": "둘"},
            {"corpus_id": "article#1", "title": "하나", "text": "첫"},
        ]
        batch = build_taxonomy_generator_batch(records)
        ordered = [
            {"request_index": 0, "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            {"request_index": 1, "label_id": "topic.b", "score": 0.8, "status": "assigned"},
        ]
        expected = [
            {"corpus_id": "article#1", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
            {"corpus_id": "article#2", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
        ]
        self.assertEqual(rejoin_taxonomy_generator_outputs(batch, ordered), expected)
        self.assertEqual(rejoin_taxonomy_generator_outputs(batch, list(reversed(ordered))), expected)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            rejoin_taxonomy_generator_outputs(
                batch,
                [
                    {"label_id": "topic.a", "score": 0.9, "status": "assigned"},
                    {"label_id": "topic.b", "score": 0.8, "status": "assigned"},
                ],
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
                    rejoin_taxonomy_generator_outputs(batch, invalid)

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
        batch = build_taxonomy_generator_batch(forward)
        self.assertEqual(batch, build_taxonomy_generator_batch(reverse))
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
            rejoin_taxonomy_generator_outputs(batch, outputs),
            [
                {"corpus_id": "article#1", "label_id": "topic.a", "score": 0.9, "status": "assigned"},
                {"corpus_id": "article#2", "label_id": "topic.b", "score": 0.8, "status": "assigned"},
            ],
        )
        for outputs_with_wrong_count in (outputs[:1], outputs + outputs[:1]):
            with self.subTest(output_count=len(outputs_with_wrong_count)):
                with self.assertRaisesRegex(ValueError, "count"):
                    rejoin_taxonomy_generator_outputs(batch, outputs_with_wrong_count)
        with self.assertRaisesRegex(ValueError, "duplicate corpus_id"):
            build_taxonomy_generator_batch(forward + [dict(forward[0])])

        leaking = copy.deepcopy(batch)
        leaking["semantic_payload"][0]["corpus_id"] = "article#1"
        with self.assertRaisesRegex(ValueError, "semantic payload"):
            validate_taxonomy_generator_batch(leaking)
        mismatched = copy.deepcopy(batch)
        mismatched["mapping_envelope"] = list(reversed(mismatched["mapping_envelope"]))
        with self.assertRaisesRegex(ValueError, "mapping envelope"):
            rejoin_taxonomy_generator_outputs(mismatched, outputs)

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
            code_contract = build_generator_code_contract(root, ["generator.py"])
            full = source_artifact()
            plan = generation_plan_for(
                full,
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
                "approved_source_git_commit": "f" * 40,
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
                actual_source_git_commit="f" * 40,
                git_status_porcelain="",
                verified_input_provenance=plan["input_provenance"],
                generator_repo_root=root,
            )
            with self.assertRaisesRegex(FileNotFoundError, "approval"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=None,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            other_plan = copy.deepcopy(plan)
            other_plan["generator"]["parameters"]["label_count"] = 3
            with self.assertRaisesRegex(ValueError, "generation plan"):
                validate_taxonomy_generation_preflight(
                    other_plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            changed_input = dict(plan["input_provenance"])
            changed_input["input_110k_corpus_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "input provenance"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=changed_input,
                    generator_repo_root=root,
                )
            with self.assertRaisesRegex(ValueError, "actual source commit"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="0" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            with self.assertRaisesRegex(ValueError, "dirty"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain=" M generator.py",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            wrong_approval = copy.deepcopy(approval)
            wrong_approval["approved_generator_code_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "generator code"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=wrong_approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            generator_path.write_text("# modified after contract\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code file hash"):
                validate_taxonomy_generation_preflight(
                    plan,
                    approval_record=approval,
                    generator_code_contract=code_contract,
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )

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
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )

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
            manifest = build_taxonomy_artifact_manifest(
                full_artifact=artifact,
                artifact_records={
                    scale: file_record(path, relative_to=data_dir) for scale in (20_000, 50_000, 110_000)
                },
                source_git_commit="f" * 40,
                generator_contract_sha256="f" * 64,
                generation_plan_sha256="f" * 64,
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
            manifest = build_taxonomy_artifact_manifest(
                full_artifact=full,
                artifact_records={
                    scale: file_record(path, relative_to=data_dir)
                    for scale, path in paths.items()
                },
                source_git_commit="f" * 40,
                generator_contract_sha256="f" * 64,
                generation_plan_sha256="f" * 64,
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
            generator_path = root / "generator.py"
            generator_path.write_text("# mock generator\n", encoding="utf-8")
            code_contract = build_generator_code_contract(root, ["generator.py"])
            full = source_artifact()
            full["provenance"]["generator"]["generator_code_sha256"] = code_contract["generator_code_sha256"]
            plan = generation_plan_for(
                full,
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
            manifest = build_taxonomy_artifact_manifest(
                full_artifact=full,
                artifact_records={scale: file_record(path, relative_to=data_dir) for scale, path in paths.items()},
                source_git_commit="f" * 40,
                generator_contract_sha256=generator_contract_sha256(code_contract),
                generation_plan_sha256=generation_plan_sha256(plan),
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
                    actual_source_git_commit="f" * 40,
                    git_status_porcelain="",
                    verified_input_provenance=plan["input_provenance"],
                    generator_repo_root=root,
                )
            approval = {
                "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
                "approval_kind": "actual_execution",
                "status": "approved_for_generation",
                "approved_generation_plan_sha256": generation_plan_sha256(plan),
                "approved_source_git_commit": "f" * 40,
                "approved_generator_contract_sha256": generator_contract_sha256(code_contract),
                "approved_generator_code_sha256": code_contract["generator_code_sha256"],
                "approved_by": "named-approver",
                "approved_at": "2026-07-24T12:00:00+00:00",
                "approval_basis": "separate approved change request",
            }
            authorized = load_authorized_taxonomy_projection(
                manifest_path,
                data_dir=data_dir,
                expected_ids_by_scale=expected_ids,
                scale=20_000,
                generation_plan=plan,
                approval_record=approval,
                generator_code_contract=code_contract,
                actual_source_git_commit="f" * 40,
                git_status_porcelain="",
                verified_input_provenance=plan["input_provenance"],
                generator_repo_root=root,
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
