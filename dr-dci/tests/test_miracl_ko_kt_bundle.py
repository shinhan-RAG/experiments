import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.miracl_ko.kt_bundle import (
    KT_BUNDLE_SOURCE_MANIFEST_SCHEMA,
    load_bundle_controls,
    parse_env_file,
    require_operation_start,
    validate_execution_lock,
)
from src.miracl_ko.preparation import sha256_file, sha256_json
from src.miracl_ko.taxonomy_artifact import (
    TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
    approval_record_sha256,
    build_generator_code_contract,
    generator_contract_sha256,
    generation_plan_sha256,
    project_taxonomy_artifact,
    validate_taxonomy_generation_receipt,
    validate_taxonomy_projection,
)
from tests.test_miracl_ko_taxonomy_artifact import (
    generation_plan_for,
    source_artifact,
    write_complete_receipt,
)


class MiraclKoKtBundleTests(unittest.TestCase):
    def _fixture(self, root: Path):
        bundle = root / "bundle"
        generator = bundle / "generator"
        source = bundle / "source"
        control = root / "control"
        data = root / "data"
        output = root / "output"
        cache = root / "cache"
        for directory in (
            generator, source, control, data / "subsets" / "20k", data / "subsets" / "50k",
            data / "subsets" / "110k", output, cache,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (generator / "generator.py").write_text("# synthetic generator\n", encoding="utf-8")
        contract = build_generator_code_contract(generator, ["generator.py"])
        (source / "generator_code_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        revision_lock = source / "miracl_ko_revision_lock.json"
        revision_lock.write_text("{}\n", encoding="utf-8")
        corpus = data / "subsets" / "110k" / "corpus.jsonl"
        corpus.write_text('{"corpus_id":"a#0","title":"A","text":"text"}\n', encoding="utf-8")
        corpus_20k = data / "subsets" / "20k" / "corpus.jsonl"
        corpus_20k.write_text('{"corpus_id":"a#0","title":"A","text":"text"}\n', encoding="utf-8")
        corpus_50k = data / "subsets" / "50k" / "corpus.jsonl"
        corpus_50k.write_text('{"corpus_id":"a#0","title":"A","text":"text"}\n', encoding="utf-8")
        subset_manifest = {
            "preparation_contract_sha256": "a" * 64,
            "source_revisions": {"topics_qrels": "1" * 40, "corpus": "2" * 40},
            "subsets": {
                "20000": {"corpus": {
                    "relative_path": "subsets/20k/corpus.jsonl",
                    "sha256": sha256_file(corpus_20k),
                }},
                "50000": {"corpus": {
                    "relative_path": "subsets/50k/corpus.jsonl",
                    "sha256": sha256_file(corpus_50k),
                }},
                "110000": {"corpus": {
                    "relative_path": "subsets/110k/corpus.jsonl",
                    "sha256": sha256_file(corpus),
                }},
            },
        }
        subset_path = data / "subsets" / "manifest.json"
        subset_path.write_text(json.dumps(subset_manifest), encoding="utf-8")
        provenance = {
            "revision_lock_sha256": sha256_file(revision_lock),
            "preparation_contract_sha256": subset_manifest["preparation_contract_sha256"],
            "input_110k_corpus_sha256": sha256_file(corpus),
            "input_subset_manifest_sha256": sha256_file(subset_path),
        }
        artifact = source_artifact()
        artifact["provenance"].update(provenance)
        source_commit = "f" * 40
        artifact["provenance"]["generator"]["generator_code_sha256"] = contract["generator_code_sha256"]
        plan = generation_plan_for(
            artifact,
            generator_source_commit=source_commit,
            generator_contract_hash=generator_contract_sha256(contract),
            generator_code_hash=contract["generator_code_sha256"],
        )
        approval = {
            "schema_version": TAXONOMY_APPROVAL_RECORD_SCHEMA_VERSION,
            "approval_kind": "actual_execution",
            "status": "approved_for_generation",
            "approved_generation_plan_sha256": generation_plan_sha256(plan),
            "approved_generator_source_commit": source_commit,
            "approved_generator_contract_sha256": generator_contract_sha256(contract),
            "approved_generator_code_sha256": contract["generator_code_sha256"],
            "approved_by": "named-approver",
            "approved_at": "2026-07-24T12:00:00+00:00",
            "approval_basis": "synthetic KT bundle test fixture",
        }
        source_manifest = {
            "schema_version": KT_BUNDLE_SOURCE_MANIFEST_SCHEMA,
            "source_git_commit": source_commit,
            "generator_contract_sha256": generator_contract_sha256(contract),
            "generator_code_sha256": contract["generator_code_sha256"],
            "files": contract["generator_files"],
        }
        (source / "clean_generator_source_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
        for name, value in (
            ("generation_plan.json", plan),
            ("approval_record.json", approval),
            ("generator_code_contract.json", contract),
        ):
            (control / name).write_text(json.dumps(value), encoding="utf-8")
        config = {
            "KT_DATA_DIR": str(data),
            "KT_CONTROL_DIR": str(control),
            "KT_OUTPUT_DIR": str(output),
            "KT_MODEL_CACHE_DIR": str(cache),
            "KT_VLLM_IMAGE": "synthetic/local-vllm",
            "KT_CONTAINER_DIGEST": "sha256:" + "a" * 64,
            "KT_VLLM_HOST": "127.0.0.1",
            "KT_VLLM_PORT": "8000",
        }
        return bundle, control, data, output, config, plan, approval, contract, artifact

    def _lock(self, bundle, control, config):
        plan, approval, contract = load_bundle_controls(control)
        return validate_execution_lock(
            bundle_root=bundle, config=config, plan=plan, approval=approval, supplied_contract=contract,
        )

    def test_git_free_clean_directory_accepts_only_matching_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, control, data, _output, config, _plan, _approval, _contract, _artifact = self._fixture(root)
            self.assertFalse((bundle / ".git").exists())
            lock = self._lock(bundle, control, config)
            self.assertEqual(lock["input_provenance"]["input_110k_corpus_sha256"], sha256_file(data / "subsets" / "110k" / "corpus.jsonl"))

            missing = control / "generation_plan.json"
            missing.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "generation plan"):
                load_bundle_controls(control)

    def test_preflight_lock_rejects_revision_container_and_dataset_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, control, data, _output, config, plan, _approval, _contract, _artifact = self._fixture(root)
            self._lock(bundle, control, config)
            wrong_container = dict(config, KT_CONTAINER_DIGEST="sha256:" + "b" * 64)
            with self.assertRaisesRegex(ValueError, "container digest"):
                self._lock(bundle, control, wrong_container)

            mutable = copy.deepcopy(plan)
            mutable["generator"]["model_revision"] = "main"
            mutable["generator"]["runtime"]["model_revision"] = "main"
            arguments = mutable["generator"]["vllm_execution"]["server_launch"]["arguments"]
            arguments[arguments.index("--revision") + 1] = "main"
            mutable["generator"]["vllm_execution"]["server_launch"]["arguments_sha256"] = sha256_json(arguments)
            with self.assertRaisesRegex(ValueError, "immutable"):
                validate_execution_lock(
                    bundle_root=bundle,
                    config=config,
                    plan=mutable,
                    approval=load_bundle_controls(control)[1],
                    supplied_contract=load_bundle_controls(control)[2],
                )

            corpus = data / "subsets" / "110k" / "corpus.jsonl"
            corpus.write_text(corpus.read_text(encoding="utf-8") + "# tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus bytes"):
                self._lock(bundle, control, config)

    def test_operation_receipt_blocks_completed_duplicate_and_permits_partial_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            first = require_operation_start(output, plan_sha256="a" * 64, run_sha256="b" * 64)
            self.assertEqual(first["status"], "partial")
            resumed = require_operation_start(output, plan_sha256="a" * 64, run_sha256="b" * 64)
            self.assertEqual(resumed["status"], "partial")
            completed = dict(resumed, status="complete")
            (output / "generation" / "operation_receipt.json").write_text(json.dumps(completed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be run twice"):
                require_operation_start(output, plan_sha256="a" * 64, run_sha256="b" * 64)

    def test_receipt_tamper_retry_partial_and_projection_invariants_fail_loud(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, control, _data, _output, _config, plan, approval, contract, artifact = self._fixture(root)
            run, receipt, response_dir = write_complete_receipt(root, artifact, plan, approval, contract)
            partial = copy.deepcopy(receipt)
            partial["status"] = "partial"
            partial["assignment_canonical_sha256"] = None
            partial["batch_summary"]["failed"] = 1
            partial["batch_summary"]["succeeded"] -= 1
            partial["raw_response_files"] = partial["raw_response_files"][:-1]
            partial["batch_summary"]["retries"] = 0
            self.assertIsNone(validate_taxonomy_generation_receipt(
                partial,
                generation_plan=plan,
                approval_record=approval,
                generator_code_contract=contract,
                generation_run=run,
                raw_response_dir=response_dir,
            ))
            retry_exceeded = copy.deepcopy(receipt)
            retry_exceeded["raw_response_files"][0]["attempt_count"] = 2
            retry_exceeded["raw_response_files"][0]["retry_count"] = 1
            retry_exceeded["batch_summary"]["retries"] = 1
            with self.assertRaisesRegex(ValueError, "retry_count exceeds"):
                validate_taxonomy_generation_receipt(
                    retry_exceeded,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            raw = response_dir / receipt["raw_response_files"][0]["relative_path"]
            raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte_size|sha256"):
                validate_taxonomy_generation_receipt(
                    receipt,
                    generation_plan=plan,
                    approval_record=approval,
                    generator_code_contract=contract,
                    generation_run=run,
                    raw_response_dir=response_dir,
                )
            source = source_artifact()
            target_ids = {"article#0", "article#1"}
            projected = project_taxonomy_artifact(source, target_ids, target_scale=20_000)
            projected["assignments"][0]["label_id"] = "topic.b"
            with self.assertRaisesRegex(ValueError, "changes"):
                validate_taxonomy_projection(source, projected, expected_corpus_ids=target_ids, target_scale=20_000)

    def test_config_env_rejects_credential_ingress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.env"
            path.write_text("KT_API_KEY=forbidden\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credentials"):
                parse_env_file(path)


if __name__ == "__main__":
    unittest.main()
