"""RED→GREEN contracts for the approved MIRACL-ko Flat-L1 KT run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.miracl_ko.flat_l1 import (
    build_flat_l1_generation_plan,
    validate_flat_l1_plan,
)
from src.miracl_ko.kt_bundle import (
    approved_vllm_entrypoint_arguments,
    image_matches_entrypoint,
    order_successful_batch_records,
)
from src.miracl_ko.preparation import sha256_json
from src.miracl_ko.taxonomy_artifact import (
    build_generator_code_contract,
    generator_contract_sha256,
    rejoin_taxonomy_generator_run_outputs,
    validate_flat_l1_assignments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class MiraclKoFlatL1ExecutionTests(unittest.TestCase):
    def _plan(self) -> dict:
        contract = build_generator_code_contract(
            REPO_ROOT,
            [
                "scripts/kt_bundle/taxonomy_vllm_generator.py",
                "src/miracl_ko/__init__.py",
                "src/miracl_ko/flat_l1.py",
                "src/miracl_ko/kt_bundle.py",
                "src/miracl_ko/preparation.py",
                "src/miracl_ko/taxonomy_artifact.py",
                "config/miracl_ko_taxonomy/flat_l1_catalog.json",
                "config/miracl_ko_taxonomy/flat_l1_output_schema.json",
                "config/miracl_ko_taxonomy/flat_l1_system_prompt.txt",
                "config/miracl_ko_taxonomy/vllm_v0_9_0_runtime_identity.json",
            ],
        )
        synthetic_input_provenance = {
            "revision_lock_sha256": "a" * 64,
            "preparation_contract_sha256": "b" * 64,
            "input_110k_corpus_sha256": "c" * 64,
            "input_subset_manifest_sha256": "d" * 64,
        }
        # Flat-L1 control tests are independent from the byte-level fixture
        # validator and must not read locally untracked MIRACL data.
        with patch(
            "src.miracl_ko.flat_l1.input_provenance",
            return_value=synthetic_input_provenance,
        ):
            return build_flat_l1_generation_plan(
                repo_root=REPO_ROOT,
                generator_source_commit="a" * 40,
                generator_code_contract_sha256=generator_contract_sha256(contract),
                generator_code_sha256=contract["generator_code_sha256"],
            )

    def test_plan_fixes_one_passage_requests_and_transport_concurrency(self):
        plan = self._plan()
        self.assertEqual(plan["run_controls"]["batch_size"], 1)
        self.assertEqual(plan["run_controls"]["transport_max_concurrency"], 32)
        self.assertEqual(plan["generator"]["generation_controls"], {"temperature": 0, "max_tokens": 100})
        self.assertEqual(plan["generator"]["vllm_execution"]["thinking"]["enable_thinking"], "disabled")

        wrong_batch = copy.deepcopy(plan)
        wrong_batch["run_controls"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "batch_size must be 1"):
            validate_flat_l1_plan(wrong_batch)

        wrong_concurrency = copy.deepcopy(plan)
        wrong_concurrency["run_controls"]["transport_max_concurrency"] = 31
        with self.assertRaisesRegex(ValueError, "transport_max_concurrency"):
            validate_flat_l1_plan(wrong_concurrency)

    def test_closed_set_has_no_failure_fallback_and_scores_are_fixed(self):
        plan = self._plan()
        parameters = plan["generator"]["parameters"]
        self.assertEqual(parameters["failure_policy"], "fail_loud_retry_no_label_fallback_v1")
        self.assertEqual(parameters["assignment_score_policy"], "assigned_1_unknown_0_v1")
        self.assertNotIn("confidence", parameters)
        self.assertNotIn("confidence", plan["generator"]["vllm_execution"]["structured_output"]["json_schema"]["properties"])
        with self.assertRaisesRegex(ValueError, "score 1.0"):
            validate_flat_l1_assignments(
                [{"corpus_id": "a#0", "label_id": "other", "score": 0.4, "status": "assigned"}],
                generator=plan["generator"],
            )

    def test_official_entrypoint_receives_model_option_first_and_drift_fails(self):
        plan = self._plan()
        arguments = plan["generator"]["vllm_execution"]["server_launch"]["arguments"]
        self.assertEqual(arguments[0], "--model")
        self.assertEqual(approved_vllm_entrypoint_arguments(arguments), arguments)
        with self.assertRaisesRegex(ValueError, "begin with --model"):
            approved_vllm_entrypoint_arguments(["container_default_openai_api_entrypoint_v1", *arguments])
        self.assertTrue(image_matches_entrypoint(
            {"entrypoint": ["python3", "-m", "vllm.entrypoints.openai.api_server"]},
            ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        ))
        self.assertFalse(image_matches_entrypoint(
            {"entrypoint": ["/bin/sh"]},
            ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        ))

    def test_concurrent_completion_order_rejoins_to_the_same_sorted_assignments(self):
        plan = self._plan()
        run = {
            "schema_version": "dr-dci.miracl-ko-taxonomy-generator-run.v2",
            "generation_plan_sha256": sha256_json(plan),
            "run_total_records": 2,
            "run_controls": plan["run_controls"],
            "batches": [],
            "generation_run_sha256": "",
        }
        # Build the real transport batches through the contract, then supply their
        # responses in reverse completion order.
        from src.miracl_ko.taxonomy_artifact import build_taxonomy_generator_run
        run = build_taxonomy_generator_run(
            [
                {"corpus_id": "b#0", "title": "B", "text": "B"},
                {"corpus_id": "a#0", "title": "A", "text": "A"},
            ],
            generation_plan_sha256=sha256_json(plan),
            batch_size=1,
            run_controls=plan["run_controls"],
        )
        catalog = {row["label_id"] for row in plan["generator"]["parameters"]["label_catalog"]}
        responses = []
        for batch in reversed(run["batches"]):
            label_id = "science" if batch["mapping_envelope"][0]["corpus_id"] == "a#0" else "history"
            self.assertIn(label_id, catalog)
            responses.append({
                "generation_request_sha256": batch["generation_request_sha256"],
                "outputs": [{"request_index": 0, "label_id": label_id, "score": 1.0, "status": "assigned"}],
            })
        assignments = rejoin_taxonomy_generator_run_outputs(run, responses)
        self.assertEqual([row["corpus_id"] for row in assignments], ["a#0", "b#0"])
        ordered = order_successful_batch_records([
            {"batch_ordinal": 1, "value": "second"},
            {"batch_ordinal": 0, "value": "first"},
        ])
        self.assertEqual([row["value"] for row in ordered], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
