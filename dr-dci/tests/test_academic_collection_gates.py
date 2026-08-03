"""RED-to-GREEN tests for the recomputing acceptance/retrieval gates.

Covers the independent-review counterexamples across rounds: BatchEncoding
field-count false passes, optional manifest pins, artifact path escapes,
arbitrary tokenizer-loader bypasses (constant-length counters), floating
revisions, unlisted snapshot files, and unverified approval artifacts. All
fixtures are synthetic; no private data and no model/API calls.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collection_academic as academic
from src.data import tokenizer_runtime
from src.eval.collection_contract import sha256_file

from tests.test_academic_collection_alignment import build_fixture, run_convert
from tests.test_academic_collection_publication import (
    CLEAN_RUNTIME,
    fixture_acceptance_contract,
)


from tests._retrieval_fixtures import (
    IMMUTABLE_REV,
    build_reference_snapshot,
    make_fixture_authority,
    make_signed_approval,
)


class FakeBatchEncoding(dict):
    """Mapping like Hugging Face BatchEncoding: len() is the field count."""


def make_tokenizer_contract(root: Path, *, add_special_tokens: bool = True) -> dict:
    """A real reference-whitespace snapshot loaded through the production path."""
    return build_reference_snapshot(
        root / "tokenizer-snapshot", add_special_tokens=add_special_tokens
    )


def rewrite_manifest(target: Path, mutate) -> Path:
    path = target / academic.MANIFEST_FILE_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


class TokenCountingTests(unittest.TestCase):
    def test_batchencoding_field_count_false_pass_is_closed(self):
        encoding = FakeBatchEncoding(
            {"input_ids": [7] * 1000, "attention_mask": [1] * 1000}
        )
        self.assertEqual(len(encoding), 2)
        self.assertEqual(academic._token_count(lambda _: encoding, "x"), 1000)

    def test_missing_input_ids_fails(self):
        with self.assertRaisesRegex(academic.ConversionError, "lacks input_ids"):
            academic._token_count(
                lambda _: FakeBatchEncoding({"attention_mask": [1]}), "x"
            )

    def test_non_mapping_outputs_fail(self):
        for bad in ([1, 2, 3], 7, "tokens", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(
                    academic.ConversionError, "BatchEncoding-like"
                ):
                    academic._token_count(lambda _: bad, "x")

    def test_batched_output_with_multiple_sequences_fails(self):
        encoding = FakeBatchEncoding({"input_ids": [[1, 2], [3, 4]]})
        with self.assertRaisesRegex(academic.ConversionError, "batched"):
            academic._token_count(lambda _: encoding, "x")

    def test_single_sequence_batch_is_unwrapped(self):
        encoding = FakeBatchEncoding(
            {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        )
        self.assertEqual(academic._token_count(lambda _: encoding, "x"), 3)

    def test_attention_mask_length_mismatch_fails(self):
        encoding = FakeBatchEncoding(
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1]}
        )
        with self.assertRaisesRegex(academic.ConversionError, "attention_mask"):
            academic._token_count(lambda _: encoding, "x")

    def test_truncation_markers_fail(self):
        overflowing = FakeBatchEncoding(
            {"input_ids": [1, 2], "overflowing_tokens": [9, 9]}
        )
        with self.assertRaisesRegex(academic.ConversionError, "overflowing"):
            academic._token_count(lambda _: overflowing, "x")
        truncated = FakeBatchEncoding(
            {"input_ids": [1, 2], "num_truncated_tokens": 3}
        )
        with self.assertRaisesRegex(academic.ConversionError, "truncated"):
            academic._token_count(lambda _: truncated, "x")

    def test_non_integer_ids_fail(self):
        encoding = FakeBatchEncoding({"input_ids": [1, "2", 3]})
        with self.assertRaisesRegex(academic.ConversionError, "integers only"):
            academic._token_count(lambda _: encoding, "x")


class AcceptedTargetHarness(unittest.TestCase):
    def setUp(self):
        self._directory = TemporaryDirectory()
        root = Path(self._directory.name)
        self.root = root
        self.fixture = build_fixture(root)
        probe = run_convert(self.fixture, root / "probe", runtime_identity=CLEAN_RUNTIME)
        self.contract = fixture_acceptance_contract(self.fixture, probe)
        self.target = root / "accepted"
        self.manifest = run_convert(
            self.fixture,
            self.target,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=self.contract,
        )
        self.manifest_path = self.target / academic.MANIFEST_FILE_NAME
        self.pin = sha256_file(self.manifest_path)
        assert self.manifest["acceptance"]["eligible"] is True

    def tearDown(self):
        self._directory.cleanup()

    def gate(self, *, pin: str | None = None, contract: dict | None = None):
        return academic.require_accepted_conversion(
            self.manifest_path,
            acceptance_contract=contract if contract is not None else self.contract,
            expected_manifest_sha256=(
                pin if pin is not None else sha256_file(self.manifest_path)
            ),
        )


class GateForgeryTests(AcceptedTargetHarness):
    def test_untampered_gate_passes_with_pin(self):
        manifest = self.gate(pin=self.pin)
        self.assertTrue(manifest["acceptance"]["eligible"])

    def test_pin_is_mandatory_and_validated(self):
        with self.assertRaises(TypeError):
            academic.require_accepted_conversion(
                self.manifest_path, acceptance_contract=self.contract
            )
        for malformed in (None, "", "not-a-sha", "F" * 64):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(
                    academic.ConversionError, "lowercase SHA-256"
                ):
                    academic.require_accepted_conversion(
                        self.manifest_path,
                        acceptance_contract=self.contract,
                        expected_manifest_sha256=malformed,
                    )

    def test_runtime_commit_swap_fails_against_pin(self):
        def forge(manifest):
            manifest["run_identity"]["runtime"]["git_commit"] = "b" * 40

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            self.gate(pin=self.pin)

    def test_smoke_flag_flip_fails(self):
        smoke_target = self.root / "smoke"
        run_convert(
            self.fixture,
            smoke_target,
            limit_documents=1,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=self.contract,
        )

        def flip(manifest):
            manifest["acceptance"]["eligible"] = True
            manifest["acceptance"]["reasons"] = []

        path = rewrite_manifest(smoke_target, flip)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            academic.require_accepted_conversion(
                path,
                acceptance_contract=self.contract,
                expected_manifest_sha256=sha256_file(path),
            )

    def test_deleted_reasons_key_fails(self):
        def forge(manifest):
            del manifest["acceptance"]["reasons"]

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            self.gate()

    def test_selection_tamper_fails(self):
        def forge(manifest):
            manifest["selection"]["limit_documents_per_cell"] = 1

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "selection does not match run identity"
        ):
            self.gate()

    def test_totals_tamper_fails(self):
        def forge(manifest):
            manifest["totals"]["documents"] += 1

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            self.gate()

    def test_runtime_dirty_tamper_fails(self):
        def forge(manifest):
            manifest["run_identity"]["runtime"]["git_dirty"] = True

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            self.gate()

    def test_options_tamper_fails(self):
        def forge(manifest):
            manifest["options"]["verify_archive_sha256"] = False

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "options do not match run identity"
        ):
            self.gate()

    def test_wrong_contract_binding_fails(self):
        other = json.loads(json.dumps(self.contract))
        other["totals"]["documents"] += 1
        with self.assertRaisesRegex(
            academic.ConversionError, "contract bound at publication"
        ):
            self.gate(contract=other)

    def test_identity_self_hash_tamper_fails(self):
        def forge(manifest):
            manifest["run_identity"]["identity_sha256"] = "f" * 64

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(academic.ConversionError, "self-hash mismatch"):
            self.gate()

    def test_reuse_path_rejects_tampered_acceptance(self):
        def forge(manifest):
            manifest["acceptance"]["eligible"] = False
            manifest["acceptance"]["reasons"] = ["forged"]

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            run_convert(
                self.fixture,
                self.target,
                runtime_identity=CLEAN_RUNTIME,
                acceptance_contract=self.contract,
            )

    def test_reuse_path_rejects_tampered_selection(self):
        def forge(manifest):
            manifest["selection"]["splits"] = ["Training"]

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "selection does not match run identity"
        ):
            run_convert(
                self.fixture,
                self.target,
                runtime_identity=CLEAN_RUNTIME,
                acceptance_contract=self.contract,
            )


class ArtifactPathConfinementTests(AcceptedTargetHarness):
    def test_absolute_artifact_path_fails(self):
        outside = self.root / "outside-elements.jsonl"
        outside.write_bytes(
            (self.target / "academic_zz" / "elements.jsonl").read_bytes()
        )

        def forge(manifest):
            entry = manifest["collections"]["academic_zz"]["artifacts"]["elements"]
            entry["path"] = str(outside)

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(academic.ConversionError, "must be exactly"):
            self.gate()

    def test_parent_traversal_artifact_path_fails(self):
        outside = self.target.parent / "outside-elements.jsonl"
        outside.write_bytes(
            (self.target / "academic_zz" / "elements.jsonl").read_bytes()
        )

        def forge(manifest):
            entry = manifest["collections"]["academic_zz"]["artifacts"]["elements"]
            entry["path"] = "../outside-elements.jsonl"

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(academic.ConversionError, "must be exactly"):
            self.gate()

    def test_symlinked_artifact_file_fails(self):
        real = self.target / "academic_zz" / "elements.jsonl"
        moved = self.root / "moved-elements.jsonl"
        moved.write_bytes(real.read_bytes())
        real.unlink()
        real.symlink_to(moved)
        with self.assertRaisesRegex(academic.ConversionError, "symlink"):
            self.gate()

    def test_symlinked_collection_directory_fails(self):
        real_dir = self.target / "academic_zz"
        moved_dir = self.root / "moved-academic_zz"
        real_dir.rename(moved_dir)
        real_dir.symlink_to(moved_dir, target_is_directory=True)
        with self.assertRaisesRegex(academic.ConversionError, "symlink"):
            self.gate()

    def test_extra_declared_artifact_fails(self):
        def forge(manifest):
            artifacts = manifest["collections"]["academic_zz"]["artifacts"]
            artifacts["extra"] = dict(artifacts["elements"])

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(academic.ConversionError, "declare exactly"):
            self.gate()


class CodeIdentityCoverageTests(unittest.TestCase):
    @staticmethod
    def local_imports(module_path: Path, seen: set[str]) -> set[str]:
        source = module_path.read_text(encoding="utf-8")
        found = set()
        for match in re.finditer(r"^import\s+(src[\w.]+)", source, re.MULTILINE):
            found.add(match.group(1))
        from_imports = re.findall(
            r"^from\s+(src[\w.]*)\s+import\s+\(([^)]*)\)", source, re.MULTILINE
        ) + re.findall(
            r"^from\s+(src[\w.]*)\s+import\s+([^(\n]+)$", source, re.MULTILINE
        )
        for module, names in from_imports:
            found.add(module)
            for name in names.split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    found.add(f"{module}.{name}")
        modules = set()
        for dotted in found:
            path = ROOT / (dotted.replace(".", "/") + ".py")
            if not path.is_file():
                continue
            if dotted in seen:
                continue
            seen.add(dotted)
            modules.add(dotted)
            modules |= CodeIdentityCoverageTests.local_imports(path, seen)
        return modules

    def test_every_local_src_import_is_in_code_identity(self):
        seen: set[str] = set()
        adapter_path = ROOT / "src" / "data" / "collection_academic.py"
        dotted_modules = self.local_imports(adapter_path, seen)
        dotted_modules.add("src.data.collection_academic")
        identity_files = {
            path.resolve() for path in academic._CODE_IDENTITY_FILES.values()
        }
        for dotted in sorted(dotted_modules):
            path = (ROOT / (dotted.replace(".", "/") + ".py")).resolve()
            self.assertTrue(path.is_file(), f"cannot resolve import {dotted}")
            self.assertIn(
                path,
                identity_files,
                f"behavior-affecting module {dotted} is missing from "
                "_CODE_IDENTITY_FILES",
            )
        self.assertIn("src.data.tokenizer_runtime", dotted_modules)
        self.assertIn("src.data.retrieval_approval", dotted_modules)

    def test_contract_module_and_clis_are_covered(self):
        keys = set(academic._CODE_IDENTITY_FILES)
        self.assertLessEqual(
            {
                "adapter",
                "publication",
                "tokenizer_runtime",
                "retrieval_approval",
                "collection_contract",
                "build_cli",
                "compat_cli",
                "verify_cli",
            },
            keys,
        )
        for name, path in academic._CODE_IDENTITY_FILES.items():
            self.assertTrue(path.is_file(), f"{name} identity file missing: {path}")
        hashes = academic.code_identity_hashes()
        self.assertEqual(set(hashes), keys)
        for digest in hashes.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class RetrievalApprovalTests(AcceptedTargetHarness):
    def setUp(self):
        super().setUp()
        self.tokenizer_contract = make_tokenizer_contract(self.root)
        self.authority = make_fixture_authority(self.root)
        self.input_template = "passage: {text}"
        self.policy_id = "shinhan.retrieval-approval-policy"
        self.policy_version = "v1"
        self.request_id = "req-0001"

    def _approval(self, *, tokenizer_contract=None, token_budget=10_000, **overrides):
        params = dict(
            authority=self.authority,
            manifest=self.manifest,
            manifest_sha256=self.pin,
            acceptance_contract=self.contract,
            model_id="fixture/embedding",
            revision=IMMUTABLE_REV,
            tokenizer_contract=tokenizer_contract or self.tokenizer_contract,
            token_budget=token_budget,
            input_template_sha256=academic.sha256_text(self.input_template),
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            approval_request_id=self.request_id,
        )
        params.update(overrides)
        return make_signed_approval(**params)

    def attest(self, *, token_budget=10_000, tokenizer_contract=None, approval=None):
        contract = tokenizer_contract or self.tokenizer_contract
        return academic.build_retrieval_approval_attestation(
            target_dir=self.target,
            acceptance_contract=self.contract,
            expected_manifest_sha256=self.pin,
            model_id="fixture/embedding",
            revision=IMMUTABLE_REV,
            token_budget=token_budget,
            input_template=self.input_template,
            tokenizer_contract=contract,
            approval=approval or self._approval(
                tokenizer_contract=contract, token_budget=token_budget
            ),
            trust_root_path=self.authority["trust_root_path"],
            expected_trust_root_sha256=self.authority["trust_root_sha256"],
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            approval_request_id=self.request_id,
        )

    def require(self, attestation, *, attestation_pin=None, trust_root_path=None,
                trust_root_sha256=None):
        return academic.require_retrieval_approved(
            attestation,
            target_dir=self.target,
            acceptance_contract=self.contract,
            expected_manifest_sha256=self.pin,
            expected_attestation_sha256=(
                attestation_pin
                if attestation_pin is not None
                else academic.attestation_sha256(
                    attestation
                    if isinstance(attestation, dict)
                    else json.loads(Path(attestation).read_text(encoding="utf-8"))
                )
            ),
            trust_root_path=trust_root_path or self.authority["trust_root_path"],
            expected_trust_root_sha256=(
                trust_root_sha256 or self.authority["trust_root_sha256"]
            ),
        )

    # ---- GREEN ----
    def test_fully_bound_signed_approval_passes_gate(self):
        attestation = self.attest(token_budget=10_000)
        self.assertEqual(attestation["result"], "approved")
        self.assertRegex(attestation["approval_subject_sha256"], r"^[0-9a-f]{64}$")
        path = academic.write_retrieval_attestation(
            attestation, self.root / "attestation.json"
        )
        sidecar = Path(str(path) + academic.ATTESTATION_FILE_SUFFIX)
        pin = sidecar.read_text(encoding="utf-8").split()[0]
        verified = self.require(path, attestation_pin=pin)
        self.assertEqual(verified["result"], "approved")

    # ---- C1: no runtime loader seam; production builds the real tokenizer ----
    def test_no_mutable_loader_override_exists(self):
        self.assertFalse(
            hasattr(tokenizer_runtime, "TEST_ONLY_LOADER_OVERRIDE"),
            "the mutable-global loader override must not exist",
        )
        import inspect

        for gate in (
            academic.build_retrieval_approval_attestation,
            academic.require_retrieval_approved,
            tokenizer_runtime.load_frozen_tokenizer,
        ):
            params = inspect.signature(gate).parameters
            self.assertFalse(
                [n for n in params if "loader" in n or "tokenizer_fn" in n],
                f"{gate.__name__} must not accept a caller loader",
            )

    def test_same_fake_loader_for_build_and_gate_cannot_bypass(self):
        # The review's C1 vector: try to use one fake constant-count loader for
        # BOTH build and gate. There is no seam, so the only way to inject a
        # tokenizer is via snapshot files; a snapshot whose real tokenizer does
        # not reproduce the bound self-test fails loud. Simulate the attempt by
        # binding a self-test that the real reference tokenizer will not match.
        contract = dict(self.tokenizer_contract)
        contract["self_test"] = {
            "probe_text": "aaa bbb ccc",
            "expected_input_ids": [1, 1],  # a "constant-2" fake claim
        }
        with self.assertRaisesRegex(academic.ConversionError, "self-test failed"):
            self.attest(token_budget=10, tokenizer_contract=contract)

    def test_substituted_self_consistent_tokenizer_is_rejected(self):
        # The real C1 trust boundary: even a fake snapshot whose OWN self-test
        # passes cannot be swapped in, because the tokenizer identity is signed
        # into the approval subject via tokenizer_contract_sha256.
        fake = build_reference_snapshot(self.root / "fake-tokenizer")
        approval_for_honest = self._approval()  # signed over the honest tokenizer
        with self.assertRaisesRegex(
            academic.ConversionError, "tokenizer_contract_sha256 does not match"
        ):
            self.attest(
                token_budget=10_000,
                tokenizer_contract=fake,
                approval=approval_for_honest,
            )

    def test_tokenizer_self_test_binds_real_execution(self):
        contract = dict(self.tokenizer_contract)
        contract["self_test"] = dict(contract["self_test"], expected_input_ids=[9, 9, 9])
        with self.assertRaisesRegex(academic.ConversionError, "self-test failed"):
            self.attest(token_budget=10_000, tokenizer_contract=contract)

    def test_no_global_state_leak_around_retrieval(self):
        before = {
            k: getattr(tokenizer_runtime, k)
            for k in dir(tokenizer_runtime)
            if k.isupper()
        }
        self.attest(token_budget=10_000)
        after = {
            k: getattr(tokenizer_runtime, k)
            for k in dir(tokenizer_runtime)
            if k.isupper()
        }
        self.assertEqual(before, after)

    # ---- C2: approval binding + authority ----
    def test_unrelated_or_missing_subject_fails(self):
        approval = self._approval()
        approval["subject"].pop("conversion_manifest_sha256")
        with self.assertRaisesRegex(academic.ConversionError, "binding set|match"):
            self.attest(token_budget=10_000, approval=approval)

    def test_manifest_mismatch_in_subject_fails(self):
        approval = self._approval(manifest_sha256="0" * 64)
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.attest(token_budget=10_000, approval=approval)

    def test_model_revision_mismatch_in_subject_fails(self):
        approval = self._approval(revision="f" * 40)
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.attest(token_budget=10_000, approval=approval)

    def test_token_budget_mismatch_in_subject_fails(self):
        approval = self._approval(token_budget=99)
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.attest(token_budget=10_000, approval=approval)

    def test_input_template_mismatch_in_subject_fails(self):
        approval = self._approval(input_template_sha256=academic.sha256_text("other {text}"))
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.attest(token_budget=10_000, approval=approval)

    def test_unauthorized_signer_fails(self):
        other = make_fixture_authority(self.root / "other", signer_id="intruder")
        approval = self._approval(
            signer_id="intruder", private_b64=other["private_b64"]
        )
        with self.assertRaisesRegex(academic.ConversionError, "not authorized"):
            self.attest(token_budget=10_000, approval=approval)

    def test_invalid_signature_fails(self):
        other = make_fixture_authority(self.root / "other2")
        # keep the trust-root's signer id but sign with a non-allowlisted key
        approval = self._approval(private_b64=other["private_b64"])
        with self.assertRaisesRegex(academic.ConversionError, "signature is invalid"):
            self.attest(token_budget=10_000, approval=approval)

    def test_trust_root_pin_mismatch_fails(self):
        approval = self._approval()
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            academic.build_retrieval_approval_attestation(
                target_dir=self.target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=self.pin,
                model_id="fixture/embedding",
                revision=IMMUTABLE_REV,
                token_budget=10_000,
                input_template=self.input_template,
                tokenizer_contract=self.tokenizer_contract,
                approval=approval,
                trust_root_path=self.authority["trust_root_path"],
                expected_trust_root_sha256="f" * 64,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                approval_request_id=self.request_id,
            )

    def test_missing_trust_root_fails_closed(self):
        approval = self._approval()
        with self.assertRaisesRegex(academic.ConversionError, "trust root missing"):
            academic.build_retrieval_approval_attestation(
                target_dir=self.target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=self.pin,
                model_id="fixture/embedding",
                revision=IMMUTABLE_REV,
                token_budget=10_000,
                input_template=self.input_template,
                tokenizer_contract=self.tokenizer_contract,
                approval=approval,
                trust_root_path=self.root / "no_such_trust_root.json",
                expected_trust_root_sha256="0" * 64,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                approval_request_id=self.request_id,
            )

    def test_approval_replay_under_different_identity_fails(self):
        # An approval signed for THIS attestation cannot be replayed onto a
        # different token budget: the subject digest no longer matches.
        approval = self._approval(token_budget=10_000)
        attestation = self.attest(token_budget=10_000, approval=approval)
        # Rebuild attestation for a different budget but reuse the old approval.
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.attest(token_budget=5, approval=approval)

    def test_gate_rejects_swapped_trust_root(self):
        attestation = self.attest(token_budget=10_000)
        other = make_fixture_authority(self.root / "swap")
        with self.assertRaisesRegex(academic.ConversionError, "trust root|pinned|authorized|signature"):
            self.require(
                attestation,
                trust_root_path=other["trust_root_path"],
                trust_root_sha256=other["trust_root_sha256"],
            )

    def test_gate_rejects_tampered_approval_after_build(self):
        attestation = self.attest(token_budget=10_000)
        pin = academic.attestation_sha256(attestation)
        forged = json.loads(json.dumps(attestation))
        forged["approval"]["subject"]["model_id"] = "swapped/model"
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            self.require(forged, attestation_pin=pin)

    # ---- existing invariants retained ----
    def test_attestation_pin_is_mandatory_and_validated(self):
        attestation = self.attest(token_budget=10_000)
        with self.assertRaises(TypeError):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=self.pin,
                trust_root_path=self.authority["trust_root_path"],
                expected_trust_root_sha256=self.authority["trust_root_sha256"],
            )
        with self.assertRaisesRegex(academic.ConversionError, "lowercase SHA-256"):
            self.require(attestation, attestation_pin="not-a-sha")

    def test_rejected_attestation_when_over_budget(self):
        # token budget 3 -> approval subject for budget 3, but chunks exceed
        # -> result rejected. The build returns a rejected attestation; the
        # gate must refuse it.
        attestation = self.attest(token_budget=3)
        self.assertEqual(attestation["result"], "rejected")
        with self.assertRaisesRegex(academic.ConversionError, "zero-violation"):
            self.require(attestation)

    def test_unlisted_extra_tokenizer_file_rejected(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        (snapshot / "special_tokens_map.json").write_bytes(b"{}")
        with self.assertRaisesRegex(academic.ConversionError, "not in the frozen inventory"):
            self.require(attestation)

    def test_symlinked_tokenizer_file_rejected(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        moved = self.root / "moved-vocab.json"
        original = snapshot / "vocab.json"
        moved.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(moved)
        with self.assertRaisesRegex(academic.ConversionError, "symlink"):
            self.require(attestation)

    def test_floating_revision_rejected(self):
        floating = dict(self.tokenizer_contract, revision="floating-release-tag")
        with self.assertRaisesRegex(academic.ConversionError, "immutable content identifier"):
            self.attest(token_budget=10_000, tokenizer_contract=floating)

    def test_option_keys_must_be_exact(self):
        missing = json.loads(json.dumps(self.tokenizer_contract))
        del missing["options"]["normalization"]
        with self.assertRaisesRegex(academic.ConversionError, "exactly"):
            self.attest(token_budget=10_000, tokenizer_contract=missing)

    def test_tokenizer_snapshot_tamper_fails(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        (snapshot / "vocab.json").write_bytes(b'{"x": 1}')
        with self.assertRaisesRegex(
            academic.ConversionError, "tokenizer file (sha256|bytes) mismatch"
        ):
            self.require(attestation)

    def test_special_token_overhead_is_included_in_budget(self):
        with_special = self.attest(token_budget=100_000)
        without_contract = build_reference_snapshot(
            self.root / "nospecial", add_special_tokens=False
        )
        without_special = self.attest(
            token_budget=100_000, tokenizer_contract=without_contract
        )
        self.assertEqual(
            with_special["max_tokens_observed"],
            without_special["max_tokens_observed"] + 2,
        )
        exact = self.attest(token_budget=with_special["max_tokens_observed"])
        self.assertEqual(exact["result"], "approved")

    def test_tampered_chunks_fail_gate(self):
        attestation = self.attest(token_budget=10_000)
        chunk_path = self.target / "academic_zz" / "chunks.jsonl"
        chunk_path.write_text(
            chunk_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            self.require(attestation)

    def test_code_identity_mismatch_fails_gate(self):
        attestation = self.attest(token_budget=10_000)
        attestation["code_identity"] = dict(
            attestation["code_identity"], retrieval_approval="f" * 64
        )
        with self.assertRaisesRegex(academic.ConversionError, "code identity"):
            self.require(attestation)

    def test_smoke_corpus_cannot_be_attested(self):
        smoke_target = self.root / "smoke"
        run_convert(
            self.fixture,
            smoke_target,
            limit_documents=1,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=self.contract,
        )
        smoke_manifest = smoke_target / academic.MANIFEST_FILE_NAME
        smoke_pin = sha256_file(smoke_manifest)
        approval = self._approval(manifest_sha256=smoke_pin)
        with self.assertRaisesRegex(academic.ConversionError, "not acceptance-eligible"):
            academic.build_retrieval_approval_attestation(
                target_dir=smoke_target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=smoke_pin,
                model_id="fixture/embedding",
                revision=IMMUTABLE_REV,
                token_budget=10_000,
                input_template=self.input_template,
                tokenizer_contract=self.tokenizer_contract,
                approval=approval,
                trust_root_path=self.authority["trust_root_path"],
                expected_trust_root_sha256=self.authority["trust_root_sha256"],
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                approval_request_id=self.request_id,
            )


class CompatCliExitCodeTests(unittest.TestCase):
    def run_cli(self, chunks: Path, contract_path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validate_chunk_model_compatibility.py"),
                str(chunks),
                "--contract",
                str(contract_path),
            ],
            capture_output=True,
            text=True,
        )

    def test_exit_codes_for_all_outcomes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            run_convert(fixture, root / "out")
            chunks = root / "out" / "academic_zz" / "chunks.jsonl"

            base = {
                "schema_version": academic.COMPAT_CONTRACT_SCHEMA_VERSION,
                "model_id": "fixture/embedding",
                "revision": "0123456789abcdef",
                "max_input_chars": 100000,
            }
            compatible_path = root / "compatible.yaml"
            compatible_path.write_text(yaml.safe_dump(base), encoding="utf-8")
            result = self.run_cli(chunks, compatible_path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("compatible", result.stdout)

            tight_path = root / "tight.yaml"
            tight_path.write_text(
                yaml.safe_dump(dict(base, max_input_chars=10)), encoding="utf-8"
            )
            result = self.run_cli(chunks, tight_path)
            self.assertEqual(result.returncode, 2, result.stderr)

            policy_path = root / "split_policy.yaml"
            policy_path.write_text(
                "policy_id: fixture.split.v1\nmax_chars: 10\n", encoding="utf-8"
            )
            approved_path = root / "approved.yaml"
            approved_path.write_text(
                yaml.safe_dump(
                    dict(
                        base,
                        max_input_chars=10,
                        approved_long_element_split_policy={
                            "policy_id": "fixture.split.v1",
                            "approved_by": "owner",
                            "path": "split_policy.yaml",
                            "bytes": policy_path.stat().st_size,
                            "sha256": sha256_file(policy_path),
                        },
                    )
                ),
                encoding="utf-8",
            )
            result = self.run_cli(chunks, approved_path)
            self.assertEqual(result.returncode, 4, result.stderr)
            self.assertIn("requires_rebuild", result.stdout)


class VerifyCliTests(AcceptedTargetHarness):
    def test_replay_cli_accepts_and_rejects(self):
        contract_path = self.root / "contract.yaml"
        contract_path.write_text(
            yaml.safe_dump(self.contract, allow_unicode=True), encoding="utf-8"
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "verify_accepted_conversion.py"),
            "--target",
            str(self.target),
            "--acceptance-contract",
            str(contract_path),
            "--expected-manifest-sha256",
        ]
        result = subprocess.run(
            [*command, self.pin], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "accepted"', result.stdout)

        result = subprocess.run(
            [*command, "f" * 64], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("pinned SHA-256", result.stderr)


if __name__ == "__main__":
    unittest.main()
