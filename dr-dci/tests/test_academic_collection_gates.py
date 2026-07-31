"""RED-to-GREEN tests for the recomputing acceptance/retrieval gates.

Forged manifests (flag flips, deleted reasons, tampered selection/totals/
runtime/options/contract hashes) must never pass; the gates recompute every
decision instead of trusting persisted flags. All fixtures are synthetic.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collection_academic as academic
from src.eval.collection_contract import sha256_file

from tests.test_academic_collection_alignment import build_fixture, run_convert
from tests.test_academic_collection_publication import (
    CLEAN_RUNTIME,
    fixture_acceptance_contract,
)


def word_tokenizer(text: str) -> list[str]:
    return text.split()


def rewrite_manifest(target: Path, mutate) -> Path:
    path = target / academic.MANIFEST_FILE_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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
        assert self.manifest["acceptance"]["eligible"] is True

    def tearDown(self):
        self._directory.cleanup()

    def gate(self, **kwargs):
        return academic.require_accepted_conversion(
            self.manifest_path, acceptance_contract=self.contract, **kwargs
        )


class GateForgeryTests(AcceptedTargetHarness):
    def test_untampered_gate_passes_with_pin(self):
        pinned = sha256_file(self.manifest_path)
        manifest = self.gate(expected_manifest_sha256=pinned)
        self.assertTrue(manifest["acceptance"]["eligible"])

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

        rewrite_manifest(smoke_target, flip)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            academic.require_accepted_conversion(
                smoke_target / academic.MANIFEST_FILE_NAME,
                acceptance_contract=self.contract,
            )

    def test_smoke_flag_flip_with_cleared_reasons_fails(self):
        smoke_target = self.root / "smoke2"
        run_convert(
            self.fixture,
            smoke_target,
            limit_documents=1,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=self.contract,
        )

        def forge(manifest):
            manifest["acceptance"]["eligible"] = True
            manifest["acceptance"]["reasons"] = []

        rewrite_manifest(smoke_target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "does not match recomputation"
        ):
            academic.require_accepted_conversion(
                smoke_target / academic.MANIFEST_FILE_NAME,
                acceptance_contract=self.contract,
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

    def test_runtime_tamper_fails(self):
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
            academic.require_accepted_conversion(
                self.manifest_path, acceptance_contract=other
            )

    def test_identity_self_hash_tamper_fails(self):
        def forge(manifest):
            manifest["run_identity"]["identity_sha256"] = "f" * 64

        rewrite_manifest(self.target, forge)
        with self.assertRaisesRegex(
            academic.ConversionError, "self-hash mismatch"
        ):
            self.gate()

    def test_manifest_pin_mismatch_fails(self):
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            self.gate(expected_manifest_sha256="f" * 64)

    def test_contract_is_required(self):
        with self.assertRaisesRegex(academic.ConversionError, "contract is required"):
            academic.require_accepted_conversion(
                self.manifest_path, acceptance_contract=None
            )

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


class CodeIdentityCoverageTests(unittest.TestCase):
    @staticmethod
    def local_imports(module_path: Path, seen: set[str]) -> set[str]:
        source = module_path.read_text(encoding="utf-8")
        found = set()
        for match in re.finditer(
            r"^(?:from\s+(src(?:\.\w+)+)\s+import|import\s+(src(?:\.\w+)+))",
            source,
            re.MULTILINE,
        ):
            found.add(match.group(1) or match.group(2))
        modules = set()
        for dotted in found:
            if dotted in seen:
                continue
            seen.add(dotted)
            path = ROOT / (dotted.replace(".", "/") + ".py")
            modules.add(dotted)
            if path.is_file():
                modules |= CodeIdentityCoverageTests.local_imports(path, seen)
        return modules

    def test_every_local_src_import_is_in_code_identity(self):
        seen: set[str] = set()
        adapter_path = ROOT / "src" / "data" / "collection_academic.py"
        dotted_modules = self.local_imports(adapter_path, seen)
        dotted_modules.add("src.data.collection_academic")
        identity_files = {path.resolve() for path in academic._CODE_IDENTITY_FILES.values()}
        for dotted in sorted(dotted_modules):
            path = (ROOT / (dotted.replace(".", "/") + ".py")).resolve()
            self.assertTrue(path.is_file(), f"cannot resolve import {dotted}")
            self.assertIn(
                path,
                identity_files,
                f"behavior-affecting module {dotted} is missing from "
                "_CODE_IDENTITY_FILES",
            )

    def test_contract_module_and_clis_are_covered(self):
        keys = set(academic._CODE_IDENTITY_FILES)
        self.assertLessEqual(
            {"adapter", "publication", "collection_contract", "build_cli", "compat_cli"},
            keys,
        )
        for name, path in academic._CODE_IDENTITY_FILES.items():
            self.assertTrue(path.is_file(), f"{name} identity file missing: {path}")
        hashes = academic.code_identity_hashes()
        self.assertEqual(set(hashes), keys)
        for digest in hashes.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class RetrievalApprovalTests(AcceptedTargetHarness):
    def attest(self, *, token_budget: int, tokenizer=word_tokenizer):
        return academic.build_retrieval_approval_attestation(
            target_dir=self.target,
            acceptance_contract=self.contract,
            model_id="fixture/embedding",
            revision="0123456789abcdef",
            token_budget=token_budget,
            input_template="passage: {text}",
            tokenizer=tokenizer,
            approved_by="owner",
        )

    def test_approved_attestation_passes_gate(self):
        attestation = self.attest(token_budget=10_000)
        self.assertEqual(attestation["result"], "approved")
        self.assertEqual(attestation["token_violations"], 0)
        verified = academic.require_retrieval_approved(
            attestation,
            target_dir=self.target,
            acceptance_contract=self.contract,
            tokenizer=word_tokenizer,
        )
        self.assertEqual(verified["result"], "approved")

    def test_rejected_attestation_fails_gate(self):
        attestation = self.attest(token_budget=3)
        self.assertEqual(attestation["result"], "rejected")
        self.assertGreater(attestation["token_violations"], 0)
        with self.assertRaisesRegex(
            academic.ConversionError, "approved zero-violation"
        ):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                tokenizer=word_tokenizer,
            )

    def test_gate_reverifies_token_counts_with_frozen_tokenizer(self):
        attestation = self.attest(token_budget=100_000)

        def char_tokenizer(text: str) -> int:
            return len(text)

        with self.assertRaisesRegex(
            academic.ConversionError, "recomputed token scan does not match"
        ):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                tokenizer=char_tokenizer,
            )

    def test_tampered_chunks_fail_gate(self):
        attestation = self.attest(token_budget=10_000)
        chunk_path = self.target / "academic_zz" / "chunks.jsonl"
        chunk_path.write_text(
            chunk_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(academic.ConversionError, "does not match"):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                tokenizer=word_tokenizer,
            )

    def test_code_identity_mismatch_fails_gate(self):
        attestation = self.attest(token_budget=10_000)
        attestation["code_identity"] = dict(
            attestation["code_identity"], adapter="f" * 64
        )
        with self.assertRaisesRegex(academic.ConversionError, "code identity"):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                tokenizer=word_tokenizer,
            )

    def test_template_tamper_fails_gate(self):
        attestation = self.attest(token_budget=10_000)
        attestation["input_template"] = "passage: {text} extra"
        with self.assertRaisesRegex(academic.ConversionError, "template hash"):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                tokenizer=word_tokenizer,
            )

    def test_mutable_revision_rejected(self):
        with self.assertRaisesRegex(academic.ConversionError, "immutable"):
            academic.build_retrieval_approval_attestation(
                target_dir=self.target,
                acceptance_contract=self.contract,
                model_id="fixture/embedding",
                revision="main",
                token_budget=10,
                input_template="{text}",
                tokenizer=word_tokenizer,
                approved_by="owner",
            )

    def test_smoke_corpus_cannot_be_attested(self):
        smoke_target = self.root / "smoke"
        run_convert(
            self.fixture,
            smoke_target,
            limit_documents=1,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=self.contract,
        )
        with self.assertRaisesRegex(
            academic.ConversionError, "not acceptance-eligible"
        ):
            academic.build_retrieval_approval_attestation(
                target_dir=smoke_target,
                acceptance_contract=self.contract,
                model_id="fixture/embedding",
                revision="0123456789abcdef",
                token_budget=10_000,
                input_template="{text}",
                tokenizer=word_tokenizer,
                approved_by="owner",
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


if __name__ == "__main__":
    unittest.main()
