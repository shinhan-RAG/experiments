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


IMMUTABLE_REV = "0123456789abcdef0123456789abcdef01234567"


class FakeBatchEncoding(dict):
    """Mapping like Hugging Face BatchEncoding: len() is the field count."""


def make_tokenizer_contract(root: Path, *, add_special_tokens: bool = True) -> dict:
    snapshot = root / "tokenizer-snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    names = ("tokenizer.json", "tokenizer_config.json")
    for name in names:
        (snapshot / name).write_bytes(f"synthetic-{name}".encode())
    return {
        "tokenizer_id": "fixture/tokenizer",
        "revision": IMMUTABLE_REV,
        "tokenizer_class": "SyntheticTokenizerFast",
        "trust_remote_code": False,
        "local_snapshot_path": str(snapshot),
        "options": {
            "add_special_tokens": add_special_tokens,
            "truncation": False,
            "padding": False,
            "max_length": None,
            "normalization": tokenizer_runtime.NORMALIZATION_TOKENIZER_BUILTIN,
        },
        "files": [
            {
                "path": name,
                "bytes": (snapshot / name).stat().st_size,
                "sha256": sha256_file(snapshot / name),
            }
            for name in names
        ],
    }


def synthetic_loader_factory(contract: dict, snapshot_root: Path):
    """Test-only stand-in for tokenizer_runtime.load_frozen_tokenizer."""
    add_special = contract["options"]["add_special_tokens"]

    def encode(text: str) -> FakeBatchEncoding:
        ids = [1000 + index for index, _ in enumerate(text.split())]
        if add_special:
            ids = [101, *ids, 102]
        return FakeBatchEncoding(
            {"input_ids": ids, "attention_mask": [1] * len(ids)}
        )

    return encode


def constant_count_loader_factory(count: int):
    """The review's bypass reproduction: a constant-length token counter."""

    def factory(contract: dict, snapshot_root: Path):
        def encode(text: str) -> FakeBatchEncoding:
            return FakeBatchEncoding(
                {"input_ids": [1] * count, "attention_mask": [1] * count}
            )

        return encode

    return factory


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

    def test_contract_module_and_clis_are_covered(self):
        keys = set(academic._CODE_IDENTITY_FILES)
        self.assertLessEqual(
            {
                "adapter",
                "publication",
                "tokenizer_runtime",
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
        self.approval_root = self.root / "approvals"
        approval_file = self.approval_root / "reviews" / "PR13_APPROVAL.md"
        approval_file.parent.mkdir(parents=True, exist_ok=True)
        approval_file.write_text(
            "# Synthetic reviewed approval record\n", encoding="utf-8"
        )
        self.approval = {
            "approved_by": "owner",
            "approval_ref": "reviews/PR13_APPROVAL.md",
            "bytes": approval_file.stat().st_size,
            "approval_artifact_sha256": sha256_file(approval_file),
        }
        patcher = mock.patch.object(
            tokenizer_runtime, "TEST_ONLY_LOADER_OVERRIDE", synthetic_loader_factory
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def attest(self, *, token_budget: int, tokenizer_contract: dict | None = None,
               approval: dict | None = None):
        return academic.build_retrieval_approval_attestation(
            target_dir=self.target,
            acceptance_contract=self.contract,
            expected_manifest_sha256=self.pin,
            model_id="fixture/embedding",
            revision=IMMUTABLE_REV,
            token_budget=token_budget,
            input_template="passage: {text}",
            tokenizer_contract=tokenizer_contract or self.tokenizer_contract,
            approval=approval or self.approval,
            approval_root=self.approval_root,
        )

    def require(self, attestation, *, attestation_pin: str | None = None):
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
            approval_root=self.approval_root,
        )

    def test_approved_attestation_passes_gate_from_file_with_sidecar(self):
        attestation = self.attest(token_budget=10_000)
        self.assertEqual(attestation["result"], "approved")
        self.assertRegex(
            attestation["token_count_digest_sha256"], r"^[0-9a-f]{64}$"
        )
        path = academic.write_retrieval_attestation(
            attestation, self.root / "attestation.json"
        )
        sidecar = Path(str(path) + academic.ATTESTATION_FILE_SUFFIX)
        self.assertTrue(sidecar.is_file())
        pin = sidecar.read_text(encoding="utf-8").split()[0]
        verified = self.require(path, attestation_pin=pin)
        self.assertEqual(verified["result"], "approved")

    def test_constant_length_bypass_loader_fails(self):
        attestation = self.attest(token_budget=10_000)
        bypass = constant_count_loader_factory(attestation["max_tokens_observed"])
        with mock.patch.object(
            tokenizer_runtime, "TEST_ONLY_LOADER_OVERRIDE", bypass
        ):
            with self.assertRaisesRegex(
                academic.ConversionError, "token-count digest does not match"
            ):
                self.require(attestation)

    def test_production_path_refuses_arbitrary_loader(self):
        import inspect

        for gate in (
            academic.build_retrieval_approval_attestation,
            academic.require_retrieval_approved,
        ):
            parameters = inspect.signature(gate).parameters
            self.assertFalse(
                [name for name in parameters if "loader" in name or "tokenizer_fn" in name],
                f"{gate.__name__} must not accept a caller-provided loader",
            )
        with mock.patch.object(tokenizer_runtime, "TEST_ONLY_LOADER_OVERRIDE", None):
            with self.assertRaises(
                (tokenizer_runtime.TokenizerRuntimeError, ValueError)
            ):
                self.attest(token_budget=10_000)

    def test_floating_revision_rejected(self):
        floating = json.loads(json.dumps(self.tokenizer_contract))
        floating["revision"] = "floating-release-tag"
        with self.assertRaisesRegex(
            academic.ConversionError, "immutable content identifier"
        ):
            self.attest(token_budget=10_000, tokenizer_contract=floating)
        with self.assertRaisesRegex(
            academic.ConversionError, "immutable content identifier"
        ):
            academic.build_retrieval_approval_attestation(
                target_dir=self.target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=self.pin,
                model_id="fixture/embedding",
                revision="v1.2.3",
                token_budget=10_000,
                input_template="{text}",
                tokenizer_contract=self.tokenizer_contract,
                approval=self.approval,
                approval_root=self.approval_root,
            )

    def test_unlisted_extra_tokenizer_file_rejected(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        (snapshot / "special_tokens_map.json").write_bytes(b"{}")
        with self.assertRaisesRegex(
            academic.ConversionError, "not in the frozen inventory"
        ):
            self.require(attestation)
        with self.assertRaisesRegex(
            academic.ConversionError, "not in the frozen inventory"
        ):
            self.attest(token_budget=10_000)

    def test_symlinked_tokenizer_file_rejected(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        moved = self.root / "moved-tokenizer.json"
        original = snapshot / "tokenizer.json"
        moved.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(moved)
        with self.assertRaisesRegex(academic.ConversionError, "symlink"):
            self.require(attestation)

    def test_option_keys_must_be_exact(self):
        missing = json.loads(json.dumps(self.tokenizer_contract))
        del missing["options"]["normalization"]
        with self.assertRaisesRegex(academic.ConversionError, "exactly"):
            self.attest(token_budget=10_000, tokenizer_contract=missing)

        extra = json.loads(json.dumps(self.tokenizer_contract))
        extra["options"]["lowercase"] = True
        with self.assertRaisesRegex(academic.ConversionError, "exactly"):
            self.attest(token_budget=10_000, tokenizer_contract=extra)

        wrong = json.loads(json.dumps(self.tokenizer_contract))
        wrong["options"]["normalization"] = "external"
        with self.assertRaisesRegex(academic.ConversionError, "normalization"):
            self.attest(token_budget=10_000, tokenizer_contract=wrong)

    def test_tokenizer_snapshot_tamper_fails(self):
        attestation = self.attest(token_budget=10_000)
        snapshot = Path(self.tokenizer_contract["local_snapshot_path"])
        (snapshot / "tokenizer.json").write_bytes(b"mutated-tokenizer")
        with self.assertRaisesRegex(
            academic.ConversionError, "tokenizer file (sha256|bytes) mismatch"
        ):
            self.require(attestation)

    def test_approval_artifact_is_actually_verified(self):
        missing = dict(self.approval, approval_ref="reviews/DOES_NOT_EXIST.md")
        with self.assertRaisesRegex(academic.ConversionError, "approval artifact missing"):
            self.attest(token_budget=10_000, approval=missing)

        wrong_hash = dict(self.approval, approval_artifact_sha256="f" * 64)
        with self.assertRaisesRegex(academic.ConversionError, "sha256 mismatch"):
            self.attest(token_budget=10_000, approval=wrong_hash)

        for bad_ref in ("../escape.md", "/etc/passwd"):
            with self.subTest(bad_ref=bad_ref):
                broken = dict(self.approval, approval_ref=bad_ref)
                with self.assertRaisesRegex(academic.ConversionError, "safe path"):
                    self.attest(token_budget=10_000, approval=broken)

    def test_symlinked_approval_artifact_rejected(self):
        approval_file = self.approval_root / "reviews" / "PR13_APPROVAL.md"
        moved = self.root / "moved-approval.md"
        moved.write_bytes(approval_file.read_bytes())
        approval_file.unlink()
        approval_file.symlink_to(moved)
        with self.assertRaisesRegex(academic.ConversionError, "symlink"):
            self.attest(token_budget=10_000)

    def test_approval_tamper_after_attestation_fails_gate(self):
        attestation = self.attest(token_budget=10_000)
        approval_file = self.approval_root / "reviews" / "PR13_APPROVAL.md"
        approval_file.write_text("tampered approval\n", encoding="utf-8")
        with self.assertRaisesRegex(academic.ConversionError, "mismatch"):
            self.require(attestation)

    def test_attestation_pin_is_mandatory_and_validated(self):
        attestation = self.attest(token_budget=10_000)
        with self.assertRaises(TypeError):
            academic.require_retrieval_approved(
                attestation,
                target_dir=self.target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=self.pin,
                approval_root=self.approval_root,
            )
        with self.assertRaisesRegex(academic.ConversionError, "lowercase SHA-256"):
            self.require(attestation, attestation_pin="not-a-sha")

    def test_rejected_attestation_fails_gate(self):
        attestation = self.attest(token_budget=3)
        self.assertEqual(attestation["result"], "rejected")
        with self.assertRaisesRegex(academic.ConversionError, "approved zero-violation"):
            self.require(attestation)

    def test_model_id_and_approved_by_tamper_fail_against_pin(self):
        attestation = self.attest(token_budget=10_000)
        pin = academic.attestation_sha256(attestation)
        forged = json.loads(json.dumps(attestation))
        forged["model_id"] = "swapped/model"
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            self.require(forged, attestation_pin=pin)
        forged = json.loads(json.dumps(attestation))
        forged["approval"]["approved_by"] = "intruder"
        with self.assertRaisesRegex(academic.ConversionError, "pinned SHA-256"):
            self.require(forged, attestation_pin=pin)

    def test_sidecar_tamper_fails(self):
        attestation = self.attest(token_budget=10_000)
        path = academic.write_retrieval_attestation(
            attestation, self.root / "attestation.json"
        )
        sidecar = Path(str(path) + academic.ATTESTATION_FILE_SUFFIX)
        sidecar.write_text("f" * 64 + "  attestation.json\n", encoding="utf-8")
        with self.assertRaisesRegex(academic.ConversionError, "sidecar mismatch"):
            self.require(path, attestation_pin=academic.attestation_sha256(attestation))

    def test_digest_tamper_with_recomputed_pin_still_fails(self):
        attestation = self.attest(token_budget=10_000)
        attestation["token_count_digest_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            academic.ConversionError, "token-count digest does not match"
        ):
            self.require(attestation)

    def test_template_tamper_with_recomputed_pin_still_fails(self):
        attestation = self.attest(token_budget=10_000)
        attestation["input_template"] = "passage: {text} extra"
        with self.assertRaisesRegex(academic.ConversionError, "template hash"):
            self.require(attestation)

    def test_special_token_overhead_is_included_in_budget(self):
        with_special = self.attest(token_budget=100_000)
        without_special_contract = make_tokenizer_contract(
            self.root / "nospecial", add_special_tokens=False
        )
        without_special = self.attest(
            token_budget=100_000, tokenizer_contract=without_special_contract
        )
        self.assertEqual(
            with_special["max_tokens_observed"],
            without_special["max_tokens_observed"] + 2,
        )
        exact = self.attest(token_budget=with_special["max_tokens_observed"])
        self.assertEqual(exact["result"], "approved")
        under = self.attest(token_budget=with_special["max_tokens_observed"] - 1)
        self.assertEqual(under["result"], "rejected")

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
            attestation["code_identity"], tokenizer_runtime="f" * 64
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
        with self.assertRaisesRegex(academic.ConversionError, "not acceptance-eligible"):
            academic.build_retrieval_approval_attestation(
                target_dir=smoke_target,
                acceptance_contract=self.contract,
                expected_manifest_sha256=sha256_file(smoke_manifest),
                model_id="fixture/embedding",
                revision=IMMUTABLE_REV,
                token_budget=10_000,
                input_template="{text}",
                tokenizer_contract=self.tokenizer_contract,
                approval=self.approval,
                approval_root=self.approval_root,
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
