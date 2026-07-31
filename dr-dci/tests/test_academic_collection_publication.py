"""RED-to-GREEN tests for atomic publication, acceptance gating, and scale.

All fixtures are synthetic; no private raw data is read or reproduced here.
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tracemalloc
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collection_academic as academic
from src.data.collection_publication import (
    LockConflictError,
    TargetLock,
    lock_path_for,
    staging_path_for,
)
from src.eval.collection_contract import sha256_file

from tests.test_academic_collection_alignment import (
    build_fixture,
    label_members,
    make_zip,
    run_convert,
    synthetic_label,
)


CLEAN_RUNTIME = {
    "git_commit": "a" * 40,
    "git_dirty": False,
    "python_version": "3.13.5",
    "platform": "synthetic",
    "pyyaml_version": "6.0",
}


def snapshot_tree(target: Path) -> dict[str, tuple[int, int, str]]:
    snapshot = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(target))] = (
                stat.st_mtime_ns,
                stat.st_size,
                sha256_file(path),
            )
    return snapshot


def fixture_acceptance_contract(fixture: dict, manifest: dict) -> dict:
    """Build a synthetic acceptance contract matching the fixture's true counts."""
    collections = sorted(fixture["config"]["collections"])
    cells = {}
    for cell_name, cell in manifest["cells"].items():
        cells[cell_name] = {
            "documents": cell["converted_documents"],
            "elements": cell["elements_retained"],
            "images_excluded": cell["image_records_excluded"],
        }
    return {
        "schema_version": academic.ACCEPTANCE_CONTRACT_SCHEMA_VERSION,
        "collections": collections,
        "splits": ["Training", "Validation"],
        "archive_count": len(collections) * 4,
        "totals": {
            "documents": manifest["totals"]["documents"],
            "elements": manifest["totals"]["elements"],
            "images_excluded": manifest["totals"]["image_records_excluded"],
        },
        "cells": cells,
    }


class AtomicPublicationTests(unittest.TestCase):
    def test_publish_reuse_and_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            first = run_convert(fixture, target)
            self.assertEqual(first["publication_status"], "published")
            self.assertFalse(staging_path_for(target).exists())
            self.assertTrue(lock_path_for(target).exists())

            metrics = json.loads(
                (target / academic.METRICS_FILE_NAME).read_text(encoding="utf-8")
            )
            for key in ("timings_seconds", "peak_rss_bytes", "bytes_read", "bytes_written"):
                self.assertIn(key, metrics)
            manifest_text = (target / academic.MANIFEST_FILE_NAME).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("timings_seconds", manifest_text)
            self.assertNotIn("publication_status", manifest_text)

            before = snapshot_tree(target)
            second = run_convert(fixture, target)
            self.assertEqual(second["publication_status"], "reused")
            self.assertEqual(snapshot_tree(target), before)
            self.assertEqual(
                first["run_identity"]["identity_sha256"],
                second["run_identity"]["identity_sha256"],
            )

    def test_mid_conversion_failure_leaves_no_partial_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            label_path = fixture["data_root"] / "labels" / "academic_zz_Training.zip"
            label = synthetic_label("ZZ_0001", ["합성 문장 하나."])
            label["training_data_info"]["section_info"][0]["original_text"] = " "
            make_zip(
                label_path,
                {"ZZ_0001.json": json.dumps(label, ensure_ascii=False).encode()},
            )
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"] = {
                "path": str(label_path.relative_to(fixture["data_root"])),
                "bytes": label_path.stat().st_size,
                "sha256": sha256_file(label_path),
            }
            target = root / "out"
            with self.assertRaises(academic.ConversionError):
                run_convert(fixture, target, source_config=config)
            self.assertFalse(target.exists())
            self.assertFalse(staging_path_for(target).exists())

    def test_verification_failure_leaves_no_partial_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            with mock.patch.object(
                academic,
                "verify_collection_outputs",
                side_effect=academic.ConversionError("injected verification failure"),
            ):
                with self.assertRaisesRegex(
                    academic.ConversionError, "injected verification failure"
                ):
                    run_convert(fixture, target)
            self.assertFalse(target.exists())
            self.assertFalse(staging_path_for(target).exists())

    def test_failed_rerun_leaves_accepted_target_unchanged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            run_convert(fixture, target)
            before = snapshot_tree(target)
            with self.assertRaisesRegex(
                academic.ConversionError, "different input identity"
            ):
                run_convert(fixture, target, limit_documents=1)
            self.assertEqual(snapshot_tree(target), before)
            self.assertFalse(staging_path_for(target).exists())

    def test_stale_broader_selection_cannot_survive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            broad_target = root / "out"
            run_convert(fixture, broad_target)
            with self.assertRaisesRegex(
                academic.ConversionError, "different input identity"
            ):
                run_convert(fixture, broad_target, collections=["academic_zz"])
            self.assertTrue((broad_target / "academic_yy").is_dir())

            narrow_target = root / "narrow"
            manifest = run_convert(fixture, narrow_target, collections=["academic_zz"])
            self.assertEqual(manifest["selection"]["collections"], ["academic_zz"])
            self.assertTrue((narrow_target / "academic_zz").is_dir())
            self.assertFalse((narrow_target / "academic_yy").exists())

    def test_partial_target_fails_loud(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            run_convert(fixture, target)
            (target / academic.MANIFEST_FILE_NAME).unlink()
            with self.assertRaisesRegex(
                academic.ConversionError, "incomplete or corrupt"
            ):
                run_convert(fixture, target)

    def test_corrupt_published_artifact_fails_reuse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            run_convert(fixture, target)
            chunk_path = target / "academic_zz" / "chunks.jsonl"
            chunk_path.write_text(
                chunk_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                academic.ConversionError, "does not match manifest"
            ):
                run_convert(fixture, target)

    def test_held_lock_yields_stable_conflict_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            with TargetLock(target):
                with self.assertRaisesRegex(LockConflictError, "held by another"):
                    run_convert(fixture, target)
            self.assertFalse(target.exists())
            manifest = run_convert(fixture, target)
            self.assertEqual(manifest["publication_status"], "published")

    def test_concurrent_same_target_processes_do_not_corrupt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            target = root / "out"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_academic_collections.py"),
                "--data-root",
                str(fixture["data_root"]),
                "--config",
                str(fixture["config_path"]),
                "--chunking-config",
                str(fixture["chunking_path"]),
                "--output-dir",
                str(target),
            ]
            processes = [
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            results = [process.communicate() for process in processes]
            codes = [process.returncode for process in processes]
            self.assertTrue(
                set(codes) <= {0, 3},
                f"unexpected exit codes {codes}: {[r[1][-300:] for r in results]}",
            )
            self.assertIn(0, codes)
            summary = academic.verify_collection_outputs(
                target, "academic_zz", separator="\n\n"
            )
            self.assertEqual(summary["status"], "ok")


class AcceptanceGateTests(unittest.TestCase):
    def build_accepted(self, root: Path):
        fixture = build_fixture(root)
        target = root / "probe"
        probe = run_convert(fixture, target, runtime_identity=CLEAN_RUNTIME)
        contract = fixture_acceptance_contract(fixture, probe)
        accepted_target = root / "accepted"
        manifest = run_convert(
            fixture,
            accepted_target,
            runtime_identity=CLEAN_RUNTIME,
            acceptance_contract=contract,
        )
        return fixture, contract, accepted_target, manifest

    def test_full_synthetic_run_is_acceptance_eligible(self):
        with TemporaryDirectory() as directory:
            _, _, target, manifest = self.build_accepted(Path(directory))
            self.assertTrue(manifest["acceptance"]["eligible"], manifest["acceptance"])
            self.assertEqual(manifest["acceptance"]["reasons"], [])
            loaded = academic.require_accepted_conversion(
                target / academic.MANIFEST_FILE_NAME
            )
            self.assertEqual(
                loaded["run_identity"]["identity_sha256"],
                manifest["run_identity"]["identity_sha256"],
            )

    def test_smoke_and_partial_runs_are_ineligible_with_reasons(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, contract, _, manifest = self.build_accepted(root)

            smoke = run_convert(
                fixture,
                root / "smoke",
                limit_documents=1,
                runtime_identity=CLEAN_RUNTIME,
                acceptance_contract=contract,
            )
            self.assertFalse(smoke["acceptance"]["eligible"])
            self.assertTrue(
                any("limit_documents_per_cell" in reason for reason in smoke["acceptance"]["reasons"])
            )

            no_hash = run_convert(
                fixture,
                root / "nohash",
                verify_archive_sha256=False,
                runtime_identity=CLEAN_RUNTIME,
                acceptance_contract=contract,
            )
            self.assertFalse(no_hash["acceptance"]["eligible"])
            self.assertTrue(
                any("SHA-256" in reason for reason in no_hash["acceptance"]["reasons"])
            )

            dirty = dict(manifest)
            dirty["run_identity"] = {
                **manifest["run_identity"],
                "runtime": {**CLEAN_RUNTIME, "git_dirty": True},
            }
            decision = academic.evaluate_acceptance(dirty, contract)
            self.assertFalse(decision["eligible"])
            self.assertTrue(
                any("dirty" in reason for reason in decision["reasons"])
            )

            wrong_totals = dict(contract, totals={**contract["totals"], "documents": 1})
            decision = academic.evaluate_acceptance(manifest, wrong_totals)
            self.assertFalse(decision["eligible"])
            self.assertTrue(
                any("total documents" in reason for reason in decision["reasons"])
            )

            missing_contract = academic.evaluate_acceptance(manifest, None)
            self.assertFalse(missing_contract["eligible"])

    def test_require_accepted_conversion_rejects_smoke_and_tamper(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, contract, target, _ = self.build_accepted(root)
            smoke_target = root / "smoke"
            run_convert(
                fixture,
                smoke_target,
                limit_documents=1,
                runtime_identity=CLEAN_RUNTIME,
                acceptance_contract=contract,
            )
            with self.assertRaisesRegex(
                academic.ConversionError, "not acceptance-eligible"
            ):
                academic.require_accepted_conversion(
                    smoke_target / academic.MANIFEST_FILE_NAME
                )

            elements_path = target / "academic_zz" / "elements.jsonl"
            elements_path.write_text(
                elements_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                academic.ConversionError, "does not match manifest hash"
            ):
                academic.require_accepted_conversion(
                    target / academic.MANIFEST_FILE_NAME
                )

    def test_acceptance_contract_loader_validates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "schema_version": academic.ACCEPTANCE_CONTRACT_SCHEMA_VERSION,
                "collections": ["academic_zz"],
                "splits": ["Training", "Validation"],
                "archive_count": 4,
                "totals": {"documents": 1, "elements": 1, "images_excluded": 0},
                "cells": {
                    "academic_zz/Training": {
                        "documents": 1, "elements": 1, "images_excluded": 0
                    },
                    "academic_zz/Validation": {
                        "documents": 0, "elements": 0, "images_excluded": 0
                    },
                },
            }
            path = root / "contract.yaml"
            path.write_text(yaml.safe_dump(contract), encoding="utf-8")
            loaded = academic.load_acceptance_contract(path)
            self.assertEqual(loaded["archive_count"], 4)

            broken = dict(contract)
            broken["cells"] = {"academic_zz/Training": contract["cells"]["academic_zz/Training"]}
            path.write_text(yaml.safe_dump(broken), encoding="utf-8")
            with self.assertRaisesRegex(
                academic.ConversionError, "collections x splits"
            ):
                academic.load_acceptance_contract(path)

    def test_real_contract_file_matches_accepted_eda_denominators(self):
        contract = academic.load_acceptance_contract(
            ROOT / "config" / "collection_academic" / "accepted_full_run.v1.yaml"
        )
        self.assertEqual(contract["totals"]["documents"], 9000)
        self.assertEqual(contract["totals"]["elements"], 177341)
        self.assertEqual(contract["totals"]["images_excluded"], 35883)
        self.assertEqual(
            sum(cell["documents"] for cell in contract["cells"].values()), 9000
        )
        self.assertEqual(
            sum(cell["elements"] for cell in contract["cells"].values()), 177341
        )
        self.assertEqual(
            sum(cell["images_excluded"] for cell in contract["cells"].values()), 35883
        )
        self.assertEqual(contract["archive_count"], 12)


class ChunkCompatibilityTests(unittest.TestCase):
    def test_manifest_reports_chunk_stats_and_unapproved_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            manifest = run_convert(fixture, root / "out")
            self.assertEqual(
                manifest["retrieval_compatibility"]["status"], "retrieval_unapproved"
            )
            stats = manifest["collections"]["academic_zz"]["chunk_length_stats"]
            self.assertGreater(stats["count"], 0)
            self.assertGreaterEqual(stats["max"], stats["p50"])
            self.assertGreater(stats["oversized_count"], 0)
            for sample in stats["oversized_element_id_sha256_samples"]:
                self.assertRegex(sample, r"^[0-9a-f]{64}$")
            cell_stats = manifest["cells"]["academic_zz/Training"]["chunk_length_stats"]
            self.assertGreater(cell_stats["count"], 0)

    def test_compat_validator_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            run_convert(fixture, root / "out")
            chunks = root / "out" / "academic_zz" / "chunks.jsonl"
            base_contract = {
                "schema_version": academic.COMPAT_CONTRACT_SCHEMA_VERSION,
                "model_id": "fixture/embedding",
                "revision": "0123456789abcdef",
                "max_input_chars": 100000,
            }
            result = academic.validate_chunk_model_compatibility(chunks, base_contract)
            self.assertEqual(result["status"], "compatible")

            tight = dict(base_contract, max_input_chars=10)
            with self.assertRaisesRegex(academic.ConversionError, "exceed the frozen"):
                academic.validate_chunk_model_compatibility(chunks, tight)

            approved = dict(
                tight,
                approved_long_element_split_policy={
                    "policy_id": "fixture.split.v1",
                    "approved_by": "owner",
                },
            )
            result = academic.validate_chunk_model_compatibility(chunks, approved)
            self.assertEqual(result["status"], "approved_split_policy_recorded")
            self.assertEqual(result["policy_id"], "fixture.split.v1")

    def test_compat_contract_loader_rejects_mutable_revision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "contract.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": academic.COMPAT_CONTRACT_SCHEMA_VERSION,
                        "model_id": "fixture/embedding",
                        "revision": "main",
                        "max_input_chars": 100,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(academic.ConversionError, "immutable"):
                academic.load_compat_contract(path)


class NormalizedOffsetTests(unittest.TestCase):
    def test_offsets_are_original_document_coordinates(self):
        start, end = academic.locate_unique_normalized("a  b", "a b")
        self.assertEqual((start, end), (0, 4))
        self.assertEqual("a  b"[start:end], "a  b")

    def test_mixed_whitespace_round_trips(self):
        document = "  머리말\n\n합성  문장\t테스트\n끝  "
        element = "합성 문장 테스트"
        start, end = academic.locate_unique_normalized(document, element)
        piece = document[start:end]
        self.assertEqual(piece, "합성  문장\t테스트")
        normalized, _ = academic._normalize_with_index_map(piece)
        self.assertEqual(normalized.strip(), element)

    def test_leading_whitespace_document(self):
        start, end = academic.locate_unique_normalized("  x y", "x y")
        self.assertEqual((start, end), (2, 5))

    def test_zero_or_many_matches_fail(self):
        with self.assertRaises(academic.AmbiguousAlignmentError):
            academic.locate_unique_normalized("반복. 반복.", "반복.")
        with self.assertRaises(academic.AmbiguousAlignmentError):
            academic.locate_unique_normalized("문서 내용", "없는 문장")


class ScaleTests(unittest.TestCase):
    @staticmethod
    def scaled_docs(factor: int) -> dict:
        sentences = [
            "합성 문장 A 입니다. 규모 시험을 위한 문장입니다.",
            "합성 문장 B 입니다. 규모 시험을 위한 문장입니다.",
            "합성 문장 C 입니다. 규모 시험을 위한 문장입니다.",
        ]
        training = {
            f"ZZ_{index:04d}": [f"{sentence} #{index}" for sentence in sentences]
            for index in range(1, 1 + 4 * factor)
        }
        validation = {
            f"ZV_{index:04d}": [f"{sentence} V#{index}" for sentence in sentences]
            for index in range(1, 1 + 2 * factor)
        }
        return {"academic_zz": {"Training": training, "Validation": validation}}

    def test_linear_growth_and_bounded_verifier_memory(self):
        peaks = {}
        elements = {}
        for factor in (1, 2, 4):
            with TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = build_fixture(root, self.scaled_docs(factor), max_chars=60)
                target = root / "out"
                manifest = run_convert(fixture, target)
                elements[factor] = manifest["totals"]["elements"]
                tracemalloc.start()
                academic.verify_collection_outputs(
                    target, "academic_zz", separator="\n\n"
                )
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peaks[factor] = peak
        self.assertEqual(elements[2], 2 * elements[1])
        self.assertEqual(elements[4], 4 * elements[1])
        self.assertLessEqual(
            peaks[4],
            3 * peaks[1] + 256 * 1024,
            f"verifier peak memory grew with corpus size: {peaks}",
        )


if __name__ == "__main__":
    unittest.main()
