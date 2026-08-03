"""RED-to-GREEN tests for the academic collection element_alignment stage.

Every fixture is synthetic; no private raw data is read or reproduced here.
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import collection_academic as academic
from src.eval.collection_contract import ContractError, sha256_file


SUMMARY_SENTINEL = "요약-필드-사용-금지-센티널"
CAPTION_SENTINEL = "캡션-필드-사용-금지-센티널"


def synthetic_section(paragraph_id: str, text: str) -> dict:
    return {
        "paragraph_id": paragraph_id,
        "page": "1",
        "location": "[1, 2, 3, 4]",
        "procede": "Y",
        "original_text": text,
        "summary_text": SUMMARY_SENTINEL,
        "original_cnt": "3",
        "summary_cnt": "1",
    }


def synthetic_label(stem: str, texts: list[str], *, images: int = 1) -> dict:
    return {
        "raw_data_meta_info": {"doc_id": f"doc-{stem}", "doc_title": "합성 제목"},
        "source_data_meta_info": {"source_data_id": f"ART-{stem}"},
        "training_data_info": {
            "section_info": [
                synthetic_section(f"para_{index + 1}", text)
                for index, text in enumerate(texts)
            ],
            "image_info": [
                {
                    "image_id": index + 1,
                    "image_name": "그림",
                    "image_caption": CAPTION_SENTINEL,
                    "image_category": "graph",
                    "image_page": "1",
                    "image_location": "[5, 6, 7, 8]",
                    "image_file_name": f"{stem}_그림 {index + 1}.png",
                }
                for index in range(images)
            ],
        },
    }


def make_zip(path: Path, members: dict[str, bytes], *, prefix: str = "/") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(prefix + name, data)


def label_members(docs: dict[str, list[str]], *, images: int = 1) -> dict[str, bytes]:
    members = {}
    for stem, texts in docs.items():
        members[f"{stem}.json"] = json.dumps(
            synthetic_label(stem, texts, images=images), ensure_ascii=False
        ).encode("utf-8")
        members[f"{stem}_그림 1.png"] = b"synthetic-png-bytes-" + stem.encode()
    return members


def source_members(stems: list[str]) -> dict[str, bytes]:
    members = {}
    for stem in stems:
        members[f"{stem}.pdf"] = b"%PDF-1.4 synthetic " + stem.encode()
        members[f"{stem}.pptx"] = b"PK-synthetic-pptx " + stem.encode()
    return members


DEFAULT_DOCS = {
    "academic_zz": {
        "Training": {
            "ZZ_0001": [
                "합성 서론 문장입니다. 데이터는 전부 만들어진 것입니다.",
                "짧은 절",
                "합성 본문 문장이 이어집니다. 반복되는 표현을 검증합니다.",
                "합성 본문 문장이 이어집니다. 반복되는 표현을 검증합니다.",
                "합성 결론 문단입니다. 이 문단은 길이를 늘리기 위한 합성 문장을 "
                "여러 번 포함합니다. 합성 문장, 합성 문장, 합성 문장.",
            ],
            "ZZ_0003": [
                "다른 합성 문서의 첫 문단입니다.",
                "다른 합성 문서의 둘째 문단입니다.",
            ],
        },
        "Validation": {
            "ZZ_0002": [
                "검증 분할의 합성 문단 하나.",
                "검증 분할의 합성 문단 둘. 조금 더 길게 작성된 합성 문장입니다.",
            ]
        },
    },
    "academic_yy": {
        "Training": {
            "YY_0001": [
                "두 번째 컬렉션의 합성 문단입니다.",
                "두 번째 컬렉션의 추가 합성 문단입니다.",
            ]
        },
        "Validation": {
            "YY_0002": [
                "두 번째 컬렉션 검증 분할의 합성 문단입니다.",
            ]
        },
    },
}


def archive_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_fixture(
    root: Path,
    docs_by_collection: dict | None = None,
    *,
    prefix: str = "/",
    max_chars: int = 60,
) -> dict:
    docs_by_collection = docs_by_collection or DEFAULT_DOCS
    data_root = root / "raw"
    collections_config = {}
    for collection_id, splits in docs_by_collection.items():
        domain = collection_id.rsplit("_", 1)[-1].upper()
        split_config = {}
        for split, docs in splits.items():
            label_path = data_root / "labels" / f"{collection_id}_{split}.zip"
            source_path = data_root / "sources" / f"{collection_id}_{split}.zip"
            make_zip(label_path, label_members(docs), prefix=prefix)
            make_zip(source_path, source_members(sorted(docs)), prefix=prefix)
            split_config[split] = {
                "label_archive": archive_entry(label_path, data_root),
                "source_archive": archive_entry(source_path, data_root),
            }
        collections_config[collection_id] = {"domain": domain, "splits": split_config}

    config = {
        "schema_version": academic.SOURCE_CONFIG_SCHEMA_VERSION,
        "dataset_name": "synthetic-academic-fixture",
        "eda": {
            "handoff_manifest_sha256": "0" * 64,
            "decision": "alignment_layer",
        },
        "collections": collections_config,
    }
    config_path = root / "source_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    chunking = {
        "schema_version": academic.CHUNK_POLICY_SCHEMA_VERSION,
        "policy_id": academic.CHUNK_POLICY_ID,
        "separator": "\n\n",
        "max_chars": max_chars,
        "element_atomic": True,
    }
    chunking_path = root / "chunking_config.yaml"
    chunking_path.write_text(
        yaml.safe_dump(chunking, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    return {
        "data_root": data_root,
        "config_path": config_path,
        "chunking_path": chunking_path,
        "config": config,
        "chunking": chunking,
    }


def run_convert(fixture: dict, output_dir: Path, **overrides) -> dict:
    kwargs = {
        "data_root": fixture["data_root"],
        "source_config": academic.load_source_config(fixture["config_path"]),
        "chunking_config": academic.load_chunking_config(fixture["chunking_path"]),
        "output_dir": output_dir,
        "verify_archive_sha256": True,
    }
    kwargs.update(overrides)
    return academic.convert_collections(**kwargs)


def artifact_hashes(output_dir: Path, collection_id: str) -> dict[str, str]:
    return {
        name: sha256_file(output_dir / collection_id / f"{name}.jsonl")
        for name in academic.ARTIFACT_NAMES
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rewrite_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class ConvertGreenPathTests(unittest.TestCase):
    def test_convert_verify_deterministic_rerun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            first = run_convert(fixture, root / "out1")
            second = run_convert(fixture, root / "out2")

            for collection_id in ("academic_yy", "academic_zz"):
                self.assertEqual(
                    artifact_hashes(root / "out1", collection_id),
                    artifact_hashes(root / "out2", collection_id),
                )
                summary = academic.verify_collection_outputs(
                    root / "out1", collection_id, separator="\n\n"
                )
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["element_to_chunk_coverage"], 1.0)
                self.assertEqual(summary["source_location"]["verified"], 0)

            self.assertEqual(first["totals"], second["totals"])
            self.assertEqual(first["totals"]["documents"], 5)
            self.assertEqual(first["totals"]["elements"], 12)
            self.assertEqual(
                first["totals"]["image_records_excluded"],
                first["exclusions"]["image_record_excluded_pending_provenance"],
            )
            self.assertGreater(first["totals"]["image_records_excluded"], 0)
            zz_training = first["cells"]["academic_zz/Training"]
            self.assertEqual(zz_training["converted_documents"], 2)
            self.assertEqual(zz_training["failed_documents"], 0)
            self.assertGreater(zz_training["label_png_members_ignored"], 0)

    def test_forbidden_fields_never_reach_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            run_convert(fixture, root / "out")
            for path in sorted((root / "out").rglob("*.jsonl")):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn(SUMMARY_SENTINEL, content, path.name)
                self.assertNotIn(CAPTION_SENTINEL, content, path.name)
                self.assertNotIn("합성 제목", content, path.name)

    def test_repeated_element_text_keeps_distinct_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            run_convert(fixture, root / "out")
            elements = load_jsonl(root / "out" / "academic_zz" / "elements.jsonl")
            repeated = [
                record
                for record in elements
                if record["text"].startswith("합성 본문 문장이 이어집니다")
            ]
            self.assertEqual(len(repeated), 2)
            self.assertEqual(len({record["element_id"] for record in repeated}), 2)

    def test_multi_element_chunks_and_single_membership(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            run_convert(fixture, root / "out")
            chunks = load_jsonl(root / "out" / "academic_zz" / "chunks.jsonl")
            alignment = load_jsonl(root / "out" / "academic_zz" / "alignment.jsonl")
            document_id = "academic_zz::ZZ_0001"
            doc_chunks = [c for c in chunks if c["document_id"] == document_id]
            self.assertGreater(len(doc_chunks), 1)
            for record in alignment:
                self.assertEqual(len(record["chunk_memberships"]), 1)

    def test_zip_member_order_and_prefix_invariance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = build_fixture(root / "a")
            run_convert(baseline, root / "a" / "out")

            reordered_root = root / "b"
            data_root = reordered_root / "raw"
            collections_config = {}
            for collection_id, splits in DEFAULT_DOCS.items():
                split_config = {}
                for split, docs in splits.items():
                    label_path = data_root / "labels" / f"{collection_id}_{split}.zip"
                    source_path = data_root / "sources" / f"{collection_id}_{split}.zip"
                    members = label_members(docs)
                    reordered = dict(reversed(list(members.items())))
                    make_zip(label_path, reordered, prefix="")
                    make_zip(source_path, source_members(sorted(docs)), prefix="")
                    split_config[split] = {
                        "label_archive": archive_entry(label_path, data_root),
                        "source_archive": archive_entry(source_path, data_root),
                    }
                collections_config[collection_id] = {
                    "domain": collection_id.rsplit("_", 1)[-1].upper(),
                    "splits": split_config,
                }
            config = dict(baseline["config"], collections=collections_config)
            config_path = reordered_root / "source_config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
                encoding="utf-8",
            )
            fixture = dict(
                baseline,
                data_root=data_root,
                config_path=config_path,
            )
            run_convert(fixture, reordered_root / "out")
            for collection_id in DEFAULT_DOCS:
                self.assertEqual(
                    artifact_hashes(root / "a" / "out", collection_id),
                    artifact_hashes(reordered_root / "out", collection_id),
                )

    def test_split_selection_keeps_split_and_collection_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            manifest = run_convert(fixture, root / "out", splits=["Training"])
            self.assertEqual(sorted(manifest["cells"]), [
                "academic_yy/Training",
                "academic_zz/Training",
            ])
            documents = load_jsonl(root / "out" / "academic_zz" / "documents.jsonl")
            self.assertTrue(documents)
            for record in documents:
                self.assertEqual(record["metadata"]["split"], "Training")
                self.assertEqual(record["collection_id"], "academic_zz")

    def test_cli_smoke_runs_on_synthetic_fixture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_academic_collections.py"),
                    "--data-root",
                    str(fixture["data_root"]),
                    "--config",
                    str(fixture["config_path"]),
                    "--chunking-config",
                    str(fixture["chunking_path"]),
                    "--output-dir",
                    str(root / "out"),
                    "--limit-documents",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout[result.stdout.index("{"):])
            self.assertEqual(
                set(summary["verification"].values()), {"ok"}
            )
            self.assertNotIn(SUMMARY_SENTINEL, result.stdout)


class ConvertRedPathTests(unittest.TestCase):
    def convert_expecting_error(self, fixture, output_dir, pattern, **overrides):
        with self.assertRaisesRegex(
            (academic.ConversionError, ContractError), pattern
        ):
            run_convert(fixture, output_dir, **overrides)

    def test_archive_bytes_pin_mismatch_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"]["bytes"] += 1
            with self.assertRaisesRegex(academic.ConversionError, "bytes mismatch"):
                run_convert(fixture, root / "out", source_config=config)

    def test_archive_sha256_pin_mismatch_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(academic.ConversionError, "sha256 mismatch"):
                run_convert(fixture, root / "out", source_config=config)

    def test_duplicate_paragraph_id_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나.", "합성 문장 둘."]},
                    "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            label_path = fixture["data_root"] / "labels" / "academic_zz_Training.zip"
            label = synthetic_label("ZZ_0001", ["합성 문장 하나.", "합성 문장 둘."])
            for section in label["training_data_info"]["section_info"]:
                section["paragraph_id"] = "para_1"
            make_zip(
                label_path,
                {
                    "ZZ_0001.json": json.dumps(label, ensure_ascii=False).encode(),
                },
            )
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"] = archive_entry(label_path, fixture["data_root"])
            self.convert_expecting_error(
                fixture, root / "out", "duplicate paragraph_id", source_config=config
            )

    def test_duplicate_stem_across_cells_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나."]},
                    "Validation": {"ZZ_0001": ["다른 내용의 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            self.convert_expecting_error(
                fixture, root / "out", "duplicate json stem across cells"
            )

    def test_missing_source_pair_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나."]},
                    "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            source_path = fixture["data_root"] / "sources" / "academic_zz_Training.zip"
            make_zip(source_path, {"ZZ_0001.pdf": b"%PDF-1.4 synthetic"})
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["source_archive"] = archive_entry(source_path, fixture["data_root"])
            self.convert_expecting_error(
                fixture, root / "out", "pdf/pptx pair", source_config=config
            )

    def test_empty_section_text_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나."]},
                    "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            label_path = fixture["data_root"] / "labels" / "academic_zz_Training.zip"
            label = synthetic_label("ZZ_0001", ["합성 문장 하나."])
            label["training_data_info"]["section_info"][0]["original_text"] = "   "
            make_zip(
                label_path,
                {"ZZ_0001.json": json.dumps(label, ensure_ascii=False).encode()},
            )
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"] = archive_entry(label_path, fixture["data_root"])
            self.convert_expecting_error(
                fixture, root / "out", "original_text", source_config=config
            )

    def test_unsafe_paragraph_id_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나."]},
                    "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            label_path = fixture["data_root"] / "labels" / "academic_zz_Training.zip"
            label = synthetic_label("ZZ_0001", ["합성 문장 하나."])
            label["training_data_info"]["section_info"][0]["paragraph_id"] = "../para"
            make_zip(
                label_path,
                {"ZZ_0001.json": json.dumps(label, ensure_ascii=False).encode()},
            )
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"] = archive_entry(label_path, fixture["data_root"])
            self.convert_expecting_error(
                fixture, root / "out", "unsafe identifier", source_config=config
            )

    def test_unsupported_label_member_type_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            docs = {
                "academic_zz": {
                    "Training": {"ZZ_0001": ["합성 문장 하나."]},
                    "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                }
            }
            fixture = build_fixture(root, docs)
            label_path = fixture["data_root"] / "labels" / "academic_zz_Training.zip"
            members = label_members(docs["academic_zz"]["Training"])
            members["notes.txt"] = b"synthetic"
            make_zip(label_path, members)
            config = academic.load_source_config(fixture["config_path"])
            spec = config["collections"]["academic_zz"]["splits"]["Training"]
            spec["label_archive"] = archive_entry(label_path, fixture["data_root"])
            self.convert_expecting_error(
                fixture, root / "out", "unsupported label member", source_config=config
            )

    def test_traversal_and_nested_member_names_fail(self):
        for bad_name in ("../evil.json", "nested/evil.json"):
            with self.subTest(bad_name=bad_name), TemporaryDirectory() as directory:
                root = Path(directory)
                docs = {
                    "academic_zz": {
                        "Training": {"ZZ_0001": ["합성 문장 하나."]},
                        "Validation": {"ZZ_0002": ["검증 합성 문장."]},
                    }
                }
                fixture = build_fixture(root, docs)
                label_path = (
                    fixture["data_root"] / "labels" / "academic_zz_Training.zip"
                )
                members = label_members(docs["academic_zz"]["Training"])
                members[bad_name] = b"{}"
                make_zip(label_path, members, prefix="")
                config = academic.load_source_config(fixture["config_path"])
                spec = config["collections"]["academic_zz"]["splits"]["Training"]
                spec["label_archive"] = archive_entry(
                    label_path, fixture["data_root"]
                )
                self.convert_expecting_error(
                    fixture,
                    root / "out",
                    "traverses|nested",
                    source_config=config,
                )

    def test_config_requires_both_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            config = yaml.safe_load(
                fixture["config_path"].read_text(encoding="utf-8")
            )
            del config["collections"]["academic_zz"]["splits"]["Validation"]
            broken = root / "broken_config.yaml"
            broken.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                academic.ConversionError, "splits, not collections"
            ):
                academic.load_source_config(broken)

    def test_wrong_eda_decision_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = build_fixture(root)
            config = yaml.safe_load(
                fixture["config_path"].read_text(encoding="utf-8")
            )
            config["eda"]["decision"] = "parser"
            broken = root / "broken_config.yaml"
            broken.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(academic.ConversionError, "alignment_layer"):
                academic.load_source_config(broken)


class VerifierTamperTests(unittest.TestCase):
    def setUp(self):
        self._directory = TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.fixture = build_fixture(self.root)
        self.output = self.root / "out"
        run_convert(self.fixture, self.output)
        self.collection = "academic_zz"
        self.collection_dir = self.output / self.collection

    def tearDown(self):
        self._directory.cleanup()

    def verify_expecting_error(self, pattern):
        with self.assertRaisesRegex(
            (academic.ConversionError, ContractError), pattern
        ):
            academic.verify_collection_outputs(
                self.output, self.collection, separator="\n\n"
            )

    def tamper(self, artifact: str, mutate) -> None:
        path = self.collection_dir / f"{artifact}.jsonl"
        records = load_jsonl(path)
        mutate(records)
        rewrite_jsonl(path, records)

    def test_verify_passes_before_tampering(self):
        summary = academic.verify_collection_outputs(
            self.output, self.collection, separator="\n\n"
        )
        self.assertEqual(summary["status"], "ok")

    def test_non_namespaced_element_id_fails(self):
        def mutate(records):
            records[0]["element_id"] = "ZZ_0001::para_1"

        self.tamper("elements", mutate)
        self.verify_expecting_error("must start")

    def test_duplicate_element_id_fails(self):
        def mutate(records):
            records[1]["element_id"] = records[0]["element_id"]

        self.tamper("elements", mutate)
        self.verify_expecting_error("duplicate element_id")

    def test_element_referencing_absent_document_fails(self):
        def mutate(records):
            for record in records:
                if record["document_id"].endswith("ZZ_0003"):
                    record["document_id"] = "academic_zz::GHOST"
                    record["element_id"] = "academic_zz::GHOST::para_1"
                    break

        self.tamper("elements", mutate)
        self.verify_expecting_error("absent document")

    def test_alignment_referencing_absent_chunk_fails(self):
        def mutate(records):
            records[0]["chunk_memberships"][0]["chunk_id"] = (
                "academic_zz::ZZ_0001::chunk::9999"
            )

        self.tamper("alignment", mutate)
        self.verify_expecting_error("absent chunk")

    def test_inverted_document_range_fails(self):
        def mutate(records):
            record = records[0]
            record["document_range"]["char_start"] = 5
            record["document_range"]["char_end"] = 2

        self.tamper("alignment", mutate)
        self.verify_expecting_error("offsets are invalid")

    def test_numeric_offsets_on_unavailable_location_fail(self):
        def mutate(records):
            records[0]["source_location"] = {
                "status": "unavailable",
                "basis": "unavailable",
                "char_start": 0,
                "char_end": 4,
            }

        self.tamper("elements", mutate)
        self.verify_expecting_error("unavailable offsets must be null")

    def test_verified_range_with_unavailable_basis_fails(self):
        def mutate(records):
            records[0]["document_range"]["basis"] = "unavailable"

        self.tamper("alignment", mutate)
        self.verify_expecting_error("basis")

    def test_dropped_source_tail_fails(self):
        def mutate(records):
            record = records[0]
            record["text"] = record["text"][:-4]
            record["content_sha256"] = academic.sha256_text(record["text"])

        self.tamper("documents", mutate)
        self.verify_expecting_error("document_range|tail|reconstruct")

    def test_lost_separator_fails(self):
        def mutate(records):
            for record in records:
                if "\n\n" in record["text"]:
                    record["text"] = record["text"].replace("\n\n", "\n", 1)
                    record["content_sha256"] = academic.sha256_text(record["text"])
                    break

        self.tamper("documents", mutate)
        self.verify_expecting_error("document_range|dropped|tail|reconstruct")

    def test_element_with_zero_chunks_fails(self):
        def mutate(records):
            records[0]["chunk_memberships"] = []

        self.tamper("alignment", mutate)
        self.verify_expecting_error("non-empty")

    def test_element_in_multiple_chunks_fails(self):
        def mutate(records):
            records[0]["chunk_memberships"] = (
                records[0]["chunk_memberships"] * 2
            )

        self.tamper("alignment", mutate)
        self.verify_expecting_error("multiple chunks")

    def test_missing_alignment_record_fails(self):
        def mutate(records):
            del records[0]

        self.tamper("alignment", mutate)
        self.verify_expecting_error("without any chunk/document alignment")

    def test_mutated_chunk_text_fails(self):
        def mutate(records):
            records[0]["text"] = records[0]["text"] + "변조"

        self.tamper("chunks", mutate)
        self.verify_expecting_error("chunk")

    def test_duplicate_document_content_fails(self):
        def mutate(records):
            twin = dict(records[0])
            twin["document_id"] = "academic_zz::ZZ_9999"
            twin["metadata"] = dict(records[0]["metadata"], json_stem="ZZ_9999")
            records.append(twin)

        self.tamper("documents", mutate)
        self.verify_expecting_error("duplicate document content")


class UnitBehaviourTests(unittest.TestCase):
    def test_pack_elements_into_chunks_is_deterministic_and_atomic(self):
        groups = academic.pack_elements_into_chunks(
            [10, 10, 50], separator_length=2, max_chars=25
        )
        self.assertEqual(groups, [[0, 1], [2]])
        oversized = academic.pack_elements_into_chunks(
            [100, 5], separator_length=2, max_chars=25
        )
        self.assertEqual(oversized, [[0], [1]])
        self.assertEqual(
            academic.pack_elements_into_chunks(
                [10, 10, 50], separator_length=2, max_chars=25
            ),
            groups,
        )

    def test_locate_unique_normalized_flags_ambiguity(self):
        document = "반복 합성 문장.\n\n반복 합성 문장.\n\n고유한 합성 문장."
        with self.assertRaises(academic.AmbiguousAlignmentError):
            academic.locate_unique_normalized(document, "반복 합성 문장.")
        start, end = academic.locate_unique_normalized(document, "고유한 합성 문장.")
        self.assertGreater(end, start)

    def test_normalize_member_name(self):
        self.assertEqual(academic.normalize_member_name("/a.json"), "a.json")
        self.assertEqual(academic.normalize_member_name("a.json"), "a.json")
        for bad in ("//a.json", "/../a.json", "a/b.json", "/C:evil.json", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(academic.ConversionError):
                    academic.normalize_member_name(bad)

    def test_output_dir_guard(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / ".git").mkdir(parents=True)
            code_base = repo / "dr-dci"
            code_base.mkdir()
            with self.assertRaisesRegex(academic.ConversionError, "inside the repository"):
                academic.ensure_output_dir_outside_git(
                    repo / "results", code_base=code_base
                )
            academic.ensure_output_dir_outside_git(
                code_base / "data" / "run", code_base=code_base
            )
            academic.ensure_output_dir_outside_git(
                root / "elsewhere", code_base=code_base
            )

    def test_no_hardcoded_private_data_paths(self):
        for relative in (
            Path("src") / "data" / "collection_academic.py",
            Path("scripts") / "build_academic_collections.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("/Users/", source, relative.name)


if __name__ == "__main__":
    unittest.main()
