import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from src.eval.collection_contract import ContractError, sha256_file, validate_bundle


PROMPT_SHA = hashlib.sha256(b"frozen prompt").hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return {
        "path": str(path.relative_to(path.parents[2])),
        "sha256": sha256_file(path),
        "records": len(records),
    }


def collection_records(collection_id: str, *, content: str = "보험계약의 보장 내용") -> dict:
    document_id = f"{collection_id}::doc-1"
    chunk_id = f"{collection_id}::chunk-1"
    element_id = f"{collection_id}::element-1"
    unavailable = {
        "status": "unavailable",
        "basis": "unavailable",
        "char_start": None,
        "char_end": None,
    }
    return {
        "documents": [
            {
                "schema_version": "shinhan.collection-document.v1",
                "collection_id": collection_id,
                "document_id": document_id,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "text": content,
            }
        ],
        "chunks": [
            {
                "schema_version": "shinhan.collection-chunk.v1",
                "collection_id": collection_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": content,
                "source_location": unavailable,
            }
        ],
        "elements": [
            {
                "schema_version": "shinhan.collection-element.v1",
                "collection_id": collection_id,
                "element_id": element_id,
                "document_id": document_id,
                "element_type": "paragraph",
                "ordinal": 0,
                "text": content,
                "source_location": unavailable,
            }
        ],
        "semantic_tags": [
            {
                "schema_version": "shinhan.collection-semantic-tag.v1",
                "collection_id": collection_id,
                "tag_id": f"{collection_id}::tag-1",
                "document_id": document_id,
                "element_id": element_id,
                "tags": ["보험계약", "보장"],
                "generation": {
                    "generator_id": "fixture-generator",
                    "revision": "fixture-revision-1",
                    "prompt_sha256": PROMPT_SHA,
                    "input_fields": ["element_text", "element_type"],
                    "qa_access": False,
                    "qrel_access": False,
                    "gold_access": False,
                },
            }
        ],
        "qa": [
            {
                "schema_version": "shinhan.collection-qa.v1",
                "collection_id": collection_id,
                "qa_id": f"{collection_id}::qa-1",
                "query": "보험계약의 보장 내용은 무엇인가?",
                "split": "test",
                "gold_document_ids": [document_id],
                "gold_element_ids": [element_id],
                "gold_chunk_ids": [chunk_id],
                "evidence": [
                    {
                        "document_id": document_id,
                        "element_id": element_id,
                        "chunk_ids": [chunk_id],
                        "quote": content,
                        "source_location": unavailable,
                        "alignment": {
                            "status": "verified",
                            "method": "exact_text",
                        },
                    }
                ],
                "generation": {
                    "generator_id": "fixture-qa-generator",
                    "revision": "fixture-revision-1",
                    "prompt_sha256": PROMPT_SHA,
                    "anchor_element_id": element_id,
                },
            }
        ],
    }


def manifest(collections: list[dict]) -> dict:
    return {
        "schema_version": "shinhan.collection-eval-manifest.v1",
        "experiment_id": "fixture-evaluation",
        "collections": collections,
        "semantic_tag_generation": {
            "allowed_input_fields": [
                "element_text",
                "element_type",
                "document_title",
                "document_metadata",
            ],
            "qa_access": False,
            "qrel_access": False,
            "gold_access": False,
        },
        "cross_collection_duplicates": {"policy": "fail_on_duplicate"},
        "evaluation": {
            "scopes": ["per_collection", "all_collections"],
            "retrieval_modes": ["chunk", "semantic_tag", "combined"],
            "shared_qa_across_modes": True,
            "all_collections_qa": "union_of_per_collection_qa",
            "candidate_budget": {
                "chunk_pool": 100,
                "semantic_tag_pool": 100,
                "combined_chunk_pool": 50,
                "combined_semantic_tag_pool": 50,
                "final_top_k": 20,
            },
            "fusion": {"method": "rrf", "rrf_k": 60},
            "models": {
                "embedding": {
                    "model_id": "fixture/embedding",
                    "revision": "0123456789abcdef",
                },
                "reranker": {
                    "model_id": "fixture/reranker",
                    "revision": "fedcba9876543210",
                },
            },
            "metrics": [
                "evidence_ndcg@10",
                "evidence_recall@20",
                "parent_hit@10",
                "evidence_coverage@20",
                "latency_p95_ms",
            ],
            "report_per_collection": True,
            "report_macro_average": True,
            "report_micro_average": True,
        },
    }


class CollectionContractTests(unittest.TestCase):
    def build_bundle(
        self,
        root: Path,
        *,
        collection_ids: tuple[str, ...] = ("collection-a",),
        duplicate_content: bool = False,
    ) -> tuple[Path, dict, dict[str, dict]]:
        collection_specs = []
        records_by_collection = {}
        for index, collection_id in enumerate(collection_ids):
            content = (
                "보험계약의 보장 내용"
                if duplicate_content or index == 0
                else "보험금 지급 제외 조건"
            )
            records = collection_records(collection_id, content=content)
            records_by_collection[collection_id] = records
            artifacts = {}
            for artifact_name, rows in records.items():
                artifact_path = root / "artifacts" / collection_id / f"{artifact_name}.jsonl"
                artifacts[artifact_name] = write_jsonl(artifact_path, rows)
            collection_specs.append(
                {
                    "collection_id": collection_id,
                    "source_revision": f"{collection_id}-revision-1",
                    "artifacts": artifacts,
                }
            )
        value = manifest(collection_specs)
        manifest_path = root / "collection_eval_manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return manifest_path, value, records_by_collection

    def rewrite_artifact(
        self,
        root: Path,
        value: dict,
        collection_index: int,
        artifact_name: str,
        records: list[dict],
    ) -> Path:
        spec = value["collections"][collection_index]["artifacts"][artifact_name]
        path = root / spec["path"]
        replacement = write_jsonl(path, records)
        value["collections"][collection_index]["artifacts"][artifact_name] = replacement
        manifest_path = root / "collection_eval_manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return manifest_path

    def test_valid_bundle_accepts_explicitly_unavailable_positions(self):
        with TemporaryDirectory() as directory:
            path, _, _ = self.build_bundle(Path(directory))
            result = validate_bundle(path)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["collections"]["collection-a"]["qa"], 1)

    def test_same_qa_and_all_three_retrieval_modes_are_mandatory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, value, _ = self.build_bundle(root)
            value["evaluation"]["retrieval_modes"] = ["chunk", "combined"]
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "retrieval_modes"):
                validate_bundle(path)

    def test_combined_pool_must_equal_single_source_pool(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, value, _ = self.build_bundle(root)
            value["evaluation"]["candidate_budget"]["combined_chunk_pool"] = 40
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "combined source pools"):
                validate_bundle(path)

    def test_qa_without_chunk_gold_fails_loud(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, value, records = self.build_bundle(root)
            qa = copy.deepcopy(records["collection-a"]["qa"])
            qa[0]["gold_chunk_ids"] = []
            path = self.rewrite_artifact(root, value, 0, "qa", qa)
            with self.assertRaisesRegex(ContractError, "gold_chunk_ids"):
                validate_bundle(path)

    def test_semantic_tag_cannot_access_query_or_gold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, value, records = self.build_bundle(root)
            tags = copy.deepcopy(records["collection-a"]["semantic_tags"])
            tags[0]["generation"]["input_fields"] = ["element_text", "query"]
            path = self.rewrite_artifact(root, value, 0, "semantic_tags", tags)
            with self.assertRaisesRegex(ContractError, "not allowed"):
                validate_bundle(path)

    def test_ids_must_be_namespaced_by_collection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, value, records = self.build_bundle(root)
            chunks = copy.deepcopy(records["collection-a"]["chunks"])
            chunks[0]["chunk_id"] = "chunk-1"
            path = self.rewrite_artifact(root, value, 0, "chunks", chunks)
            with self.assertRaisesRegex(ContractError, "must start"):
                validate_bundle(path)

    def test_artifact_hash_mismatch_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = self.build_bundle(root)
            chunk_path = root / "artifacts" / "collection-a" / "chunks.jsonl"
            chunk_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "sha256 mismatch"):
                validate_bundle(path)

    def test_document_content_hash_is_recomputed_from_text(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, value, records = self.build_bundle(root)
            documents = copy.deepcopy(records["collection-a"]["documents"])
            documents[0]["content_sha256"] = hashlib.sha256(b"different").hexdigest()
            path = self.rewrite_artifact(root, value, 0, "documents", documents)
            with self.assertRaisesRegex(ContractError, "does not match text"):
                validate_bundle(path)

    def test_v1_does_not_accept_unverified_duplicate_resolution_policy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path, value, _ = self.build_bundle(root)
            value["cross_collection_duplicates"]["policy"] = "mark_all_relevant"
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "v1 requires"):
                validate_bundle(path)

    def test_cross_collection_duplicate_content_fails_by_default(self):
        with TemporaryDirectory() as directory:
            path, _, _ = self.build_bundle(
                Path(directory),
                collection_ids=("collection-a", "collection-b"),
                duplicate_content=True,
            )
            with self.assertRaisesRegex(ContractError, "duplicate document content"):
                validate_bundle(path)


if __name__ == "__main__":
    unittest.main()
