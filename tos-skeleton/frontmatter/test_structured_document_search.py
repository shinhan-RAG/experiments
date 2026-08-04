#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from structured_document_search import (
    Arm, ArtifactIsolationError, DocumentSelector, FrontmatterResult, SearchHit,
)


class FileStub:
    def __init__(self):
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        return [SearchHit("file-1", 1.0)]


class IndexStub:
    def __init__(self, ids=("d1", "d2", "d3", "d4")):
        self.ids = ids
        self.calls = 0

    def rank_candidates(self, query):
        self.calls += 1
        return [SearchHit(doc_id, 1.0 / (rank + 1)) for rank, doc_id in enumerate(self.ids)]


class FrontmatterStub:
    def __init__(self, matching_id=None, leak_id=None):
        self.matching_id = matching_id
        self.leak_id = leak_id
        self.calls = []

    def search(self, query, candidate_ids, top_k):
        self.calls.append(None if candidate_ids is None else list(candidate_ids))
        if self.leak_id:
            return FrontmatterResult((SearchHit(self.leak_id, 1.0, ("헤딩",)),), True)
        if candidate_ids is None and self.matching_id:
            return FrontmatterResult((SearchHit(self.matching_id, 1.0, ("헤딩",)),), True)
        if candidate_ids and self.matching_id in candidate_ids:
            return FrontmatterResult((SearchHit(self.matching_id, 1.0, ("헤딩",)),), True)
        return FrontmatterResult((), False)


class DocumentSelectorTest(unittest.TestCase):
    def test_a_loads_neither_artifact(self):
        files = FileStub()
        result = DocumentSelector(Arm.A_BASELINE, files).select("질문")
        self.assertEqual(["file"], result.audit.artifact_reads)
        self.assertEqual("file-1", result.hits[0].document_id)

    def test_b_never_accepts_frontmatter_tool(self):
        with self.assertRaises(ArtifactIsolationError):
            DocumentSelector(Arm.B_INDEX_ONLY, FileStub(), IndexStub(), FrontmatterStub())

    def test_c_never_accepts_index_tool(self):
        with self.assertRaises(ArtifactIsolationError):
            DocumentSelector(Arm.C_FRONTMATTER_ONLY, FileStub(), IndexStub(), FrontmatterStub("d1"))

    def test_d_retries_next_index_batch(self):
        index = IndexStub()
        frontmatter = FrontmatterStub(matching_id="d3")
        selector = DocumentSelector(
            Arm.D_CASCADE, FileStub(), index, frontmatter,
            candidate_batch_size=2, max_candidate_batches=2,
        )
        result = selector.select("두 번째 배치에서 찾는 질문")
        self.assertEqual([["d1", "d2"], ["d3", "d4"]], frontmatter.calls)
        self.assertEqual("d3", result.hits[0].document_id)
        self.assertIsNone(result.audit.fallback_reason)

    def test_d_rejects_frontmatter_result_outside_current_batch(self):
        selector = DocumentSelector(
            Arm.D_CASCADE, FileStub(), IndexStub(), FrontmatterStub(leak_id="outside"),
            candidate_batch_size=2,
        )
        with self.assertRaises(ArtifactIsolationError):
            selector.select("질문")

    def test_d_returns_not_found_after_all_batches(self):
        frontmatter = FrontmatterStub()
        selector = DocumentSelector(
            Arm.D_CASCADE, FileStub(), IndexStub(), frontmatter,
            candidate_batch_size=2, max_candidate_batches=2,
        )
        result = selector.select("없는 내용")
        self.assertEqual([["d1", "d2"], ["d3", "d4"]], frontmatter.calls)
        self.assertFalse(result.hits)
        self.assertEqual("frontmatter_no_evidence_in_index_batches", result.audit.fallback_reason)


if __name__ == "__main__":
    unittest.main()
