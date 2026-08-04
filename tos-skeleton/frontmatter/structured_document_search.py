#!/usr/bin/env python3
"""Artifact-aware document selection orchestration.

This module controls *which* artifact a search arm may consume. Concrete index,
frontmatter, filename, and vector implementations are injected as tools so the
same orchestration can be used in evaluation and production experiments.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence


class Arm(str, Enum):
    A_BASELINE = "A"
    B_INDEX_ONLY = "B"
    C_FRONTMATTER_ONLY = "C"
    D_CASCADE = "D"


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    score: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrontmatterResult:
    hits: tuple[SearchHit, ...]
    has_evidence: bool


@dataclass
class SearchAudit:
    artifact_reads: list[str] = field(default_factory=list)
    index_batches: list[list[str]] = field(default_factory=list)
    frontmatter_candidate_batches: list[list[str]] = field(default_factory=list)
    fallback_reason: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    hits: tuple[SearchHit, ...]
    audit: SearchAudit


class FileSearchTool(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[SearchHit]: ...


class IndexSearchTool(Protocol):
    def rank_candidates(self, query: str) -> Sequence[SearchHit]: ...


class FrontmatterSearchTool(Protocol):
    def search(self, query: str, candidate_ids: Sequence[str] | None,
               top_k: int) -> FrontmatterResult: ...


class ArtifactIsolationError(RuntimeError):
    pass


class DocumentSelector:
    """Run isolated A/B/C/D document-selection arms.

    D reads Index candidates in batches. Frontmatter is restricted to the
    current batch; when it returns no evidence, the next Index batch is tried.
    No gold labels or QA annotations are accepted by this API.
    """

    def __init__(self, arm: Arm, file_search: FileSearchTool,
                 index_search: IndexSearchTool | None = None,
                 frontmatter_search: FrontmatterSearchTool | None = None,
                 candidate_batch_size: int = 20, max_candidate_batches: int = 10,
                 top_k: int = 5):
        if candidate_batch_size < 1 or max_candidate_batches < 1 or top_k < 1:
            raise ValueError("batch sizes and top_k must be positive")
        self.arm = arm
        self.file_search = file_search
        self.index_search = index_search
        self.frontmatter_search = frontmatter_search
        self.candidate_batch_size = candidate_batch_size
        self.max_candidate_batches = max_candidate_batches
        self.top_k = top_k
        self._validate_tools()

    def _validate_tools(self):
        if self.arm == Arm.A_BASELINE and (self.index_search or self.frontmatter_search):
            raise ArtifactIsolationError("A must not load Index or Frontmatter")
        if self.arm == Arm.B_INDEX_ONLY and (not self.index_search or self.frontmatter_search):
            raise ArtifactIsolationError("B requires Index and must not load Frontmatter")
        if self.arm == Arm.C_FRONTMATTER_ONLY and (self.index_search or not self.frontmatter_search):
            raise ArtifactIsolationError("C requires Frontmatter and must not load Index")
        if self.arm == Arm.D_CASCADE and (not self.index_search or not self.frontmatter_search):
            raise ArtifactIsolationError("D requires both Index and Frontmatter")

    def select(self, query: str) -> SelectionResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        audit = SearchAudit()

        if self.arm == Arm.A_BASELINE:
            audit.artifact_reads.append("file")
            return SelectionResult(tuple(self.file_search.search(query, self.top_k)), audit)

        if self.arm == Arm.B_INDEX_ONLY:
            audit.artifact_reads.append("index")
            ranked = tuple(self.index_search.rank_candidates(query))
            return SelectionResult(ranked[:self.top_k], audit)

        if self.arm == Arm.C_FRONTMATTER_ONLY:
            audit.artifact_reads.append("frontmatter")
            result = self.frontmatter_search.search(query, candidate_ids=None, top_k=self.top_k)
            if result.has_evidence:
                return SelectionResult(result.hits[:self.top_k], audit)
            audit.fallback_reason = "frontmatter_no_evidence"
            return SelectionResult((), audit)

        return self._select_cascade(query, audit)

    def _select_cascade(self, query: str, audit: SearchAudit) -> SelectionResult:
        audit.artifact_reads.append("index")
        ranked = tuple(self.index_search.rank_candidates(query))
        seen: set[str] = set()

        for batch_number in range(self.max_candidate_batches):
            start = batch_number * self.candidate_batch_size
            raw_batch = ranked[start:start + self.candidate_batch_size]
            batch = [hit.document_id for hit in raw_batch if hit.document_id not in seen]
            if not batch:
                break
            seen.update(batch)
            audit.index_batches.append(batch)

            audit.artifact_reads.append("frontmatter")
            audit.frontmatter_candidate_batches.append(batch.copy())
            result = self.frontmatter_search.search(query, candidate_ids=batch, top_k=self.top_k)
            if result.has_evidence and result.hits:
                batch_ids = set(batch)
                leaked = [hit.document_id for hit in result.hits if hit.document_id not in batch_ids]
                if leaked:
                    raise ArtifactIsolationError(
                        f"Frontmatter returned documents outside current Index batch: {leaked[:3]}"
                    )
                return SelectionResult(result.hits[:self.top_k], audit)

        audit.fallback_reason = "frontmatter_no_evidence_in_index_batches"
        return SelectionResult((), audit)
