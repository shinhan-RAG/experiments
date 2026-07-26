"""S0 parsed-chunk retrieval and character-evidence diagnostics.

``evaluate_ranked_source_chunks`` applies only to a real unique-ID ranked list
with one known source chunk.  Agent workspaces are pull-order accumulations,
so ``evaluate_workspace_evidence`` reports evidence only and never nDCG.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import math
import re
import unicodedata


MIN_PARTIAL_MATCH_CHARS = 8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _norm(value: str) -> str:
    """NFC, casefold, and one space per whitespace run."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value or "").casefold()).strip()


def _ranked_chunk_ids(ranked_chunks: list[dict]) -> list[str]:
    ids = [str(chunk["chunk_id"]) for chunk in ranked_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("ranked list must have unique chunk IDs for rank metrics")
    return ids


def _source_id(value: str) -> str:
    source_id = str(value or "")
    if not source_id:
        raise ValueError("known_positive_source_chunk_id is required")
    return source_id


def source_chunk_recall_at_k(ranked_ids: list[str], source_id: str, k: int) -> float:
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked list must have unique chunk IDs for rank metrics")
    return 1.0 if _source_id(source_id) in ranked_ids[:k] else 0.0


def source_chunk_ndcg_at_k(ranked_ids: list[str], source_id: str, k: int) -> float:
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked list must have unique chunk IDs for rank metrics")
    source_id = _source_id(source_id)
    value = sum(
        1.0 / math.log2(rank + 2)
        for rank, chunk_id in enumerate(ranked_ids[:k])
        if chunk_id == source_id
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"source_chunk_ndcg@{k} outside [0, 1]: {value}")
    return value


def _canonical_evidence_texts(supporting_spans: list[dict] | None) -> list[str]:
    """Deduplicate normalized spans and collapse overlap without double counting."""
    raw = []
    for span in supporting_spans or []:
        text = _norm(str(span.get("text", "")))
        if not text:
            continue
        start, end = span.get("char_start"), span.get("char_end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
            raw.append((text, start, end))
        else:
            raw.append((text, None, None))

    unique: dict[str, tuple[int | None, int | None]] = {}
    for text, start, end in raw:
        candidate = (start, end)
        existing = unique.get(text)
        if existing is None or (candidate[0] is not None and (existing[0] is None or candidate < existing)):
            unique[text] = candidate
    offset_spans = sorted(
        [(text, start, end) for text, (start, end) in unique.items() if start is not None],
        key=lambda item: (item[1], item[2], item[0]),
    )
    selected = []
    index = 0
    while index < len(offset_spans):
        group = [offset_spans[index]]
        group_end = offset_spans[index][2]
        index += 1
        while index < len(offset_spans) and offset_spans[index][1] < group_end:
            group.append(offset_spans[index])
            group_end = max(group_end, offset_spans[index][2])
            index += 1
        selected.append(max((item[0] for item in group), key=lambda value: (len(value), value)))
    text_only = [text for text, (start, _) in unique.items() if start is None]
    for text in sorted(text_only, key=lambda value: (-len(value), value)):
        if not any(text in kept for kept in selected):
            selected.append(text)
    return sorted(set(selected), key=lambda value: (-len(value), value))


def _partial_match_chars(evidence_text: str, retrieved_text: str) -> int:
    if evidence_text in retrieved_text:
        return len(evidence_text)
    size = SequenceMatcher(None, evidence_text, retrieved_text, autojunk=False).find_longest_match().size
    return size if size >= MIN_PARTIAL_MATCH_CHARS else 0


def evidence_scores(ranked_chunks: list[dict], supporting_spans: list[dict] | None, k: int) -> dict | None:
    """Evidence coverage/density under a fixed normalized-character contract."""
    _ranked_chunk_ids(ranked_chunks)
    evidence_texts = _canonical_evidence_texts(supporting_spans)
    if not evidence_texts:
        return None
    retrieved_texts = [_norm(str(chunk.get("text", ""))) for chunk in ranked_chunks[:k]]
    evidence_total = sum(len(text) for text in evidence_texts)
    matched = sum(
        max((_partial_match_chars(evidence, text) for text in retrieved_texts), default=0)
        for evidence in evidence_texts
    )
    retrieved_total = sum(len(text) for text in retrieved_texts)
    coverage = matched / evidence_total if evidence_total else 0.0
    density = min(matched / retrieved_total, 1.0) if retrieved_total else 0.0
    hmean = 2.0 * coverage * density / (coverage + density) if coverage + density else 0.0
    return {
        "evidence_coverage": round(coverage, 4),
        "evidence_density": round(density, 4),
        "evidence_hmean": round(hmean, 4),
        "evidence_total_char_count": evidence_total,
        "evidence_matched_char_count": matched,
        "retrieved_char_count": retrieved_total,
    }


def evaluate_ranked_source_chunks(
    ranked_chunks: list[dict],
    known_positive_source_chunk_id: str,
    supporting_spans: list[dict] | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    ranked_ids = _ranked_chunk_ids(ranked_chunks)
    source_id = _source_id(known_positive_source_chunk_id)
    metrics = {}
    for k in ks:
        metrics[f"source_chunk_recall@{k}"] = source_chunk_recall_at_k(ranked_ids, source_id, k)
        metrics[f"source_chunk_ndcg@{k}"] = source_chunk_ndcg_at_k(ranked_ids, source_id, k)
        evidence = evidence_scores(ranked_chunks, supporting_spans, k)
        if evidence is not None:
            metrics.update({f"{name}@{k}": value for name, value in evidence.items()})
    return metrics


def evaluate_workspace_evidence(
    workspace_chunks: list[dict],
    supporting_spans: list[dict] | None,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Evidence-only diagnostics for an Agent workspace; no ranked nDCG exists."""
    _ranked_chunk_ids(workspace_chunks)
    metrics = {}
    for k in ks:
        evidence = evidence_scores(workspace_chunks, supporting_spans, k)
        if evidence is not None:
            metrics.update({f"{name}@{k}": value for name, value in evidence.items()})
    return metrics


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def build_s0_result_manifest(*, input_corpus_manifest_sha256: str, output_corpus_manifest_sha256: str, query_qrel_manifest_sha256: str) -> dict:
    return {
        "evaluation_track": "s0_exploratory_chunk_smoke",
        "retrieval_unit": "parsed_markdown_text_chunk",
        "qrel_contract": "single_known_source_chunk_incomplete",
        "primary_metric": "source_chunk_recall@k",
        "evidence_metrics": ["evidence_coverage@k", "evidence_density@k", "evidence_hmean@k"],
        "confirmatory_gate_status": "not_advanced",
        "advances_hypotheses": {"H1": False, "H2a": False, "H2b": False, "H3": False, "H4": False},
        "input_corpus_manifest_sha256": _require_sha256("input_corpus_manifest_sha256", input_corpus_manifest_sha256),
        "output_corpus_manifest_sha256": _require_sha256("output_corpus_manifest_sha256", output_corpus_manifest_sha256),
        "query_qrel_manifest_sha256": _require_sha256("query_qrel_manifest_sha256", query_qrel_manifest_sha256),
    }


def aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    keys = set().union(*(row.keys() for row in per_query))
    return {
        key: round(sum(row[key] for row in per_query if key in row) / sum(1 for row in per_query if key in row), 4)
        for key in sorted(keys)
    }
