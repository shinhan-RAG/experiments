"""Synthetic RED→GREEN contracts for the non-confirmatory S0 track."""

import json

import pytest

from scripts import build_shinhan_corpus as chunk_builder
from src.eval import span_metrics
from src.eval.retrieval_metrics import ndcg_at_k


def test_source_chunk_metric_names_do_not_claim_complete_relevance():
    metrics = span_metrics.evaluate_ranked_source_chunks(
        [{"chunk_id": "other", "text": "x"}, {"chunk_id": "source", "text": "evidence"}],
        "source", ks=(1, 2),
    )
    assert metrics["source_chunk_recall@1"] == 0.0
    assert metrics["source_chunk_recall@2"] == 1.0
    assert "recall@1" not in metrics


def test_duplicate_ranked_ids_fail_loud_before_ndcg_can_exceed_one():
    with pytest.raises(ValueError, match="unique"):
        span_metrics.evaluate_ranked_source_chunks(
            [{"chunk_id": "A", "text": "e"}, {"chunk_id": "A", "text": "e"}], "A", ks=(2,)
        )
    with pytest.raises(ValueError, match="unique"):
        ndcg_at_k(["A", "A"], {"A": 1.0}, 2)


def test_source_chunk_ndcg_has_the_unit_interval_guard():
    value = span_metrics.source_chunk_ndcg_at_k(["A"], "A", 1)
    assert value == 1.0
    assert 0.0 <= value <= 1.0


def test_workspace_emits_evidence_only_not_ndcg():
    metrics = span_metrics.evaluate_workspace_evidence(
        [{"chunk_id": "A", "text": "Evidence text"}], [{"text": "evidence text"}], ks=(1,)
    )
    assert metrics["evidence_coverage@1"] == 1.0
    assert "source_chunk_ndcg@1" not in metrics
    assert "ndcg@1" not in metrics


def test_evidence_nfc_overlap_and_rehit_are_counted_once():
    metrics = span_metrics.evidence_scores(
        [
            {"chunk_id": "one", "text": "Alpha beta gamma"},
            {"chunk_id": "two", "text": "ALPHA  BETA gamma"},
        ],
        [
            {"text": "Alpha beta gamma", "char_start": 0, "char_end": 16},
            {"text": "beta gamma", "char_start": 6, "char_end": 16},
        ],
        2,
    )
    assert metrics["evidence_total_char_count"] == len("alpha beta gamma")
    assert metrics["evidence_matched_char_count"] == len("alpha beta gamma")
    assert metrics["evidence_coverage"] == 1.0
    assert "evidence_hmean" in metrics


def test_s0_manifest_is_not_allowed_to_advance_any_hypothesis():
    manifest = span_metrics.build_s0_result_manifest(
        input_corpus_manifest_sha256="a" * 64,
        output_corpus_manifest_sha256="b" * 64,
        query_qrel_manifest_sha256="c" * 64,
    )
    assert manifest["evaluation_track"] == "s0_exploratory_chunk_smoke"
    assert manifest["confirmatory_gate_status"] == "not_advanced"
    assert manifest["advances_hypotheses"] == {"H1": False, "H2a": False, "H2b": False, "H3": False, "H4": False}


def test_chunk_builder_records_short_fragments_and_manifest_hashes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "short.md").write_text("짧다", encoding="utf-8")
    (source / "long.md").write_text("본문 " * 400, encoding="utf-8")
    stats = chunk_builder.build_corpus(source, output)
    saved = json.loads((output / "corpus_stats.json").read_text(encoding="utf-8"))
    assert stats == saved
    assert saved["dropped_fragment_count"] >= 1
    assert saved["zero_chunk_document_count"] >= 1
    assert len(saved["input_corpus_manifest_sha256"]) == 64
    assert len(saved["output_corpus_manifest_sha256"]) == 64


def test_chunk_builder_splits_text_but_preserves_oversized_markdown_tables():
    stats = {}
    long_text = ("가나다라마바사 " * 400).strip()
    assert all(len(chunk) <= chunk_builder.CHUNK_CHARS for chunk in chunk_builder.pack_chunks(long_text, stats=stats))
    table = "\n".join(["| a | b |"] * 300)
    assert chunk_builder.pack_chunks(table, stats=stats) == [table]
    assert stats["oversized_table_chunk_count"] == 1
