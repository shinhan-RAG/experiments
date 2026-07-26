"""No-cost tests for AIHub Gate L0 and legal/longdoc execution boundaries."""

import hashlib
import json

import pytest

import run_experiment
from scripts.clean_corpus_headers import normalize_corpus
from src.eval.legal_execution_gate import (
    EXPECTED_L0_FINDINGS,
    LegalExecutionBlocked,
    require_legal_execution_gate,
    require_longdoc_diagnostic_contract,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _approval(**extra):
    return {"status": "approved", "approved_by": "reviewer", "approved_at": "2026-07-26T00:00:00+00:00", "basis": "synthetic test", **extra}


def _approved_gate(tmp_path, *, with_tags=False):
    inputs = {}
    for name in ("corpus", "subset", "final_query", "exclusion_list"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        inputs[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    l0_ref = _write_json(tmp_path / "l0.json", _approval(
        referential_integrity="passed", license_internal_use="approved",
        findings=EXPECTED_L0_FINDINGS, immutable_inputs={name: item["sha256"] for name, item in inputs.items()},
    ))
    metric_ref = _write_json(tmp_path / "metric.json", _approval(retrieval_unit="parent_document"))
    execution_ref = _write_json(tmp_path / "execution.json", _approval(gate_l0_manifest_sha256=l0_ref["sha256"], part="part1"))
    gate = {"status": "approved", "gate_l0_approval_manifest": l0_ref, "immutable_inputs": inputs, "metric_effect_approval": metric_ref, "execution_approval": execution_ref}
    if with_tags:
        gate["tag_execution_contract"] = _write_json(tmp_path / "tags.json", {
            "element_universe": "parser_derived", "structural_type_predicate": "exact_typed_predicate",
            "semantic_role_predicate": "exact_typed_predicate", "failure_policy": "fail_loud",
        })
    return {"parts": {"part1_stacking": {"dataset": "aihub-full"}, "part2_scaling": {"dataset": "aihub-full"}, "part4_generalization": {"datasets": []}}, "legal_execution_gate": gate}


def test_missing_l0_blocks_before_query_loading_or_endpoint_construction(monkeypatch):
    config = run_experiment.load_config("config/experiment_legal.yaml")
    monkeypatch.setattr(run_experiment, "load_queries", lambda *_: pytest.fail("query load must not occur"))
    monkeypatch.setattr(run_experiment, "PullRetriever", lambda *_: pytest.fail("endpoint client must not occur"))
    with pytest.raises(LegalExecutionBlocked):
        run_experiment.run_part1(config)


def test_tag_treatment_is_not_enabled_without_parser_derived_exact_contract(tmp_path):
    config = _approved_gate(tmp_path)
    with pytest.raises(LegalExecutionBlocked, match="tag execution contract"):
        require_legal_execution_gate(config, artifact_root=tmp_path, part="part1", steps=[{"tags": "A"}])
    config = _approved_gate(tmp_path, with_tags=True)
    require_legal_execution_gate(config, artifact_root=tmp_path, part="part1", steps=[{"tags": "A"}])


def test_legal_config_is_not_an_automatic_confirmatory_promotion():
    config = run_experiment.load_config("config/experiment_legal.yaml")
    assert config["legal_execution_gate"]["status"] == "pending"
    assert config["parts"]["part4_generalization"]["datasets"][-1]["confirmatory_gate_status"] == "not_advanced"


def test_longdoc_is_rejected_unless_explicitly_diagnostic_with_provenance():
    with pytest.raises(LegalExecutionBlocked):
        require_longdoc_diagnostic_contract({"name": "longdoc", "evaluation_role": "confirmatory"})
    require_longdoc_diagnostic_contract({
        "name": "longdoc", "evaluation_role": "diagnostic_only", "confirmatory_gate_status": "not_advanced",
        "source_revision_sha256": "a" * 64, "lexical_overlap_stratum": "low_overlap",
    })


def test_header_cleaning_never_overwrites_raw_input(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"_id": "x", "text": "[판시사항]\n【판시사항】\n본문"}, ensure_ascii=False) + "\n", encoding="utf-8")
    before = raw.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        normalize_corpus(raw, raw)
    normalized = tmp_path / "normalized.jsonl"
    report = normalize_corpus(raw, normalized)
    assert raw.read_bytes() == before
    assert report["changed_rows"] == 1
    assert "【판시사항】" not in normalized.read_text(encoding="utf-8")


def test_pr3_parent_pr4_s0_and_pr6_legal_config_stay_separate(monkeypatch):
    """The integration must not confuse parent qrels with S0 chunks or legal approval."""
    from src.eval import span_metrics

    parent_map = run_experiment.parent_map_from_corpus([
        {"_id": "parent:A#0", "parent_id": "parent:A"},
        {"_id": "parent:A#1", "parent_id": "parent:A"},
    ])
    assert run_experiment.to_parent_ids(["parent:A#1", "parent:A#0"], parent_map) == ["parent:A"]
    s0 = span_metrics.evaluate_ranked_source_chunks(
        [{"chunk_id": "parent:A#1", "text": "evidence"}], "parent:A#1", ks=(1,)
    )
    assert s0["source_chunk_recall@1"] == 1.0
    assert "ndcg@1" not in span_metrics.evaluate_workspace_evidence(
        [{"chunk_id": "parent:A#1", "text": "evidence"}], [{"text": "evidence"}], ks=(1,)
    )
    config = run_experiment.load_config("config/experiment_legal.yaml")
    monkeypatch.setattr(run_experiment, "load_queries", lambda *_: pytest.fail("L0 must block first"))
    with pytest.raises(LegalExecutionBlocked):
        run_experiment.run_part1(config)
