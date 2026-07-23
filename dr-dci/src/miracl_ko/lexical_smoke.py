"""Standalone lexical plumbing contract for MIRACL Korean passage retrieval.

This module deliberately stays outside the focused Part 1/2 runner.  It
validates a single pinned Anserini/Lucene CJK backend and computes only
passage-level retrieval plumbing metrics from already-produced rankings.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from src.eval.comparison import paired_bootstrap_delta


LEXICAL_SMOKE_CONFIG_SCHEMA = "dr-dci.miracl-ko-lexical-smoke.v1"
LEXICAL_SMOKE_RESULT_SCHEMA = "dr-dci.miracl-ko-lexical-smoke-result.v1"
RETRIEVAL_UNIT = "passage"
EXPECTED_BACKEND = {
    "name": "anserini_lucene_cjk",
    "package": "anserini",
    "version": "2.1.1",
    "distribution_package": "pyserini",
    "distribution_version": "2.1.0",
    "analyzer_language": "ko",
    "analyzer_class": "org.apache.lucene.analysis.cjk.CJKAnalyzer",
    "distribution_source_archive_sha256": "384fb783c52ac1605caabe8a75f520323dfed2b5595072911c87f6cfca8bf15f",
    "jar_relative_path": "pyserini/resources/jars/anserini-2.1.1-fatjar.jar",
    "jar_path": "/opt/anserini/anserini-2.1.1-fatjar.jar",
    "jar_sha256": "3c83883246d0fb2326c8a9291572b969467cf478d1fc65f517cbf37fd9b0d914",
}
EXPECTED_RUNTIME_EXECUTION_MODE = "anserini_java_cli_via_pyserini_distribution"
SMOKE_METRIC_KEYS = (
    "passage_ndcg_at_10",
    "passage_recall_at_5",
    "passage_recall_at_20",
    "passage_recall_at_100",
    "passage_hit_at_5",
    "passage_hit_at_10",
    "passage_precision_at_20",
    "passage_mrr",
    "query_latency_seconds",
)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")


def validate_lexical_smoke_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("MIRACL lexical smoke config must be an object")
    for key, expected in {
        "schema_version": LEXICAL_SMOKE_CONFIG_SCHEMA,
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
        "scope": "standalone_lexical_plumbing_smoke_only",
    }.items():
        if config.get(key) != expected:
            if key == "retrieval_unit":
                raise ValueError("MIRACL lexical smoke must use passage retrieval_unit")
            raise ValueError(f"MIRACL lexical smoke config has invalid {key}")
    backend = config.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("MIRACL lexical smoke config is missing backend")
    for key, expected in EXPECTED_BACKEND.items():
        if backend.get(key) != expected:
            raise ValueError("MIRACL lexical smoke requires pinned Anserini Lucene CJK backend")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("MIRACL lexical smoke config is missing runtime")
    for key in ("container_base_image", "container_base_image_sha256", "java_runtime_version"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"MIRACL lexical smoke config is missing runtime {key}")
    _require_sha256(runtime["container_base_image_sha256"], label="MIRACL lexical smoke base image sha256")
    if runtime.get("execution_mode") != EXPECTED_RUNTIME_EXECUTION_MODE:
        raise ValueError("MIRACL lexical smoke must use the declared Anserini Java CLI runtime")
    if runtime.get("supporting_runtime_packages") != {"numpy": "2.4.2"}:
        raise ValueError("MIRACL lexical smoke must pin NumPy for paired bootstrap")
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("MIRACL lexical smoke config is missing retrieval controls")
    if retrieval.get("top_k") != 100:
        raise ValueError("MIRACL lexical smoke top_k must be fixed at 100")
    for key in ("bm25_k1", "bm25_b"):
        if not isinstance(retrieval.get(key), (float, int)):
            raise ValueError(f"MIRACL lexical smoke retrieval is missing {key}")
    for key, expected in {
        "index_threads": 1,
        "index_memory_buffer_mb": 256,
        "search_threads": 1,
    }.items():
        if retrieval.get(key) != expected:
            raise ValueError(f"MIRACL lexical smoke retrieval must fix {key} at {expected}")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("split") != "dev":
        raise ValueError("MIRACL lexical smoke must score the fixed dev split")
    if evaluation.get("bootstrap_iterations") != 10_000:
        raise ValueError("MIRACL lexical smoke bootstrap iterations must be fixed at 10,000")
    if not isinstance(evaluation.get("bootstrap_seed"), int):
        raise ValueError("MIRACL lexical smoke requires bootstrap_seed")
    forbidden = {"taxonomy", "agent", "llm", "embedding", "dense", "qrel_input_to_retriever"}
    if forbidden & set(config):
        raise ValueError("MIRACL lexical smoke config contains out-of-scope treatment controls")


def _positive_ids(qrels: dict[str, int | float]) -> set[str]:
    return {str(corpus_id) for corpus_id, relevance in qrels.items() if float(relevance) > 0}


def _ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> float:
    def gain(relevance: int | float) -> float:
        return (2.0 ** float(relevance)) - 1.0

    observed = sum(
        gain(qrels.get(corpus_id, 0)) / math.log2(rank + 1)
        for rank, corpus_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal_gains = sorted((gain(relevance) for relevance in qrels.values()), reverse=True)[:k]
    ideal = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal_gains, start=1))
    return observed / ideal if ideal else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def evaluate_passage_rankings(
    queries: Iterable[dict[str, Any]],
    qrels_by_qid: dict[str, dict[str, int | float]],
    rankings_by_qid: dict[str, list[tuple[str, float]]],
    *,
    corpus_ids: set[str],
    latencies_by_qid: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score already-retrieved passage rankings; qrels never enter retrieval."""
    rows: list[dict[str, Any]] = []
    latencies_by_qid = latencies_by_qid or {}
    seen_qids: set[str] = set()
    for query in queries:
        qid = str(query["qid"])
        if qid in seen_qids:
            raise ValueError(f"duplicate query ID in smoke scoring input: {qid}")
        seen_qids.add(qid)
        if qid not in rankings_by_qid:
            raise ValueError(f"missing ranking for query {qid}")
        if qid not in qrels_by_qid:
            raise ValueError(f"missing qrels for query {qid}")
        ranking = rankings_by_qid[qid]
        ranked_ids = [str(corpus_id) for corpus_id, _score in ranking]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"duplicate passage ID in ranking for query {qid}")
        orphan_ids = sorted(set(ranked_ids) - corpus_ids)
        if orphan_ids:
            raise ValueError(f"orphan retrieval passage ID for query {qid}: {orphan_ids[0]}")
        positives = _positive_ids(qrels_by_qid[qid])
        if not positives:
            raise ValueError(f"query {qid} has no positive qrels for passage scoring")

        def recall_at(k: int) -> float:
            return len(set(ranked_ids[:k]) & positives) / len(positives)

        def hit_at(k: int) -> float:
            return 1.0 if set(ranked_ids[:k]) & positives else 0.0

        first_positive_rank = next(
            (rank for rank, corpus_id in enumerate(ranked_ids, start=1) if corpus_id in positives), None
        )
        rows.append({
            "query_id": qid,
            "retrieval_unit": RETRIEVAL_UNIT,
            "ranked_passage_ids": ranked_ids,
            "scores": [_round(float(score)) for _corpus_id, score in ranking],
            "retrieved_passage_count": len(ranked_ids),
            "positive_passage_count": len(positives),
            "passage_ndcg_at_10": _round(_ndcg_at_k(ranked_ids, qrels_by_qid[qid], 10)),
            "passage_recall_at_5": _round(recall_at(5)),
            "passage_recall_at_20": _round(recall_at(20)),
            "passage_recall_at_100": _round(recall_at(100)),
            "passage_hit_at_5": _round(hit_at(5)),
            "passage_hit_at_10": _round(hit_at(10)),
            "passage_precision_at_20": _round(len(set(ranked_ids[:20]) & positives) / 20),
            "passage_mrr": _round(1.0 / first_positive_rank if first_positive_rank else 0.0),
            "query_latency_seconds": _round(float(latencies_by_qid.get(qid, 0.0))),
        })
    unexpected = sorted(set(rankings_by_qid) - seen_qids)
    if unexpected:
        raise ValueError(f"ranking input contains unknown query ID: {unexpected[0]}")
    if not rows:
        raise ValueError("MIRACL lexical smoke requires at least one query")
    aggregate = {
        "aggregation": "macro_mean_over_dev_queries",
        "query_count": len(rows),
        "precision_at_20_denominator": 20,
        **{
            key: _round(sum(float(row[key]) for row in rows) / len(rows))
            for key in SMOKE_METRIC_KEYS
        },
    }
    return rows, aggregate


def parse_anserini_trec_run(path: Path, *, query_ids: set[str]) -> dict[str, list[tuple[str, float]]]:
    """Parse one Anserini TREC run without silently accepting foreign IDs."""
    rankings: dict[str, list[tuple[str, float]]] = {qid: [] for qid in query_ids}
    previous_rank: dict[str, int] = {}
    seen_ids: dict[str, set[str]] = {qid: set() for qid in query_ids}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"invalid Anserini TREC line at {path}:{line_number}")
            qid, _q0, corpus_id, rank_text, score_text, _run_tag = fields
            if qid not in rankings:
                raise ValueError(f"Anserini run contains unknown query ID: {qid}")
            try:
                rank = int(rank_text)
                score = float(score_text)
            except ValueError as error:
                raise ValueError(f"invalid Anserini rank or score at {path}:{line_number}") from error
            if rank < 1 or not math.isfinite(score):
                raise ValueError(f"invalid Anserini rank or score at {path}:{line_number}")
            if qid in previous_rank and rank <= previous_rank[qid]:
                raise ValueError(f"Anserini ranks are not strictly increasing for query {qid}")
            if corpus_id in seen_ids[qid]:
                raise ValueError(f"Anserini run has duplicate passage ID for query {qid}: {corpus_id}")
            previous_rank[qid] = rank
            seen_ids[qid].add(corpus_id)
            rankings[qid].append((corpus_id, score))
    return rankings


def validate_standalone_smoke_result(result: dict[str, Any], *, config: dict[str, Any]) -> None:
    validate_lexical_smoke_config(config)
    for key, expected in {
        "schema_version": LEXICAL_SMOKE_RESULT_SCHEMA,
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": RETRIEVAL_UNIT,
    }.items():
        if result.get(key) != expected:
            raise ValueError(f"MIRACL lexical smoke result has invalid {key}")
    if result.get("scale") not in {20_000, 50_000, 110_000}:
        raise ValueError("MIRACL lexical smoke result has invalid scale")
    source_git_commit = result.get("source_git_commit")
    if not isinstance(source_git_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_git_commit):
        raise ValueError("MIRACL lexical smoke result requires source_git_commit")
    rows = result.get("raw_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("MIRACL lexical smoke result requires raw per-query rows")
    if any(row.get("retrieval_unit") != RETRIEVAL_UNIT for row in rows):
        raise ValueError("MIRACL lexical smoke raw rows must be passage-level")
    qids = [str(row.get("query_id", "")) for row in rows]
    if not all(qids) or len(qids) != len(set(qids)):
        raise ValueError("MIRACL lexical smoke raw rows require unique query IDs")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or any(key not in metrics for key in SMOKE_METRIC_KEYS[:-1]):
        raise ValueError("MIRACL lexical smoke result is missing passage metrics")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("MIRACL lexical smoke result requires provenance")
    for key in (
        "subset_sha256", "query_qrel_sha256", "backend_config_sha256",
        "backend_runtime_sha256", "raw_rows_sha256", "container_recipe_sha256",
    ):
        _require_sha256(provenance.get(key), label=f"MIRACL lexical smoke {key}")
    if provenance["backend_config_sha256"] != sha256_json(config):
        raise ValueError("MIRACL lexical smoke backend_config_sha256 does not match config")
    if provenance["raw_rows_sha256"] != sha256_json(rows):
        raise ValueError("MIRACL lexical smoke raw_rows_sha256 does not match raw rows")


def compare_smoke_scales(
    rows_20k: list[dict[str, Any]],
    rows_110k: list[dict[str, Any]],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    control = {str(row["query_id"]): row for row in rows_20k}
    treatment = {str(row["query_id"]): row for row in rows_110k}
    if set(control) != set(treatment) or not control:
        raise ValueError("MIRACL smoke scale comparison requires identical non-empty query IDs")
    query_ids = sorted(control)
    return {
        "delta_direction": "110k_minus_20k",
        "paired_query_count": len(query_ids),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
        "metrics": {
            key: paired_bootstrap_delta(
                [float(control[qid][key]) for qid in query_ids],
                [float(treatment[qid][key]) for qid in query_ids],
                seed=seed,
                iterations=iterations,
            )
            for key in SMOKE_METRIC_KEYS
        },
        "interpretation": "plumbing diagnostic only; no practical-effect decision or Part 2 claim",
    }
