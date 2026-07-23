#!/usr/bin/env python3
"""Revalidate existing MIRACL Korean lexical-smoke raw results without search.

This script deliberately does not import or invoke Anserini.  It validates the
already-recorded rankings and metrics against the pinned fixtures, then writes a
replacement result-contract report that treats legacy batch latency only as a
single batch observation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.lexical_smoke import (
    EXPERIMENT_CONTRACT_PATHS,
    LEGACY_BATCH_LATENCY_MEASUREMENT,
    SmokeResultValidationInputs,
    build_experiment_contract,
    compare_smoke_scales,
    sha256_json,
    validate_standalone_smoke_result,
)
from src.miracl_ko.preparation import SCALE_SIZES, sha256_file, validate_miracl_ko_subset_files


REVALIDATION_SCHEMA = "dr-dci.miracl-ko-lexical-smoke-revalidation.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            yield row


def git_blob(commit: str, relative_path: str) -> bytes:
    try:
        git_root = Path(subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip())
        repo_relative_path = (REPO_ROOT.relative_to(git_root) / relative_path).as_posix()
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{repo_relative_path}"],
            capture_output=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"historical source blob is unavailable: {commit}:{relative_path}") from error


def git_blob_sha256(commit: str, relative_path: str) -> str:
    return hashlib.sha256(git_blob(commit, relative_path)).hexdigest()


def git_blob_json(commit: str, relative_path: str) -> dict[str, Any]:
    try:
        value = json.loads(git_blob(commit, relative_path).decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"historical JSON is invalid: {commit}:{relative_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"historical JSON object required: {commit}:{relative_path}")
    return value


def require_manifest_file(data_dir: Path, record: dict[str, Any], *, label: str) -> Path:
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} relative_path is invalid")
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
        raise ValueError(f"{label} is missing or outside MIRACL data directory")
    if path.stat().st_size != record.get("byte_size"):
        raise ValueError(f"{label} byte size does not match subset manifest")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA-256 does not match subset manifest")
    return path


def load_dev_inputs(data_dir: Path, subset_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int | float]], str]:
    records = subset_manifest["subsets"][str(SCALE_SIZES[0])]["query_qrel_inputs"]
    queries = list(iter_jsonl(require_manifest_file(data_dir, records["queries_dev"], label="dev queries")))
    qrels: dict[str, dict[str, int | float]] = defaultdict(dict)
    for row in iter_jsonl(require_manifest_file(data_dir, records["qrels_dev"], label="dev qrels")):
        qid = str(row["qid"])
        corpus_id = str(row["corpus_id"])
        relevance = row["relevance"]
        if corpus_id in qrels[qid] and qrels[qid][corpus_id] != relevance:
            raise ValueError(f"conflicting dev qrel: {qid}/{corpus_id}")
        qrels[qid][corpus_id] = relevance
    if {str(row["qid"]) for row in queries} != set(qrels):
        raise ValueError("dev queries and qrels do not have identical query IDs")
    return queries, dict(qrels), sha256_json({
        "queries_dev": records["queries_dev"]["sha256"],
        "qrels_dev": records["qrels_dev"]["sha256"],
    })


def corpus_ids_for_scale(data_dir: Path, subset_manifest: dict[str, Any], scale: int) -> set[str]:
    record = subset_manifest["subsets"][str(scale)]["corpus"]
    corpus_ids = {str(row["corpus_id"]) for row in iter_jsonl(require_manifest_file(data_dir, record, label=f"{scale} corpus"))}
    if len(corpus_ids) != scale:
        raise ValueError(f"{scale} corpus has duplicate or missing passage IDs")
    return corpus_ids


def legacy_latency_record(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    if metrics.get("query_latency_measurement") != LEGACY_BATCH_LATENCY_MEASUREMENT:
        raise ValueError("existing raw result does not use the known legacy batch latency representation")
    query_count = metrics.get("query_count")
    batch_seconds = metrics.get("search_batch_seconds")
    mean_seconds = metrics.get("query_latency_seconds")
    if not isinstance(query_count, int) or query_count < 1:
        raise ValueError("existing raw result has invalid legacy latency query count")
    if not isinstance(batch_seconds, (int, float)) or not isinstance(mean_seconds, (int, float)):
        raise ValueError("existing raw result has invalid legacy latency values")
    return {
        "measurement": "legacy_batch_elapsed_seconds_only",
        "search_batch_seconds": batch_seconds,
        "batch_mean_per_query_seconds": mean_seconds,
        "query_count": query_count,
        "paired_bootstrap_allowed": False,
        "note": "legacy raw rows repeat this batch/query descriptive mean; no query-latency CI is reported",
    }


def historical_contract(source_git_commit: str, *, subset_manifest_sha256: str) -> dict[str, Any]:
    return build_experiment_contract(
        {relative_path: git_blob_sha256(source_git_commit, relative_path) for relative_path in EXPERIMENT_CONTRACT_PATHS},
        subset_manifest_sha256=subset_manifest_sha256,
    )


def historical_provenance(
    *,
    result: dict[str, Any],
    source_git_commit: str,
    config: dict[str, Any],
    subset_manifest: dict[str, Any],
    subset_manifest_sha256: str,
    query_qrel_sha256: str,
) -> dict[str, str]:
    scale = result["scale"]
    return {
        "subset_sha256": subset_manifest["subsets"][str(scale)]["corpus"]["sha256"],
        "query_qrel_sha256": query_qrel_sha256,
        "backend_config_sha256": sha256_json(config),
        "backend_runtime_sha256": sha256_json(result["runtime"]),
        "raw_rows_sha256": sha256_json(result["raw_rows"]),
        "subset_manifest_sha256": subset_manifest_sha256,
        "runner_code_sha256": git_blob_sha256(source_git_commit, "scripts/run_miracl_ko_lexical_smoke.py"),
        "contract_code_sha256": git_blob_sha256(source_git_commit, "src/miracl_ko/lexical_smoke.py"),
        "container_recipe_sha256": git_blob_sha256(source_git_commit, "docker/miracl_ko_lexical_smoke.Dockerfile"),
    }


def assert_original_report_matches_commit(path: Path, commit: str) -> str:
    relative_path = path.resolve().relative_to(REPO_ROOT).as_posix()
    historical_bytes = git_blob(commit, relative_path)
    if historical_bytes != path.read_bytes():
        raise ValueError("original lexical smoke report does not match the declared report commit")
    return hashlib.sha256(historical_bytes).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    paired = report["paired_110k_minus_20k"]
    recall = paired["metrics"]["passage_recall_at_20"]
    scales = report["scales"]
    text = "\n".join([
        "# MIRACL Korean lexical smoke — result-contract revalidation", "",
        "## Scope", "",
        "This report revalidates existing raw rankings without running Anserini again. It remains a standalone Korean passage-retrieval plumbing smoke, not a taxonomy, Agentic RAG, focused Part 1/2, insurance-domain, or physical-document operations result.", "",
        "## Provenance", "",
        f"- Smoke-run Git commit: `{report['smoke_run_git_commit']}`.",
        f"- Original report Git commit: `{report['original_report_git_commit']}`.",
        f"- Original report SHA-256: `{report['original_report_sha256']}`.",
        f"- Historical execution contract SHA-256: `{report['historical_experiment_contract']['experiment_contract_sha256']}`.",
        "- Historical contract covers runner, lexical result validation, paired-bootstrap code, MIRACL preparation validator, backend config, Docker recipe, revision lock, and current subset-manifest hash.", "",
        "## Contract checks", "",
        "- Every raw per-query ranking was rescored against the verified dev qrels and its scale corpus; stored raw-row and aggregate passage metrics matched.",
        "- Raw-result, runtime, runner, validation-code, Docker-recipe, query/qrel, subset, and subset-manifest hashes matched the historical source/data contract.",
        "- No Anserini command, taxonomy generation, Agent/LLM call, embedding call, or focused Part 1/2 run occurred.", "",
        "## Corrected latency handling", "",
        "The historical rows repeat one batch elapsed/query descriptive mean. It is retained as batch metadata below, but is excluded from paired bootstrap and has no query-level CI.",
        *[
            f"- {scale}: batch `{item['latency']['search_batch_seconds']:.6f}s`; batch/query mean `{item['latency']['batch_mean_per_query_seconds']:.6f}s`; paired latency CI `not reported`."
            for scale, item in sorted(scales.items(), key=lambda pair: int(pair[0]))
        ], "",
        "## Revalidated retrieval comparison", "",
        f"- 110K−20K passage Recall@20: `{recall['mean_delta']:.6f}`; 95% CI [`{recall['ci95_low']:.6f}`, `{recall['ci95_high']:.6f}`]; paired queries `{recall['n']}`.",
        "- The paired section includes retrieval metrics only; no latency metric appears.", "",
        "## Interpretation boundary", "",
        "The fixture fixes all judged passages and increases unjudged distractors. It supports controlled distractor-scaling plumbing only, not natural-corpus growth, taxonomy effectiveness, Agentic RAG effectiveness, insurance-domain transfer, or Shinhan 110K physical-document performance.",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/miracl_ko_lexical_smoke/20260723"))
    parser.add_argument("--original-summary", type=Path, default=Path("docs/MIRACL_KO_LEXICAL_SMOKE_20260723.json"))
    parser.add_argument("--original-report-git-commit", default="ee916d3f4e8858521d83df5388021f50f1b2ede6")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stamp", default="20260723")
    args = parser.parse_args()
    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    results_dir = (REPO_ROOT / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir
    original_summary_path = (REPO_ROOT / args.original_summary).resolve() if not args.original_summary.is_absolute() else args.original_summary
    docs_dir = (REPO_ROOT / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir

    original_summary = load_json(original_summary_path)
    original_report_sha256 = assert_original_report_matches_commit(original_summary_path, args.original_report_git_commit)
    subset_manifest_path = data_dir / "subsets" / "manifest.json"
    subset_manifest = load_json(subset_manifest_path)
    validate_miracl_ko_subset_files(data_dir, subset_manifest)
    subset_manifest_sha256 = sha256_file(subset_manifest_path)
    queries, qrels_by_qid, query_qrel_sha256 = load_dev_inputs(data_dir, subset_manifest)

    results: dict[int, dict[str, Any]] = {}
    source_commits: set[str] = set()
    for scale in SCALE_SIZES:
        path = results_dir / f"miracl_ko_anserini_smoke_{scale // 1000}k.json"
        result = load_json(path)
        source_commit = result.get("source_git_commit")
        if not isinstance(source_commit, str):
            raise ValueError(f"{scale} raw result is missing source_git_commit")
        source_commits.add(source_commit)
        results[scale] = {"path": path, "result": result}
    if len(source_commits) != 1:
        raise ValueError("raw lexical smoke scales do not share exactly one source Git commit")
    smoke_run_git_commit = next(iter(source_commits))
    historical_config = git_blob_json(smoke_run_git_commit, "config/miracl_ko_lexical_smoke.json")
    contract = historical_contract(smoke_run_git_commit, subset_manifest_sha256=subset_manifest_sha256)

    validated_scales: dict[str, Any] = {}
    for scale, output in results.items():
        result = output["result"]
        provenance = historical_provenance(
            result=result,
            source_git_commit=smoke_run_git_commit,
            config=historical_config,
            subset_manifest=subset_manifest,
            subset_manifest_sha256=subset_manifest_sha256,
            query_qrel_sha256=query_qrel_sha256,
        )
        validate_standalone_smoke_result(
            result,
            config=historical_config,
            inputs=SmokeResultValidationInputs(
                queries=queries,
                qrels_by_qid=qrels_by_qid,
                corpus_ids=corpus_ids_for_scale(data_dir, subset_manifest, scale),
                expected_provenance=provenance,
                expected_experiment_contract_sha256=contract["experiment_contract_sha256"],
                allow_legacy_contract=True,
            ),
        )
        summary_scale = original_summary.get("scales", {}).get(str(scale))
        if not isinstance(summary_scale, dict) or summary_scale.get("raw_result_sha256") != sha256_file(output["path"]):
            raise ValueError(f"original summary raw result SHA-256 does not match {scale} raw result")
        validated_scales[str(scale)] = {
            "raw_result_path": output["path"].relative_to(REPO_ROOT).as_posix(),
            "raw_result_sha256": sha256_file(output["path"]),
            "metrics": {key: result["metrics"][key] for key in result["metrics"] if key.startswith("passage_")},
            "latency": legacy_latency_record(result),
            "provenance": provenance,
        }

    comparison = compare_smoke_scales(
        results[20_000]["result"]["raw_rows"],
        results[110_000]["result"]["raw_rows"],
        seed=historical_config["evaluation"]["bootstrap_seed"],
        iterations=historical_config["evaluation"]["bootstrap_iterations"],
    )
    if "query_latency_seconds" in comparison["metrics"]:
        raise AssertionError("legacy batch latency entered paired retrieval comparison")
    report = {
        "schema_version": REVALIDATION_SCHEMA,
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": "passage",
        "scope": "existing raw lexical smoke revalidation only; no Anserini rerun",
        "smoke_run_git_commit": smoke_run_git_commit,
        "original_report_git_commit": args.original_report_git_commit,
        "original_report_sha256": original_report_sha256,
        "historical_experiment_contract": contract,
        "scales": validated_scales,
        "paired_110k_minus_20k": comparison,
        "validation": {
            "raw_rows_and_aggregate_metrics_recomputed": True,
            "runtime_and_source_data_provenance_verified": True,
            "legacy_batch_latency_excluded_from_paired_bootstrap": True,
            "anserini_rerun": False,
        },
        "interpretation_limits": [
            "controlled distractor scaling only; judged passages are fixed across scales",
            "no taxonomy, embedding, query rewrite, Agent/LLM, or focused Part 1/2 treatment",
            "no insurance-domain or physical-document operations claim",
        ],
    }
    json_path = docs_dir / f"MIRACL_KO_LEXICAL_SMOKE_REVALIDATION_{args.stamp}.json"
    markdown_path = docs_dir / f"MIRACL_KO_LEXICAL_SMOKE_REVALIDATION_{args.stamp}.md"
    write_json(json_path, report)
    write_markdown(markdown_path, report)
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MIRACL Korean lexical smoke revalidation failed loudly: {error}", file=sys.stderr)
        raise
