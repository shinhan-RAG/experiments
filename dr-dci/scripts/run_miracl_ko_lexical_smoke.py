#!/usr/bin/env python3
"""Run only the pinned Anserini lexical plumbing smoke for MIRACL Korean.

Run this script inside docker/miracl_ko_lexical_smoke.Dockerfile.  It does not
create taxonomy, embeddings, agent traces, answers, or focused Part 1/2 output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.miracl_ko.lexical_smoke import (
    EXPERIMENT_CONTRACT_PATHS,
    SmokeResultValidationInputs,
    build_experiment_contract,
    compare_smoke_scales,
    evaluate_passage_rankings,
    parse_anserini_trec_run,
    sha256_json,
    validate_source_archive_manifest,
    validate_lexical_smoke_config,
    validate_standalone_smoke_result,
)
from src.miracl_ko.preparation import (
    SCALE_SIZES,
    sha256_file,
    validate_miracl_ko_subset_files,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                raise ValueError(f"JSONL object required at {path}:{line_number}")
            yield row


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def resolve_source_provenance(
    value: str | None,
    *,
    source_manifest_path: Path | None,
    source_archive_path: Path | None,
) -> dict[str, Any]:
    try:
        actual_head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except FileNotFoundError:
        actual_head = None
    except subprocess.CalledProcessError as error:
        actual_head = None
    else:
        if dirty:
            raise RuntimeError("MIRACL lexical smoke rejects a dirty Git checkout")
        if value is not None and value != actual_head:
            raise RuntimeError("source_git_commit does not match the clean checkout HEAD")
        if len(actual_head) != 40 or any(character not in "0123456789abcdef" for character in actual_head):
            raise ValueError("source_git_commit must be a 40-character lowercase Git SHA-1")
        return {
            "mode": "git_checkout",
            "source_git_commit": actual_head,
            "git_clean": True,
        }

    if source_manifest_path is None or source_archive_path is None:
        raise RuntimeError("Git-free smoke execution requires --source-manifest and --source-archive")
    if not source_archive_path.is_file():
        raise RuntimeError("Git-free smoke execution source archive is missing")
    source_manifest = load_json(source_manifest_path)
    source_archive_sha256 = sha256_file(source_archive_path)
    source_git_commit = source_manifest.get("source_git_commit")
    if value is not None and value != source_git_commit:
        raise RuntimeError("source_git_commit does not match the source manifest")
    return {
        "mode": "source_archive_manifest",
        "source_git_commit": source_git_commit,
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_archive_sha256": source_archive_sha256,
    }


def build_current_experiment_contract(data_dir: Path) -> dict[str, Any]:
    return build_experiment_contract(
        {relative_path: sha256_file(REPO_ROOT / relative_path) for relative_path in EXPERIMENT_CONTRACT_PATHS},
        subset_manifest_sha256=sha256_file(data_dir / "subsets" / "manifest.json"),
    )


def require_file_record(data_dir: Path, record: dict[str, Any], *, label: str) -> Path:
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} has invalid relative_path")
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
        raise ValueError(f"{label} is missing or outside data directory")
    if path.stat().st_size != record.get("byte_size"):
        raise ValueError(f"{label} byte_size does not match manifest")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} sha256 does not match manifest")
    return path


def runtime_provenance(config: dict[str, Any]) -> dict[str, Any]:
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    backend = config["backend"]
    jar_path = Path(backend["jar_path"])
    if not jar_path.is_file() or sha256_file(jar_path) != backend["jar_sha256"]:
        raise RuntimeError("pinned Anserini fat JAR is missing or has an unexpected SHA-256")
    supporting_packages = {
        name: importlib.metadata.version(name)
        for name in config["runtime"]["supporting_runtime_packages"]
    }
    if supporting_packages != config["runtime"]["supporting_runtime_packages"]:
        raise RuntimeError("installed numerical runtime package does not match pinned smoke config")
    return {
        "python_version": sys.version,
        "java_version_output": java.stderr.strip() or java.stdout.strip(),
        "pyserini_distribution": {
            "package": backend["distribution_package"],
            "version": backend["distribution_version"],
            "source_archive_sha256": backend["distribution_source_archive_sha256"],
        },
        "anserini": {
            "version": backend["version"],
            "jar_relative_path": backend["jar_relative_path"],
            "jar_sha256": sha256_file(jar_path),
            "analyzer_language": backend["analyzer_language"],
            "analyzer_class": backend["analyzer_class"],
        },
        "supporting_runtime_packages": supporting_packages,
        "execution_mode": config["runtime"]["execution_mode"],
    }


def load_dev_inputs(data_dir: Path, subset_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int | float]], str]:
    records = subset_manifest["subsets"][str(SCALE_SIZES[0])]["query_qrel_inputs"]
    queries_path = require_file_record(data_dir, records["queries_dev"], label="dev queries")
    qrels_path = require_file_record(data_dir, records["qrels_dev"], label="dev qrels")
    queries = list(iter_jsonl(queries_path))
    qrels: dict[str, dict[str, int | float]] = defaultdict(dict)
    for row in iter_jsonl(qrels_path):
        qid = str(row["qid"])
        corpus_id = str(row["corpus_id"])
        relevance = row["relevance"]
        if corpus_id in qrels[qid] and qrels[qid][corpus_id] != relevance:
            raise ValueError(f"conflicting dev qrel: {qid}/{corpus_id}")
        qrels[qid][corpus_id] = relevance
    if {str(row["qid"]) for row in queries} != set(qrels):
        raise ValueError("dev queries and qrels do not have identical query IDs")
    query_qrel_sha256 = sha256_json({
        "queries_dev": records["queries_dev"]["sha256"],
        "qrels_dev": records["qrels_dev"]["sha256"],
    })
    return queries, dict(qrels), query_qrel_sha256


def anserini_jar_path(config: dict[str, Any]) -> Path:
    backend = config["backend"]
    jar_path = Path(backend["jar_path"])
    if not jar_path.is_file() or sha256_file(jar_path) != backend["jar_sha256"]:
        raise RuntimeError("pinned Anserini fat JAR is missing or has an unexpected SHA-256")
    return jar_path


def write_anserini_index_input(corpus_path: Path, input_dir: Path, *, scale: int) -> set[str]:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path = input_dir / "documents.jsonl"
    corpus_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8") as stream:
        for row in iter_jsonl(corpus_path):
            corpus_id = str(row["corpus_id"])
            if corpus_id in corpus_ids:
                raise ValueError(f"duplicate passage ID in {scale} corpus: {corpus_id}")
            corpus_ids.add(corpus_id)
            json.dump(
                {
                    "id": corpus_id,
                    "contents": f"{row.get('title', '')}\n{row.get('text', '')}",
                },
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.write("\n")
    if len(corpus_ids) != scale:
        raise ValueError(f"{scale} corpus row count does not match fixture")
    return corpus_ids


def write_anserini_topics(queries: list[dict[str, Any]], path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("w", encoding="utf-8") as stream:
        for row in queries:
            qid = str(row["qid"])
            query = str(row["query"])
            if not qid or qid in qids:
                raise ValueError(f"invalid or duplicate dev query ID for Anserini topics: {qid}")
            if any(character in qid for character in "\t\r\n") or any(character in query for character in "\t\r\n"):
                raise ValueError(f"Anserini TSV topic cannot losslessly represent dev query {qid}")
            qids.add(qid)
            stream.write(f"{qid}\t{query}\n")
    return qids


def run_anserini(command: list[str], *, log_path: Path, label: str) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_stream:
        try:
            subprocess.run(command, stdout=log_stream, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Anserini {label} failed; inspect {log_path}") from error
    return time.perf_counter() - started


def run_scale(
    *,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    subset_manifest: dict[str, Any],
    scale: int,
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int | float]],
    query_qrel_sha256: str,
    runtime: dict[str, Any],
    source_git_commit: str,
    source_provenance: dict[str, Any],
    container_recipe_sha256: str,
    experiment_contract: dict[str, Any],
) -> dict[str, Any]:
    subset = subset_manifest["subsets"][str(scale)]
    corpus_path = require_file_record(data_dir, subset["corpus"], label=f"{scale} passage corpus")
    scale_root = data_dir / "lexical_smoke_work" / f"{scale // 1000}k"
    index_path = scale_root / "lucene_index"
    if scale_root.exists():
        shutil.rmtree(scale_root)
    scale_root.mkdir(parents=True, exist_ok=True)
    input_dir = scale_root / "index_input"
    corpus_ids = write_anserini_index_input(corpus_path, input_dir, scale=scale)
    topics_path = scale_root / "topics.tsv"
    qids = write_anserini_topics(queries, topics_path)
    if qids != {str(row["qid"]) for row in queries}:
        raise ValueError("Anserini topic qids do not match dev queries")
    run_path = scale_root / "anserini.run"
    jar_path = anserini_jar_path(config)
    backend = config["backend"]
    retrieval = config["retrieval"]
    index_build_seconds = run_anserini(
        [
            "java", "-cp", str(jar_path), "io.anserini.index.IndexCollection",
            "-collection", "JsonCollection",
            "-input", str(input_dir),
            "-index", str(index_path),
            "-language", backend["analyzer_language"],
            "-threads", str(retrieval["index_threads"]),
            "-memoryBuffer", str(retrieval["index_memory_buffer_mb"]),
        ],
        log_path=output_dir / "logs" / f"anserini_index_{scale // 1000}k.log",
        label=f"{scale} index",
    )
    search_batch_seconds = run_anserini(
        [
            "java", "-cp", str(jar_path), "io.anserini.search.SearchCollection",
            "-index", str(index_path),
            "-topics", str(topics_path),
            "-topicReader", "TsvString",
            "-output", str(run_path),
            "-language", backend["analyzer_language"],
            "-hits", str(retrieval["top_k"]),
            "-bm25",
            "-bm25.k1", str(retrieval["bm25_k1"]),
            "-bm25.b", str(retrieval["bm25_b"]),
            "-threads", str(retrieval["search_threads"]),
            "-runtag", "miracl_ko_lexical_smoke",
        ],
        log_path=output_dir / "logs" / f"anserini_search_{scale // 1000}k.log",
        label=f"{scale} search",
    )
    rankings = parse_anserini_trec_run(run_path, query_ids=qids)

    raw_rows, metrics = evaluate_passage_rankings(
        queries, qrels, rankings, corpus_ids=corpus_ids
    )
    index_size_bytes = sum(path.stat().st_size for path in index_path.rglob("*") if path.is_file())
    metrics.update({
        "index_build_seconds": round(index_build_seconds, 6),
        "index_size_bytes": index_size_bytes,
        "orphan_retrieval_id_count": 0,
        "search_batch_seconds": round(search_batch_seconds, 6),
    })
    result = {
        "schema_version": "dr-dci.miracl-ko-lexical-smoke-result.v1",
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": "passage",
        "scale": scale,
        "source_git_commit": source_git_commit,
        "source_provenance": source_provenance,
        "experiment_contract_sha256": experiment_contract["experiment_contract_sha256"],
        "backend": config["backend"],
        "runtime": runtime,
        "retrieval": config["retrieval"],
        "raw_rows": raw_rows,
        "metrics": metrics,
        "latency": {
            "measurement": "search_batch_elapsed_seconds_only",
            "search_batch_seconds": round(search_batch_seconds, 6),
            "batch_mean_per_query_seconds": round(search_batch_seconds / len(qids), 6),
            "query_count": len(qids),
            "paired_bootstrap_allowed": False,
        },
        "provenance": {
            "subset_sha256": subset["corpus"]["sha256"],
            "query_qrel_sha256": query_qrel_sha256,
            "backend_config_sha256": sha256_json(config),
            "backend_runtime_sha256": sha256_json(runtime),
            "raw_rows_sha256": sha256_json(raw_rows),
            "subset_manifest_sha256": sha256_file(data_dir / "subsets" / "manifest.json"),
            "runner_code_sha256": sha256_file(Path(__file__).resolve()),
            "contract_code_sha256": sha256_file(REPO_ROOT / "src" / "miracl_ko" / "lexical_smoke.py"),
            "container_recipe_sha256": container_recipe_sha256,
        },
    }
    validate_standalone_smoke_result(
        result,
        config=config,
        inputs=SmokeResultValidationInputs(
            queries=queries,
            qrels_by_qid=qrels,
            corpus_ids=corpus_ids,
            expected_provenance=result["provenance"],
            expected_experiment_contract_sha256=experiment_contract["experiment_contract_sha256"],
        ),
    )
    result_path = output_dir / f"miracl_ko_anserini_smoke_{scale // 1000}k.json"
    atomic_write_json(result_path, result)
    return {
        "path": str(result_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(result_path),
        "result": result,
    }


def write_report(
    *,
    docs_dir: Path,
    stamp: str,
    config: dict[str, Any],
    runtime: dict[str, Any],
    outputs: dict[int, dict[str, Any]],
    comparison: dict[str, Any],
) -> tuple[Path, Path]:
    summary = {
        "schema_version": "dr-dci.miracl-ko-lexical-smoke-summary.v1",
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": "passage",
        "scope": "standalone lexical plumbing smoke; not a focused Part 1/2 or Agentic RAG result",
        "source_git_commit": outputs[20_000]["result"]["source_git_commit"],
        "smoke_run_git_commit": outputs[20_000]["result"]["source_git_commit"],
        "source_provenance": outputs[20_000]["result"]["source_provenance"],
        "experiment_contract_sha256": outputs[20_000]["result"]["experiment_contract_sha256"],
        "backend_config": config,
        "backend_runtime": runtime,
        "scales": {
            str(scale): {
                "raw_result_path": output["path"],
                "raw_result_sha256": output["sha256"],
                "metrics": output["result"]["metrics"],
                "provenance": output["result"]["provenance"],
            }
            for scale, output in outputs.items()
        },
        "paired_110k_minus_20k": comparison,
        "interpretation_limits": [
            "controlled distractor scaling only; judged passages are fixed across scales",
            "no taxonomy, embedding, query rewrite, Agent/LLM, or focused Part 1/2 treatment",
            "no insurance-domain or physical-document operations claim",
            "no practical-effect decision is applied to smoke metrics",
        ],
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    json_path = docs_dir / f"MIRACL_KO_LEXICAL_SMOKE_{stamp}.json"
    markdown_path = docs_dir / f"MIRACL_KO_LEXICAL_SMOKE_{stamp}.md"
    atomic_write_json(json_path, summary)
    rows = [
        f"- {scale}: index build `{output['result']['metrics']['index_build_seconds']:.3f}s`; "
        f"index bytes `{output['result']['metrics']['index_size_bytes']}`; "
        f"passage Recall@20 `{output['result']['metrics']['passage_recall_at_20']:.6f}`; "
        f"raw result SHA-256 `{output['sha256']}`"
        for scale, output in sorted(outputs.items())
    ]
    markdown = "\n".join([
        f"# MIRACL Korean lexical plumbing smoke — {stamp}", "",
        "## Scope", "",
        "This is a standalone passage-retrieval plumbing smoke. It is not a taxonomy, Agentic RAG, focused Part 1/2, insurance-domain, or physical-document operations result.",
        "",
        "## Pinned backend", "",
        f"- Backend: `{config['backend']['package']}=={config['backend']['version']}` fat JAR distributed in `{config['backend']['distribution_package']}=={config['backend']['distribution_version']}`, with `{config['backend']['analyzer_class']}` for `ko`.",
        f"- Container base: `{config['runtime']['container_base_image']}@sha256:{config['runtime']['container_base_image_sha256']}`; Java package `{config['runtime']['java_runtime_version']}`.",
        f"- Runtime execution mode: `{config['runtime']['execution_mode']}`; only NumPy is added for paired bootstrap. No embedding, model, or external API client is installed or invoked.",
        f"- Smoke-run source commit: `{outputs[20_000]['result']['source_git_commit']}`; experiment contract SHA-256 `{outputs[20_000]['result']['experiment_contract_sha256']}`.",
        "",
        "## Completed checks", "",
        "- Passage IDs returned by Lucene were checked against each scale corpus.",
        "- Dev qids, qrels, raw per-query rows, retrieval metrics, batch latency, index size, and raw-result hashes were recorded.",
        "- 110K−20K comparison is paired by query for retrieval metrics only; batch latency has no paired confidence interval.",
        *rows,
        "",
        "## Interpretation boundary", "",
        "The fixtures fix all judged passages and increase only unjudged distractors. These numbers test adapter and metric wiring, not natural-corpus growth or Shinhan-domain retrieval quality.",
        "",
    ])
    markdown_path.write_text(markdown, encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/miracl-ko"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/miracl_ko_lexical_smoke/20260723"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--config", type=Path, default=Path("config/miracl_ko_lexical_smoke.json"))
    parser.add_argument("--stamp", default="20260723")
    parser.add_argument("--source-git-commit", help="40-character source commit; required outside a Git checkout")
    parser.add_argument("--source-manifest", type=Path, help="Git-free source manifest generated before bundling")
    parser.add_argument("--source-archive", type=Path, help="Git-free source archive whose SHA-256 is in the manifest")
    args = parser.parse_args()
    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    docs_dir = (REPO_ROOT / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir
    config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_json(config_path)
    validate_lexical_smoke_config(config)
    source_manifest_path = (
        (REPO_ROOT / args.source_manifest).resolve()
        if args.source_manifest is not None and not args.source_manifest.is_absolute()
        else args.source_manifest
    )
    source_archive_path = (
        (REPO_ROOT / args.source_archive).resolve()
        if args.source_archive is not None and not args.source_archive.is_absolute()
        else args.source_archive
    )
    source_provenance = resolve_source_provenance(
        args.source_git_commit,
        source_manifest_path=source_manifest_path,
        source_archive_path=source_archive_path,
    )
    source_git_commit = source_provenance["source_git_commit"]
    container_recipe_sha256 = sha256_file(REPO_ROOT / "docker" / "miracl_ko_lexical_smoke.Dockerfile")

    subset_manifest = load_json(data_dir / "subsets" / "manifest.json")
    validate_miracl_ko_subset_files(data_dir, subset_manifest)
    queries, qrels, query_qrel_sha256 = load_dev_inputs(data_dir, subset_manifest)
    runtime = runtime_provenance(config)
    experiment_contract = build_current_experiment_contract(data_dir)
    if source_provenance["mode"] == "source_archive_manifest":
        validate_source_archive_manifest(
            source_provenance["source_manifest"],
            source_archive_sha256=source_provenance["source_archive_sha256"],
            expected_experiment_contract_sha256=experiment_contract["experiment_contract_sha256"],
        )
        source_provenance = {
            key: value for key, value in source_provenance.items() if key != "source_manifest"
        }
    outputs = {
        scale: run_scale(
            config=config,
            data_dir=data_dir,
            output_dir=output_dir,
            subset_manifest=subset_manifest,
            scale=scale,
            queries=queries,
            qrels=qrels,
            query_qrel_sha256=query_qrel_sha256,
            runtime=runtime,
            source_git_commit=source_git_commit,
            source_provenance=source_provenance,
            container_recipe_sha256=container_recipe_sha256,
            experiment_contract=experiment_contract,
        )
        for scale in SCALE_SIZES
    }
    comparison = compare_smoke_scales(
        outputs[20_000]["result"]["raw_rows"],
        outputs[110_000]["result"]["raw_rows"],
        seed=config["evaluation"]["bootstrap_seed"],
        iterations=config["evaluation"]["bootstrap_iterations"],
    )
    markdown_path, json_path = write_report(
        docs_dir=docs_dir,
        stamp=args.stamp,
        config=config,
        runtime=runtime,
        outputs=outputs,
        comparison=comparison,
    )
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MIRACL Korean lexical smoke failed loudly: {error}", file=sys.stderr)
        raise
