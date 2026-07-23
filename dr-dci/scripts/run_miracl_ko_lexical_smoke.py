#!/usr/bin/env python3
"""Run only the pinned Pyserini lexical plumbing smoke for MIRACL Korean.

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
    compare_smoke_scales,
    evaluate_passage_rankings,
    sha256_json,
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
    from pyserini.analysis import get_lucene_analyzer

    analyzer = get_lucene_analyzer(config["backend"]["analyzer_language"])
    analyzed_probe = [str(token) for token in analyzer.analyze("한국어 검색 검증")]
    if not analyzed_probe:
        raise RuntimeError("Pyserini Korean CJK analyzer returned no probe tokens")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("pyserini", "pyjnius", "numpy", "pandas", "tqdm")
    }
    if packages["pyserini"] != config["backend"]["version"]:
        raise RuntimeError("installed Pyserini version does not match pinned smoke config")
    return {
        "python_version": sys.version,
        "java_version_output": java.stderr.strip() or java.stdout.strip(),
        "packages": packages,
        "analyzer_language": config["backend"]["analyzer_language"],
        "analyzer_class": config["backend"]["analyzer_class"],
        "analyzer_probe_tokens": analyzed_probe,
        "dependency_mode": config["runtime"]["dependency_mode"],
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
) -> dict[str, Any]:
    from pyserini.index.lucene import LuceneIndexer
    from pyserini.search.lucene import LuceneSearcher

    subset = subset_manifest["subsets"][str(scale)]
    corpus_path = require_file_record(data_dir, subset["corpus"], label=f"{scale} passage corpus")
    scale_root = data_dir / "lexical_smoke_work" / f"{scale // 1000}k"
    index_path = scale_root / "lucene_index"
    if index_path.exists():
        shutil.rmtree(index_path)
    scale_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    indexer = LuceneIndexer(
        args=[
            "-index", str(index_path),
            "-storePositions", "-storeDocvectors", "-storeRaw",
            "-language", config["backend"]["analyzer_language"],
        ],
        threads=1,
    )
    corpus_ids: set[str] = set()
    for row in iter_jsonl(corpus_path):
        corpus_id = str(row["corpus_id"])
        if corpus_id in corpus_ids:
            raise ValueError(f"duplicate passage ID in {scale} corpus: {corpus_id}")
        corpus_ids.add(corpus_id)
        indexer.add_doc_dict({
            "id": corpus_id,
            "contents": f"{row.get('title', '')}\n{row.get('text', '')}",
        })
    indexer.close()
    index_build_seconds = time.perf_counter() - started
    if len(corpus_ids) != scale:
        raise ValueError(f"{scale} corpus row count does not match fixture")

    searcher = LuceneSearcher(str(index_path))
    searcher.set_language(config["backend"]["analyzer_language"])
    searcher.set_bm25(config["retrieval"]["bm25_k1"], config["retrieval"]["bm25_b"])
    rankings: dict[str, list[tuple[str, float]]] = {}
    latencies: dict[str, float] = {}
    for query in queries:
        qid = str(query["qid"])
        query_started = time.perf_counter()
        hits = searcher.search(str(query["query"]), k=config["retrieval"]["top_k"])
        latencies[qid] = time.perf_counter() - query_started
        rankings[qid] = [(str(hit.docid), float(hit.score)) for hit in hits]

    raw_rows, metrics = evaluate_passage_rankings(
        queries, qrels, rankings, corpus_ids=corpus_ids, latencies_by_qid=latencies
    )
    index_size_bytes = sum(path.stat().st_size for path in index_path.rglob("*") if path.is_file())
    metrics.update({
        "index_build_seconds": round(index_build_seconds, 6),
        "index_size_bytes": index_size_bytes,
        "orphan_retrieval_id_count": 0,
    })
    result = {
        "schema_version": "dr-dci.miracl-ko-lexical-smoke-result.v1",
        "generated_at": utc_now(),
        "dataset": "MIRACL",
        "language": "ko",
        "retrieval_unit": "passage",
        "scale": scale,
        "backend": config["backend"],
        "runtime": runtime,
        "retrieval": config["retrieval"],
        "raw_rows": raw_rows,
        "metrics": metrics,
        "provenance": {
            "subset_sha256": subset["corpus"]["sha256"],
            "query_qrel_sha256": query_qrel_sha256,
            "backend_config_sha256": sha256_json(config),
            "backend_runtime_sha256": sha256_json(runtime),
            "raw_rows_sha256": sha256_json(raw_rows),
            "subset_manifest_sha256": sha256_file(data_dir / "subsets" / "manifest.json"),
            "code_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    validate_standalone_smoke_result(result, config=config)
    result_path = output_dir / f"miracl_ko_pyserini_smoke_{scale // 1000}k.json"
    atomic_write_json(result_path, result)
    return {"path": str(result_path), "sha256": sha256_file(result_path), "result": result}


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
        f"- Backend: `{config['backend']['package']}=={config['backend']['version']}` with `{config['backend']['analyzer_class']}` for `ko`.",
        f"- Container base: `{config['runtime']['container_base_image']}@sha256:{config['runtime']['container_base_image_sha256']}`; Java package `{config['runtime']['java_runtime_version']}`.",
        f"- Runtime dependency mode: `{config['runtime']['dependency_mode']}`; no embedding, model, or external API client is invoked.",
        "",
        "## Completed checks", "",
        "- Passage IDs returned by Lucene were checked against each scale corpus.",
        "- Dev qids, qrels, raw per-query rows, metrics, latency, index size, and raw-result hashes were recorded.",
        "- 110K−20K comparison is paired by query and is a plumbing diagnostic only.",
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
    args = parser.parse_args()
    data_dir = (REPO_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    docs_dir = (REPO_ROOT / args.docs_dir).resolve() if not args.docs_dir.is_absolute() else args.docs_dir
    config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_json(config_path)
    validate_lexical_smoke_config(config)

    subset_manifest = load_json(data_dir / "subsets" / "manifest.json")
    validate_miracl_ko_subset_files(data_dir, subset_manifest)
    queries, qrels, query_qrel_sha256 = load_dev_inputs(data_dir, subset_manifest)
    runtime = runtime_provenance(config)
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
