"""
DR-DCI Augmentation Experiment Runner
- Part 1: 기법 적층 (TREC-COVID, config subset 크기)
- Part 2: 규모 확장 비교 (controlled distractor scaling)
- Part 3: @el: 태그 방식 비교
- Part 4: 일반화 검증
- Part 5: pull backend 비교 (dense vs hybrid_rrf)
"""

import json
import os
import yaml
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.agent.retriever import PullRetriever, RetrieverConfig
from src.agent.dci_agent import DCIAgent
from src.hybrid.pipeline import HybridRAG
from src.eval.comparison import (
    compare_paired_results,
    compare_probe_rows,
    compare_result_rows,
)
from src.eval.judge import Judge, compute_metrics
from src.eval.part12_contracts import audit_part12
from src.eval.retrieval_metrics import rank_metrics


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"


def load_config():
    with open(CONFIG_DIR / "experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_corpus(dataset: str, subset_size: int = None):
    """corpus 로드 (서브셋 적용)"""
    raw_dir = DATA_DIR / "raw" / dataset
    corpus = []
    with open(raw_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    if subset_size:
        subset_path = DATA_DIR / "subsets" / dataset / f"{subset_size // 1000}k.json"
        with open(subset_path, encoding="utf-8") as f:
            doc_ids = set(json.load(f)["doc_ids"])
        corpus = [doc for doc in corpus if doc["_id"] in doc_ids]

    return corpus


def load_queries(dataset: str):
    """쿼리 + qrels 로드"""
    raw_dir = DATA_DIR / "raw" / dataset
    queries = []
    with open(raw_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    qrels = []
    with open(raw_dir / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    return queries, qrels


def load_augmentations(dataset: str, subset_size: int | None, step_config: dict):
    """step 설정에 따라 augmentation을 로드한다.

    요청한 처치가 없는 상태로 같은 arm 이름을 사용하면 실험 라벨이
    어긋난다. 그러므로 요청된 산출물 누락은 즉시 실패한다.
    """
    taxonomy = None
    tags = None
    prefix = None
    metadata = None

    if subset_size is None:
        return taxonomy, tags, prefix, metadata

    size_key = f"{subset_size // 1000}k"

    def required(path: Path, feature: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"requested augmentation '{feature}' is missing: {path}"
            )
        return path

    if step_config.get("taxonomy"):
        path = DATA_DIR / "taxonomy" / f"{dataset}_{size_key}.json"
        required(path, "taxonomy")
        with open(path, encoding="utf-8") as f:
            taxonomy = json.load(f)

    if step_config.get("tags"):
        approach = step_config["tags"].lower()
        path = DATA_DIR / "tags" / dataset / f"approach_{approach}" / f"{size_key}.json"
        required(path, f"tags({approach})")
        with open(path, encoding="utf-8") as f:
            raw_tags = json.load(f)
        tags = {}
        for elem in raw_tags:
            did = elem["doc_id"]
            tags.setdefault(did, []).append(elem)

    if step_config.get("prefix"):
        path = DATA_DIR / "prefix" / f"{dataset}_{size_key}.json"
        required(path, "prefix")
        with open(path, encoding="utf-8") as f:
            prefix = json.load(f)

    if step_config.get("metadata"):
        path = DATA_DIR / "metadata" / f"{dataset}_{size_key}.json"
        required(path, "metadata")
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)

    return taxonomy, tags, prefix, metadata


def build_pull_retriever(config: dict, step_config: dict, corpus: list,
                         prefix: dict = None, taxonomy: dict = None) -> PullRetriever:
    """pull retriever 구성 단일화 — agent 실행과 retrieval-only probe가 같은
    구성·인덱스(임베딩 캐시 포함)를 공유해 backend 외 변인이 생기지 않게 한다."""
    models = config["models"]
    agent_cfg = config["agent"]
    pull_backend = step_config.get(
        "pull_backend", agent_cfg.get("pull_backend", "dense")
    )
    retriever_config = RetrieverConfig(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        top_k=agent_cfg["pull_top_k"],
        use_prefix=step_config.get("prefix", False),
        backend=pull_backend,
        bm25_top_k=agent_cfg.get("bm25_top_k", agent_cfg["pull_top_k"]),
        rrf_k=agent_cfg.get("rrf_k", 60),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        query_instruction=models["embedding"].get("query_instruction"),
        max_top_k=agent_cfg.get("max_pull_top_k", 200),
    )
    retriever = PullRetriever(retriever_config)
    print("    Indexing corpus...")
    retriever.index(corpus, prefixes=prefix, taxonomy=taxonomy)
    return retriever


def positive_gold_by_query(qrels: list) -> dict:
    """질의별 positive gold 집합 — gold 없는 질의는 retrieval 분모에서 제외
    (EDA §3 원칙: unjudged를 miss로 세면 결과가 왜곡된다)."""
    from collections import defaultdict
    gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            gold[str(entry["query-id"])].add(str(entry["corpus-id"]))
    return dict(gold)


def positive_gold_gains_by_query(qrels: list) -> dict:
    """질의별 graded gain(doc_id→score) — nDCG용. positive(score>=1)만 포함해
    분모 정책을 positive_gold_by_query와 일치시킨다."""
    from collections import defaultdict
    gains = defaultdict(dict)
    for entry in qrels:
        if entry["score"] >= 1:
            gains[str(entry["query-id"])][str(entry["corpus-id"])] = float(entry["score"])
    return dict(gains)


def run_pull_probe(retriever: PullRetriever, queries: list,
                   query_gold: dict, query_gains: dict = None) -> list:
    """retrieval-only probe: 원 질의 텍스트로 backend를 직접 1회 pull해
    rank 지표(Recall@5/20·Hit@5/10·P@20·nDCG@10)를 잰다 — agent의 질의
    재작성과 독립인 검색 품질 축. LLM/judge 불요(임베딩 endpoint만 필요)."""
    rows = []
    for q in queries:
        qid = str(q["_id"])
        gold = query_gold.get(qid)
        if not gold:
            continue                      # gold 없는 질의는 분모 제외
        gains = query_gains.get(qid) if query_gains else None
        text = q.get("title") or q.get("text", "")
        started = time.perf_counter()
        pulled = retriever.pull(text)
        candidates = pulled["results"] if isinstance(pulled, dict) else pulled
        ranked = [r["doc_id"] for r in candidates]
        latency = time.perf_counter() - started
        rows.append({
            "query_id": qid,
            **rank_metrics(ranked, gold, gains=gains),
            "probe_latency_seconds": latency,
            "ranked_top20": ranked[:20],
        })
    return rows


def failed_query_row(query: dict, exc: Exception,
                     requested_features: dict = None,
                     single_pull: bool = False) -> dict:
    """실행에 실패한 질의를 결과 행으로 기록한다 — 한 질의의 장애가
    이미 소비한 나머지 결과를 유실시키지 않도록."""
    return {
        "query_id": str(query.get("_id", "")),
        "query_text": query.get("title") or query.get("text", ""),
        "answer": "",
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "termination_reason": "harness_error",
        "gold_recall": 0.0,
        "read_recall": 0.0,
        "efficiency": 0.0,
        "pull_count": 0,
        "retrieved_candidates": 0,
        "added_documents": 0,
        "workspace_docs": [],
        "read_docs": [],
        "turns": 0,
        "tool_call_counts": {},
        "tool_calls_total": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "taxonomy_filtered_pulls": 0,
        "system_fingerprints": [],
        "latency_seconds": 0.0,
        "distinct_pull_queries": 0,
        "pull_stats": [],
        "budget_exhausted": False,
        "rule_violations": [],
        "trace": [],
        "requested_features": dict(requested_features or {}),
        "single_pull": single_pull,
    }


def check_arm_failure_rate(results: list, threshold: float = 0.2):
    """arm 내 실패율이 임계값을 넘으면 결과 해석이 불가하므로 중단한다."""
    if not results:
        return
    failed = [
        r for r in results
        if r.get("termination_reason") in {"llm_error", "harness_error"}
    ]
    rate = len(failed) / len(results)
    if rate > threshold:
        errors = [r.get("error", "") for r in failed[:5]]
        raise RuntimeError(
            f"arm aborted: {len(failed)}/{len(results)} queries failed "
            f"(threshold {threshold:.0%}). first errors: {errors}"
        )


def assign_judgment(r: dict, ref_answers: dict, judge) -> None:
    """빈 답변(실패/프로토콜 위반)은 judge에 보내지 않고 명시적으로 구분한다."""
    if not r.get("answer"):
        r["judgment"] = "agent_error"
    elif r["query_id"] in ref_answers:
        r["judgment"] = judge.evaluate_accuracy(
            r["query_text"], ref_answers[r["query_id"]], r["answer"]
        )
    else:
        r["judgment"] = "n/a"


def run_dr_dci(config: dict, corpus: list, queries: list, qrels: list,
               ref_answers: dict, step_config: dict, subset_size: int,
               dataset: str, cached_retriever: PullRetriever = None,
               single_pull: bool = False) -> list:
    """DR-DCI 에이전트 실행"""
    models = config["models"]
    agent_cfg = config["agent"]

    taxonomy, tags, prefix, metadata = load_augmentations(dataset, subset_size, step_config)

    if cached_retriever:
        retriever = cached_retriever
        retriever.set_taxonomy(taxonomy)
        print("    Using cached embeddings")
    else:
        retriever = build_pull_retriever(
            config, step_config, corpus, prefix=prefix, taxonomy=taxonomy
        )

    corpus_dict = {doc["_id"]: doc for doc in corpus}

    # Load schemas for agent prompt
    taxonomy_schema = None
    metadata_schema = None
    if step_config.get("taxonomy"):
        schema_path = CONFIG_DIR / "taxonomy_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path) as f:
                taxonomy_schema = yaml.safe_load(f)
    if step_config.get("metadata"):
        schema_path = CONFIG_DIR / "metadata_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path) as f:
                metadata_schema = yaml.safe_load(f)

    # Agent
    agent = DCIAgent(
        llm_url=models["agent_llm"]["url"],
        model_name=models["agent_llm"]["name"],
        retriever=retriever,
        corpus=corpus_dict,
        tags_data=tags,
        taxonomy_data=taxonomy,
        metadata_data=metadata,
        prefix_data=prefix,
        max_turns=agent_cfg["max_turns"],
        workspace_max_docs=agent_cfg["workspace_max_docs"],
        min_pulls=agent_cfg.get("min_pulls", 2),
        single_pull=single_pull,
        taxonomy_schema=taxonomy_schema,
        metadata_schema=metadata_schema,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    # Judge
    with open(CONFIG_DIR / "judge_prompt.txt") as f:
        judge_prompt = f.read()
    judge = Judge(
        llm_url=models["judge_llm"]["url"],
        model_name=models["judge_llm"]["name"],
        prompt_template=judge_prompt,
    )

    # qrels → query별 gold docs
    from collections import defaultdict
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(str(entry["corpus-id"]))

    # 에이전트 병렬 실행
    def run_single_query(i, q):
        qid = str(q["_id"])
        query_text = q.get("title") or q.get("text", "")
        started = time.perf_counter()
        result = agent.run(query_text)
        latency_seconds = time.perf_counter() - started
        gold_docs = list(query_gold.get(qid, []))
        gold_recall = Judge.gold_recall_at_workspace(result["workspace_docs"], gold_docs)
        read_recall = Judge.gold_recall_at_workspace(result["read_docs"], gold_docs)
        efficiency = Judge.efficiency(gold_recall, result["pull_count"])
        print(f"    [{i+1}/{len(queries)}] {query_text[:50]}...")
        return {
            "query_id": qid,
            "query_text": query_text,
            "answer": result["answer"],
            "gold_recall": gold_recall,
            "read_recall": read_recall,
            "efficiency": efficiency,
            "pull_count": result["pull_count"],
            "retrieved_candidates": result["retrieved_candidates"],
            "added_documents": result["added_documents"],
            "workspace_docs": result["workspace_docs"],
            "read_docs": result["read_docs"],
            "turns": result["turns"],
            "tool_call_counts": result.get("tool_call_counts", {}),
            "tool_calls_total": result.get("tool_calls_total", 0),
            "llm_prompt_tokens": result.get("llm_prompt_tokens", 0),
            "llm_completion_tokens": result.get("llm_completion_tokens", 0),
            "taxonomy_filtered_pulls": result.get("taxonomy_filtered_pulls", 0),
            "system_fingerprints": result.get("system_fingerprints", []),
            "latency_seconds": latency_seconds,
            "distinct_pull_queries": result["distinct_pull_queries"],
            "pull_stats": result["pull_stats"],
            "budget_exhausted": result["budget_exhausted"],
            "rule_violations": result["rule_violations"],
            "trace": result["trace"],
            "requested_features": {
                key: step_config.get(key, False)
                for key in ("taxonomy", "tags", "prefix", "metadata")
            },
            "single_pull": single_pull,
        }

    requested_features = {
        key: step_config.get(key, False)
        for key in ("taxonomy", "tags", "prefix", "metadata")
    }
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(run_single_query, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = failed_query_row(
                    queries[idx], exc,
                    requested_features=requested_features,
                    single_pull=single_pull,
                )
                print(f"    [FAILED] query {queries[idx].get('_id')}: {exc}")

    try:
        check_arm_failure_rate(results)
    except RuntimeError:
        # 이미 소비한 API 비용의 결과는 중단 전에 보존한다
        dump_path = RESULTS_DIR / f"aborted_arm_{datetime.now():%Y%m%d_%H%M%S}.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"    [ABORTED] partial results saved: {dump_path}")
        raise

    # Judge 병렬 실행
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(
            lambda r: assign_judgment(r, ref_answers, judge), results
        ))

    return results


def run_hybrid(config: dict, corpus: list, queries: list, qrels: list,
               ref_answers: dict) -> list:
    """Hybrid RAG baseline 실행"""
    models = config["models"]
    hybrid_cfg = config["hybrid"]

    pipeline = HybridRAG(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        reranker_url=models["reranker"]["url"],
        reranker_model=models["reranker"]["name"],
        llm_url=models["agent_llm"]["url"],
        llm_model=models["agent_llm"]["name"],
        dense_top_k=hybrid_cfg["dense_top_k"],
        bm25_top_k=hybrid_cfg["bm25_top_k"],
        rerank_top_k=hybrid_cfg["rerank_top_k"],
        api_key=os.getenv("OPENAI_API_KEY", ""),
        query_instruction=models["embedding"].get("query_instruction"),
    )

    print("    Indexing corpus...")
    pipeline.index(corpus)

    # Judge
    with open(CONFIG_DIR / "judge_prompt.txt") as f:
        judge_prompt = f.read()
    judge = Judge(
        llm_url=models["judge_llm"]["url"],
        model_name=models["judge_llm"]["name"],
        prompt_template=judge_prompt,
    )

    from collections import defaultdict
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(str(entry["corpus-id"]))

    # Hybrid 병렬 실행
    def run_single_hybrid(i, q):
        qid = str(q["_id"])
        query_text = q.get("title") or q.get("text", "")
        started = time.perf_counter()
        result = pipeline.run(query_text)
        latency_seconds = time.perf_counter() - started
        gold_docs = list(query_gold.get(qid, []))
        gold_recall = Judge.gold_recall_at_workspace(result["retrieved_docs"], gold_docs)
        efficiency = Judge.efficiency(gold_recall, result["pull_count"])
        print(f"    [{i+1}/{len(queries)}] {query_text[:50]}...")
        return {
            "query_id": qid,
            "query_text": query_text,
            "answer": result["answer"],
            "gold_recall": gold_recall,
            "efficiency": efficiency,
            "pull_count": result["pull_count"],
            "retrieved_candidates": len(result["retrieved_docs"]),
            "retrieved_docs": result["retrieved_docs"],
            "turns": 1,
            "latency_seconds": latency_seconds,
            "reranker_used": result.get("reranker_used"),
            "reranker_error": result.get("reranker_error"),
        }

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_hybrid, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = failed_query_row(queries[idx], exc)
                print(f"    [FAILED] query {queries[idx].get('_id')}: {exc}")

    check_arm_failure_rate(results)

    # Judge 순차 실행
    for r in results:
        assign_judgment(r, ref_answers, judge)

    return results


def _part12_preflight(config: dict, *, step_names: set[str], sizes: list[int]):
    report = audit_part12(
        config,
        DATA_DIR,
        step_names=step_names,
        sizes=sizes,
    )
    if report["blockers"]:
        details = "\n".join(f"- {item}" for item in report["blockers"])
        raise RuntimeError(f"Part 1/2 preflight failed:\n{details}")
    return report


def run_part1(config: dict, *, focused: bool = False):
    """Part 1: 기법 적층"""
    part_cfg = config["parts"]["part1_stacking"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]
    steps = part_cfg["steps"]
    if focused:
        focused_names = {"baseline", "taxonomy_only"}
        steps = [step for step in steps if step["name"] in focused_names]

    print("\n" + "=" * 60)
    print(f"Part 1: Technique Stacking ({dataset}, {subset_size // 1000}K)")
    print("=" * 60)

    # Validate treatments before loading a corpus or calling any model endpoint.
    preflight = _part12_preflight(
        config,
        step_names={str(step["name"]) for step in steps},
        sizes=[subset_size],
    )

    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset)

    # reference answers
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path) as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    # 임베딩 캐싱: prefix off / prefix on 두 가지만 빌드
    models = config["models"]
    agent_cfg = config["agent"]

    retriever_no_prefix = None
    if any(not step.get("prefix") for step in steps):
        print("\n  Building cached embeddings (prefix=off)...")
        retriever_no_prefix = PullRetriever(RetrieverConfig(
            embedding_url=models["embedding"]["url"],
            embedding_model=models["embedding"]["name"],
            top_k=agent_cfg["pull_top_k"],
            use_prefix=False,
            backend=agent_cfg.get("pull_backend", "dense"),
            bm25_top_k=agent_cfg.get("bm25_top_k", agent_cfg["pull_top_k"]),
            rrf_k=agent_cfg.get("rrf_k", 60),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            query_instruction=models["embedding"].get("query_instruction"),
            max_top_k=agent_cfg.get("max_pull_top_k", 200),
        ))
        retriever_no_prefix.index(corpus)

    retriever_with_prefix = None
    if any(step.get("prefix") for step in steps):
        _, _, prefix_data, _ = load_augmentations(
            dataset, subset_size, {"prefix": True}
        )
        print("  Building cached embeddings (prefix=on)...")
        retriever_with_prefix = PullRetriever(RetrieverConfig(
            embedding_url=models["embedding"]["url"],
            embedding_model=models["embedding"]["name"],
            top_k=agent_cfg["pull_top_k"],
            use_prefix=True,
            backend=agent_cfg.get("pull_backend", "dense"),
            bm25_top_k=agent_cfg.get("bm25_top_k", agent_cfg["pull_top_k"]),
            rrf_k=agent_cfg.get("rrf_k", 60),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            query_instruction=models["embedding"].get("query_instruction"),
            max_top_k=agent_cfg.get("max_pull_top_k", 200),
        ))
        retriever_with_prefix.index(corpus, prefixes=prefix_data)

    all_results = {}
    for step in steps:
        step_name = step["name"]
        print(f"\n  --- {step_name}: {step['description']} ---")

        cached = retriever_with_prefix if step.get("prefix") else retriever_no_prefix
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step, subset_size, dataset, cached_retriever=cached)
        metrics = compute_metrics(results)
        all_results[step_name] = {"results": results, "metrics": metrics}

        print(f"    Metrics: {metrics}")

    analysis = {}
    if "baseline" in all_results and "taxonomy_only" in all_results:
        analysis["taxonomy_only_minus_baseline"] = compare_result_rows(
            all_results["baseline"]["results"],
            all_results["taxonomy_only"]["results"],
            seed=config["seed"],
        )
    save_results(
        "part1_taxonomy_focused" if focused else "part1_stacking",
        all_results,
        manifest={
            "git_commit": current_git_commit(),
            "dataset": dataset,
            "subset_size": subset_size,
            "focused": focused,
            "arms": [str(step["name"]) for step in steps],
            "preflight": preflight,
        },
        analysis=analysis,
    )


def run_part2(config: dict, *, focused: bool = False):
    """Part 2: 규모 확장 비교"""
    print("\n" + "=" * 60)
    print("Part 2: Scale Comparison (TREC-COVID)")
    print("=" * 60)

    part_cfg = config["parts"]["part2_scaling"]
    dataset = part_cfg["dataset"]

    queries, qrels = load_queries(dataset)
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path) as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    # The focused path tests one mechanism at a time. The historical path keeps
    # Peter's original stack_all comparison for reproducibility.
    best_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}
    focused_arms = [
        {"name": "baseline", "taxonomy": False, "tags": False,
         "prefix": False, "metadata": False},
        {"name": "taxonomy_only", "taxonomy": True, "tags": False,
         "prefix": False, "metadata": False},
    ]
    selected_steps = focused_arms if focused else [{"name": "stack_all", **best_config}]
    preflight = _part12_preflight(
        config,
        step_names={"baseline", "taxonomy_only"} if focused else {"stack_all"},
        sizes=[int(size) for size in part_cfg["subsets"]],
    )

    all_results = {}
    for subset_size in part_cfg["subsets"]:
        size_key = f"{subset_size // 1000}k"
        corpus = load_corpus(dataset, subset_size)

        for arm in selected_steps:
            print(f"\n  --- DR-DCI {arm['name']} @ {size_key} ---")
            results = run_dr_dci(
                config, corpus, queries, qrels, ref_answers,
                arm, subset_size, dataset,
            )
            metrics = compute_metrics(results)
            all_results[f"{arm['name']}_{size_key}"] = {
                "results": results,
                "metrics": metrics,
            }
            print(f"    Metrics: {metrics}")

            if part_cfg.get("include_single_pull", True):
                print(f"\n  --- Single Pull {arm['name']} @ {size_key} ---")
                single_results = run_dr_dci(
                    config, corpus, queries, qrels, ref_answers,
                    arm, subset_size, dataset, single_pull=True,
                )
                single_key = f"single-pull_{arm['name']}_{size_key}"
                all_results[single_key] = {
                    "results": single_results,
                    "metrics": compute_metrics(single_results),
                }
                print(f"    Metrics: {all_results[single_key]['metrics']}")

        if not focused:
            print(f"\n  --- Hybrid RAG @ {size_key} ---")
            results = run_hybrid(config, corpus, queries, qrels, ref_answers)
            metrics = compute_metrics(results)
            all_results[f"hybrid_{size_key}"] = {"results": results, "metrics": metrics}
            print(f"    Metrics: {metrics}")

    analysis = {}
    sizes = [int(size) for size in part_cfg["subsets"]]
    smallest_key = f"{sizes[0] // 1000}k"
    if focused:
        for size in sizes:
            size_key = f"{size // 1000}k"
            analysis[f"taxonomy_minus_baseline_{size_key}"] = compare_result_rows(
                all_results[f"baseline_{size_key}"]["results"],
                all_results[f"taxonomy_only_{size_key}"]["results"],
                seed=config["seed"],
            )
        for arm in ("baseline", "taxonomy_only"):
            for size in sizes[1:]:
                size_key = f"{size // 1000}k"
                analysis[f"{arm}_{size_key}_minus_{smallest_key}"] = compare_result_rows(
                    all_results[f"{arm}_{smallest_key}"]["results"],
                    all_results[f"{arm}_{size_key}"]["results"],
                    seed=config["seed"],
                )
    if part_cfg.get("include_single_pull", True):
        for arm in selected_steps:
            arm_name = arm["name"]
            for size in sizes:
                size_key = f"{size // 1000}k"
                analysis[f"dynamic_minus_single_{arm_name}_{size_key}"] = compare_result_rows(
                    all_results[f"single-pull_{arm_name}_{size_key}"]["results"],
                    all_results[f"{arm_name}_{size_key}"]["results"],
                    seed=config["seed"],
                )

    save_results(
        "part2_taxonomy_scaling_focused" if focused else "part2_scaling",
        all_results,
        manifest={
            "git_commit": current_git_commit(),
            "dataset": dataset,
            "subsets": sizes,
            "focused": focused,
            "arms": [str(step["name"]) for step in selected_steps],
            "include_single_pull": part_cfg.get("include_single_pull", True),
            "preflight": preflight,
        },
        analysis=analysis,
    )


def run_part2_scale_probe(config: dict):
    """Part 2 선행 probe: 같은 질의를 nested subset(20K/50K/110K)에서 dense
    pull만으로 재측정한다 — distractor 희석에 따른 검색단 저하를 agent/LLM
    없이 국소화하는 단계. augmentation 산출물과 무관하게 실행 가능하다.

    지표 선택 근거: TREC-COVID는 질의당 positive gold 중앙값 478로 Recall@20
    상한이 낮고 Hit@k가 포화되므로 nDCG@10(graded)·P@20을 주 지표로 쓴다.
    주의: 지표가 스케일에 평탄해도 후단(agent/LLM) 원인 확정이 아니라 검색단
    병목 가설의 약화로만 해석한다(국소화이지 귀속 확정이 아님)."""
    print("\n" + "=" * 60)
    print("Part 2 Scale Probe: Dense Retrieval vs Distractor Scaling")
    print("=" * 60)

    part_cfg = config["parts"]["part2_scaling"]
    dataset = part_cfg["dataset"]
    sizes = [int(size) for size in part_cfg["subsets"]]
    # 모델 호출 전 무비용 계약 검사(nested·gold 보존). baseline은 산출물이
    # 필요 없으므로 augmentation 부재로 차단되지 않는다.
    preflight = _part12_preflight(config, step_names={"baseline"}, sizes=sizes)

    queries, qrels = load_queries(dataset)
    query_gold = positive_gold_by_query(qrels)
    query_gains = positive_gold_gains_by_query(qrels)

    all_results = {}
    probes = {}
    for size in sizes:
        size_key = f"{size // 1000}k"
        corpus = load_corpus(dataset, size)
        print(f"\n  --- dense probe @ {size_key} ---")
        retriever = build_pull_retriever(config, {"pull_backend": "dense"}, corpus)
        probes[size_key] = run_pull_probe(retriever, queries, query_gold,
                                          query_gains=query_gains)
        all_results[f"dense_{size_key}"] = {"probe_rows": probes[size_key]}

    analysis = {}
    size_keys = [f"{size // 1000}k" for size in sizes]
    for i in range(len(size_keys)):
        for j in range(i + 1, len(size_keys)):
            analysis[f"dense_{size_keys[j]}_minus_{size_keys[i]}"] = compare_probe_rows(
                probes[size_keys[i]], probes[size_keys[j]], seed=config["seed"]
            )

    manifest = {
        "hypothesis": (
            "Top-rank dense retrieval quality degrades as distractors grow "
            "from 20K to 110K over fixed queries and fixed gold."
        ),
        "single_variable": "distractor_count",
        "design": "controlled distractor scaling (nested subsets, gold preserved)",
        "backend": "dense",
        "dataset": dataset,
        "subset_sizes": sizes,
        "primary_metrics": ["ndcg_at_10", "precision_at_20"],
        "secondary_metrics": ["recall_at_5", "recall_at_20", "hit_at_5",
                              "hit_at_10", "probe_latency_seconds"],
        "probe": (
            "retrieval-only rank metrics on the original query text; "
            "denominator = queries with positive gold only"
        ),
        "seed": config["seed"],
        "git_commit": current_git_commit(),
        "embedding_endpoint": {
            "url": config["models"]["embedding"]["url"],
            "model": config["models"]["embedding"]["name"],
        },
        "preflight": preflight,
    }
    save_results("part2_scale_probe", all_results, manifest=manifest,
                 analysis=analysis)


def run_part3(config: dict):
    """Part 3: @el: 태그 방식 비교"""
    print("\n" + "=" * 60)
    print("Part 3: Tag Approach Comparison (A vs B vs C)")
    print("=" * 60)

    part_cfg = config["parts"]["part3_tags"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]

    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset)

    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path) as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    all_results = {}
    for approach in part_cfg["approaches"]:
        step_config = {"taxonomy": True, "tags": approach, "prefix": True, "metadata": True}
        print(f"\n  --- Approach {approach} ---")

        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step_config, subset_size, dataset)
        metrics = compute_metrics(results)
        all_results[f"approach_{approach}"] = {"results": results, "metrics": metrics}
        print(f"    Metrics: {metrics}")

    save_results("part3_tags", all_results)


def run_part4(config: dict):
    """Part 4: 일반화 검증"""
    print("\n" + "=" * 60)
    print("Part 4: Generalization (FiQA, Ko-StrategyQA)")
    print("=" * 60)

    part_cfg = config["parts"]["part4_generalization"]

    all_results = {}
    for dataset in part_cfg["datasets"]:
        subset_size = 20_000
        corpus = load_corpus(dataset, subset_size)
        queries, qrels = load_queries(dataset)

        # sampled queries만 사용
        sampled_path = DATA_DIR / "subsets" / dataset / "sampled_queries.json"
        if sampled_path.exists():
            with open(sampled_path) as f:
                sampled = json.load(f)
            query_ids = set(str(qid) for qid in sampled["query_ids"])
            queries = [q for q in queries if str(q["_id"]) in query_ids]

        ref_answers = {}  # FiQA/Ko-StrategyQA는 reference answer 없음 → recall만 평가

        step_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}

        print(f"\n  --- DR-DCI @ {dataset} ---")
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step_config, subset_size, dataset)
        metrics = compute_metrics(results)
        all_results[f"dr-dci_{dataset}"] = {"results": results, "metrics": metrics}
        print(f"    Metrics: {metrics}")

        print(f"\n  --- Hybrid RAG @ {dataset} ---")
        results = run_hybrid(config, corpus, queries, qrels, ref_answers)
        metrics = compute_metrics(results)
        all_results[f"hybrid_{dataset}"] = {"results": results, "metrics": metrics}
        print(f"    Metrics: {metrics}")

    save_results("part4_generalization", all_results)


def run_part5(config: dict, probe_only: bool = False):
    """Part 5: compare pull backends while holding the agent loop fixed."""
    print("\n" + "=" * 60)
    print("Part 5: Pull Backend (Dense vs Hybrid RRF)")
    print("=" * 60)

    part_cfg = config["parts"]["part5_pull_backend"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]
    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset)

    ref_answers = {}
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    if ref_path.exists():
        with open(ref_path) as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    fixed = dict(part_cfg["fixed_augmentations"])
    query_gold = positive_gold_by_query(qrels)
    query_gains = positive_gold_gains_by_query(qrels)
    all_results = {}
    probes = {}
    for backend in part_cfg["backends"]:
        step_config = {**fixed, "pull_backend": backend}
        print(f"\n  --- pull backend: {backend} ---")
        taxonomy, tags, prefix, metadata = load_augmentations(
            dataset, subset_size, step_config
        )
        retriever = build_pull_retriever(
            config, step_config, corpus, prefix=prefix, taxonomy=taxonomy
        )
        # retrieval-only probe(진단 축): 같은 retriever 인스턴스로 agent 실행과
        # backend 외 변인 없이 rank 지표를 먼저 잰다.
        probes[backend] = run_pull_probe(retriever, queries, query_gold,
                                         query_gains=query_gains)
        if probe_only:
            all_results[backend] = {"probe_rows": probes[backend]}
            continue
        results = run_dr_dci(
            config,
            corpus,
            queries,
            qrels,
            ref_answers,
            step_config,
            subset_size,
            dataset,
            cached_retriever=retriever,
        )
        metrics = compute_metrics(results)
        all_results[backend] = {"results": results, "metrics": metrics,
                                "probe_rows": probes[backend]}
        print(f"    Metrics: {metrics}")

    manifest = {
        "hypothesis": (
            "Adding BM25 through RRF improves pull coverage for exact lexical "
            "clues without changing the agent loop."
        ),
        "single_variable": "pull_backend",
        "backends": list(part_cfg["backends"]),
        "fixed_augmentations": fixed,
        "dataset": dataset,
        "subset_size": subset_size,
        "query_count": len(queries),
        "seed": config["seed"],
        "git_commit": current_git_commit(),
    }
    comparison = None
    if not probe_only:
        comparison = compare_paired_results(
            all_results["dense"]["results"],
            all_results["hybrid_rrf"]["results"],
            seed=config["seed"],
        )
    probe_analysis = compare_probe_rows(
        probes["dense"], probes["hybrid_rrf"], seed=config["seed"]
    )
    manifest["probe"] = (
        "retrieval-only rank metrics on the original query text; "
        "denominator = queries with positive gold only"
    )
    manifest["probe_only"] = probe_only
    manifest["embedding_endpoint"] = {
        "url": config["models"]["embedding"]["url"],
        "model": config["models"]["embedding"]["name"],
    }
    save_results(
        "part5_pull_backend",
        all_results,
        manifest=manifest,
        analysis={"dense_vs_hybrid_rrf": comparison,
                  "probe_dense_vs_hybrid_rrf": probe_analysis},
    )


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_results(
    part_name: str,
    results: dict,
    manifest: dict = None,
    analysis: dict = None,
):
    out_dir = RESULTS_DIR / part_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.json"

    # results에서 큰 리스트 제거 (요약만 저장)
    summary = {}
    for key, val in results.items():
        summary[key] = val.get("metrics", "probe_only")

    with open(out_path, "w") as f:
        json.dump(
            {
                "manifest": manifest or {},
                "analysis": analysis or {},
                "summary": summary,
                "full_results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DR-DCI Experiment Runner")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4, 5], help="Run specific part")
    parser.add_argument("--probe-only", action="store_true",
                        help="part5: retrieval-only probe만 실행(agent/judge 생략)")
    parser.add_argument("--embedding-url", default="",
                        help="임베딩 endpoint 오버라이드(결과 manifest에 기록)")
    parser.add_argument("--embedding-model", default="",
                        help="임베딩 모델 오버라이드(결과 manifest에 기록)")
    parser.add_argument("--dataset", default="",
                        help="part5 데이터셋 오버라이드(예: fiqa)")
    parser.add_argument("--subset", type=int, default=-1,
                        help="part5 subset 크기 오버라이드(0=전체 코퍼스)")
    parser.add_argument("--all", action="store_true", help="Run all parts")
    parser.add_argument("--focused", action="store_true",
                        help="part1/2: run the narrow baseline vs taxonomy experiment")
    parser.add_argument("--scale-probe", action="store_true",
                        help="part2: retrieval-only distractor scale probe"
                             "(agent/judge 생략, 임베딩 endpoint만 필요)")
    args = parser.parse_args()

    config = load_config()

    if args.all:
        run_part1(config, focused=args.focused)
        run_part2(config, focused=args.focused)
        run_part3(config)
        run_part4(config)
        if args.embedding_url:
            config["models"]["embedding"]["url"] = args.embedding_url
        if args.embedding_model:
            config["models"]["embedding"]["name"] = args.embedding_model
        if args.dataset:
            config["parts"]["part5_pull_backend"]["dataset"] = args.dataset
        if args.subset >= 0:
            config["parts"]["part5_pull_backend"]["subset"] = (
                args.subset if args.subset > 0 else None
            )
        run_part5(config, probe_only=args.probe_only)
    elif args.part == 1:
        run_part1(config, focused=args.focused)
    elif args.part == 2:
        if args.scale_probe:
            if args.embedding_url:
                config["models"]["embedding"]["url"] = args.embedding_url
            if args.embedding_model:
                config["models"]["embedding"]["name"] = args.embedding_model
            run_part2_scale_probe(config)
        else:
            run_part2(config, focused=args.focused)
    elif args.part == 3:
        run_part3(config)
    elif args.part == 4:
        run_part4(config)
    elif args.part == 5:
        if args.embedding_url:
            config["models"]["embedding"]["url"] = args.embedding_url
        if args.embedding_model:
            config["models"]["embedding"]["name"] = args.embedding_model
        if args.dataset:
            config["parts"]["part5_pull_backend"]["dataset"] = args.dataset
        if args.subset >= 0:
            config["parts"]["part5_pull_backend"]["subset"] = (
                args.subset if args.subset > 0 else None
            )
        run_part5(config, probe_only=args.probe_only)
    else:
        print("Usage: python run_experiment.py --part {1,2,3,4,5} or --all")


if __name__ == "__main__":
    main()
