"""
DR-DCI Augmentation Experiment Runner
- Part 1: 기법 적층 (TREC-COVID, 20K)
- Part 2: 규모 확장 비교 (+ Single Pull baseline)
- Part 3: @el: 태그 방식 비교
- Part 4: 일반화 검증 (retrieval-only, nullable accuracy)

가이드 반영 사항:
- P0-4: augmentation 파일 누락 시 즉시 실패 (--allow-missing-aug로만 우회),
        coverage 검증, requested/active features 기록
- P1-5: condition별 retriever taxonomy state 명시적 설정/해제
- P0-9: workspace/turn 예산을 config에서 주입
- P1-3: read recall 기록
- P2-7: experiment manifest 저장
"""

import json
import os
import subprocess
import sys
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.agent.retriever import PullRetriever, RetrieverConfig
from src.agent.dci_agent import DCIAgent
from src.hybrid.pipeline import HybridRAG
from src.eval.judge import Judge, compute_metrics

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"

ALLOW_MISSING_AUG = False  # --allow-missing-aug로만 True


class AugmentationMissingError(RuntimeError):
    pass


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


def _require_aug_file(path: Path, feature: str):
    """P0-4: 요청된 augmentation 파일이 없으면 즉시 실패."""
    if not path.exists():
        msg = (f"Augmentation '{feature}' was requested but file is missing: {path}\n"
               f"Generate it first (scripts/build_*.py) or rerun with --allow-missing-aug "
               f"to run WITHOUT this feature (condition will be renamed accordingly).")
        if ALLOW_MISSING_AUG:
            print(f"    [WARN] {msg}")
            return False
        raise AugmentationMissingError(msg)
    return True


def _check_coverage(feature: str, keys: set, corpus_ids: set):
    """augmentation이 subset 문서를 100% 커버하는지 검증 (P0-4)."""
    missing = corpus_ids - keys
    if missing:
        msg = (f"Augmentation '{feature}' coverage {len(corpus_ids) - len(missing)}/{len(corpus_ids)} "
               f"({len(missing)} docs missing, e.g. {sorted(missing)[:3]})")
        if ALLOW_MISSING_AUG:
            print(f"    [WARN] {msg}")
        else:
            raise AugmentationMissingError(msg)


def load_augmentations(dataset: str, subset_size: int | None, step_config: dict,
                       corpus_ids: set = None):
    """step 설정에 따라 augmentation 데이터 로드.

    반환: (taxonomy, tags, prefix, metadata, active_features)
    """
    taxonomy = None
    tags = None
    prefix = None
    metadata = None
    active = {"taxonomy": False, "tags": False, "prefix": False, "metadata": False}

    if subset_size is None:
        return taxonomy, tags, prefix, metadata, active

    size_key = f"{subset_size // 1000}k"

    if step_config.get("taxonomy"):
        path = DATA_DIR / "taxonomy" / f"{dataset}_{size_key}.json"
        if _require_aug_file(path, "taxonomy"):
            with open(path, encoding="utf-8") as f:
                taxonomy = json.load(f)
            if corpus_ids:
                _check_coverage("taxonomy", set(taxonomy.keys()), corpus_ids)
            active["taxonomy"] = True

    if step_config.get("tags"):
        approach = step_config["tags"].lower()
        path = DATA_DIR / "tags" / dataset / f"approach_{approach}" / f"{size_key}.json"
        if _require_aug_file(path, f"tags({approach})"):
            with open(path, encoding="utf-8") as f:
                raw_tags = json.load(f)
            tags = {}
            for elem in raw_tags:
                did = elem["doc_id"]
                tags.setdefault(did, []).append(elem)
            if corpus_ids:
                _check_coverage("tags", set(tags.keys()), corpus_ids)
            active["tags"] = step_config["tags"]

    if step_config.get("prefix"):
        path = DATA_DIR / "prefix" / f"{dataset}_{size_key}.json"
        if _require_aug_file(path, "prefix"):
            with open(path, encoding="utf-8") as f:
                prefix = json.load(f)
            if corpus_ids:
                _check_coverage("prefix", set(prefix.keys()), corpus_ids)
            active["prefix"] = True

    if step_config.get("metadata"):
        path = DATA_DIR / "metadata" / f"{dataset}_{size_key}.json"
        if _require_aug_file(path, "metadata"):
            with open(path, encoding="utf-8") as f:
                metadata = json.load(f)
            if corpus_ids:
                _check_coverage("metadata", set(metadata.keys()), corpus_ids)
            active["metadata"] = True

    return taxonomy, tags, prefix, metadata, active


def load_reference_answers(dataset: str) -> dict:
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]
    return ref_answers


def build_query_gold(qrels: list) -> dict:
    from collections import defaultdict
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(str(entry["corpus-id"]))
    return query_gold


def run_dr_dci(config: dict, corpus: list, queries: list, qrels: list,
               ref_answers: dict, step_config: dict, subset_size: int,
               dataset: str, cached_retriever: PullRetriever = None,
               single_pull: bool = False) -> list:
    """DR-DCI 에이전트 실행"""
    models = config["models"]
    agent_cfg = config["agent"]

    corpus_ids = {doc["_id"] for doc in corpus}
    taxonomy, tags, prefix, metadata, active = load_augmentations(
        dataset, subset_size, step_config, corpus_ids=corpus_ids)

    if cached_retriever:
        retriever = cached_retriever
        # P1-5: condition별 taxonomy state를 명시적으로 설정/해제 (누출 방지)
        retriever.set_taxonomy(taxonomy)
        print("    Using cached embeddings")
    else:
        retriever_config = RetrieverConfig(
            embedding_url=models["embedding"]["url"],
            embedding_model=models["embedding"]["name"],
            top_k=agent_cfg["pull_top_k"],
            use_prefix=bool(active.get("prefix")),
        )
        retriever = PullRetriever(retriever_config)
        print("    Indexing corpus...")
        retriever.index(corpus, prefixes=prefix, taxonomy=taxonomy)

    corpus_dict = {doc["_id"]: doc for doc in corpus}

    # Load schemas for agent prompt
    taxonomy_schema = None
    metadata_schema = None
    if active.get("taxonomy"):
        schema_path = CONFIG_DIR / "taxonomy_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                taxonomy_schema = yaml.safe_load(f)
    if active.get("metadata"):
        schema_path = CONFIG_DIR / "metadata_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                metadata_schema = yaml.safe_load(f)

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
        workspace_max_docs=agent_cfg.get("workspace_max_docs", 100),
        min_pulls=agent_cfg.get("min_pulls", 2),
        single_pull=single_pull,
        taxonomy_schema=taxonomy_schema,
        metadata_schema=metadata_schema,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    with open(CONFIG_DIR / "judge_prompt.txt", encoding="utf-8") as f:
        judge_prompt = f.read()
    judge = Judge(
        llm_url=models["judge_llm"]["url"],
        model_name=models["judge_llm"]["name"],
        prompt_template=judge_prompt,
    )

    query_gold = build_query_gold(qrels)

    def run_single_query(i, q):
        qid = str(q["_id"])
        query_text = q.get("title") or q.get("text", "")
        result = agent.run(query_text)
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
            "distinct_pull_queries": result["distinct_pull_queries"],
            "pull_stats": result["pull_stats"],
            "workspace_docs": result["workspace_docs"],
            "read_docs": result["read_docs"],
            "turns": result["turns"],
            "budget_exhausted": result["budget_exhausted"],
            "rule_violations": result["rule_violations"],
            "trace": result["trace"],
            "requested_features": {k: step_config.get(k, False) for k in
                                   ("taxonomy", "tags", "prefix", "metadata")},
            "active_features": active,
            "single_pull": single_pull,
        }

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(run_single_query, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    def judge_single(r):
        if r["query_id"] in ref_answers:
            r["judgment"] = judge.evaluate_accuracy(
                r["query_text"], ref_answers[r["query_id"]], r["answer"]
            )
        else:
            r["judgment"] = "n/a"

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(judge_single, results))

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
    )

    print("    Indexing corpus...")
    pipeline.index(corpus)

    with open(CONFIG_DIR / "judge_prompt.txt", encoding="utf-8") as f:
        judge_prompt = f.read()
    judge = Judge(
        llm_url=models["judge_llm"]["url"],
        model_name=models["judge_llm"]["name"],
        prompt_template=judge_prompt,
    )

    query_gold = build_query_gold(qrels)

    def run_single_hybrid(i, q):
        qid = str(q["_id"])
        query_text = q.get("title") or q.get("text", "")
        result = pipeline.run(query_text)
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
            "retrieved_docs": result["retrieved_docs"],
        }

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_hybrid, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    for r in results:
        r["judgment"] = "n/a"
        if r["query_id"] in ref_answers:
            r["judgment"] = judge.evaluate_accuracy(
                r["query_text"], ref_answers[r["query_id"]], r["answer"]
            )

    return results


def run_part1(config: dict):
    """Part 1: 기법 적층"""
    print("\n" + "=" * 60)
    print("Part 1: Technique Stacking (TREC-COVID, 20K)")
    print("=" * 60)

    part_cfg = config["parts"]["part1_stacking"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]

    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset)
    ref_answers = load_reference_answers(dataset)
    corpus_ids = {doc["_id"] for doc in corpus}

    models = config["models"]
    agent_cfg = config["agent"]

    print("\n  Building cached embeddings (prefix=off)...")
    retriever_no_prefix = PullRetriever(RetrieverConfig(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        top_k=agent_cfg["pull_top_k"],
        use_prefix=False,
    ))
    retriever_no_prefix.index(corpus)

    _, _, prefix_data, _, _ = load_augmentations(
        dataset, subset_size, {"prefix": True}, corpus_ids=corpus_ids)
    print("  Building cached embeddings (prefix=on)...")
    retriever_with_prefix = PullRetriever(RetrieverConfig(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        top_k=agent_cfg["pull_top_k"],
        use_prefix=True,
    ))
    retriever_with_prefix.index(corpus, prefixes=prefix_data)

    all_results = {}
    for step in part_cfg["steps"]:
        step_name = step["name"]
        print(f"\n  --- {step_name}: {step['description']} ---")

        cached = retriever_with_prefix if step.get("prefix") else retriever_no_prefix
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step,
                             subset_size, dataset, cached_retriever=cached)
        metrics = compute_metrics(results)
        all_results[step_name] = {"results": results, "metrics": metrics}

        print(f"    Metrics: {metrics}")

    save_results("part1_stacking", all_results, config)


def run_part2(config: dict):
    """Part 2: 규모 확장 비교 (+ Single Pull baseline)"""
    print("\n" + "=" * 60)
    print("Part 2: Scale Comparison (TREC-COVID)")
    print("=" * 60)

    part_cfg = config["parts"]["part2_scaling"]
    dataset = part_cfg["dataset"]

    queries, qrels = load_queries(dataset)
    ref_answers = load_reference_answers(dataset)

    best_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}

    all_results = {}
    for subset_size in part_cfg["subsets"]:
        size_key = f"{subset_size // 1000}k"
        corpus = load_corpus(dataset, subset_size)

        print(f"\n  --- DR-DCI Best @ {size_key} ---")
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                             best_config, subset_size, dataset)
        all_results[f"dr-dci_{size_key}"] = {"results": results,
                                             "metrics": compute_metrics(results)}
        print(f"    Metrics: {all_results[f'dr-dci_{size_key}']['metrics']}")

        if part_cfg.get("include_single_pull", True):
            print(f"\n  --- Single Pull @ {size_key} ---")
            results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                                 best_config, subset_size, dataset, single_pull=True)
            all_results[f"single-pull_{size_key}"] = {"results": results,
                                                      "metrics": compute_metrics(results)}
            print(f"    Metrics: {all_results[f'single-pull_{size_key}']['metrics']}")

        print(f"\n  --- Hybrid RAG @ {size_key} ---")
        results = run_hybrid(config, corpus, queries, qrels, ref_answers)
        all_results[f"hybrid_{size_key}"] = {"results": results,
                                             "metrics": compute_metrics(results)}
        print(f"    Metrics: {all_results[f'hybrid_{size_key}']['metrics']}")

    save_results("part2_scaling", all_results, config)


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
    ref_answers = load_reference_answers(dataset)

    all_results = {}
    for approach in part_cfg["approaches"]:
        step_config = {"taxonomy": True, "tags": approach, "prefix": True, "metadata": True}
        print(f"\n  --- Approach {approach} ---")

        results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                             step_config, subset_size, dataset)
        all_results[f"approach_{approach}"] = {"results": results,
                                               "metrics": compute_metrics(results)}
        print(f"    Metrics: {all_results[f'approach_{approach}']['metrics']}")

    save_results("part3_tags", all_results, config)


def run_part4(config: dict):
    """Part 4: 일반화 검증 (reference 없으면 retrieval-only, accuracy=None)"""
    print("\n" + "=" * 60)
    print("Part 4: Generalization (FiQA, Ko-StrategyQA)")
    print("=" * 60)

    part_cfg = config["parts"]["part4_generalization"]

    all_results = {}
    for dataset in part_cfg["datasets"]:
        subset_size = 20_000
        corpus = load_corpus(dataset, subset_size)
        queries, qrels = load_queries(dataset)

        sampled_path = DATA_DIR / "subsets" / dataset / "sampled_queries.json"
        if sampled_path.exists():
            with open(sampled_path, encoding="utf-8") as f:
                sampled = json.load(f)
            query_ids = set(str(qid) for qid in sampled["query_ids"])
            queries = [q for q in queries if str(q["_id"]) in query_ids]

        # reference answer가 있으면 사용, 없으면 retrieval-only (accuracy=None)
        ref_answers = load_reference_answers(dataset)
        if not ref_answers:
            print(f"  [NOTE] No reference answers for {dataset}: retrieval-only evaluation")

        step_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}

        print(f"\n  --- DR-DCI @ {dataset} ---")
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                             step_config, subset_size, dataset)
        all_results[f"dr-dci_{dataset}"] = {"results": results,
                                            "metrics": compute_metrics(results)}
        print(f"    Metrics: {all_results[f'dr-dci_{dataset}']['metrics']}")

        print(f"\n  --- Hybrid RAG @ {dataset} ---")
        results = run_hybrid(config, corpus, queries, qrels, ref_answers)
        all_results[f"hybrid_{dataset}"] = {"results": results,
                                            "metrics": compute_metrics(results)}
        print(f"    Metrics: {all_results[f'hybrid_{dataset}']['metrics']}")

    save_results("part4_generalization", all_results, config)


def build_manifest(config: dict) -> dict:
    """P2-7: 실행 환경 manifest."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  cwd=BASE_DIR, timeout=10).stdout.strip()
        except Exception:
            return None

    return {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "config_snapshot": config,
        "allow_missing_aug": ALLOW_MISSING_AUG,
    }


def save_results(part_name: str, results: dict, config: dict = None):
    out_dir = RESULTS_DIR / part_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.json"

    summary = {key: val["metrics"] for key, val in results.items()}
    payload = {"summary": summary, "full_results": results}
    if config is not None:
        payload["manifest"] = build_manifest(config)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved: {out_path}")
    return out_path


def main():
    global ALLOW_MISSING_AUG
    parser = argparse.ArgumentParser(description="DR-DCI Experiment Runner")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], help="Run specific part")
    parser.add_argument("--all", action="store_true", help="Run all parts")
    parser.add_argument("--allow-missing-aug", action="store_true",
                        help="Continue with missing augmentation files (feature disabled + warning)")
    args = parser.parse_args()

    ALLOW_MISSING_AUG = args.allow_missing_aug

    config = load_config()

    if args.all:
        run_part1(config)
        run_part2(config)
        run_part3(config)
        run_part4(config)
    elif args.part == 1:
        run_part1(config)
    elif args.part == 2:
        run_part2(config)
    elif args.part == 3:
        run_part3(config)
    elif args.part == 4:
        run_part4(config)
    else:
        print("Usage: python run_experiment.py --part {1,2,3,4} or --all")


if __name__ == "__main__":
    main()
