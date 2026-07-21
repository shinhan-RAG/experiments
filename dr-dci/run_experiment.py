"""
DR-DCI Augmentation Experiment Runner
- Part 1: 기법 적층 (TREC-COVID, 10K)
- Part 2: 규모 확장 비교
- Part 3: @el: 태그 방식 비교
- Part 4: 일반화 검증
"""

import json
import os
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


def load_config():
    with open(CONFIG_DIR / "experiment.yaml") as f:
        return yaml.safe_load(f)


def load_corpus(dataset: str, subset_size: int = None):
    """corpus 로드 (서브셋 적용)"""
    raw_dir = DATA_DIR / "raw" / dataset
    corpus = []
    with open(raw_dir / "corpus.jsonl") as f:
        for line in f:
            corpus.append(json.loads(line))

    if subset_size:
        subset_path = DATA_DIR / "subsets" / dataset / f"{subset_size // 1000}k.json"
        with open(subset_path) as f:
            doc_ids = set(json.load(f)["doc_ids"])
        corpus = [doc for doc in corpus if doc["_id"] in doc_ids]

    return corpus


def load_queries(dataset: str):
    """쿼리 + qrels 로드"""
    raw_dir = DATA_DIR / "raw" / dataset
    queries = []
    with open(raw_dir / "queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))

    qrels = []
    with open(raw_dir / "qrels.jsonl") as f:
        for line in f:
            qrels.append(json.loads(line))

    return queries, qrels


def load_augmentations(dataset: str, subset_size: int | None, step_config: dict):
    """step 설정에 따라 augmentation 데이터 로드"""
    taxonomy = None
    tags = None
    prefix = None
    metadata = None

    if subset_size is None:
        return taxonomy, tags, prefix, metadata

    size_key = f"{subset_size // 1000}k"

    if step_config.get("taxonomy"):
        path = DATA_DIR / "taxonomy" / f"{dataset}_{size_key}.json"
        if path.exists():
            with open(path) as f:
                taxonomy = json.load(f)

    if step_config.get("tags"):
        approach = step_config["tags"].lower()
        path = DATA_DIR / "tags" / dataset / f"approach_{approach}" / f"{size_key}.json"
        if path.exists():
            with open(path) as f:
                raw_tags = json.load(f)
            # doc_id별로 그룹핑
            tags = {}
            for elem in raw_tags:
                did = elem["doc_id"]
                if did not in tags:
                    tags[did] = []
                tags[did].append(elem)

    if step_config.get("prefix"):
        path = DATA_DIR / "prefix" / f"{dataset}_{size_key}.json"
        if path.exists():
            with open(path) as f:
                prefix = json.load(f)

    if step_config.get("metadata"):
        path = DATA_DIR / "metadata" / f"{dataset}_{size_key}.json"
        if path.exists():
            with open(path) as f:
                metadata = json.load(f)

    return taxonomy, tags, prefix, metadata


def run_dr_dci(config: dict, corpus: list, queries: list, qrels: list,
               ref_answers: dict, step_config: dict, subset_size: int,
               dataset: str, cached_retriever: PullRetriever = None) -> list:
    """DR-DCI 에이전트 실행"""
    models = config["models"]
    agent_cfg = config["agent"]

    taxonomy, tags, prefix, metadata = load_augmentations(dataset, subset_size, step_config)

    if cached_retriever:
        retriever = cached_retriever
        # taxonomy가 있으면 retriever에 설정 (soft boost용)
        if taxonomy:
            retriever.doc_taxonomy = {did: t for did, t in taxonomy.items()}
        print("    Using cached embeddings")
    else:
        retriever_config = RetrieverConfig(
            embedding_url=models["embedding"]["url"],
            embedding_model=models["embedding"]["name"],
            top_k=agent_cfg["pull_top_k"],
            use_prefix=step_config.get("prefix", False),
        )
        retriever = PullRetriever(retriever_config)
        print("    Indexing corpus...")
        retriever.index(corpus, prefixes=prefix, taxonomy=taxonomy)

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
        result = agent.run(query_text)
        gold_docs = list(query_gold.get(qid, []))
        gold_recall = Judge.gold_recall_at_workspace(result["workspace_docs"], gold_docs)
        efficiency = Judge.efficiency(gold_recall, result["pull_count"])
        print(f"    [{i+1}/{len(queries)}] {query_text[:50]}...")
        return {
            "query_id": qid,
            "query_text": query_text,
            "answer": result["answer"],
            "gold_recall": gold_recall,
            "efficiency": efficiency,
            "pull_count": result["pull_count"],
            "workspace_docs": result["workspace_docs"],
            "turns": result["turns"],
        }

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(run_single_query, i, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Judge 병렬 실행
    def judge_single(r):
        if r["query_id"] in ref_answers:
            r["judgment"] = judge.evaluate_accuracy(
                r["query_text"], ref_answers[r["query_id"]], r["answer"]
            )
        else:
            r["judgment"] = "n/a"

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(judge_single, results)

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

    # Judge 순차 실행
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
    print("Part 1: Technique Stacking (TREC-COVID, 10K)")
    print("=" * 60)

    part_cfg = config["parts"]["part1_stacking"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]

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

    print("\n  Building cached embeddings (prefix=off)...")
    retriever_no_prefix = PullRetriever(RetrieverConfig(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        top_k=agent_cfg["pull_top_k"],
        use_prefix=False,
    ))
    retriever_no_prefix.index(corpus)

    _, _, prefix_data, _ = load_augmentations(dataset, subset_size, {"prefix": True})
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
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step, subset_size, dataset, cached_retriever=cached)
        metrics = compute_metrics(results)
        all_results[step_name] = {"results": results, "metrics": metrics}

        print(f"    Metrics: {metrics}")

    save_results("part1_stacking", all_results)


def run_part2(config: dict):
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

    # DR-DCI best config from Part 1 (stack_all)
    best_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}

    all_results = {}
    for subset_size in part_cfg["subsets"]:
        size_key = f"{subset_size // 1000}k"
        corpus = load_corpus(dataset, subset_size)

        print(f"\n  --- DR-DCI Best @ {size_key} ---")
        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, best_config, subset_size, dataset)
        metrics = compute_metrics(results)
        all_results[f"dr-dci_{size_key}"] = {"results": results, "metrics": metrics}
        print(f"    Metrics: {metrics}")

        print(f"\n  --- Hybrid RAG @ {size_key} ---")
        results = run_hybrid(config, corpus, queries, qrels, ref_answers)
        metrics = compute_metrics(results)
        all_results[f"hybrid_{size_key}"] = {"results": results, "metrics": metrics}
        print(f"    Metrics: {metrics}")

    save_results("part2_scaling", all_results)


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


def save_results(part_name: str, results: dict):
    out_dir = RESULTS_DIR / part_name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}.json"

    # results에서 큰 리스트 제거 (요약만 저장)
    summary = {}
    for key, val in results.items():
        summary[key] = val["metrics"]

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "full_results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DR-DCI Experiment Runner")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4], help="Run specific part")
    parser.add_argument("--all", action="store_true", help="Run all parts")
    args = parser.parse_args()

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
