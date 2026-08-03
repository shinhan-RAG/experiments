"""
DR-DCI Augmentation Experiment Runner
- Part 1: 기법 적층 (TREC-COVID, config subset 크기)
- Part 2: 규모 확장 비교 (controlled distractor scaling)
- Part 3: @el: 태그 방식 비교
- Part 4: 일반화 검증
- Part 5: pull backend 비교 (dense vs hybrid_rrf)
"""

import copy
import json
import os
import random
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
from src.eval import span_metrics
from src.eval.part12_contracts import audit_part12
from src.eval.retrieval_metrics import rank_metrics


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"


def load_config(path: str | None = None):
    """실험 config 로드. path 미지정 시 기본 experiment.yaml.

    법률 파트는 --config config/experiment_legal.yaml 로 기존 config를 건드리지
    않고 병렬 실행한다. 상대경로는 프로젝트 루트 기준으로도 해석한다."""
    if path:
        cfg_path = Path(path)
        if not cfg_path.is_absolute() and not cfg_path.exists():
            cfg_path = BASE_DIR / path
    else:
        cfg_path = CONFIG_DIR / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# aihub 계열(법률)은 BEIR 표준 위치(data/raw/<dataset>)가 아닌
# data/aihub/<variant>에 빌드되어 있다 (prepare_aihub.py 산출물).
AIHUB_DATASET_DIRS = {
    "aihub-full": ("aihub", "full"),
    "aihub-smoke20k": ("aihub", "smoke20k"),
}

# corpus를 갖지 않고 다른 데이터셋의 corpus를 공유하는 질의 전용 데이터셋.
# legal-qa는 자체 corpus 없이 aihub 판례 corpus를 gold로 쓴다(qrels가 precedent
# parent를 가리킴). 질의/qrels/answers는 legal-qa 디렉토리에서 로드한다.
CORPUS_ALIAS = {"legal-qa": "aihub-full"}


def dataset_dir(dataset: str) -> Path:
    """dataset 이름 → 데이터 디렉토리."""
    if dataset in AIHUB_DATASET_DIRS:
        group, variant = AIHUB_DATASET_DIRS[dataset]
        return DATA_DIR / group / variant
    return DATA_DIR / "raw" / dataset


def load_corpus(dataset: str, subset_size: int = None):
    """corpus 로드 (서브셋 적용). corpus 미보유 데이터셋은 CORPUS_ALIAS로 공유."""
    corpus_dataset = CORPUS_ALIAS.get(dataset, dataset)
    raw_dir = dataset_dir(corpus_dataset)
    corpus = []
    with open(raw_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    if subset_size:
        size_key = f"{subset_size // 1000}k"
        parent_subset = raw_dir / f"{size_key}_parent_ids.json"
        if parent_subset.exists():
            # 청크형 corpus: 서브셋은 parent 문서 ID 목록으로 정의된다
            with open(parent_subset, encoding="utf-8") as f:
                parent_ids = set(json.load(f))
            corpus = [doc for doc in corpus
                      if doc.get("parent_id", doc["_id"]) in parent_ids]
        else:
            subset_path = DATA_DIR / "subsets" / corpus_dataset / f"{size_key}.json"
            with open(subset_path, encoding="utf-8") as f:
                doc_ids = set(json.load(f)["doc_ids"])
            corpus = [doc for doc in corpus if doc["_id"] in doc_ids]

    return corpus


def load_queries(dataset: str, variant: str = None):
    """쿼리 + qrels 로드.

    층화 표본(agent_queries_50.jsonl)이 있으면 그것을 질의 집합으로 쓴다 —
    법률 데이터셋의 6천+ 전체 질의 대신 기존 50질의 실험 설계와 맞춘다.

    variant(예: "paraphrase"): queries_{variant}.jsonl 등 변형 질의 파일을
    명시적으로 선택한다. 원 질의 파일을 덮어쓰는 방식 대신 별도 파일을 두어
    원질의/변형 두 세트를 같은 코퍼스·qrels 위에서 병행 비교할 수 있게 한다.
    변형 파일이 없으면 조용히 원본으로 폴백하지 않고 즉시 실패한다 —
    폴백하면 결과 라벨(query_variant)이 실제 질의와 어긋난다.
    """
    raw_dir = dataset_dir(dataset)
    if variant:
        candidates = [
            raw_dir / f"agent_queries_50_{variant}.jsonl",
            raw_dir / f"queries_{variant}.jsonl",
        ]
        query_file = next((p for p in candidates if p.exists()), None)
        if query_file is None:
            raise FileNotFoundError(
                f"query variant '{variant}' missing for {dataset}: "
                f"{' or '.join(p.name for p in candidates)} — "
                f"python scripts/paraphrase_queries.py {dataset} 로 먼저 생성할 것"
            )
        print(f"    Using query variant: {query_file.name}")
    else:
        query_file = raw_dir / "agent_queries_50.jsonl"
        if query_file.exists():
            print(f"    Using stratified query sample: {query_file.name}")
        else:
            query_file = raw_dir / "queries.jsonl"

    queries = []
    with open(query_file, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    qrels = []
    with open(raw_dir / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    return queries, qrels


def load_supporting_spans(dataset: str) -> dict:
    """qa_meta.jsonl에서 질의별 근거 span 로드 (없으면 빈 dict).

    반환: {qid(str): [{"text": ...}, ...]}  — span 기반 지표(coverage/density/f1)용.
    span이 없는 데이터셋(trec-covid 등)은 빈 dict → span 지표는 자동으로 건너뜀.
    """
    # aihub 계열은 QA를 청킹 variant와 무관하게 data/aihub/qa/ 에 공유 저장한다.
    path = DATA_DIR / "raw" / dataset / "qa_meta.jsonl"
    if not path.exists():
        alt = DATA_DIR / "aihub" / "qa" / "qa_meta.jsonl"
        if alt.exists():
            path = alt
    spans = {}
    if not path.exists():
        return spans
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m.get("supporting_spans"):
                spans[str(m["qid"])] = m["supporting_spans"]
    return spans


def parent_map_from_corpus(corpus: list) -> dict:
    """청크형 corpus의 {chunk_id → parent_id}. BEIR corpus면 빈 dict.

    법률 corpus는 `_id`가 청크 ID이고 qrels의 corpus-id는 parent 문서를
    가리키므로, 평가 전에 검색 결과를 parent 수준으로 사상해야 한다."""
    return {
        str(doc["_id"]): str(doc["parent_id"])
        for doc in corpus
        if doc.get("parent_id") and doc["parent_id"] != doc["_id"]
    }


def to_parent_ids(ids: list, parent_map: dict) -> list:
    """ID 목록을 parent 수준으로 사상하고 첫 등장 순서 유지로 중복 제거."""
    if not parent_map:
        return list(ids)
    seen = set()
    out = []
    for doc_id in ids:
        pid = parent_map.get(str(doc_id), str(doc_id))
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


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
    # 리랭커 패리티(F): 기본 off = 기존 동작. on이면 hybrid와 같은 리랭커를
    # pull 경로에도 붙여 "리랭커가 hybrid에만 부착된" 구성요소 불일치를 제거한다.
    pull_reranker = step_config.get(
        "pull_reranker", agent_cfg.get("pull_reranker", False)
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
        # boost 강도(E): 임의 고정값 +0.15를 config/CLI로 노출해 민감도
        # 스윕이 가능하게 한다. 기본값은 기존 실험과 동일한 0.15.
        taxonomy_bonus=float(
            step_config.get("taxonomy_bonus",
                            agent_cfg.get("taxonomy_bonus", 0.15))
        ),
        reranker_url=models["reranker"]["url"] if pull_reranker else None,
        reranker_model=models["reranker"]["name"] if pull_reranker else None,
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
                   query_gold: dict, query_gains: dict = None,
                   parent_map: dict = None, routing: dict = None) -> list:
    """retrieval-only probe: 원 질의 텍스트로 backend를 직접 1회 pull해
    rank 지표(Recall@5/20·Hit@5/10·P@20·nDCG@10)를 잰다 — agent의 질의
    재작성과 독립인 검색 품질 축. LLM/judge 불요(임베딩 endpoint만 필요).

    routing: 질의별 taxonomy_filter 맵 {qid: {"L1": ...}} — taxonomy_routed/
    taxonomy_boosted backend에서 질의 라우팅으로 전달된다."""
    rows = []
    for q in queries:
        qid = str(q["_id"])
        gold = query_gold.get(qid)
        if not gold:
            continue                      # gold 없는 질의는 분모 제외
        gains = query_gains.get(qid) if query_gains else None
        text = q.get("title") or q.get("text", "")
        tax_filter = routing.get(qid) if routing else None
        started = time.perf_counter()
        pulled = retriever.pull(text, taxonomy_filter=tax_filter)
        candidates = pulled["results"] if isinstance(pulled, dict) else pulled
        ranked = to_parent_ids([r["doc_id"] for r in candidates],
                               parent_map or {})
        latency = time.perf_counter() - started
        row = {
            "query_id": qid,
            **rank_metrics(ranked, gold, gains=gains),
            "probe_latency_seconds": latency,
            "ranked_top20": ranked[:20],
        }
        if tax_filter is not None:
            row["routed_filter"] = tax_filter
        rows.append(row)
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
    # 청크형(법률) corpus면 qrels가 parent를 가리키므로 평가를 parent 수준으로
    parent_map = parent_map_from_corpus(corpus)

    # Load schemas for agent prompt
    taxonomy_schema = None
    metadata_schema = None
    if step_config.get("taxonomy"):
        schema_path = CONFIG_DIR / "taxonomy_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                taxonomy_schema = yaml.safe_load(f)
    if step_config.get("metadata"):
        schema_path = CONFIG_DIR / "metadata_schemas" / f"{dataset}.yaml"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
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
    with open(CONFIG_DIR / "judge_prompt.txt", encoding="utf-8") as f:
        judge_prompt = f.read()
    judge = Judge(
        llm_url=models["judge_llm"]["url"],
        model_name=models["judge_llm"]["name"],
        prompt_template=judge_prompt,
    )

    # qrels → query별 gold docs (청크 단위: corpus-id = 정답 청크)
    from collections import defaultdict
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(str(entry["corpus-id"]))
    # 근거 span (있으면 coverage/density/f1 계산; 없으면 건너뜀)
    query_spans = load_supporting_spans(dataset)

    # 에이전트 병렬 실행
    def run_single_query(i, q):
        qid = str(q["_id"])
        query_text = q.get("title") or q.get("text", "")
        started = time.perf_counter()
        result = agent.run(query_text)
        latency_seconds = time.perf_counter() - started
        gold_docs = list(query_gold.get(qid, []))
        gold_recall = Judge.gold_recall_at_workspace(
            to_parent_ids(result["workspace_docs"], parent_map), gold_docs)
        read_recall = Judge.gold_recall_at_workspace(
            to_parent_ids(result["read_docs"], parent_map), gold_docs)
        efficiency = Judge.efficiency(gold_recall, result["pull_count"])
        # 청크단위 검색품질(recall/ndcg 항상) + 근거 span 지표(span 있을 때만)
        span_eval = None
        if gold_docs:
            ranked_chunks = [
                {"chunk_id": did, "text": corpus_dict.get(did, {}).get("text", "")}
                for did in result["workspace_docs"]
            ]
            # parent 단위 corpus(aihub)면 gold가 parent라 청크 recall과 단위가
            # 다르다 → 청크 recall은 생략하고 span 지표(coverage/density)만 잰다.
            chunk_gold = set() if parent_map else set(gold_docs)
            span_eval = span_metrics.evaluate_query(
                ranked_chunks, chunk_gold, query_spans.get(qid)
            )
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
            "span_metrics": span_eval,
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

    _report_span_metrics(results)
    return results


def _report_span_metrics(results: list):
    """질의별 span_metrics를 평균내어 출력 (있을 때만)."""
    have = [r["span_metrics"] for r in results if r.get("span_metrics")]
    if not have:
        return
    agg = span_metrics.aggregate(have)
    keys = ["recall@5", "recall@20", "ndcg@10", "coverage@20", "density@20", "span_f1@20"]
    line = "  ".join(f"{k}={agg[k]}" for k in keys if k in agg)
    print(f"    [청크단위 지표 n={len(have)}] {line}")


def run_hybrid(config: dict, corpus: list, queries: list, qrels: list,
               ref_answers: dict, dataset: str = None) -> list:
    """Hybrid RAG baseline 실행"""
    models = config["models"]
    hybrid_cfg = config["hybrid"]
    corpus_dict = {doc["_id"]: doc for doc in corpus}
    query_spans = load_supporting_spans(dataset) if dataset else {}

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
    parent_map = parent_map_from_corpus(corpus)

    # Judge
    with open(CONFIG_DIR / "judge_prompt.txt", encoding="utf-8") as f:
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
        gold_recall = Judge.gold_recall_at_workspace(
            to_parent_ids(result["retrieved_docs"], parent_map), gold_docs)
        efficiency = Judge.efficiency(gold_recall, result["pull_count"])
        span_eval = None
        if gold_docs:
            ranked_chunks = [
                {"chunk_id": did, "text": corpus_dict.get(did, {}).get("text", "")}
                for did in result["retrieved_docs"]
            ]
            # parent 단위 corpus(aihub)면 gold가 parent라 청크 recall과 단위가
            # 다르다 → 청크 recall은 생략하고 span 지표(coverage/density)만 잰다.
            chunk_gold = set() if parent_map else set(gold_docs)
            span_eval = span_metrics.evaluate_query(
                ranked_chunks, chunk_gold, query_spans.get(qid)
            )
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
            "span_metrics": span_eval,
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

    _report_span_metrics(results)
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


def run_part1(config: dict, *, focused: bool = False, probe_only: bool = False):
    """Part 1: 기법 적층.

    단계 분리 측정(A): 각 arm의 retriever로 retrieval-only probe를 먼저 실행해
    "증강이 pull 단계 검색 품질을 바꿨는가"를 agent 루프의 비결정성과 분리해
    잰다. 최종 Gold R@W와 probe recall이 반대로 움직이면 원인은 agent 단이다.
    """
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
    queries, qrels = load_queries(dataset, variant=config.get("query_variant"))

    # reference answers
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    # 임베딩 캐싱: prefix off / prefix on 두 가지만 빌드.
    # 구성 단일화를 위해 인라인 RetrieverConfig 대신 build_pull_retriever를 쓴다
    # — taxonomy_bonus/pull_reranker 배선이 probe·agent 양쪽에 일관 적용된다.
    retriever_no_prefix = None
    if any(not step.get("prefix") for step in steps):
        print("\n  Building cached embeddings (prefix=off)...")
        retriever_no_prefix = build_pull_retriever(config, {"prefix": False}, corpus)

    retriever_with_prefix = None
    if any(step.get("prefix") for step in steps):
        _, _, prefix_data, _ = load_augmentations(
            dataset, subset_size, {"prefix": True}
        )
        print("  Building cached embeddings (prefix=on)...")
        retriever_with_prefix = build_pull_retriever(
            config, {"prefix": True}, corpus, prefix=prefix_data
        )

    query_gold = positive_gold_by_query(qrels)
    query_gains = positive_gold_gains_by_query(qrels)
    parent_map = parent_map_from_corpus(corpus)

    all_results = {}
    probes = {}
    for step in steps:
        step_name = step["name"]
        print(f"\n  --- {step_name}: {step['description']} ---")

        cached = retriever_with_prefix if step.get("prefix") else retriever_no_prefix
        # probe는 agent와 같은 retriever 인스턴스·taxonomy state를 공유한다.
        # (원 질의를 filter 없이 1회 pull — Part 1 taxonomy는 agent가 스스로
        # filter를 골라야 발동하므로, probe에서 taxonomy arm과 baseline의 probe가
        # 같게 나오는 것 자체가 "차이는 agent 행동"이라는 분리 증거다)
        taxonomy_data, _, _, _ = load_augmentations(dataset, subset_size, step)
        cached.set_taxonomy(taxonomy_data)
        probes[step_name] = run_pull_probe(
            cached, queries, query_gold,
            query_gains=query_gains, parent_map=parent_map,
        )
        if probe_only:
            all_results[step_name] = {"probe_rows": probes[step_name]}
            continue

        results = run_dr_dci(config, corpus, queries, qrels, ref_answers, step, subset_size, dataset, cached_retriever=cached)
        metrics = compute_metrics(results)
        all_results[step_name] = {
            "results": results,
            "metrics": metrics,
            "probe_rows": probes[step_name],
        }

        print(f"    Metrics: {metrics}")

    # baseline을 기준축으로 전 arm paired 비교 — probe(검색단)와 최종 결과
    # (agent 포함)를 각각 잰다. 기존 키 이름(taxonomy_only_minus_baseline)은
    # 일반화된 루프가 그대로 생성하므로 하위 호환이다.
    analysis = {}
    baseline_name = "baseline"
    for step in steps:
        step_name = step["name"]
        if step_name == baseline_name or baseline_name not in probes:
            continue
        analysis[f"probe_{step_name}_minus_baseline"] = compare_probe_rows(
            probes[baseline_name], probes[step_name], seed=config["seed"]
        )
        if (not probe_only and baseline_name in all_results
                and "results" in all_results.get(step_name, {})):
            analysis[f"{step_name}_minus_baseline"] = compare_result_rows(
                all_results[baseline_name]["results"],
                all_results[step_name]["results"],
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
            "probe_only": probe_only,
            "arms": [str(step["name"]) for step in steps],
            "preflight": preflight,
            "seed": config["seed"],
            "replicate_index": config.get("replicate_index", 0),
            "query_count": len(queries),
            "query_variant": config.get("query_variant"),
            "taxonomy_bonus": config["agent"].get("taxonomy_bonus", 0.15),
            "pull_reranker": config["agent"].get("pull_reranker", False),
            "probe": (
                "retrieval-only rank metrics on the original query text; "
                "denominator = queries with positive gold only"
            ),
            "embedding_endpoint": {
                "url": config["models"]["embedding"]["url"],
                "model": config["models"]["embedding"]["name"],
            },
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

    queries, qrels = load_queries(dataset, variant=config.get("query_variant"))
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
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
            "seed": config["seed"],
            "replicate_index": config.get("replicate_index", 0),
            "query_variant": config.get("query_variant"),
            "taxonomy_bonus": config["agent"].get("taxonomy_bonus", 0.15),
            "pull_reranker": config["agent"].get("pull_reranker", False),
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

    queries, qrels = load_queries(dataset, variant=config.get("query_variant"))
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
                                          query_gains=query_gains,
                                          parent_map=parent_map_from_corpus(corpus))
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


def run_part3(config: dict, *, probe_only: bool = False):
    """Part 3: @el: 태그 방식 비교.

    tags는 워크스페이스 grep 전용이라 pull에 관여하지 않는다 — probe가 arm 간
    동일하게 나오는 것 자체가 "approach 간 차이는 전부 agent 단"이라는 구조적
    분리 증거다(같은 인덱스·같은 prefix·같은 taxonomy 공유)."""
    print("\n" + "=" * 60)
    print("Part 3: Tag Approach Comparison (A vs B vs C)")
    print("=" * 60)

    part_cfg = config["parts"]["part3_tags"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]

    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset, variant=config.get("query_variant"))

    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    ref_answers = {}
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    query_gold = positive_gold_by_query(qrels)
    query_gains = positive_gold_gains_by_query(qrels)
    parent_map = parent_map_from_corpus(corpus)

    all_results = {}
    probes = {}
    for approach in part_cfg["approaches"]:
        step_config = {"taxonomy": True, "tags": approach, "prefix": True, "metadata": True}
        print(f"\n  --- Approach {approach} ---")

        # retriever를 러너에서 만들어 probe와 agent가 같은 인스턴스를 공유한다
        taxonomy, tags, prefix, metadata = load_augmentations(
            dataset, subset_size, step_config
        )
        retriever = build_pull_retriever(
            config, step_config, corpus, prefix=prefix, taxonomy=taxonomy
        )
        probes[approach] = run_pull_probe(
            retriever, queries, query_gold,
            query_gains=query_gains, parent_map=parent_map,
        )
        if probe_only:
            all_results[f"approach_{approach}"] = {"probe_rows": probes[approach]}
            continue

        results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                             step_config, subset_size, dataset,
                             cached_retriever=retriever)
        metrics = compute_metrics(results)
        all_results[f"approach_{approach}"] = {
            "results": results,
            "metrics": metrics,
            "probe_rows": probes[approach],
        }
        print(f"    Metrics: {metrics}")

    # approach 간 pairwise 비교 (첫 approach를 기준축으로)
    analysis = {}
    approaches = list(part_cfg["approaches"])
    base = approaches[0] if approaches else None
    for other in approaches[1:]:
        analysis[f"probe_{base}_vs_{other}"] = compare_probe_rows(
            probes[base], probes[other], seed=config["seed"]
        )
        if not probe_only:
            analysis[f"approach_{other}_minus_{base}"] = compare_result_rows(
                all_results[f"approach_{base}"]["results"],
                all_results[f"approach_{other}"]["results"],
                seed=config["seed"],
            )

    save_results(
        "part3_tags",
        all_results,
        manifest={
            "git_commit": current_git_commit(),
            "dataset": dataset,
            "subset_size": subset_size,
            "probe_only": probe_only,
            "arms": [f"approach_{a}" for a in approaches],
            "fixed_augmentations": {"taxonomy": True, "prefix": True, "metadata": True},
            "single_variable": "tags_approach",
            "seed": config["seed"],
            "replicate_index": config.get("replicate_index", 0),
            "query_count": len(queries),
            "query_variant": config.get("query_variant"),
            "taxonomy_bonus": config["agent"].get("taxonomy_bonus", 0.15),
            "pull_reranker": config["agent"].get("pull_reranker", False),
            "embedding_endpoint": {
                "url": config["models"]["embedding"]["url"],
                "model": config["models"]["embedding"]["name"],
            },
        },
        analysis=analysis,
    )


def _sample_generalization_queries(queries: list, qrels: list, corpus: list,
                                   n: int, seed: int) -> list:
    """gold가 (서브)corpus 안에 존재하는 질의만 남겨 결정적으로 n개 표본.

    sampled_queries.json이 없는 데이터셋(ruling-anon/legal-qa 등)에서 기존 50질의
    설계와 맞추기 위한 폴백. gold가 corpus에 없으면 recall 분모에서 빠지므로 제외."""
    from collections import defaultdict
    parent_map = parent_map_from_corpus(corpus)
    present = set(parent_map.values()) if parent_map else {doc["_id"] for doc in corpus}
    gold_by_q = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            gold_by_q[str(entry["query-id"])].add(str(entry["corpus-id"]))
    eligible = [
        q for q in queries
        if gold_by_q.get(str(q["_id"])) and (gold_by_q[str(q["_id"])] & present)
    ]
    random.Random(seed).shuffle(eligible)
    return eligible[:n]


def run_part4(config: dict, *, probe_only: bool = False):
    """Part 4: 일반화 검증.

    datasets 항목은 문자열(fiqa 등, 20K·full-stack 기본) 또는
    {name, subset, augment, queries} dict(법률 데이터셋)를 모두 허용한다.

    단계 분리(A): DR-DCI arm은 retrieval-only probe를 병행하고, hybrid arm은
    파이프라인 출력(retrieved_docs)에 같은 rank 지표를 사후 계산해 검색단끼리
    비교 가능하게 한다(hybrid의 probe_latency는 파이프라인 경유라 비교 불가 —
    latency delta는 무시할 것)."""
    part_cfg = config["parts"]["part4_generalization"]
    seed = config.get("seed", 42)
    names = [e if isinstance(e, str) else e["name"] for e in part_cfg["datasets"]]
    print("\n" + "=" * 60)
    print(f"Part 4: Generalization ({', '.join(names)})")
    print("=" * 60)

    all_results = {}
    analysis = {}
    for entry in part_cfg["datasets"]:
        if isinstance(entry, str):
            dataset, subset_size, augment, n_q = entry, 20_000, True, 50
        else:
            dataset = entry["name"]
            subset_size = entry.get("subset", 20_000)
            augment = entry.get("augment", True)
            n_q = entry.get("queries", 50)

        corpus = load_corpus(dataset, subset_size)
        queries, qrels = load_queries(dataset, variant=config.get("query_variant"))

        # 질의 표본: sampled_queries.json 있으면 사용, 없으면 gold 보유 질의에서 결정적 샘플
        sampled_path = DATA_DIR / "subsets" / dataset / "sampled_queries.json"
        if sampled_path.exists():
            with open(sampled_path) as f:
                sampled = json.load(f)
            query_ids = set(str(qid) for qid in sampled["query_ids"])
            queries = [q for q in queries if str(q["_id"]) in query_ids]
        elif len(queries) > n_q:
            queries = _sample_generalization_queries(queries, qrels, corpus, n_q, seed)
        print(f"  {dataset}: corpus={len(corpus)} queries={len(queries)} "
              f"subset={subset_size} augment={augment}")

        # reference answers (있으면 accuracy judge; 없으면 recall-only)
        ref_answers = {}
        ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
        if ref_path.exists():
            with open(ref_path, encoding="utf-8") as f:
                for item in json.load(f):
                    ref_answers[item["query_id"]] = item["reference_answer"]

        if augment:
            step_config = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}
        else:
            step_config = {"taxonomy": False, "tags": False, "prefix": False, "metadata": False}

        query_gold = positive_gold_by_query(qrels)
        query_gains = positive_gold_gains_by_query(qrels)
        parent_map = parent_map_from_corpus(corpus)

        print(f"\n  --- DR-DCI @ {dataset} ---")
        taxonomy, tags, prefix, metadata = load_augmentations(
            dataset, subset_size, step_config
        )
        retriever = build_pull_retriever(
            config, step_config, corpus, prefix=prefix, taxonomy=taxonomy
        )
        probe_rows = run_pull_probe(
            retriever, queries, query_gold,
            query_gains=query_gains, parent_map=parent_map,
        )
        if probe_only:
            # hybrid는 LLM 파이프라인 경유라 probe-only 배선 검증에서 제외
            all_results[f"dr-dci_{dataset}"] = {"probe_rows": probe_rows}
            continue

        results = run_dr_dci(config, corpus, queries, qrels, ref_answers,
                             step_config, subset_size, dataset,
                             cached_retriever=retriever)
        metrics = compute_metrics(results)
        all_results[f"dr-dci_{dataset}"] = {
            "results": results, "metrics": metrics, "probe_rows": probe_rows,
        }
        print(f"    Metrics: {metrics}")

        print(f"\n  --- Hybrid RAG @ {dataset} ---")
        hybrid_results = run_hybrid(config, corpus, queries, qrels, ref_answers, dataset=dataset)
        hybrid_metrics = compute_metrics(hybrid_results)
        # hybrid 검색단 rank 지표: 파이프라인이 반환한 retrieved_docs를
        # probe와 같은 지표로 사후 채점 — DR-DCI probe와 검색단끼리 비교 가능
        hybrid_rank_rows = []
        for r in hybrid_results:
            if r.get("failed"):
                continue
            gold = query_gold.get(str(r["query_id"]))
            if not gold:
                continue
            ranked = to_parent_ids(r.get("retrieved_docs", []), parent_map)
            hybrid_rank_rows.append({
                "query_id": str(r["query_id"]),
                **rank_metrics(ranked, gold,
                               gains=query_gains.get(str(r["query_id"]))),
                "probe_latency_seconds": 0.0,   # 파이프라인 경유 — 비교 무효
                "ranked_top20": ranked[:20],
            })
        all_results[f"hybrid_{dataset}"] = {
            "results": hybrid_results, "metrics": hybrid_metrics,
            "probe_rows": hybrid_rank_rows,
        }
        print(f"    Metrics: {hybrid_metrics}")

        # paired 비교: 최종 결과(agent 포함)와 검색단(rank 지표) 각각
        analysis[f"hybrid_minus_drdci_{dataset}"] = compare_result_rows(
            all_results[f"dr-dci_{dataset}"]["results"],
            hybrid_results,
            seed=seed,
        )
        if probe_rows and hybrid_rank_rows:
            analysis[f"probe_drdci_vs_hybrid_{dataset}"] = compare_probe_rows(
                probe_rows, hybrid_rank_rows, seed=seed,
            )

    save_results(
        "part4_generalization",
        all_results,
        manifest={
            "git_commit": current_git_commit(),
            "datasets": names,
            "probe_only": probe_only,
            "seed": seed,
            "replicate_index": config.get("replicate_index", 0),
            "query_variant": config.get("query_variant"),
            "taxonomy_bonus": config["agent"].get("taxonomy_bonus", 0.15),
            "pull_reranker": config["agent"].get("pull_reranker", False),
            "hybrid_reranker": True,   # hybrid 파이프라인은 리랭커 상시 부착
            "probe": (
                "dr-dci: retrieval-only pull probe on original query text; "
                "hybrid: rank metrics re-scored from pipeline retrieved_docs "
                "(latency delta is not comparable)"
            ),
            "embedding_endpoint": {
                "url": config["models"]["embedding"]["url"],
                "model": config["models"]["embedding"]["name"],
            },
        },
        analysis=analysis,
    )


TAXONOMY_BACKENDS = {"taxonomy_routed", "taxonomy_boosted", "taxonomy_partitioned"}


def load_query_routing(dataset: str, subset_size: int | None,
                       use_top2: bool = False,
                       min_confidence: float = 0.0) -> dict:
    """질의→카테고리 라우팅 맵 로드 (scripts/build_query_routing.py 산출물).

    반환: {qid: {"L1": ...}} — retriever.pull(taxonomy_filter=...)에 그대로 쓴다.

    라우팅 정확도 병목 완화 옵션(8/2, paper-mixed 결과의 근본 원인 대응):
    - use_top2: L1 값으로 top-2 리스트를 쓴다(OR 매치) — 인접 카테고리 혼동 흡수.
    - min_confidence: 라우터 확신도가 이 값 미만인 질의는 맵에서 제외한다 →
      해당 질의는 taxonomy_filter=None으로 순수 dense 폴백(오라우팅 손실을 no-op으로).
    구버전 라우팅 파일(top2/confidence 없음)에서는 두 옵션 모두 자동 무시된다."""
    size_key = f"{(subset_size or 0) // 1000}k" if subset_size else "full"
    path = DATA_DIR / "routing" / f"{dataset}_{size_key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"query routing missing: {path} — "
            f"python scripts/build_query_routing.py {dataset} --size={subset_size} 로 먼저 생성할 것"
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    routing = {}
    skipped = 0
    for qid, entry in raw["routing"].items():
        if min_confidence and entry.get("confidence", 1.0) < min_confidence:
            skipped += 1
            continue
        l1 = entry["L1"]
        if use_top2 and entry.get("L1_top2"):
            top2 = entry["L1_top2"]
            l1 = top2 if len(top2) > 1 else top2[0]
        routing[str(qid)] = {"L1": l1}
    if skipped:
        print(f"    routing: 확신도 {min_confidence} 미만 {skipped}건 → dense 폴백")
    return routing


def run_part5(config: dict, probe_only: bool = False,
              sweep_bonus: list[float] = None):
    """Part 5: compare pull backends while holding the agent loop fixed.

    sweep_bonus(E): taxonomy_boosted backend에서 soft bonus 가중치를 스윕해
    "+0.15가 과했나 vs taxonomy 신호 자체가 해로운가"를 분리한다. bonus는
    쿼리 시점 덧셈이라 재인덱싱 없이 같은 retriever로 값만 바꿔 probe한다.
    bonus=0.0은 (filter가 있어도 가산 0이므로) 순수 dense와 동치인 기준축."""
    print("\n" + "=" * 60)
    print("Part 5: Pull Backend Comparison")
    print("=" * 60)

    part_cfg = config["parts"]["part5_pull_backend"]
    dataset = part_cfg["dataset"]
    subset_size = part_cfg["subset"]

    if sweep_bonus:
        # 데이터 로드 전에 검증 — 잘못된 조합이면 즉시 실패
        if not probe_only:
            raise ValueError("--sweep-bonus는 --probe-only와 함께만 지원한다 "
                             "(agent 루프는 bonus 스윕 대상이 아님)")
        if "taxonomy_boosted" not in part_cfg["backends"]:
            raise ValueError("--sweep-bonus는 backends에 taxonomy_boosted가 "
                             "있어야 한다 (bonus는 soft boost 경로에만 작용)")

    corpus = load_corpus(dataset, subset_size)
    queries, qrels = load_queries(dataset, variant=config.get("query_variant"))

    ref_answers = {}
    ref_path = DATA_DIR / "reference_answers" / f"{dataset}.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            for item in json.load(f):
                ref_answers[item["query_id"]] = item["reference_answer"]

    fixed = dict(part_cfg["fixed_augmentations"])
    query_gold = positive_gold_by_query(qrels)
    query_gains = positive_gold_gains_by_query(qrels)
    all_results = {}
    probes = {}
    routing = None
    if any(b in TAXONOMY_BACKENDS for b in part_cfg["backends"]):
        routing = load_query_routing(
            dataset, subset_size,
            use_top2=part_cfg.get("routing_use_top2", False),
            min_confidence=part_cfg.get("routing_min_confidence", 0.0),
        )
        print(f"  query routing: {len(routing)}건 로드 "
              f"(top2={part_cfg.get('routing_use_top2', False)}, "
              f"min_conf={part_cfg.get('routing_min_confidence', 0.0)})")

    for backend in part_cfg["backends"]:
        step_config = {**fixed, "pull_backend": backend}
        if backend in TAXONOMY_BACKENDS:
            # 라우팅/부스트는 taxonomy 아티팩트가 인덱스에 실려 있어야 작동한다
            step_config["taxonomy"] = True
        print(f"\n  --- pull backend: {backend} ---")
        taxonomy, tags, prefix, metadata = load_augmentations(
            dataset, subset_size, step_config
        )
        retriever = build_pull_retriever(
            config, step_config, corpus, prefix=prefix, taxonomy=taxonomy
        )
        # retrieval-only probe(진단 축): 같은 retriever 인스턴스로 agent 실행과
        # backend 외 변인 없이 rank 지표를 먼저 잰다.
        probes[backend] = run_pull_probe(
            retriever, queries, query_gold,
            query_gains=query_gains,
            parent_map=parent_map_from_corpus(corpus),
            routing=routing if backend in TAXONOMY_BACKENDS else None,
        )
        if sweep_bonus and backend == "taxonomy_boosted":
            # 같은 인덱스에서 bonus만 바꿔 재측정 (재임베딩 없음)
            default_bonus = retriever.config.taxonomy_bonus
            for b in sweep_bonus:
                retriever.config.taxonomy_bonus = float(b)
                key = f"taxonomy_boosted_b{b:g}"
                print(f"    --- sweep: {key} ---")
                probes[key] = run_pull_probe(
                    retriever, queries, query_gold,
                    query_gains=query_gains,
                    parent_map=parent_map_from_corpus(corpus),
                    routing=routing,
                )
                all_results[key] = {"probe_rows": probes[key]}
            retriever.config.taxonomy_bonus = default_bonus
        if probe_only or backend in TAXONOMY_BACKENDS:
            if not probe_only:
                # agent 루프는 질의별 라우팅을 모른다 — taxonomy backend는
                # 검색 축(probe)만 비교하고 agent 실행은 명시적으로 생략한다.
                print(f"    [notice] {backend}는 probe 전용 — agent 실행 생략")
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
        "replicate_index": config.get("replicate_index", 0),
        "query_variant": config.get("query_variant"),
        "taxonomy_bonus": config["agent"].get("taxonomy_bonus", 0.15),
        "pull_reranker": config["agent"].get("pull_reranker", False),
        "sweep_bonus": sweep_bonus,
        "git_commit": current_git_commit(),
    }
    # 비교는 dense를 기준축으로 나머지 backend 전부와 paired로 잰다
    analysis = {}
    baseline_backend = "dense"
    # bonus 스윕 arm: dense 기준 paired 비교 (b0은 dense와 동치인지 자가검증)
    for key in sorted(probes):
        if key.startswith("taxonomy_boosted_b") and baseline_backend in probes:
            analysis[f"probe_{baseline_backend}_vs_{key}"] = compare_probe_rows(
                probes[baseline_backend], probes[key], seed=config["seed"]
            )
    for backend in part_cfg["backends"]:
        if backend == baseline_backend or baseline_backend not in probes:
            continue
        analysis[f"probe_{baseline_backend}_vs_{backend}"] = compare_probe_rows(
            probes[baseline_backend], probes[backend], seed=config["seed"]
        )
        both_full = all(
            "results" in all_results.get(b, {})
            for b in (baseline_backend, backend)
        )
        if not probe_only and both_full:
            analysis[f"{baseline_backend}_vs_{backend}"] = compare_paired_results(
                all_results[baseline_backend]["results"],
                all_results[backend]["results"],
                seed=config["seed"],
            )
    manifest["probe"] = (
        "retrieval-only rank metrics on the original query text; "
        "denominator = queries with positive gold only"
    )
    manifest["probe_only"] = probe_only
    if routing is not None:
        manifest["routing_options"] = {
            "use_top2": part_cfg.get("routing_use_top2", False),
            "min_confidence": part_cfg.get("routing_min_confidence", 0.0),
            "routed_queries": len(routing),
        }
    manifest["embedding_endpoint"] = {
        "url": config["models"]["embedding"]["url"],
        "model": config["models"]["embedding"]["name"],
    }
    save_results(
        "part5_pull_backend",
        all_results,
        manifest=manifest,
        analysis=analysis,
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
    # multi-seed 연속 실행 등으로 같은 초에 두 번 저장돼도 덮어쓰지 않는다
    dedup = 1
    while out_path.exists():
        dedup += 1
        out_path = out_dir / f"{timestamp}-{dedup}.json"

    # results에서 큰 리스트 제거 (요약만 저장)
    summary = {}
    for key, val in results.items():
        summary[key] = val.get("metrics", "probe_only")

    with open(out_path, "w", encoding="utf-8") as f:
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
                        help="part1/3/4/5: retrieval-only probe만 실행(agent/judge 생략)")
    parser.add_argument("--seeds", default="",
                        help="쉼표 구분 seed 목록(예: 42,43,44) - 각 seed로 전체를 "
                             "반복 실행(replicate)해 run 간 분산을 잰다. "
                             "agent 경로의 분산원은 seed가 아니라 API 비결정성이므로 "
                             "seed는 bootstrap/샘플링 재현성용이다")
    parser.add_argument("--taxonomy-bonus", type=float, default=None,
                        help="taxonomy soft boost 가중치 오버라이드(기본 0.15)")
    parser.add_argument("--sweep-bonus", default="",
                        help="part5 --probe-only 전용: 쉼표 구분 bonus 값 목록"
                             "(예: 0.0,0.05,0.15,0.3) - taxonomy_boosted 스윕")
    parser.add_argument("--pull-reranker", action="store_true",
                        help="DR-DCI pull 경로에 리랭커 부착(hybrid와 패리티). "
                             "기본 off = 기존 동작")
    parser.add_argument("--query-variant", default="",
                        help="질의 변형 세트 선택(예: paraphrase -> "
                             "queries_paraphrase.jsonl). 미지정 시 원 질의")
    parser.add_argument("--embedding-url", default="",
                        help="임베딩 endpoint 오버라이드(결과 manifest에 기록)")
    parser.add_argument("--embedding-model", default="",
                        help="임베딩 모델 오버라이드(결과 manifest에 기록)")
    parser.add_argument("--dataset", default="",
                        help="part5 데이터셋 오버라이드(예: fiqa)")
    parser.add_argument("--subset", type=int, default=-1,
                        help="part5 subset 크기 오버라이드(0=전체 코퍼스)")
    parser.add_argument("--config", default="",
                        help="실험 config 경로 (예: config/experiment_legal.yaml). "
                             "미지정 시 config/experiment.yaml")
    parser.add_argument("--all", action="store_true", help="Run all parts")
    parser.add_argument("--focused", action="store_true",
                        help="part1/2: run the narrow baseline vs taxonomy experiment")
    parser.add_argument("--scale-probe", action="store_true",
                        help="part2: retrieval-only distractor scale probe"
                             "(agent/judge 생략, 임베딩 endpoint만 필요)")
    args = parser.parse_args()

    base_config = load_config(args.config)

    # CLI 오버라이드는 config에 실어 러너·manifest가 같은 값을 보게 한다
    if args.embedding_url:
        base_config["models"]["embedding"]["url"] = args.embedding_url
    if args.embedding_model:
        base_config["models"]["embedding"]["name"] = args.embedding_model
    if args.dataset:
        base_config["parts"]["part5_pull_backend"]["dataset"] = args.dataset
    if args.subset >= 0:
        base_config["parts"]["part5_pull_backend"]["subset"] = (
            args.subset if args.subset > 0 else None
        )
    if args.taxonomy_bonus is not None:
        base_config["agent"]["taxonomy_bonus"] = args.taxonomy_bonus
    if args.pull_reranker:
        base_config["agent"]["pull_reranker"] = True
    if args.query_variant:
        base_config["query_variant"] = args.query_variant

    sweep_bonus = (
        [float(v) for v in args.sweep_bonus.split(",") if v.strip()]
        if args.sweep_bonus else None
    )
    if sweep_bonus and args.part != 5:
        raise SystemExit("--sweep-bonus는 --part 5에서만 지원한다")

    seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.seeds else [base_config.get("seed", 42)]
    )

    def dispatch(config):
        if args.all:
            run_part1(config, focused=args.focused, probe_only=args.probe_only)
            run_part2(config, focused=args.focused)
            run_part3(config, probe_only=args.probe_only)
            run_part4(config, probe_only=args.probe_only)
            run_part5(config, probe_only=args.probe_only, sweep_bonus=sweep_bonus)
        elif args.part == 1:
            run_part1(config, focused=args.focused, probe_only=args.probe_only)
        elif args.part == 2:
            if args.scale_probe:
                run_part2_scale_probe(config)
            else:
                run_part2(config, focused=args.focused)
        elif args.part == 3:
            run_part3(config, probe_only=args.probe_only)
        elif args.part == 4:
            run_part4(config, probe_only=args.probe_only)
        elif args.part == 5:
            run_part5(config, probe_only=args.probe_only, sweep_bonus=sweep_bonus)
        else:
            print("Usage: python run_experiment.py --part {1,2,3,4,5} or --all")

    for replicate_index, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n{'#' * 60}\n# replicate {replicate_index + 1}/{len(seeds)} "
                  f"(seed={seed})\n{'#' * 60}")
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        config["replicate_index"] = replicate_index
        dispatch(config)


if __name__ == "__main__":
    main()
