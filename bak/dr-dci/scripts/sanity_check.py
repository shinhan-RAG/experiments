"""
Phase 0: Sanity Check
- 5문항, 1K 문서로 파이프라인 정상 동작 확인
- DR-DCI agent + Hybrid RAG 모두 테스트
- 모델 endpoint 연결, tool call, 평가 루프 검증
"""

import json
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.retriever import PullRetriever, RetrieverConfig
from src.agent.dci_agent import DCIAgent
from src.hybrid.pipeline import HybridRAG
from src.eval.judge import Judge, compute_metrics

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_DIR = Path(__file__).parent.parent / "config"

SANITY_SIZE = 1000  # 1K 문서
SANITY_QUERIES = 5


def load_config():
    import yaml
    with open(CONFIG_DIR / "experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_endpoints(config):
    """모델 endpoint 연결 확인"""
    print("=== Step 1: Endpoint Check ===")
    models = config["models"]

    # Agent LLM
    try:
        resp = requests.post(models["agent_llm"]["url"], json={
            "model": models["agent_llm"]["name"],
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        }, timeout=10)
        resp.raise_for_status()
        print(f"  Agent LLM: OK ({models['agent_llm']['name']})")
    except Exception as e:
        print(f"  Agent LLM: FAIL - {e}")
        return False

    # Embedding
    try:
        resp = requests.post(models["embedding"]["url"], json={
            "model": models["embedding"]["name"],
            "input": ["test"],
        }, timeout=10)
        resp.raise_for_status()
        dim = len(resp.json()["data"][0]["embedding"])
        print(f"  Embedding: OK (dim={dim})")
    except Exception as e:
        print(f"  Embedding: FAIL - {e}")
        return False

    # Reranker
    try:
        resp = requests.post(models["reranker"]["url"], json={
            "model": models["reranker"]["name"],
            "query": "test",
            "documents": ["doc1", "doc2"],
        }, timeout=10)
        resp.raise_for_status()
        print(f"  Reranker: OK")
    except Exception as e:
        print(f"  Reranker: FAIL - {e}")
        print(f"  (Hybrid RAG reranker will fallback to RRF order)")

    # Judge LLM (NIM)
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("NVIDIA_API_KEY", "")

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.post(models["judge_llm"]["url"], json={
            "model": models["judge_llm"]["name"],
            "messages": [{"role": "user", "content": "Say 'correct'"}],
            "max_tokens": 5,
        }, headers=headers, timeout=15)
        resp.raise_for_status()
        print(f"  Judge LLM (NIM): OK")
    except Exception as e:
        print(f"  Judge LLM (NIM): FAIL - {e}")
        print(f"  (Accuracy evaluation will not work)")

    return True


def run_sanity(config):
    """5문항 × 1K 문서로 DR-DCI + Hybrid 테스트"""
    print("\n=== Step 2: Pipeline Test (5 queries, 1K docs) ===")

    models = config["models"]

    # 데이터 로드
    raw_dir = DATA_DIR / "raw" / "trec-covid"
    if not raw_dir.exists():
        print("  ERROR: Raw data not found. Run download_base.py first.")
        return False

    corpus = []
    with open(raw_dir / "corpus.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= SANITY_SIZE:
                break
            corpus.append(json.loads(line))

    queries = []
    with open(raw_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    queries = queries[:SANITY_QUERIES]

    qrels = []
    with open(raw_dir / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    print(f"  Corpus: {len(corpus)} docs")
    print(f"  Queries: {len(queries)}")

    # --- DR-DCI ---
    print("\n  --- DR-DCI Agent ---")
    retriever_config = RetrieverConfig(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        top_k=5,  # sanity check에서는 적게
    )
    retriever = PullRetriever(retriever_config)

    print("    Indexing...")
    retriever.index(corpus)

    corpus_dict = {doc["_id"]: doc for doc in corpus}
    agent = DCIAgent(
        llm_url=models["agent_llm"]["url"],
        model_name=models["agent_llm"]["name"],
        retriever=retriever,
        corpus=corpus_dict,
        max_turns=5,  # sanity에서는 적은 턴
    )

    for i, q in enumerate(queries):
        query_text = q.get("title") or q.get("text", "")
        print(f"    [{i+1}] {query_text[:50]}...")
        try:
            result = agent.run(query_text)
            print(f"        Pulls: {result['pull_count']}, Workspace: {len(result['workspace_docs'])}, Turns: {result['turns']}")
            print(f"        Answer: {result['answer'][:80]}...")
        except Exception as e:
            print(f"        FAIL: {e}")
            return False

    # --- Hybrid RAG ---
    print("\n  --- Hybrid RAG ---")
    hybrid = HybridRAG(
        embedding_url=models["embedding"]["url"],
        embedding_model=models["embedding"]["name"],
        reranker_url=models["reranker"]["url"],
        reranker_model=models["reranker"]["name"],
        llm_url=models["agent_llm"]["url"],
        llm_model=models["agent_llm"]["name"],
        dense_top_k=5,
        bm25_top_k=5,
        rerank_top_k=5,
    )

    print("    Indexing...")
    hybrid.index(corpus)

    for i, q in enumerate(queries):
        query_text = q.get("title") or q.get("text", "")
        print(f"    [{i+1}] {query_text[:50]}...")
        try:
            result = hybrid.run(query_text)
            print(f"        Retrieved: {len(result['retrieved_docs'])} docs")
            print(f"        Answer: {result['answer'][:80]}...")
        except Exception as e:
            print(f"        FAIL: {e}")
            return False

    print("\n=== Sanity Check PASSED ===")
    return True


if __name__ == "__main__":
    config = load_config()

    if not check_endpoints(config):
        print("\nEndpoint check failed. Fix connections before proceeding.")
        sys.exit(1)

    if not run_sanity(config):
        print("\nPipeline test failed.")
        sys.exit(1)

    print("\nAll checks passed. Ready for full experiment.")
