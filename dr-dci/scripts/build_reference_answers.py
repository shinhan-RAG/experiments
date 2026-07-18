"""
3단계: Reference Answer 생성
- TREC-COVID 50 쿼리에 대해 gold docs(score=2) 텍스트를 기반으로
- LLM으로 reference answer 생성
- H200 서버에서 Qwen3-8B 또는 NIM API로 실행
"""

import json
import time
import requests
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "reference_answers"

# H200 서버의 vLLM endpoint (실행 시 환경에 맞게 수정)
VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """You are an expert medical researcher. Given a query and relevant research paper excerpts, generate a comprehensive reference answer.

Requirements:
- Answer must be grounded in the provided documents
- Be specific and cite key findings
- 2-4 sentences, concise but complete
- If documents provide conflicting information, mention both perspectives"""

USER_TEMPLATE = """Query: {query}

Relevant Documents (highly relevant, score=2):
{docs_text}

Generate a reference answer based on the above documents."""


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def call_llm(prompt: str, system: str = SYSTEM_PROMPT, max_retries: int = 3) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
            else:
                raise


def build_reference_answers():
    print("=== Building Reference Answers for TREC-COVID ===")

    queries = load_jsonl(RAW_DIR / "trec-covid" / "queries.jsonl")
    qrels = load_jsonl(RAW_DIR / "trec-covid" / "qrels.jsonl")
    corpus = load_jsonl(RAW_DIR / "trec-covid" / "corpus.jsonl")

    # corpus를 dict로
    corpus_dict = {doc["_id"]: doc for doc in corpus}
    print(f"  Corpus loaded: {len(corpus_dict)} docs")

    # query별 score=2 (highly relevant) doc IDs
    query_gold = defaultdict(list)
    for entry in qrels:
        if entry["score"] == 2:
            query_gold[str(entry["query-id"])].append(entry["corpus-id"])

    print(f"  Queries with score=2 docs: {len(query_gold)}")

    # 쿼리 dict
    query_dict = {str(q["_id"]): q for q in queries}

    results = []
    for i, (qid, doc_ids) in enumerate(sorted(query_gold.items())):
        if qid not in query_dict:
            continue

        query_text = query_dict[qid].get("title") or query_dict[qid].get("text", "")

        # gold docs 텍스트 수집 (최대 5개, 각 500자 제한)
        docs_texts = []
        for did in doc_ids[:5]:
            if did in corpus_dict:
                doc = corpus_dict[did]
                title = doc.get("title", "")
                text = doc.get("text", "")[:500]
                docs_texts.append(f"[{did}] {title}\n{text}")

        if not docs_texts:
            continue

        docs_combined = "\n\n---\n\n".join(docs_texts)
        prompt = USER_TEMPLATE.format(query=query_text, docs_text=docs_combined)

        print(f"  [{i+1}/{len(query_gold)}] Query: {query_text[:60]}...")

        try:
            answer = call_llm(prompt)
        except Exception as e:
            print(f"    FAILED: {e}")
            answer = ""

        results.append({
            "query_id": qid,
            "query_text": query_text,
            "gold_doc_ids": doc_ids,
            "gold_doc_count": len(doc_ids),
            "reference_answer": answer,
        })

        # rate limiting
        time.sleep(0.5)

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "trec-covid.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Generated {len(results)} reference answers")
    print(f"  Saved to: {out_path}")

    # 샘플 출력
    if results:
        sample = results[0]
        print(f"\n  [Sample]")
        print(f"  Query: {sample['query_text']}")
        print(f"  Answer: {sample['reference_answer'][:200]}...")


if __name__ == "__main__":
    build_reference_answers()
