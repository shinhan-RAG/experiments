"""
6단계: Contextual Prefix 생성
- 각 문서 청크에 50-100 토큰의 문맥 요약 prefix 부여
- 임베딩 품질 향상 (Pull retriever 정밀도 개선)
- Format: "[prefix] original_text"
"""

import json
import time
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "prefix"

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """Generate a brief contextual prefix (50-100 tokens) for the given document chunk.
The prefix should:
- Summarize the document's topic and key context
- Help a retriever understand what this chunk is about
- Be prepended before the chunk for embedding

Format: Return ONLY the prefix text, nothing else."""

USER_TEMPLATE = """Document title: {title}
Document text (first 500 chars):
{text}

Generate a contextual prefix for this document."""


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def call_llm(prompt: str, max_retries: int = 3) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 150,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""


def build_prefix(dataset: str = "trec-covid", subset_size: int = 10_000):
    print(f"=== Building Contextual Prefix for {dataset} ({subset_size // 1000}K) ===")

    # 서브셋 doc IDs
    subset_path = SUBSET_DIR / dataset / f"{subset_size // 1000}k.json"
    with open(subset_path) as f:
        subset_info = json.load(f)
    doc_ids = set(subset_info["doc_ids"])

    # corpus 로드
    corpus = load_jsonl(RAW_DIR / dataset / "corpus.jsonl")
    corpus_subset = [doc for doc in corpus if doc["_id"] in doc_ids]
    print(f"  Docs to process: {len(corpus_subset)}")

    results = {}
    for i, doc in enumerate(corpus_subset):
        title = doc.get("title", "")
        text = doc.get("text", "")[:500]

        prompt = USER_TEMPLATE.format(title=title, text=text)
        prefix = call_llm(prompt)
        results[doc["_id"]] = prefix

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(corpus_subset)}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    prefix_lengths = [len(p.split()) for p in results.values() if p]
    avg_len = sum(prefix_lengths) / len(prefix_lengths) if prefix_lengths else 0

    print(f"\n  Generated {len(results)} prefixes")
    print(f"  Avg prefix length: {avg_len:.1f} words")
    print(f"  Saved to: {out_path}")

    # 샘플
    sample_id = list(results.keys())[0]
    print(f"\n  [Sample] doc_id={sample_id}")
    print(f"  Prefix: {results[sample_id][:100]}...")


if __name__ == "__main__":
    import sys
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_prefix("trec-covid", size)
