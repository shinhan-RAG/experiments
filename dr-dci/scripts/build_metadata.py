"""
7단계: Metadata 생성
- 문서별 구조화된 메타데이터 추출
- Pull pre-filter에 사용 (topic, entities, year 등)
"""

import json
import time
import re
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "metadata"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """Extract structured metadata from the given medical document.

Return JSON with these fields:
{
  "topic": "primary topic (1-3 words)",
  "entities": ["key entity 1", "key entity 2", ...],  // max 5
  "year": 2020,  // publication year if detectable, else null
  "study_type": "one of: review, clinical_trial, observational, case_report, meta_analysis, commentary, other",
  "population": "target population if mentioned (e.g., 'elderly', 'children', 'healthcare workers'), else null"
}

Return ONLY valid JSON."""


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def call_llm(prompt: str, max_retries: int = 3) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, Exception):
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return {
                    "topic": "unknown",
                    "entities": [],
                    "year": None,
                    "study_type": "other",
                    "population": None,
                }


def build_metadata(dataset: str = "trec-covid", subset_size: int = 10_000):
    print(f"=== Building Metadata for {dataset} ({subset_size // 1000}K) ===")

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
        text = doc.get("text", "")[:400]

        prompt = f"Title: {title}\nText: {text}"
        metadata = call_llm(prompt)
        results[doc["_id"]] = metadata

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(corpus_subset)}")

        time.sleep(0.05)

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    study_types = {}
    all_topics = {}
    for meta in results.values():
        st = meta.get("study_type", "other")
        study_types[st] = study_types.get(st, 0) + 1
        topic = meta.get("topic", "unknown")
        all_topics[topic] = all_topics.get(topic, 0) + 1

    print(f"\n  Study type distribution:")
    for st, count in sorted(study_types.items(), key=lambda x: -x[1]):
        print(f"    {st}: {count}")

    print(f"\n  Top 10 topics:")
    for topic, count in sorted(all_topics.items(), key=lambda x: -x[1])[:10]:
        print(f"    {topic}: {count}")

    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_metadata("trec-covid", size)
