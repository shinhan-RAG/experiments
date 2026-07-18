"""
4단계: Document Taxonomy 생성
- 각 문서에 L1/L2/L3 분류 태그 부여
- LLM으로 자동 분류 (H200 Qwen3-8B)
- TREC-COVID: 의학 논문 → 주제 기반 분류
"""

import json
import time
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "taxonomy"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

# TREC-COVID L1/L2 체계
TAXONOMY = {
    "L1": ["Treatment", "Diagnosis", "Prevention", "Mechanism", "Epidemiology", "Other"],
    "L2": {
        "Treatment": ["Drug", "Vaccine", "Therapy", "Clinical_Trial"],
        "Diagnosis": ["Testing", "Imaging", "Symptoms", "Biomarker"],
        "Prevention": ["Public_Health", "PPE", "Social_Distancing", "Hygiene"],
        "Mechanism": ["Virology", "Immunology", "Pathogenesis", "Genetics"],
        "Epidemiology": ["Transmission", "Mortality", "Risk_Factors", "Modeling"],
        "Other": ["Policy", "Mental_Health", "Economics", "General"],
    }
}

SYSTEM_PROMPT = f"""Classify the given medical document into a taxonomy.

L1 categories: {', '.join(TAXONOMY['L1'])}
L2 categories (per L1): {json.dumps(TAXONOMY['L2'])}

Respond in JSON format only:
{{"L1": "...", "L2": "...", "L3": "brief_topic_keyword"}}

L3 is a free-form keyword (1-3 words) describing the specific topic."""

USER_TEMPLATE = """Title: {title}
Text (first 300 chars): {text}

Classify this document."""


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
        "max_tokens": 100,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # JSON 파싱 시도
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return {"L1": "Other", "L2": "General", "L3": "unknown"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"L1": "Other", "L2": "General", "L3": "unknown"}


def build_taxonomy(dataset: str = "trec-covid", subset_size: int = 10_000):
    """10K 서브셋 문서에 대해 taxonomy 생성"""
    print(f"=== Building Taxonomy for {dataset} ({subset_size // 1000}K) ===")

    # 서브셋 doc IDs 로드
    subset_path = SUBSET_DIR / dataset / f"{subset_size // 1000}k.json"
    with open(subset_path) as f:
        subset_info = json.load(f)
    doc_ids = set(subset_info["doc_ids"])
    print(f"  Subset docs: {len(doc_ids)}")

    # corpus 로드
    corpus = load_jsonl(RAW_DIR / dataset / "corpus.jsonl")
    corpus_subset = [doc for doc in corpus if doc["_id"] in doc_ids]
    print(f"  Loaded {len(corpus_subset)} docs from corpus")

    # 배치 처리
    results = {}
    batch_size = 100
    for i, doc in enumerate(corpus_subset):
        title = doc.get("title", "")
        text = doc.get("text", "")[:300]

        prompt = USER_TEMPLATE.format(title=title, text=text)
        taxonomy = call_llm(prompt)
        results[doc["_id"]] = taxonomy

        if (i + 1) % batch_size == 0:
            print(f"  Progress: {i + 1}/{len(corpus_subset)}")

        time.sleep(0.1)  # rate limiting

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    l1_counts = {}
    for tax in results.values():
        l1 = tax.get("L1", "Other")
        l1_counts[l1] = l1_counts.get(l1, 0) + 1

    print(f"\n  L1 distribution:")
    for l1, count in sorted(l1_counts.items(), key=lambda x: -x[1]):
        print(f"    {l1}: {count} ({count/len(results)*100:.1f}%)")

    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    sizes = [10_000, 50_000, 110_000] if "--all" in sys.argv else [10_000]
    for size in sizes:
        build_taxonomy("trec-covid", size)
