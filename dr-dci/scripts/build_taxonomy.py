"""
4단계: Document Taxonomy 생성 (Schema-driven)
- 데이터셋별 YAML 스키마에 따라 L1/L2/L3 분류
- 병렬 처리 (async, 32 concurrent)
"""

import json
import yaml
from pathlib import Path
from utils import run_batch_llm, parse_llm_content

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "taxonomy"
SCHEMA_DIR = Path(__file__).parent.parent / "config" / "taxonomy_schemas"


def load_schema(dataset: str) -> dict:
    schema_path = SCHEMA_DIR / f"{dataset}.yaml"
    with open(schema_path) as f:
        return yaml.safe_load(f)


def build_system_prompt(schema: dict) -> str:
    domain = schema["domain"]
    l1_list = schema["L1"]
    l2_map = schema["L2"]

    return f"""Classify the given {domain} document into a taxonomy.

L1 categories: {', '.join(l1_list)}
L2 categories (per L1): {json.dumps(l2_map)}

Respond in JSON format only:
{{"L1": "...", "L2": "...", "L3": "brief_topic_keyword"}}

L3 is a free-form keyword (1-3 words) describing the specific topic."""


USER_TEMPLATE = "Title: {title}\nText (first 300 chars): {text}\n\nClassify this document."


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_taxonomy(dataset: str = "trec-covid", subset_size: int = 10_000):
    """서브셋 문서에 대해 taxonomy 생성"""
    print(f"=== Building Taxonomy for {dataset} ({subset_size // 1000}K) ===")

    # 이미 존재하면 스킵
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    if out_path.exists():
        print(f"  Already exists: {out_path}, skipping.")
        return

    # 스키마 로드
    schema = load_schema(dataset)
    system_prompt = build_system_prompt(schema)
    l1_list = schema["L1"]
    print(f"  Domain: {schema['domain']}, L1 categories: {l1_list}")

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

    # 프롬프트 준비
    prompts = []
    for doc in corpus_subset:
        title = doc.get("title", "")
        text = doc.get("text", "")[:300]
        prompts.append({
            "id": doc["_id"],
            "text": USER_TEMPLATE.format(title=title, text=text),
        })

    # 배치 처리
    DEFAULT_RESULT = {"L1": "Other", "L2": "General", "L3": "unknown"}
    BATCH_SIZE = 500
    results = {}
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        raw_results = run_batch_llm(batch, system_prompt, max_tokens=100)

        for prompt_item, raw in zip(batch, raw_results):
            if raw is None:
                results[prompt_item["id"]] = DEFAULT_RESULT
            else:
                try:
                    parsed = json.loads(parse_llm_content(raw))
                    results[prompt_item["id"]] = parsed
                except json.JSONDecodeError:
                    results[prompt_item["id"]] = DEFAULT_RESULT

        done = min(batch_start + BATCH_SIZE, len(prompts))
        print(f"  Progress: {done}/{len(prompts)}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dataset = args[0] if args else "trec-covid"
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_taxonomy(dataset, size)
