"""
6단계: Contextual Prefix 생성
- 각 문서에 50-100 토큰의 문맥 요약 prefix 부여
- 임베딩 품질 향상 (Pull retriever 정밀도 개선)
- 병렬 처리 (async, 32 concurrent)
"""

import json
from pathlib import Path
from utils import run_batch_llm, strip_thinking, load_corpus_subset

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "prefix"

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


def build_prefix(dataset: str = "trec-covid", subset_size: int = 10_000):
    print(f"=== Building Contextual Prefix for {dataset} ({subset_size // 1000}K) ===")

    # 이미 존재하면 스킵
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    if out_path.exists():
        print(f"  Already exists: {out_path}, skipping.")
        return

    # 서브셋 corpus 로드 (aihub=parent-id 서브셋, BEIR=doc-id 서브셋)
    corpus_subset = load_corpus_subset(DATA_DIR, dataset, subset_size)
    print(f"  Docs to process: {len(corpus_subset)}")

    # 프롬프트 준비
    prompts = []
    for doc in corpus_subset:
        title = doc.get("title", "")
        text = doc.get("text", "")[:500]
        prompts.append({
            "id": doc["_id"],
            "text": USER_TEMPLATE.format(title=title, text=text),
        })

    # 배치 처리
    BATCH_SIZE = 500
    results = {}
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        raw_results = run_batch_llm(batch, SYSTEM_PROMPT, max_tokens=150)

        for prompt_item, raw in zip(batch, raw_results):
            if raw is None:
                results[prompt_item["id"]] = ""
            else:
                results[prompt_item["id"]] = strip_thinking(raw)

        done = min(batch_start + BATCH_SIZE, len(prompts))
        print(f"  Progress: {done}/{len(prompts)}")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
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
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dataset = args[0] if args else "trec-covid"
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_prefix(dataset, size)
