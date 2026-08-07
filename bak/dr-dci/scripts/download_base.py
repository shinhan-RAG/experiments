"""
1단계: 기본 데이터셋 다운로드
- TREC-COVID (corpus 171K, queries 50, qrels)
- FiQA (corpus 57K, queries 6648, qrels)
- Ko-StrategyQA (corpus 27.8K, queries 8.5K, qrels)
"""

from datasets import load_dataset
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_trec_covid(output_dir: Path):
    print("=== Downloading TREC-COVID ===")

    corpus = load_dataset("BeIR/trec-covid", "corpus", split="corpus")
    queries = load_dataset("BeIR/trec-covid", "queries", split="queries")
    qrels = load_dataset("BeIR/trec-covid-qrels", split="test")

    out = output_dir / "trec-covid"
    out.mkdir(parents=True, exist_ok=True)

    corpus.to_json(out / "corpus.jsonl")
    queries.to_json(out / "queries.jsonl")
    qrels.to_json(out / "qrels.jsonl")

    print(f"  corpus: {len(corpus)} docs")
    print(f"  queries: {len(queries)}")
    print(f"  qrels: {len(qrels)} judgments")

    # 샘플 확인
    print(f"\n  [Sample corpus]")
    print(f"  _id: {corpus[0]['_id']}")
    print(f"  title: {corpus[0]['title'][:80]}...")
    print(f"  text: {corpus[0]['text'][:100]}...")

    print(f"\n  [Sample query]")
    print(f"  _id: {queries[0]['_id']}")
    print(f"  text: {queries[0]['title']}")


def download_fiqa(output_dir: Path):
    print("\n=== Downloading FiQA ===")

    corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels = load_dataset("BeIR/fiqa-qrels", split="test")

    out = output_dir / "fiqa"
    out.mkdir(parents=True, exist_ok=True)

    corpus.to_json(out / "corpus.jsonl")
    queries.to_json(out / "queries.jsonl")
    qrels.to_json(out / "qrels.jsonl")

    print(f"  corpus: {len(corpus)} docs")
    print(f"  queries: {len(queries)}")
    print(f"  qrels: {len(qrels)} judgments")


def download_ko_strategyqa(output_dir: Path):
    print("\n=== Downloading Ko-StrategyQA ===")

    corpus = load_dataset("taeminlee/Ko-StrategyQA", "corpus", split="ko")
    queries = load_dataset("taeminlee/Ko-StrategyQA", "queries", split="ko")
    qrels = load_dataset("taeminlee/Ko-StrategyQA", "default", split="dev")

    out = output_dir / "ko-strategyqa"
    out.mkdir(parents=True, exist_ok=True)

    corpus.to_json(out / "corpus.jsonl")
    queries.to_json(out / "queries.jsonl")
    qrels.to_json(out / "qrels.jsonl")

    print(f"  corpus: {len(corpus)} docs")
    print(f"  queries: {len(queries)}")
    print(f"  qrels: {len(qrels)} judgments")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_trec_covid(OUTPUT_DIR)
    download_fiqa(OUTPUT_DIR)
    download_ko_strategyqa(OUTPUT_DIR)

    print("\n=== Done ===")
    print(f"Data saved to: {OUTPUT_DIR}")
