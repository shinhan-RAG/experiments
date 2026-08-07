"""
1단계: 데이터셋 다운로드
- BEIR 포맷: corpus.jsonl, queries.jsonl, qrels.jsonl
- TREC-COVID: beir/trec-covid (HuggingFace)
- FiQA: beir/fiqa
- Ko-StrategyQA: 직접 구성 (StrategyQA 한국어 번역본)
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def download_beir_dataset(dataset_name: str, hf_name: str = None):
    """HuggingFace datasets에서 BEIR 포맷 데이터셋 다운로드"""
    from datasets import load_dataset

    if hf_name is None:
        hf_name = dataset_name

    out_dir = RAW_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # corpus.jsonl이 이미 있으면 스킵
    if (out_dir / "corpus.jsonl").exists():
        print(f"  [{dataset_name}] Already exists, skipping.")
        return

    print(f"  [{dataset_name}] Downloading from BeIR/{hf_name}...")

    # Corpus
    print(f"    Loading corpus...")
    corpus_ds = load_dataset(f"BeIR/{hf_name}", "corpus", split="corpus")
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus_ds:
            doc = {
                "_id": item["_id"],
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"    Corpus: {len(corpus_ds)} docs")

    # Queries
    print(f"    Loading queries...")
    queries_ds = load_dataset(f"BeIR/{hf_name}", "queries", split="queries")
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for item in queries_ds:
            q = {
                "_id": item["_id"],
                "text": item.get("text", ""),
            }
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"    Queries: {len(queries_ds)}")

    # Qrels (test split)
    print(f"    Loading qrels...")
    try:
        qrels_ds = load_dataset(f"BeIR/{hf_name}-qrels", split="test")
    except Exception:
        # 일부 데이터셋은 다른 구조
        qrels_ds = load_dataset(f"BeIR/{hf_name}", "default", split="test")

    with open(out_dir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for item in qrels_ds:
            qrel = {
                "query-id": item.get("query-id", item.get("query_id", "")),
                "corpus-id": item.get("corpus-id", item.get("corpus_id", "")),
                "score": item.get("score", 0),
            }
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    print(f"    Qrels: {len(qrels_ds)}")

    print(f"  [{dataset_name}] Done! Saved to {out_dir}")


def download_ko_strategyqa():
    """Ko-StrategyQA 다운로드 (BEIR 포맷으로 변환)"""
    from datasets import load_dataset

    out_dir = RAW_DIR / "ko-strategyqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "corpus.jsonl").exists():
        print(f"  [ko-strategyqa] Already exists, skipping.")
        return

    print(f"  [ko-strategyqa] Downloading from mteb/Ko-StrategyQA...")

    # mteb/Ko-StrategyQA: 이미 BEIR 포맷(corpus/queries/qrels), split은 "dev"
    corpus_ds = load_dataset("mteb/Ko-StrategyQA", "corpus", split="dev")
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus_ds:
            doc = {
                "_id": item["_id"],
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    queries_ds = load_dataset("mteb/Ko-StrategyQA", "queries", split="dev")
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for item in queries_ds:
            f.write(json.dumps({"_id": item["_id"], "text": item.get("text", "")}, ensure_ascii=False) + "\n")

    qrels_ds = load_dataset("mteb/Ko-StrategyQA", "qrels", split="dev")
    with open(out_dir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for item in qrels_ds:
            f.write(json.dumps({
                "query-id": item["query-id"],
                "corpus-id": item["corpus-id"],
                "score": item.get("score", 0),
            }, ensure_ascii=False) + "\n")

    print(f"    Corpus: {len(corpus_ds)} docs")
    print(f"    Queries: {len(queries_ds)}")
    print(f"    Qrels: {len(qrels_ds)}")
    print(f"  [ko-strategyqa] Done! Saved to {out_dir}")


if __name__ == "__main__":
    import sys

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["trec-covid", "fiqa", "ko-strategyqa"]

    print("=== Downloading Datasets ===\n")

    for ds in datasets:
        if ds == "ko-strategyqa":
            download_ko_strategyqa()
        else:
            download_beir_dataset(ds)
        print()

    print("=== All downloads complete ===")
