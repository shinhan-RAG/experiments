"""
2단계: 규모별 서브셋 생성
- TREC-COVID에서 gold docs 추출 → 10K/50K/110K 서브셋
- FiQA, Ko-StrategyQA에서 50 쿼리 샘플링
"""

import json
import random
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "subsets"

SEED = 42
SUBSET_SIZES = [10_000, 50_000, 110_000]


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_trec_covid_subsets():
    print("=== Building TREC-COVID subsets ===")

    corpus = load_jsonl(RAW_DIR / "trec-covid" / "corpus.jsonl")
    qrels = load_jsonl(RAW_DIR / "trec-covid" / "qrels.jsonl")

    # gold doc IDs 추출 (score >= 1)
    gold_doc_ids = set()
    for entry in qrels:
        if entry["score"] >= 1:
            gold_doc_ids.add(entry["corpus-id"])

    print(f"  Gold docs (score >= 1): {len(gold_doc_ids)}")

    # corpus를 gold / noise로 분리
    gold_docs = []
    noise_docs = []
    corpus_id_set = set()

    for doc in corpus:
        corpus_id_set.add(doc["_id"])
        if doc["_id"] in gold_doc_ids:
            gold_docs.append(doc)
        else:
            noise_docs.append(doc)

    # gold docs 중 corpus에 없는 것 확인
    missing = gold_doc_ids - corpus_id_set
    if missing:
        print(f"  WARNING: {len(missing)} gold docs not found in corpus")

    print(f"  Gold docs in corpus: {len(gold_docs)}")
    print(f"  Noise docs available: {len(noise_docs)}")

    # 노이즈 셔플
    random.seed(SEED)
    random.shuffle(noise_docs)

    # 서브셋 생성 (누적)
    for size in SUBSET_SIZES:
        noise_needed = size - len(gold_docs)
        if noise_needed > len(noise_docs):
            print(f"  WARNING: Not enough noise for {size}. Using all {len(noise_docs)} noise docs.")
            noise_needed = len(noise_docs)

        subset_docs = gold_docs + noise_docs[:noise_needed]
        subset_ids = [doc["_id"] for doc in subset_docs]

        out_path = OUTPUT_DIR / "trec-covid" / f"{size // 1000}k.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "subset_size": size,
                "actual_size": len(subset_ids),
                "gold_doc_count": len(gold_docs),
                "noise_doc_count": noise_needed,
                "doc_ids": subset_ids,
            }, f, ensure_ascii=False, indent=2)

        print(f"  Subset {size // 1000}K: {len(subset_ids)} docs ({len(gold_docs)} gold + {noise_needed} noise)")

    # 확인: 10K ⊂ 50K ⊂ 110K
    subsets = {}
    for size in SUBSET_SIZES:
        path = OUTPUT_DIR / "trec-covid" / f"{size // 1000}k.json"
        with open(path) as f:
            subsets[size] = set(json.load(f)["doc_ids"])

    assert subsets[10_000].issubset(subsets[50_000]), "10K is not subset of 50K!"
    assert subsets[50_000].issubset(subsets[110_000]), "50K is not subset of 110K!"
    print("  ✓ Subset inclusion verified (10K ⊂ 50K ⊂ 110K)")


def sample_queries(dataset_name: str, n: int = 50):
    """FiQA, Ko-StrategyQA에서 n개 쿼리 샘플링 (gold docs가 있는 것만)"""
    print(f"\n=== Sampling {n} queries for {dataset_name} ===")

    queries = load_jsonl(RAW_DIR / dataset_name / "queries.jsonl")
    qrels = load_jsonl(RAW_DIR / dataset_name / "qrels.jsonl")

    # 쿼리 ID별 gold docs
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(entry["corpus-id"])

    # gold docs가 있는 쿼리만 필터
    valid_queries = [q for q in queries if str(q["_id"]) in query_gold]
    print(f"  Valid queries (with gold docs): {len(valid_queries)}")

    random.seed(SEED)
    sampled = random.sample(valid_queries, min(n, len(valid_queries)))

    # 샘플된 쿼리의 gold doc IDs
    sampled_ids = [q["_id"] for q in sampled]
    sampled_gold_ids = set()
    for qid in sampled_ids:
        sampled_gold_ids.update(query_gold[str(qid)])

    out_path = OUTPUT_DIR / dataset_name / "sampled_queries.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "count": len(sampled),
            "query_ids": sampled_ids,
            "gold_doc_ids": list(sampled_gold_ids),
            "gold_doc_count": len(sampled_gold_ids),
        }, f, ensure_ascii=False, indent=2)

    print(f"  Sampled: {len(sampled)} queries, {len(sampled_gold_ids)} gold docs")


if __name__ == "__main__":
    build_trec_covid_subsets()
    sample_queries("fiqa", n=50)
    sample_queries("ko-strategyqa", n=50)

    print("\n=== Done ===")
    print(f"Subsets saved to: {OUTPUT_DIR}")
