"""
2단계: 규모별 서브셋 생성
- TREC-COVID에서 gold docs 추출 → 20K/50K/110K 서브셋
  (gold docs score>=1 이 ~17.5K이므로 최소 20K부터 시작)
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
SUBSET_SIZES = [20_000, 50_000, 110_000]


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
        if noise_needed < 0:
            print(f"  WARNING: Gold docs ({len(gold_docs)}) > subset size ({size}). Skipping.")
            continue
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

    assert subsets[20_000].issubset(subsets[50_000]), "20K is not subset of 50K!"
    assert subsets[50_000].issubset(subsets[110_000]), "50K is not subset of 110K!"
    print("  ✓ Subset inclusion verified (20K ⊂ 50K ⊂ 110K)")


def build_generic_subset(dataset_name: str, subset_size: int = 20_000):
    """fiqa, ko-strategyqa 등 일반 데이터셋의 서브셋 생성
    - 50개 쿼리 샘플링
    - gold docs 전부 포함 + noise로 채워서 subset_size 맞춤
    """
    print(f"\n=== Building {dataset_name} subset ({subset_size // 1000}K) ===")

    queries = load_jsonl(RAW_DIR / dataset_name / "queries.jsonl")
    qrels = load_jsonl(RAW_DIR / dataset_name / "qrels.jsonl")
    corpus = load_jsonl(RAW_DIR / dataset_name / "corpus.jsonl")

    # 쿼리 ID별 gold docs
    query_gold = defaultdict(set)
    for entry in qrels:
        if entry["score"] >= 1:
            query_gold[str(entry["query-id"])].add(str(entry["corpus-id"]))

    # gold docs가 있는 쿼리만 필터 → 50개 샘플링
    valid_queries = [q for q in queries if str(q["_id"]) in query_gold]
    print(f"  Valid queries (with gold docs): {len(valid_queries)}")

    random.seed(SEED)
    sampled = random.sample(valid_queries, min(50, len(valid_queries)))
    sampled_ids = [q["_id"] for q in sampled]

    # 샘플된 쿼리의 gold doc IDs
    gold_doc_ids = set()
    for qid in sampled_ids:
        gold_doc_ids.update(query_gold[str(qid)])
    print(f"  Sampled: {len(sampled)} queries, {len(gold_doc_ids)} gold docs")

    # corpus에서 gold / noise 분리
    corpus_ids = {doc["_id"] for doc in corpus}
    gold_doc_ids = gold_doc_ids & corpus_ids  # corpus에 있는 것만

    gold_docs = [doc for doc in corpus if doc["_id"] in gold_doc_ids]
    noise_docs = [doc for doc in corpus if doc["_id"] not in gold_doc_ids]

    random.shuffle(noise_docs)

    # subset 구성
    actual_size = min(subset_size, len(corpus))
    noise_needed = actual_size - len(gold_docs)
    if noise_needed < 0:
        # gold docs가 subset_size보다 많으면 gold만으로 구성
        subset_docs = gold_docs[:actual_size]
        noise_needed = 0
    else:
        noise_needed = min(noise_needed, len(noise_docs))
        subset_docs = gold_docs + noise_docs[:noise_needed]

    subset_ids = [doc["_id"] for doc in subset_docs]

    # 저장: subset
    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{actual_size // 1000}k.json", "w") as f:
        json.dump({
            "subset_size": actual_size,
            "actual_size": len(subset_ids),
            "gold_doc_count": len(gold_docs),
            "noise_doc_count": noise_needed,
            "doc_ids": subset_ids,
        }, f, ensure_ascii=False, indent=2)

    # 저장: sampled queries
    with open(out_dir / "sampled_queries.json", "w") as f:
        json.dump({
            "count": len(sampled),
            "query_ids": sampled_ids,
            "queries": sampled,
            "gold_doc_ids": list(gold_doc_ids),
            "gold_doc_count": len(gold_doc_ids),
        }, f, ensure_ascii=False, indent=2)

    print(f"  Subset: {len(subset_ids)} docs ({len(gold_docs)} gold + {noise_needed} noise)")
    print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if not args or "trec-covid" in args:
        build_trec_covid_subsets()

    if not args or "fiqa" in args:
        build_generic_subset("fiqa", subset_size=20_000)

    if not args or "ko-strategyqa" in args:
        build_generic_subset("ko-strategyqa", subset_size=20_000)

    print("\n=== Done ===")
    print(f"Subsets saved to: {OUTPUT_DIR}")
