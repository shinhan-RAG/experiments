"""
8단계: 샘플 검증
- 10건 샘플로 전처리 결과 확인
- taxonomy, tags, prefix, metadata, reference answer 품질 검수
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

SEED = 42


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate():
    print("=== Sample Validation (10 docs) ===\n")

    # 10K 서브셋에서 10개 샘플
    subset = load_json(DATA_DIR / "subsets" / "trec-covid" / "20k.json")
    random.seed(SEED)
    sample_ids = random.sample(subset["doc_ids"], 10)

    # 데이터 로드
    corpus = load_jsonl(RAW_DIR / "trec-covid" / "corpus.jsonl")
    corpus_dict = {doc["_id"]: doc for doc in corpus}

    taxonomy = load_json(DATA_DIR / "taxonomy" / "trec-covid_20k.json")
    prefix = load_json(DATA_DIR / "prefix" / "trec-covid_20k.json")
    metadata = load_json(DATA_DIR / "metadata" / "trec-covid_20k.json")

    # tags (approach A만 샘플 검증)
    tags_a = load_json(DATA_DIR / "tags" / "trec-covid" / "approach_a" / "20k.json")
    tags_by_doc = {}
    for elem in tags_a:
        doc_id = elem["doc_id"]
        if doc_id not in tags_by_doc:
            tags_by_doc[doc_id] = []
        tags_by_doc[doc_id].append(elem)

    # reference answers
    ref_answers = load_json(DATA_DIR / "reference_answers" / "trec-covid.json")
    ref_by_query = {r["query_id"]: r for r in ref_answers}

    print("=" * 80)
    for i, doc_id in enumerate(sample_ids):
        doc = corpus_dict.get(doc_id)
        if not doc:
            print(f"[{i+1}] doc_id={doc_id} — NOT FOUND")
            continue

        print(f"\n[{i+1}] doc_id={doc_id}")
        print(f"  Title: {doc.get('title', '')[:80]}")
        print(f"  Text: {doc.get('text', '')[:100]}...")

        # Taxonomy
        tax = taxonomy.get(doc_id, {})
        print(f"  Taxonomy: L1={tax.get('L1')} / L2={tax.get('L2')} / L3={tax.get('L3')}")

        # Prefix
        pref = prefix.get(doc_id, "")
        print(f"  Prefix: {pref[:80]}...")

        # Metadata
        meta = metadata.get(doc_id, {})
        print(f"  Metadata: topic={meta.get('topic')}, type={meta.get('study_type')}, entities={meta.get('entities', [])[:3]}")

        # Tags
        doc_tags = tags_by_doc.get(doc_id, [])
        if doc_tags:
            print(f"  Tags ({len(doc_tags)} elements):")
            for elem in doc_tags[:3]:
                print(f"    [{elem['tag']}] {elem['text'][:60]}...")

        print("-" * 80)

    # Reference answer 샘플 (3개)
    print("\n\n=== Reference Answer Samples ===")
    for ra in ref_answers[:3]:
        print(f"\n  Query [{ra['query_id']}]: {ra['query_text']}")
        print(f"  Gold docs: {ra['gold_doc_count']}")
        print(f"  Answer: {ra['reference_answer'][:200]}...")
        print()


if __name__ == "__main__":
    validate()
