"""신한라이프 subset / 질의 표본 생성 (LLM 호출 없음).

Part 1·3은 config의 `subset`이 non-null이어야 augmentation이 적용된다.
`subset: null`이면 run_experiment.load_augmentations 가 조용히 (None,None,None,None)을
반환해 **모든 arm이 baseline으로 붕괴하는데 라벨만 그대로 남는다**. 그래서 코퍼스가
2,802청크뿐이어도 subset 파일을 반드시 만들어 둔다.

  subset_size 3000 -> size_key "3k"  (run_experiment.load_corpus:98)

출력:
  data/subsets/shinhan/3k.json              {"doc_ids": [...2802...]}
  data/subsets/shinhan/sampled_queries.json {"query_ids": [...50...]}

주의: `data/raw/shinhan/3k_parent_ids.json`은 **만들지 않는다.** 존재하면
load_corpus가 parent 분기를 타는데(run_experiment.py:91-97) 신한 corpus에는
`parent_id` 필드가 없어 잘못 필터링된다.

  python scripts/build_shinhan_subset.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "subsets"
DATASET = "shinhan"
SUBSET_SIZE = 3000  # -> "3k". 실제 2,802청크지만 size_key 규칙상 3000으로 올린다.


def build() -> None:
    ds_dir = dataset_dir(DATA_DIR, DATASET)
    corpus = load_jsonl(ds_dir / "corpus.jsonl")
    queries = load_jsonl(ds_dir / "queries.jsonl")
    qrels = load_jsonl(ds_dir / "qrels.jsonl")

    doc_ids = [doc["_id"] for doc in corpus]
    corpus_ids = set(doc_ids)
    if len(doc_ids) != len(corpus_ids):
        raise ValueError(f"corpus에 중복 _id가 있다: {len(doc_ids)} rows / {len(corpus_ids)} unique")

    if len(corpus) > SUBSET_SIZE:
        raise ValueError(
            f"corpus({len(corpus)})가 SUBSET_SIZE({SUBSET_SIZE})보다 크다. "
            "전체 코퍼스를 담는다는 전제가 깨졌으니 SUBSET_SIZE를 올릴 것."
        )

    gold_by_q = defaultdict(set)
    for entry in qrels:
        if entry.get("score", 0) >= 1:
            gold_by_q[str(entry["query-id"])].add(str(entry["corpus-id"]))

    query_ids = [str(q["_id"]) for q in queries]
    gold_doc_ids = {cid for qid in query_ids for cid in gold_by_q.get(qid, ())}
    missing_gold = gold_doc_ids - corpus_ids
    no_gold = [qid for qid in query_ids if not gold_by_q.get(qid)]

    out_dir = OUTPUT_DIR / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)

    size_key = f"{SUBSET_SIZE // 1000}k"
    subset_path = out_dir / f"{size_key}.json"
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump({
            "subset_size": SUBSET_SIZE,
            "actual_size": len(doc_ids),
            "gold_doc_count": len(gold_doc_ids & corpus_ids),
            "noise_doc_count": len(doc_ids) - len(gold_doc_ids & corpus_ids),
            # 코퍼스(2,802)가 subset_size(3000)보다 작아 서브셋이 짧을 수밖에 없다는 선언.
            # part12_contracts.audit_subsets 가 이 플래그를 보고 크기 불일치를 허용한다.
            "full_corpus": True,
            "note": "전체 코퍼스를 그대로 담는다(샘플링 없음). 실제 크기는 actual_size.",
            "doc_ids": doc_ids,
        }, f, ensure_ascii=False, indent=2)

    queries_path = out_dir / "sampled_queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump({
            "count": len(query_ids),
            "note": "질의 50개 전부 사용(샘플링 없음). Part 4의 무작위 표본 경로를 건너뛴다.",
            "query_ids": query_ids,
            "queries": queries,
            "gold_doc_ids": sorted(gold_doc_ids),
            "gold_doc_count": len(gold_doc_ids),
        }, f, ensure_ascii=False, indent=2)

    print(f"=== subset: {DATASET} ({size_key}) ===")
    print(f"  corpus  : {len(doc_ids)} chunks -> {subset_path}")
    print(f"  queries : {len(query_ids)} -> {queries_path}")
    print(f"  gold docs: {len(gold_doc_ids)}")
    if missing_gold:
        print(f"  [!] corpus에 없는 gold {len(missing_gold)}건: {sorted(missing_gold)[:5]}")
        print("    해당 질의는 recall 분모에 남지만 절대 맞힐 수 없다.")
    if no_gold:
        print(f"  [!] gold가 없는 질의 {len(no_gold)}건: {no_gold[:5]}")

    stale = ds_dir / f"{size_key}_parent_ids.json"
    if stale.exists():
        print(f"  [!] {stale} 가 존재한다. load_corpus가 parent 분기를 타서 subset이 무시된다. 삭제할 것.")


if __name__ == "__main__":
    build()
