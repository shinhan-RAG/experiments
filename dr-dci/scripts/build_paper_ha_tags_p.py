"""paper-ha 결정론 @el: 태그 (approach_p) — LLM 호출 0회.

corpus.jsonl 의 element_type(라벨 유래, docs/PAPER_HA_TAG_RULES.md)을 그대로
@el: 태그로 변환한다. 청크 1개 = element 1개 (단락이 곧 요소이므로 분할 없음).
build_tags.py 산출물과 같은 형태(플랫 배열 {"idx","text","doc_id","tag"})라서
run_experiment.load_augmentations / part12_contracts 가 그대로 소비한다.

출력: data/tags/paper-ha/approach_p/5k.json

  python scripts/build_paper_ha_tags_p.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET = "paper-ha"
SUBSET_SIZE = 5000  # -> "5k" (build_paper_ha_subset.py 와 반드시 일치)

TAG_BY_ELEMENT = {
    "heading": "@el:heading",
    "text": "@el:paragraph",
    "table": "@el:table",
    "chart": "@el:chart",
    "figure": "@el:figure",
}


def build() -> None:
    corpus = load_jsonl(dataset_dir(DATA_DIR, DATASET) / "corpus.jsonl")
    tagged, counts = [], {}
    for doc in corpus:
        et = doc.get("element_type", "text")
        tag = TAG_BY_ELEMENT.get(et)
        if tag is None:
            raise SystemExit(f"알 수 없는 element_type '{et}' ({doc['_id']})")
        tagged.append({"idx": 0, "text": doc["text"], "doc_id": doc["_id"], "tag": tag})
        counts[tag] = counts.get(tag, 0) + 1

    size_key = f"{SUBSET_SIZE // 1000}k"
    out_dir = DATA_DIR / "tags" / DATASET / "approach_p"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{size_key}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False)

    print(f"=== approach_p tags: {DATASET} ({size_key}) ===")
    print(f"  elements: {len(tagged)} -> {out_path}")
    for tag, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {n}")


if __name__ == "__main__":
    build()
