"""신한라이프 reference answer 생성 (LLM-as-Judge accuracy용).

`data/raw/shinhan/qa_meta.jsonl`에 QA 생성 단계에서 만든 `answer`가 이미 들어 있으므로
**LLM 호출 없이** 변환만 한다 (build_legal_reference_answers.build_legal_qa 와 동일한 방식).

이 파일이 없으면 judge가 한 건도 실행되지 않아 accuracy가 통째로 `None`이 된다
(run_experiment.assign_judgment). 2026-07-28 Part 4 실행이 `judged 0`으로 나온 원인이다.

출력: data/reference_answers/<dataset>.json
      [{"query_id": "1", "reference_answer": "..."}, ...]

  python scripts/build_shinhan_reference_answers.py
  python scripts/build_shinhan_reference_answers.py --dataset shinhan-uw
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "reference_answers"
DATASET = "shinhan"


def build(dataset: str = DATASET) -> None:
    ds_dir = dataset_dir(DATA_DIR, dataset)
    meta_path = ds_dir / "qa_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} 없음. {dataset} 데이터가 {ds_dir}에 있어야 한다 "
            "(run_experiment.load_supporting_spans 도 같은 경로를 읽는다)."
        )

    meta = load_jsonl(meta_path)
    queries = load_jsonl(ds_dir / "queries.jsonl")
    query_ids = {str(q["_id"]) for q in queries}

    out, skipped_empty, skipped_unknown = [], 0, 0
    for m in meta:
        qid = str(m["qid"])
        answer = (m.get("answer") or "").strip()
        if not answer:
            skipped_empty += 1
            continue
        if qid not in query_ids:
            # queries.jsonl에 없는 qid는 평가에서 조회되지 않는다
            skipped_unknown += 1
            continue
        out.append({"query_id": qid, "reference_answer": answer})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"=== reference answers: {dataset} ===")
    print(f"  qa_meta   : {len(meta)}")
    print(f"  queries   : {len(queries)}")
    print(f"  written   : {len(out)}")
    if skipped_empty:
        print(f"  skipped(빈 answer)      : {skipped_empty}")
    if skipped_unknown:
        print(f"  skipped(queries에 없음) : {skipped_unknown}")
    print(f"  -> {out_path}")

    missing = query_ids - {o["query_id"] for o in out}
    if missing:
        print(f"  [!] reference answer 없는 질의 {len(missing)}건: {sorted(missing)[:10]}")
        print("    해당 질의는 accuracy 집계에서 빠진다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET,
                    help="데이터셋 이름 (기본 shinhan, 언더라이팅 파싱본은 shinhan-uw)")
    build(ap.parse_args().dataset)
