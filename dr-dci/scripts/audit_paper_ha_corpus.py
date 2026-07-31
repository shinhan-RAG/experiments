"""paper-ha corpus 감사 (audit_shinhan_corpus 계열, LLM 없음).

검사 항목:
  1. _id 유일성, 필수 필드, element_type 어휘
  2. 중복 텍스트 청크 (완전 일치)
  3. 4,096자 초과 청크 (임베딩 절단 위험)
  4. 이미지 캡션 청크 ↔ 원본 PNG 크롭 존재 대응
  5. qrels gold가 corpus에 존재하는지 / qa_meta supporting_spans verbatim 재검증
  6. tags/approach_p 커버리지(청크 1:1)

출력: data/raw/paper-ha/audit_corpus.json  (+ 콘솔 요약, 문제 발견 시 exit 1)

  python scripts/audit_paper_ha_corpus.py [--png-dir <VL_HA 디렉토리>]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET = "paper-ha"
SIZE_KEY = "5k"
VALID_ELEMENTS = {"heading", "text", "table", "chart", "figure"}
DEFAULT_PNG_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "32.학술논문 이해 데이터" / "3.개방데이터" / "1.데이터"
    / "Validation" / "02.라벨링데이터" / "VL_인문학,예술체육학(HA)"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    args = ap.parse_args()

    ds_dir = dataset_dir(DATA_DIR, DATASET)
    corpus = load_jsonl(ds_dir / "corpus.jsonl")
    qrels = load_jsonl(ds_dir / "qrels.jsonl")
    qa_meta = load_jsonl(ds_dir / "qa_meta.jsonl")

    report, problems = {}, []

    # 1. 기본 무결성
    ids = [c["_id"] for c in corpus]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    bad_element = [c["_id"] for c in corpus if c.get("element_type") not in VALID_ELEMENTS]
    empty_text = [c["_id"] for c in corpus if not (c.get("text") or "").strip()]
    report["chunks"] = len(corpus)
    report["element_dist"] = dict(Counter(c["element_type"] for c in corpus))
    if dup_ids:
        problems.append(f"중복 _id {len(dup_ids)}건")
    if bad_element:
        problems.append(f"잘못된 element_type {len(bad_element)}건")
    if empty_text:
        problems.append(f"빈 text {len(empty_text)}건")

    # 2. 중복 텍스트
    by_text = defaultdict(list)
    for c in corpus:
        by_text[c["text"]].append(c["_id"])
    dup_texts = {ids_[0]: ids_ for ids_ in by_text.values() if len(ids_) > 1}
    report["duplicate_text_groups"] = len(dup_texts)
    report["duplicate_text_examples"] = list(dup_texts.values())[:5]

    # 3. 4,096자 초과
    over = [c["_id"] for c in corpus if len(c["text"]) > 4096]
    report["over_4096"] = len(over)

    # 4. 캡션 ↔ PNG 대응
    caption_chunks = [c for c in corpus if c["element_type"] in ("table", "chart", "figure")]
    missing_png, no_ref = [], 0
    if args.png_dir.exists():
        for c in caption_chunks:
            ref = (c.get("image_file_name") or "").strip()
            if not ref:
                no_ref += 1
                continue
            if not (args.png_dir / Path(ref).name).exists():
                missing_png.append(c["_id"])
        report["caption_chunks"] = len(caption_chunks)
        report["caption_missing_png"] = len(missing_png)
        report["caption_no_ref"] = no_ref
        if missing_png:
            problems.append(f"PNG 크롭 없는 캡션 {len(missing_png)}건")
    else:
        report["caption_png_check"] = f"skipped (png_dir 없음: {args.png_dir})"

    # 5. gold / span 재검증
    corpus_by_id = {c["_id"]: c for c in corpus}
    missing_gold = [e["corpus-id"] for e in qrels
                    if e.get("score", 0) >= 1 and e["corpus-id"] not in corpus_by_id]
    bad_spans = []
    for m in qa_meta:
        chunk = corpus_by_id.get(m["gold_chunk_id"])
        if chunk is None:
            bad_spans.append((m["qid"], "gold 청크 없음"))
            continue
        for s in m["supporting_spans"]:
            if s.get("verbatim") and s["text"] not in chunk["text"]:
                bad_spans.append((m["qid"], "verbatim span이 청크에 없음"))
    report["qrels"] = len(qrels)
    report["missing_gold"] = missing_gold
    report["bad_spans"] = bad_spans
    if missing_gold:
        problems.append(f"corpus에 없는 gold {len(missing_gold)}건")
    if bad_spans:
        problems.append(f"span 재검증 실패 {len(bad_spans)}건")

    # 6. approach_p 태그 커버리지
    tags_path = DATA_DIR / "tags" / DATASET / "approach_p" / f"{SIZE_KEY}.json"
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            tags = json.load(f)
        tagged_ids = {t["doc_id"] for t in tags}
        untagged = [i for i in ids if i not in tagged_ids]
        report["tags_p_elements"] = len(tags)
        report["tags_p_untagged_chunks"] = len(untagged)
        if untagged:
            problems.append(f"approach_p 태그 없는 청크 {len(untagged)}건")
    else:
        problems.append(f"approach_p 태그 파일 없음: {tags_path}")

    report["problems"] = problems
    out_path = ds_dir / "audit_corpus.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"=== audit: {DATASET} ===")
    for k in ("chunks", "element_dist", "duplicate_text_groups", "over_4096",
              "caption_chunks", "caption_missing_png", "qrels",
              "tags_p_elements", "tags_p_untagged_chunks"):
        if k in report:
            print(f"  {k}: {report[k]}")
    if problems:
        print("  [FAIL]", "; ".join(problems))
        raise SystemExit(1)
    print(f"  [PASS] -> {out_path}")


if __name__ == "__main__":
    main()
