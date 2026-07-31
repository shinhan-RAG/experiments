"""AI Hub 학술논문(HA) Validation 라벨 JSON → dr-dci BEIR corpus.jsonl.

규칙은 docs/PAPER_HA_TAG_RULES.md 를 따른다 (결정론, LLM 없음).
- section_info 단락 1개 = 청크 1개 (재청킹 없음), title_* 는 이후 단락의 section 으로 승계
- image_info 이미지 1개 = 캡션 청크 1개 (element_type: table/chart/figure)
- parent_id 없음 → 청크 단위 평가 (신한과 동일. aihub-full 과 절대점수 비교 금지)

출력: data/raw/paper-ha/corpus.jsonl   {_id, title, text, doc, section, element_type}
      data/raw/paper-ha/corpus_stats.json
      data/raw/paper-ha/manifest.json

  python scripts/build_paper_ha_corpus.py [--src <VL_HA 디렉토리>]
"""
import argparse
import json
from pathlib import Path

DEFAULT_SRC = (
    Path(__file__).resolve().parents[2]
    / "data" / "32.학술논문 이해 데이터" / "3.개방데이터" / "1.데이터"
    / "Validation" / "02.라벨링데이터" / "VL_인문학,예술체육학(HA)"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "paper-ha"

MIN_CHARS = 20  # 이보다 짧은 요소(목차형 제목 등)는 제외하고 section 승계만 한다

IMAGE_CATEGORY_MAP = {"TA": "table", "CH": "chart", "PI": "figure"}


def build_doc(path: Path, stats: dict) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        label = json.load(f)
    doc = path.stem
    meta = label["raw_data_meta_info"]
    doc_title = (meta.get("doc_title") or doc).strip()
    info = label["training_data_info"]

    chunks = []
    section = "(전문)"
    for entry in info.get("section_info", []):
        pid = entry["paragraph_id"]
        text = (entry.get("original_text") or "").strip()
        is_title = pid.startswith("title_")
        if is_title and text:
            section = text[:80]
        if len(text) < MIN_CHARS:
            stats["skipped_short"] += 1
            continue
        et = "heading" if is_title else "text"
        chunks.append({
            "_id": f"{doc}_{pid}",
            "title": f"{doc_title[:80]} — {text[:60] if is_title else section[:60]}",
            "text": text,
            "doc": doc,
            "section": text[:80] if is_title else section,
            "element_type": et,
        })
        if len(text) > 4096:
            stats["over_4096"] += 1

    for img in info.get("image_info", []):
        caption = (img.get("image_caption") or "").strip()
        if len(caption) < MIN_CHARS:
            stats["skipped_short"] += 1
            continue
        cat = (img.get("image_category") or "").strip().upper()
        et = IMAGE_CATEGORY_MAP.get(cat)
        if et is None:
            stats["unknown_image_category"] += 1
            et = "figure"
        name = (img.get("image_name") or f"이미지 {img.get('image_id')}").strip()
        # 드물게 한 문서 안에서 image_id가 중복되는 라벨 오류가 있다(예: SS_0132_0067410)
        # → 충돌 시 등장 순서 접미사로 유일성 보장 (Validation HA에는 중복 없음)
        img_id = f"{doc}_img_{img.get('image_id')}"
        if any(c["_id"] == img_id for c in chunks):
            img_id = f"{img_id}_dup{sum(1 for c in chunks if c['_id'].startswith(img_id))}"
        chunks.append({
            "_id": img_id,
            "title": f"{doc_title[:80]} — {name}",
            "text": caption,
            "doc": doc,
            "section": name,
            "element_type": et,
            "image_file_name": (img.get("image_file_name") or "").strip(),
        })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()

    files = sorted(args.src.glob("*.json"))
    if not files:
        raise SystemExit(f"라벨 JSON이 없다: {args.src}")

    corpus = []
    stats = {"docs": 0, "chunks": 0, "by_element": {}, "skipped_short": 0,
             "over_4096": 0, "unknown_image_category": 0, "per_doc": []}
    for f in files:
        doc_chunks = build_doc(f, stats)
        corpus.extend(doc_chunks)
        stats["docs"] += 1
        stats["per_doc"].append({"doc": f.stem, "chunks": len(doc_chunks)})
        for c in doc_chunks:
            stats["by_element"][c["element_type"]] = \
                stats["by_element"].get(c["element_type"], 0) + 1
    stats["chunks"] = len(corpus)

    ids = [c["_id"] for c in corpus]
    if len(ids) != len(set(ids)):
        raise SystemExit("corpus에 중복 _id가 있다")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    manifest = {
        "dataset": "paper-ha",
        "source": "AI Hub 32.학술논문 이해 데이터 / Validation / VL_인문학,예술체육학(HA)",
        "built_from": str(args.src),
        "docs": stats["docs"],
        "chunks": stats["chunks"],
        "by_element": stats["by_element"],
        "granularity": "chunk (parent_id 없음 — parent 단위 데이터셋과 절대점수 비교 금지)",
        "tag_rules": "docs/PAPER_HA_TAG_RULES.md",
        "limitations": [
            "original_text는 어절 간 공백이 소실된 구간이 많다(원문 유지). summary_text만 정상 띄어쓰기.",
            "image_caption은 인쇄된 실제 캡션이 아니라 본문에서 생성된 설명문이다. 표는 셀 구조 없이 설명문만 검색된다.",
            "page가 전부 '1'(단일 슬라이드 좌표계)이라 페이지 정보가 없다. location(EMU bbox)은 싣지 않았다.",
            f"4,096자 초과 단락 {stats['over_4096']}건은 임베딩 입력에서 잘릴 수 있다(재청킹하지 않음).",
            f"{MIN_CHARS}자 미만 요소 {stats['skipped_short']}건 제외(목차형 제목 등 — section 필드로 승계됨).",
        ],
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"docs={stats['docs']} chunks={stats['chunks']}")
    print("element 분포:", stats["by_element"])
    print(f"skipped(<{MIN_CHARS}자)={stats['skipped_short']} over4096={stats['over_4096']} "
          f"unknown_image_category={stats['unknown_image_category']}")


if __name__ == "__main__":
    main()
