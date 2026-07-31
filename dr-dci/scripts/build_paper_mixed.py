"""paper-mixed: 학술논문 distractor 확장 코퍼스 (taxonomy 라우팅 실험용, LLM 없음).

구성 (build_shinhan_mixed 패턴):
- gold 층: Validation HA 331편 → 4,253청크 (paper-ha와 동일 규칙·동일 _id — qrels 재사용)
- distractor 층: Training TL_HA(2,648) + TL_SS(3,637) + TL_ST(1,715) = 8,000편을
  같은 추출 규칙으로 청킹, seed 셔플한 문서 순서로 20k ⊂ 50k ⊂ 110k 중첩 적층
- taxonomy: 라벨의 doc_category(인문학/사회과학/과학기술 등)에서 **결정론적으로** 생성
  → ground-truth 분류라 "분류가 틀려서"라는 교란이 없다. L2=도메인코드, L3=저널명.

출력:
  data/raw/paper-mixed/corpus.jsonl (+ 90MB 초과 시 corpus.jsonl.partNN 분할본)
  data/raw/paper-mixed/{queries,qrels,qa_meta}.jsonl   (paper-ha에서 복사)
  data/raw/paper-mixed/manifest.json
  data/subsets/paper-mixed/{20k,50k,110k}.json         (중첩, gold 층 항상 포함)
  data/taxonomy/paper-mixed_{20k,50k,110k}.json        {chunk_id: {L1,L2,L3}}
  config/taxonomy_schemas/paper-mixed.yaml             (발견된 L1 값으로 생성)

  python scripts/build_paper_mixed.py
"""
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_paper_ha_corpus import build_doc  # noqa: E402 — 동일 추출 규칙 재사용

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = DATA_DIR / "raw" / "paper-mixed"
GOLD_RAW = DATA_DIR / "raw" / "paper-ha"
AIHUB_ROOT = BASE.parent / "data" / "32.학술논문 이해 데이터" / "3.개방데이터" / "1.데이터"

GOLD_SRC = AIHUB_ROOT / "Validation" / "02.라벨링데이터" / "VL_인문학,예술체육학(HA)"
DISTRACTOR_SRCS = [
    AIHUB_ROOT / "Training" / "02.라벨링데이터" / "TL_인문학,예술체육학(HA)",
    AIHUB_ROOT / "Training" / "02.라벨링데이터" / "TL_사회과학(SS)",
    AIHUB_ROOT / "Training" / "02.라벨링데이터" / "TL_과학기술(ST)",
]
SEED = 42
TIERS = [20_000, 50_000, 110_000]
SPLIT_BYTES = 85 * 1024 * 1024  # git 파일 한도(100MB) 미만으로 분할


def doc_category(path: Path) -> tuple[str, str]:
    """라벨 파일에서 (doc_category, doc_origin)만 가볍게 읽는다."""
    with open(path, encoding="utf-8-sig") as f:
        meta = json.load(f)["raw_data_meta_info"]
    return (meta.get("doc_category") or "기타").strip(), (meta.get("doc_origin") or "").strip()


def build_dir(src: Path, stats: dict) -> list[dict]:
    """라벨 디렉토리 하나를 문서별 청크 리스트로 (build_paper_ha_corpus 규칙)."""
    docs = []
    domain = src.name.split("(")[-1].rstrip(")")  # HA/SS/ST
    for path in sorted(src.glob("*.json")):
        chunks = build_doc(path, stats)
        if not chunks:
            continue
        cat, origin = doc_category(path)
        for c in chunks:
            c["doc_category"] = cat
            c["domain"] = domain
            c["doc_origin"] = origin
        docs.append(chunks)
    return docs


def main() -> None:
    stats = {"docs": 0, "chunks": 0, "by_element": {}, "skipped_short": 0,
             "over_4096": 0, "unknown_image_category": 0, "per_doc": []}

    print("gold 층(Validation HA) 청킹...")
    gold_docs = build_dir(GOLD_SRC, stats)
    gold_chunks = [c for doc in gold_docs for c in doc]
    print(f"  gold: {len(gold_docs)}편 {len(gold_chunks)}청크")

    # 기존 paper-ha corpus와 _id 일치 검증 — qrels 재사용의 전제
    ha_ids = {json.loads(l)["_id"] for l in open(GOLD_RAW / "corpus.jsonl", encoding="utf-8")}
    mixed_gold_ids = {c["_id"] for c in gold_chunks}
    if ha_ids != mixed_gold_ids:
        raise SystemExit(f"gold 층 _id 불일치: paper-ha {len(ha_ids)} vs mixed {len(mixed_gold_ids)}")

    distractor_docs = []
    for src in DISTRACTOR_SRCS:
        print(f"distractor 청킹: {src.name} ...")
        docs = build_dir(src, stats)
        n = sum(len(d) for d in docs)
        print(f"  {len(docs)}편 {n}청크")
        distractor_docs.extend(docs)

    rng = random.Random(SEED)
    rng.shuffle(distractor_docs)

    # 중첩 적층: gold 전량 + distractor 문서를 순서대로 tier 크기까지
    corpus = list(gold_chunks)
    tier_ids: dict[int, list[str]] = {}
    tier_iter = iter(sorted(TIERS))
    target = next(tier_iter)
    for doc in distractor_docs:
        while target is not None and len(corpus) >= target:
            tier_ids[target] = [c["_id"] for c in corpus]
            target = next(tier_iter, None)
        corpus.extend(doc)
    while target is not None:
        tier_ids[target] = [c["_id"] for c in corpus]
        target = next(tier_iter, None)

    ids = [c["_id"] for c in corpus]
    if len(ids) != len(set(ids)):
        raise SystemExit("corpus에 중복 _id가 있다")
    print(f"\n총 코퍼스: {len(corpus)}청크 / tiers: "
          f"{ {k: len(v) for k, v in tier_ids.items()} }")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # git 전송용 분할 (서버에서 cat corpus.jsonl.part* > corpus.jsonl)
    size = corpus_path.stat().st_size
    parts = []
    if size > SPLIT_BYTES:
        with open(corpus_path, "rb") as f:
            i = 0
            while True:
                buf = f.readlines(SPLIT_BYTES)
                if not buf:
                    break
                part = OUT_DIR / f"corpus.jsonl.part{i:02d}"
                with open(part, "wb") as pf:
                    pf.writelines(buf)
                parts.append(part.name)
                i += 1
        print(f"corpus {size / 1e6:.0f}MB → {len(parts)}개 분할 "
              f"(서버: cat corpus.jsonl.part* > corpus.jsonl)")

    # 질의/정답: paper-ha 그대로 (gold 층 _id 동일)
    for name in ("queries.jsonl", "qrels.jsonl", "qa_meta.jsonl"):
        shutil.copy(GOLD_RAW / name, OUT_DIR / name)

    # 서브셋 (중첩, doc_ids 순서 = corpus 순서)
    subset_dir = DATA_DIR / "subsets" / "paper-mixed"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for tier, tids in tier_ids.items():
        with open(subset_dir / f"{tier // 1000}k.json", "w", encoding="utf-8") as f:
            json.dump({
                "subset_size": tier,
                "actual_size": len(tids),
                "gold_layer": len(gold_chunks),
                "nested": True,
                "note": "gold 층(Validation HA 전량) + seed 42 셔플 distractor 문서 적층",
                "doc_ids": tids,
            }, f, ensure_ascii=False)

    # 결정론 taxonomy (tier별, L1=doc_category)
    tax_by_id = {c["_id"]: {"L1": c["doc_category"], "L2": c["domain"],
                            "L3": c["doc_origin"][:40]} for c in corpus}
    tax_dir = DATA_DIR / "taxonomy"
    tax_dir.mkdir(parents=True, exist_ok=True)
    for tier, tids in tier_ids.items():
        with open(tax_dir / f"paper-mixed_{tier // 1000}k.json", "w", encoding="utf-8") as f:
            json.dump({tid: tax_by_id[tid] for tid in tids}, f, ensure_ascii=False)

    # taxonomy 스키마 (발견된 L1 값 그대로 — 질의 라우팅 빌더가 이 enum을 쓴다)
    l1_counts = Counter(c["doc_category"] for c in corpus)
    schema_path = BASE / "config" / "taxonomy_schemas" / "paper-mixed.yaml"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("dataset: paper-mixed\ndomain: korean_academic_papers_all\n\n")
        f.write("# 라벨 doc_category에서 결정론 생성 — build_paper_mixed.py가 자동 작성\n")
        f.write("L1:\n")
        for v in sorted(l1_counts):
            f.write(f"  - {v}    # {l1_counts[v]}청크\n")
        f.write("\nL2:\n")
        for v in sorted(l1_counts):
            f.write(f"  {v}:\n    - General\n")

    manifest = {
        "dataset": "paper-mixed",
        "purpose": "taxonomy 라우팅 pull backend 실험 (Part 5 taxroute) — distractor 규모 환경",
        "gold_layer": "paper-ha(Validation HA 331편, 4,253청크) — qrels/질의 50건 재사용",
        "distractors": "Training TL_HA/TL_SS/TL_ST 8,000편, seed 42 문서 셔플 적층",
        "tiers": {f"{k // 1000}k": len(v) for k, v in tier_ids.items()},
        "total_chunks": len(corpus),
        "taxonomy": "라벨 doc_category 유래 결정론 (ground-truth). L1 분포: "
                    + json.dumps(dict(l1_counts), ensure_ascii=False),
        "corpus_parts": parts,
        "limitations": [
            "질의 50건이 전부 gold 층(인문·예체능)에서 나옴 — 라우팅은 사실상 해당 분야로 수렴",
            "paper-ha와 동일한 추출 규칙 한계(공백 소실·생성 캡션·4096자 초과) 상속",
        ],
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"L1 분포: {dict(l1_counts)}")
    print(f"element 분포: {stats['by_element']}")
    print("완료.")


if __name__ == "__main__":
    main()
