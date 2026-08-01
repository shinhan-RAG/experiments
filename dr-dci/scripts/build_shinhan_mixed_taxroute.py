"""shinhan-mixed taxonomy 라우팅 실험(Part 5 taxroute) 준비 — LLM/임베딩 호출 없음.

기존 shinhan-mixed corpus(신한 gold 2,802 + 법률 distractor, build_shinhan_mixed.py)를
그대로 쓰되, 라우팅 실험에 필요한 산출물을 결정론적으로 생성한다.

전제 — corpus 순서가 곧 subset 정의다
--------------------------------------
build_shinhan_mixed.py 는 corpus 를 [신한 2,802 전량 → distractor 선택 순서]로 쓰고,
tier cut 은 그 시점의 누적 길이였다. 따라서 manifest.subsets 의 actual_size 만큼
앞에서 자르면 원래 subset doc_ids 가 그대로 복원된다(중첩 자동 보장).
경계에서 parent 가 쪼개지지 않았는지 assert 로 재확인한다.

taxonomy (결정론 — ground truth)
--------------------------------
- 신한 gold 청크: title 접두어 → L1
    약관→Terms / 상품→Product / 보험금심사→ClaimsReview /
    고객상담→CustomerService / LICO→SalesEducation
- 법률 distractor(`distractor_source` 보유): L1=Legal, L2=출처(aihub-full 등)
"분류가 틀려서"라는 교란이 없는 라벨이므로 routing_accuracy 의 분모로 신뢰할 수 있다.

출력
----
  data/subsets/shinhan-mixed/{20k,50k,110k}.json + sampled_queries.json
  data/taxonomy/shinhan-mixed_{20k,50k,110k}.json      {chunk_id: {L1,L2,L3}}
  config/taxonomy_schemas/shinhan-mixed.yaml
  data/raw/shinhan-mixed/corpus.jsonl.partNN           (git 전송용 85MB 분할)

  python scripts/build_shinhan_mixed_taxroute.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw" / "shinhan-mixed"
SPLIT_BYTES = 85 * 1024 * 1024  # git 파일 한도(100MB) 미만

PREFIX_L1 = {
    "약관": "Terms",
    "상품": "Product",
    "보험금심사": "ClaimsReview",
    "고객상담": "CustomerService",
    "LICO": "SalesEducation",
}

# 질의 라우터(Qwen)에게 줄 L2 힌트 — shinhan.yaml 의 토픽을 요약.
# Legal 은 distractor 전용 카테고리임을 힌트로 명시한다(보험 질의가 흡수되지 않도록).
L2_HINTS = {
    "Terms": ["Coverage", "Exclusion", "PolicyChange", "Premium", "Surrender", "Definition"],
    "Product": ["ProductSpec", "Pricing_Actuarial", "SalesRule", "ProductStrategy", "Approval"],
    "ClaimsReview": ["ReviewGuideline", "PaymentCriteria", "Dispute_Precedent", "Reimbursement"],
    "CustomerService": ["ClaimIntake", "Script", "ContractChange", "WritingGuide"],
    "SalesEducation": ["TrainingMaterial", "Brochure", "SalesNews", "NeedsAnalysis"],
    "Legal": ["CourtRuling", "Statute", "LegalQA (distractor-only: court rulings, not insurer documents)"],
}


def classify(doc: dict) -> dict:
    """청크 1건의 ground-truth taxonomy."""
    if doc.get("distractor_source") or "::" in str(doc["_id"]):
        source = doc.get("distractor_source") or str(doc["_id"]).split("::", 1)[0]
        return {"L1": "Legal", "L2": source, "L3": doc.get("title", "")[:40]}
    title = doc.get("title", "")
    prefix = title.split("__", 1)[0] if "__" in title else ""
    l1 = PREFIX_L1.get(prefix)
    if l1 is None:
        raise SystemExit(f"미지의 신한 title 접두어: {prefix!r} ({doc['_id']})")
    return {"L1": l1, "L2": "General", "L3": title.split("__", 1)[-1][:40]}


def main() -> None:
    with open(RAW_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    sizes = {k: v["actual_size"] for k, v in manifest["subsets"].items()}  # {"20k": 19999, ...}
    n_shinhan = manifest["gold_axis"]["chunks"]

    print("corpus 로드...")
    corpus = load_jsonl(RAW_DIR / "corpus.jsonl")
    ids = [str(d["_id"]) for d in corpus]
    assert len(ids) == len(set(ids)), "corpus 중복 _id"
    print(f"  {len(corpus):,}청크 (신한 {n_shinhan:,} + distractor {len(corpus) - n_shinhan:,})")

    # ---- 검증: 앞 n_shinhan 은 전부 신한(비-distractor), 이후는 전부 distractor ----
    for d in corpus[:n_shinhan]:
        assert "::" not in str(d["_id"]), f"신한 구간에 distractor: {d['_id']}"
    for d in corpus[n_shinhan:]:
        assert d.get("distractor_source"), f"distractor 구간에 무출처 청크: {d['_id']}"

    qrels = load_jsonl(RAW_DIR / "qrels.jsonl")
    queries = load_jsonl(RAW_DIR / "queries.jsonl")
    gold_ids = {str(e["corpus-id"]) for e in qrels if e.get("score", 0) >= 1}
    assert gold_ids <= set(ids[:n_shinhan]), "gold 가 신한 구간에 없다"

    # ---- taxonomy (전 청크 결정론) -----------------------------------------
    tax_by_id = {str(d["_id"]): classify(d) for d in corpus}
    l1_counts = Counter(t["L1"] for t in tax_by_id.values())
    print(f"  L1 분포: {dict(l1_counts)}")

    # ---- subsets 재생성 (corpus 순서 = 원래 적층 순서) ----------------------
    subset_dir = DATA_DIR / "subsets" / "shinhan-mixed"
    subset_dir.mkdir(parents=True, exist_ok=True)
    tax_dir = DATA_DIR / "taxonomy"
    tax_dir.mkdir(parents=True, exist_ok=True)

    for size_key in sorted(sizes, key=lambda k: sizes[k]):
        n = sizes[size_key]
        # 경계에서 parent 가 쪼개지지 않았는지 (원래 빌드는 parent 단위로 잘랐다)
        if n < len(corpus):
            left = corpus[n - 1].get("parent_id", corpus[n - 1]["_id"])
            right = corpus[n].get("parent_id", corpus[n]["_id"])
            assert left != right, f"{size_key}: 경계에서 parent 분할 — 재생성 전제가 깨짐"
        doc_ids = ids[:n]
        assert gold_ids <= set(doc_ids)
        meta = manifest["subsets"][size_key]
        with open(subset_dir / f"{size_key}.json", "w", encoding="utf-8") as f:
            json.dump({
                **{k: v for k, v in meta.items()},
                "regenerated_by": "build_shinhan_mixed_taxroute.py (corpus 순서 기반 복원)",
                "doc_ids": doc_ids,
            }, f, ensure_ascii=False)
        with open(tax_dir / f"shinhan-mixed_{size_key}.json", "w", encoding="utf-8") as f:
            json.dump({tid: tax_by_id[tid] for tid in doc_ids}, f, ensure_ascii=False)
        sub_l1 = Counter(tax_by_id[tid]["L1"] for tid in doc_ids)
        print(f"  {size_key:>5s}: {n:,}청크 저장 / L1 {dict(sub_l1)}")

    with open(subset_dir / "sampled_queries.json", "w", encoding="utf-8") as f:
        json.dump({
            "count": len(queries),
            "note": "신한 질의 50개 전부 사용(샘플링 없음).",
            "query_ids": [str(q["_id"]) for q in queries],
            "queries": queries,
            "gold_doc_ids": sorted(gold_ids),
            "gold_doc_count": len(gold_ids),
        }, f, ensure_ascii=False, indent=2)

    # ---- taxonomy 스키마 (라우터 enum) -------------------------------------
    schema_path = BASE / "config" / "taxonomy_schemas" / "shinhan-mixed.yaml"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("dataset: shinhan-mixed\ndomain: korean_insurance_with_legal_distractors\n\n")
        f.write("# 결정론 생성 — build_shinhan_mixed_taxroute.py 가 자동 작성.\n")
        f.write("# 신한 gold 는 title 접두어 유래, Legal 은 distractor 전용 카테고리.\n")
        f.write("L1:\n")
        for v in sorted(l1_counts):
            f.write(f"  - {v}    # {l1_counts[v]}청크\n")
        f.write("\nL2:\n")
        for v in sorted(l1_counts):
            f.write(f"  {v}:\n")
            for hint in L2_HINTS.get(v, ["General"]):
                f.write(f"    - {hint}\n")

    # ---- corpus 분할 (git 전송용) ------------------------------------------
    corpus_path = RAW_DIR / "corpus.jsonl"
    size = corpus_path.stat().st_size
    parts = []
    for old in RAW_DIR.glob("corpus.jsonl.part*"):
        old.unlink()
    if size > SPLIT_BYTES:
        with open(corpus_path, "rb") as f:
            i = 0
            while True:
                buf = f.readlines(SPLIT_BYTES)
                if not buf:
                    break
                part = RAW_DIR / f"corpus.jsonl.part{i:02d}"
                with open(part, "wb") as pf:
                    pf.writelines(buf)
                parts.append(part.name)
                i += 1
        print(f"corpus {size / 1e6:.0f}MB → {len(parts)}개 분할 "
              f"(서버: cat corpus.jsonl.part* > corpus.jsonl)")

    # ---- manifest 에 taxroute 섹션 추가 ------------------------------------
    manifest["taxroute"] = {
        "generated_by": "scripts/build_shinhan_mixed_taxroute.py",
        "taxonomy": "결정론 ground-truth — 신한 title 접두어 L1 + 법률 distractor Legal",
        "l1_distribution": dict(l1_counts),
        "corpus_parts": parts,
        "routing_note": ("라우팅이 사실상 보험(5분류) vs Legal 이진에 가깝다 — "
                         "routing_accuracy 높게 나올 것. '라우팅이 정확할 때의 상한 효과'로 해석."),
    }
    with open(RAW_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("완료.")


if __name__ == "__main__":
    main()
