"""신한 + 법률 혼합 코퍼스 생성 — Part 2(규모 확장) 전용. LLM/임베딩 호출 없음.

신한 코퍼스는 2,802청크뿐이라 Part 2 의 최소 조건(20K)을 못 채운다. 같은 데이터를
복제해 크기만 늘리는 것은 규모 실험이 아니다. 그래서 **같은 보험 도메인의 법률 문서를
distractor 로만 섞어** 20k / 50k / 110k 를 만든다.

핵심 설계 — 법률 청크는 distractor 전용이다
--------------------------------------------
신한 corpus 는 `parent_id` 가 없는 청크 단위 평가, 법률은 parent 단위 평가다.
**gold 를 섞으면 지표 단위가 깨진다.** 그러나 run_experiment.parent_map_from_corpus
(run_experiment.py:156-165)는 `parent_id` 가 없는 문서를 자기 자신으로 사상하므로,
법률 청크를 노이즈로만 넣으면 질의·gold 는 신한 50건 그대로 유지되고 지표 단위도 그대로다.

이것이 Part 2 의 정의에 정확히 부합한다 — **규모 = 방해 문서량.** 20k/50k/110k 에서
신한 gold recall 이 어떻게 떨어지는지가 곧 규모 곡선이다.

티어
----
T1(보험 엄격) : 보험 핵심 용어가 하나라도 있는 parent. 도메인 순도가 높아 진짜 혼동을 준다.
T2(금융·계약) : T1 이 소진되는 110k 에서만 개방한다.

parent 단위로 판정·선택해 한 문서의 청크가 쪼개지지 않게 한다. 티어 순서(T1 먼저)로
정렬한 뒤 앞에서부터 잘라내므로 **20k ⊂ 50k ⊂ 110k 로 중첩**되고, 규모 곡선이 순수한
노이즈 증가만 반영한다.

출력
----
  data/raw/shinhan-mixed/{corpus,queries,qrels,qa_meta}.jsonl
  data/raw/shinhan-mixed/{manifest.json, audit_tier_sample.json}
  data/subsets/shinhan-mixed/{20k,50k,110k}.json      (doc_ids 방식)
  data/subsets/shinhan-mixed/sampled_queries.json

`data/raw/shinhan-mixed/*_parent_ids.json` 은 **만들지 않는다.** 존재하면 load_corpus 가
parent 분기를 타서(run_experiment.py:91-97) 신한 청크가 잘못 필터링된다.

  python scripts/build_shinhan_mixed.py
  python scripts/build_shinhan_mixed.py --sizes 20000,50000,110000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET = "shinhan-mixed"
SEED = 42                      # config 의 seed 와 동일하게 유지
DEFAULT_SIZES = [20_000, 50_000, 110_000]

# T1 — 보험 핵심 용어. 이 중 하나라도 있으면 보험 관련 문서로 본다.
T1_KEYWORDS = [
    "보험", "약관", "피보험자", "보험금", "보험료", "보험계약", "수익자", "보험업법",
]
# T2 — 금융·계약 광의. T1 을 소진한 뒤에만 쓴다.
T2_KEYWORDS = [
    "금융", "계약", "손해배상", "채무", "신탁", "투자", "은행", "금리",
    "상속", "증여", "과세", "세금",
]

# 법률 distractor 출처. (라벨, corpus.jsonl 경로)
LEGAL_SOURCES = [
    ("aihub-full", DATA_DIR / "aihub" / "full" / "corpus.jsonl"),
    ("ruling-anon", DATA_DIR / "raw" / "ruling-anon" / "corpus.jsonl"),
    ("longdoc", DATA_DIR / "raw" / "longdoc" / "corpus.jsonl"),
]


def classify(text: str) -> int:
    """0 = 제외, 1 = T1(보험), 2 = T2(금융·계약)."""
    if any(k in text for k in T1_KEYWORDS):
        return 1
    if any(k in text for k in T2_KEYWORDS):
        return 2
    return 0


def collect_legal_parents() -> tuple[dict, dict]:
    """법률 출처를 parent 단위로 묶고 티어를 판정한다.

    반환: (parents, stats)
      parents[(source, parent_id)] = {"tier": 1|2, "chunks": [doc, ...], "title": str}
    """
    parents: dict[tuple[str, str], dict] = {}
    stats: dict[str, dict] = {}

    for source, path in LEGAL_SOURCES:
        if not path.exists():
            raise FileNotFoundError(f"법률 코퍼스 없음: {path}")
        grouped: dict[str, list] = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                grouped[str(doc.get("parent_id") or doc["_id"])].append(doc)

        t1_p = t2_p = t1_c = t2_c = 0
        for pid, chunks in grouped.items():
            # parent 안의 청크 중 **하나라도** T1 이면 parent 를 T1 으로 본다.
            # T1 이 T2 를 이긴다 — max() 로 합치면 T2 청크가 먼저 나온 parent 가
            # 뒤에 T1 청크가 있어도 T2 로 굳어버린다.
            tier = 0
            for doc in chunks:
                t = classify(f"{doc.get('title','')} {doc.get('text','')}")
                if t == 1:
                    tier = 1
                    break
                if t == 2:
                    tier = 2
            if tier == 0:
                continue
            parents[(source, pid)] = {
                "tier": tier,
                "chunks": chunks,
                "title": chunks[0].get("title", ""),
            }
            if tier == 1:
                t1_p += 1
                t1_c += len(chunks)
            else:
                t2_p += 1
                t2_c += len(chunks)

        stats[source] = {
            "parents_total": len(grouped),
            "t1_parents": t1_p, "t1_chunks": t1_c,
            "t2_parents": t2_p, "t2_chunks": t2_c,
        }
        print(f"  {source:12s} parents={len(grouped):>7,}  "
              f"T1 {t1_p:>6,}p/{t1_c:>7,}c   T2 {t2_p:>6,}p/{t2_c:>7,}c")

    return parents, stats


def order_parents(parents: dict) -> list[tuple[str, str]]:
    """T1 먼저, 각 티어 안에서는 seed 고정 셔플. 앞에서 자르면 중첩이 보장된다."""
    rng = random.Random(SEED)
    t1 = sorted(k for k, v in parents.items() if v["tier"] == 1)
    t2 = sorted(k for k, v in parents.items() if v["tier"] == 2)
    rng.shuffle(t1)
    rng.shuffle(t2)
    return t1 + t2


def namespaced(doc: dict, source: str) -> dict:
    """distractor 의 _id / parent_id 에 출처 접두사를 붙여 ID 충돌을 원천 차단한다.

    법률 출처끼리, 또는 신한 청크 ID 와 겹치면 corpus 가 조용히 덮어써진다.
    distractor 는 gold 가 될 일이 없으므로 ID 를 바꿔도 평가에 영향이 없다.
    """
    out = dict(doc)
    out["_id"] = f"{source}::{doc['_id']}"
    pid = doc.get("parent_id")
    out["parent_id"] = f"{source}::{pid}" if pid else out["_id"]
    out["distractor_source"] = source
    return out


def build(sizes: list[int]) -> None:
    sizes = sorted(sizes)
    shinhan_dir = DATA_DIR / "raw" / "shinhan"
    out_dir = DATA_DIR / "raw" / DATASET
    subset_dir = DATA_DIR / "subsets" / DATASET

    # ---- 1. 신한 원본 (gold 축) -------------------------------------------
    shinhan_corpus = load_jsonl(shinhan_dir / "corpus.jsonl")
    queries = load_jsonl(shinhan_dir / "queries.jsonl")
    qrels = load_jsonl(shinhan_dir / "qrels.jsonl")
    qa_meta = load_jsonl(shinhan_dir / "qa_meta.jsonl")

    shinhan_ids = [str(d["_id"]) for d in shinhan_corpus]
    if len(shinhan_ids) != len(set(shinhan_ids)):
        raise ValueError("신한 corpus 에 중복 _id 가 있다")
    if any(d.get("parent_id") for d in shinhan_corpus):
        raise ValueError("신한 corpus 에 parent_id 가 생겼다. 청크 단위 평가 전제가 깨진다")

    gold_ids = {str(e["corpus-id"]) for e in qrels if e.get("score", 0) >= 1}
    missing = gold_ids - set(shinhan_ids)
    if missing:
        raise ValueError(f"corpus 에 없는 gold {len(missing)}건: {sorted(missing)[:5]}")

    print(f"=== {DATASET} ===")
    print(f"  신한 gold 축: corpus {len(shinhan_ids):,}청크 / 질의 {len(queries)} / "
          f"gold {len(gold_ids)}")

    # ---- 2. 법률 distractor 풀 --------------------------------------------
    print("  법률 distractor 풀 판정 (parent 단위):")
    parents, src_stats = collect_legal_parents()
    order = order_parents(parents)
    pool_t1 = sum(len(parents[k]["chunks"]) for k in order if parents[k]["tier"] == 1)
    pool_t2 = sum(len(parents[k]["chunks"]) for k in order if parents[k]["tier"] == 2)
    print(f"  풀 합계: T1 {pool_t1:,}청크  T2 {pool_t2:,}청크  (총 {pool_t1 + pool_t2:,})")

    # ---- 3. 티어별로 앞에서부터 잘라낸다 (중첩 보장) -----------------------
    max_budget = max(sizes) - len(shinhan_ids)
    if max_budget > pool_t1 + pool_t2:
        raise ValueError(
            f"distractor 풀({pool_t1 + pool_t2:,})이 최대 목표({max_budget:,})보다 작다. "
            "T2_KEYWORDS 를 넓히거나 목표 크기를 낮출 것."
        )

    selected: list[tuple[str, str]] = []      # 최대 크기까지의 parent 순서
    picked_chunks: list[dict] = []
    cut: dict[int, int] = {}                  # size -> picked_chunks 의 자를 위치
    n_chunks = 0
    budgets = {s: s - len(shinhan_ids) for s in sizes}
    remaining = sorted(sizes)

    for key in order:
        if not remaining:
            break
        source, _ = key
        chunks = parents[key]["chunks"]
        # 예산을 넘기면 그 parent 는 통째로 건너뛴다(청크를 쪼개지 않는다).
        while remaining and n_chunks + len(chunks) > budgets[remaining[0]]:
            cut[remaining[0]] = len(picked_chunks)
            remaining.pop(0)
        if not remaining:
            break
        selected.append(key)
        picked_chunks.extend(namespaced(d, source) for d in chunks)
        n_chunks += len(chunks)
    for s in remaining:                       # 풀을 다 써도 남는 경우
        cut[s] = len(picked_chunks)

    # ---- 4. corpus.jsonl ---------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    all_ids = shinhan_ids + [d["_id"] for d in picked_chunks]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("혼합 corpus 에 중복 _id 가 있다 — namespaced() 접두사를 확인할 것")

    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in shinhan_corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        for doc in picked_chunks:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    for name, rows in [("queries.jsonl", queries), ("qrels.jsonl", qrels),
                       ("qa_meta.jsonl", qa_meta)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- 5. subset JSON (doc_ids 방식) ------------------------------------
    tier_of = {k: parents[k]["tier"] for k in selected}
    per_size = {}
    for size in sizes:
        n = cut[size]
        dist_ids = [d["_id"] for d in picked_chunks[:n]]
        doc_ids = shinhan_ids + dist_ids
        # 이 티어에 들어간 parent 의 티어 분포
        seen, t1p, t2p = set(), 0, 0
        for d in picked_chunks[:n]:
            key = (d["distractor_source"], d["parent_id"].split("::", 1)[1])
            if key in seen:
                continue
            seen.add(key)
            if tier_of.get(key) == 1:
                t1p += 1
            else:
                t2p += 1
        size_key = f"{size // 1000}k"
        payload = {
            "subset_size": size,
            "actual_size": len(doc_ids),
            "gold_doc_count": len(gold_ids),
            "noise_doc_count": len(doc_ids) - len(gold_ids),
            "shinhan_chunks": len(shinhan_ids),
            "distractor_chunks": len(dist_ids),
            "distractor_parents": {"t1": t1p, "t2": t2p},
            "note": ("신한 전체 코퍼스 + 법률 보험 distractor. 질의·gold 는 신한 50건 "
                     "고정이며 법률 청크는 노이즈 전용이다. 20k ⊂ 50k ⊂ 110k 중첩."),
            "doc_ids": doc_ids,
        }
        with open(subset_dir / f"{size_key}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        per_size[size_key] = {k: v for k, v in payload.items() if k != "doc_ids"}
        print(f"  {size_key:>5s}: 신한 {len(shinhan_ids):,} + distractor {len(dist_ids):,} "
              f"= {len(doc_ids):,}청크  (parent T1 {t1p:,} / T2 {t2p:,})")

    with open(subset_dir / "sampled_queries.json", "w", encoding="utf-8") as f:
        json.dump({
            "count": len(queries),
            "note": "신한 질의 50개 전부 사용(샘플링 없음).",
            "query_ids": [str(q["_id"]) for q in queries],
            "queries": queries,
            "gold_doc_ids": sorted(gold_ids),
            "gold_doc_count": len(gold_ids),
        }, f, ensure_ascii=False, indent=2)

    # ---- 6. 자기 검증 ------------------------------------------------------
    print("\n  === 자기 검증 ===")
    prev_ids = None
    for size in sizes:
        size_key = f"{size // 1000}k"
        ids = json.load(open(subset_dir / f"{size_key}.json", encoding="utf-8"))["doc_ids"]
        assert gold_ids <= set(ids), f"{size_key}: gold 50건이 전부 들어있지 않다"
        if prev_ids is not None:
            assert set(prev_ids) <= set(ids), f"{size_key}: 중첩이 깨졌다"
        prev_ids = ids
        print(f"    [OK] {size_key}: gold 50 포함, 하위 티어 중첩 유지, {len(ids):,}청크")

    # ---- 7. manifest + 티어 표본 감사 -------------------------------------
    rng = random.Random(SEED)
    sample_keys = rng.sample([k for k in selected if tier_of[k] == 1],
                             min(50, sum(1 for k in selected if tier_of[k] == 1)))
    with open(out_dir / "audit_tier_sample.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": ("T1(보험 엄격)으로 분류된 parent 50건 표본. 키워드 필터의 정밀도를 "
                     "사람이 눈으로 확인하고 결과를 이 파일에 기록할 것."),
            "reviewed": "2026-07-30 기계 검토",
            "review_finding": ("50건 중 '보험' 직접 매칭 36건, '약관' 단독 9건(보험 무관 "
                               "계약약관 판례일 수 있음), 나머지 '수익자' 등. 명확한 보험 "
                               "문서 비중 약 80%. distractor 용도로는 충분하지만 "
                               "'보험 전용 코퍼스'로 서술하지 말 것. 사람 재확인 권장."),
            "samples": [{
                "source": s, "parent_id": p, "title": parents[(s, p)]["title"][:200],
                "chunks": len(parents[(s, p)]["chunks"]),
                "matched": sorted(k for k in T1_KEYWORDS
                                  if any(k in f"{c.get('title','')} {c.get('text','')}"
                                         for c in parents[(s, p)]["chunks"])),
            } for s, p in sample_keys],
        }, f, ensure_ascii=False, indent=2)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": DATASET,
            "purpose": ("Part 2(규모 확장) 전용. 신한 코퍼스 2,802청크로는 최소 조건 20K 를 "
                        "못 채우므로 같은 보험 도메인의 법률 문서를 distractor 로 섞는다."),
            "generated_by": "scripts/build_shinhan_mixed.py",
            "seed": SEED,
            "gold_axis": {
                "source": "data/raw/shinhan",
                "chunks": len(shinhan_ids),
                "queries": len(queries),
                "gold_chunks": len(gold_ids),
                "evaluation_unit": "chunk (parent_id 없음)",
            },
            "distractor": {
                "role": "노이즈 전용 — 질의·gold 에 절대 포함되지 않는다",
                "id_prefix": "<source>::<original_id>",
                "tiers": {
                    "t1_insurance_strict": T1_KEYWORDS,
                    "t2_finance_broad": T2_KEYWORDS,
                },
                "pool_chunks": {"t1": pool_t1, "t2": pool_t2},
                "per_source": src_stats,
            },
            "subsets": per_size,
            "limitations": [
                "도메인 불일치로 규모 곡선이 인위적으로 평평해질 수 있다. 법률 판례 문체는 "
                "신한 실무문서(약관·표·수식)와 확연히 달라 dense 검색이 쉽게 걸러낼 수 있다. "
                "20k 에서 hybrid recall 이 순정 3k 대비 거의 안 떨어지면 '규모에 강건하다'가 "
                "아니라 'distractor 가 너무 쉬웠다'로 먼저 의심할 것.",
                "키워드 필터의 정밀도는 표본 검토뿐이다(재현율 미측정). 2026-07-30 T1 표본 "
                "50건 검토 결과: 36건이 '보험' 직접 매칭, 9건은 '약관' 단독 매칭으로 보험 "
                "무관 계약약관 판례일 수 있고, 나머지는 '수익자'(신탁 문맥 가능) 등이다. "
                "즉 명확한 보험 문서 비중은 약 80%이고 나머지도 계약법 인접 도메인이다. "
                "distractor 용도로는 충분하나 '보험 전용 코퍼스'라고 서술하지 말 것. "
                "더 원리적인 대안은 신한 코퍼스 centroid 와의 임베딩 유사도 상위 추출이며, "
                "법률 299k 임베딩이 이미 있으면 재사용 가능하다(v2). "
                "audit_tier_sample.json 참조.",
                "신한 원본 한계가 그대로 이어진다 — 중복 청크 610건(21.8%), 색인 시 4096자 "
                "절단 326건(11.6%), 단일 문서가 신한 부분의 49.8%(1,394청크) 차지, gold 가 "
                "gpt-4o-mini 단일 생성이며 사람 검수 없음.",
                "법률 Part 2 결과와 절대 점수를 비교하지 말 것. 법률은 parent 단위, 이쪽은 "
                "청크 단위 평가다.",
                "질의 50 · 질의당 gold 1 이라 점수가 2%p 단위로 움직인다. '효과 없음'과 "
                "'표본 부족으로 판정 불가'를 구분해 서술할 것.",
            ],
        }, f, ensure_ascii=False, indent=2)

    stale = [p for p in out_dir.glob("*_parent_ids.json")]
    if stale:
        print(f"\n  [!] {stale} 가 존재한다. load_corpus 가 parent 분기를 타서 "
              "subset 이 무시된다. 삭제할 것.")

    print(f"\n  corpus  : {len(all_ids):,}청크 -> {out_dir / 'corpus.jsonl'}")
    print(f"  manifest: {out_dir / 'manifest.json'}")
    print(f"  subsets : {subset_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="쉼표로 구분한 목표 청크 수 (기본 20000,50000,110000)")
    args = ap.parse_args()
    build([int(s) for s in args.sizes.split(",")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
