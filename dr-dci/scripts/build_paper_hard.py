"""paper-hard: gold와 같은 분야(인문학·예술체육학) distractor만 남긴 변형 코퍼스.

배경 (8/2 paper-mixed 결과의 근본 원인 2 대응)
--------------------------------------------
paper-mixed에서 dense는 110k에서도 거의 안 무너졌다(R@20 0.96) — 타 분야
distractor가 임베딩 공간에서 잘 분리되기 때문. 열화의 실제 조건은 "유사문서
밀집"(법률 Part 2에서 관측)이므로, gold와 같은 분야 논문만으로 distractor를
채워 dense가 실제로 스트레스받는 환경을 만든다.

- 소스: 로컬 data/raw/paper-mixed/corpus.jsonl (AI Hub 원천 재청킹 불필요)
- 포함: gold 층(paper-ha 4,253) 전량 + doc_category ∈ {인문학, 예술체육학} distractor
- 순서: paper-mixed corpus 순서 그대로 → 10k ⊂ 20k ⊂ 38k(전량) 중첩, parent 경계 컷
- 해석 유의: 라우팅은 이 환경에서 후보를 거의 못 줄인다(전부 같은 분야) —
  이 데이터의 목적은 라우팅 이득 측정이 아니라 "규모 열화 전제" 검증이다.
  dense 곡선이 무너지는지, hybrid·routed가 그때 어떤지가 판독 대상.

출력:
  data/raw/paper-hard/{corpus,queries,qrels,qa_meta}.jsonl + manifest.json
  data/raw/paper-hard/corpus.jsonl.partNN (85MB 초과 시)
  data/subsets/paper-hard/{10k,20k,38k}.json
  data/taxonomy/paper-hard_{10k,20k,38k}.json
  config/taxonomy_schemas/paper-hard.yaml

  python scripts/build_paper_hard.py
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
SRC_DIR = DATA_DIR / "raw" / "paper-mixed"
GOLD_DIR = DATA_DIR / "raw" / "paper-ha"
OUT_DIR = DATA_DIR / "raw" / "paper-hard"
GOLD_CATEGORIES = {"인문학", "예술체육학"}
TIERS = [10_000, 20_000, 38_000]   # 38k = 전량(±parent 경계)
SPLIT_BYTES = 85 * 1024 * 1024


def main() -> None:
    gold_ids = {json.loads(l)["_id"]
                for l in open(GOLD_DIR / "corpus.jsonl", encoding="utf-8")}
    print(f"gold 층: {len(gold_ids):,}청크 (paper-ha)")

    corpus = []
    dropped = 0
    with open(SRC_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["_id"] in gold_ids or d.get("doc_category") in GOLD_CATEGORIES:
                corpus.append(d)
            else:
                dropped += 1
    ids = [c["_id"] for c in corpus]
    assert len(ids) == len(set(ids)), "중복 _id"
    assert gold_ids <= set(ids), "gold 누락"
    print(f"필터 결과: {len(corpus):,}청크 유지 / {dropped:,} 제외 "
          f"(카테고리 {dict(Counter(c.get('doc_category') for c in corpus))})")

    # 중첩 tier — 문서(doc) 경계에서 컷 (paper-mixed 순서 보존이 중첩을 보장.
    # paper 계열 청크는 parent_id가 없고 원문서 ID가 'doc' 필드에 있다)
    def parent_of(i):
        return corpus[i].get("doc") or corpus[i].get("parent_id") or corpus[i]["_id"]

    tier_cut = {}
    for target in TIERS:
        n = min(target, len(corpus))
        while 0 < n < len(corpus) and parent_of(n - 1) == parent_of(n):
            n -= 1
        tier_cut[target] = n
    print(f"tiers: { {f'{t//1000}k': n for t, n in tier_cut.items()} }")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    for name in ("queries.jsonl", "qrels.jsonl", "qa_meta.jsonl"):
        shutil.copy(SRC_DIR / name, OUT_DIR / name)

    subset_dir = DATA_DIR / "subsets" / "paper-hard"
    subset_dir.mkdir(parents=True, exist_ok=True)
    tax_dir = DATA_DIR / "taxonomy"
    tax_dir.mkdir(parents=True, exist_ok=True)
    prev = None
    for target, n in tier_cut.items():
        size_key = f"{target // 1000}k"
        doc_ids = ids[:n]
        assert gold_ids <= set(doc_ids), f"{size_key}: gold 미포함"
        if prev is not None:
            assert set(prev) <= set(doc_ids), f"{size_key}: 중첩 깨짐"
        prev = doc_ids
        with open(subset_dir / f"{size_key}.json", "w", encoding="utf-8") as f:
            json.dump({
                "subset_size": target,
                "actual_size": n,
                "gold_layer": len(gold_ids),
                "nested": True,
                "note": "gold 전량 + 같은 분야(인문·예체능) distractor. paper-mixed 순서 보존.",
                "doc_ids": doc_ids,
            }, f, ensure_ascii=False)
        tax = {c["_id"]: {"L1": c.get("doc_category", "?"),
                          "L2": c.get("domain", ""),
                          "L3": (c.get("doc_origin") or "")[:40]}
               for c in corpus[:n]}
        with open(tax_dir / f"paper-hard_{size_key}.json", "w", encoding="utf-8") as f:
            json.dump(tax, f, ensure_ascii=False)
        print(f"  {size_key}: {n:,}청크 subset+taxonomy 저장")

    l1_counts = Counter(c.get("doc_category") for c in corpus)
    schema_path = BASE / "config" / "taxonomy_schemas" / "paper-hard.yaml"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write("dataset: paper-hard\ndomain: korean_academic_papers_same_field\n\n")
        f.write("# build_paper_hard.py 자동 작성 — gold와 같은 분야만 남긴 변형\n")
        f.write("L1:\n")
        for v in sorted(l1_counts):
            f.write(f"  - {v}    # {l1_counts[v]}청크\n")
        f.write("\nL2:\n")
        for v in sorted(l1_counts):
            f.write(f"  {v}:\n    - General\n")

    size = corpus_path.stat().st_size
    parts = []
    for old in OUT_DIR.glob("corpus.jsonl.part*"):
        old.unlink()
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
        print(f"corpus {size / 1e6:.0f}MB → {len(parts)}개 분할")

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "paper-hard",
            "purpose": ("규모 열화 전제 검증 — gold와 같은 분야 distractor로 dense가 "
                        "실제로 무너지는 조건을 만든다 (paper-mixed 근본 원인 2 대응)"),
            "source": "data/raw/paper-mixed/corpus.jsonl 필터 (재청킹 없음)",
            "gold_layer": len(gold_ids),
            "categories": dict(l1_counts),
            "tiers": {f"{t // 1000}k": n for t, n in tier_cut.items()},
            "corpus_parts": parts,
            "limitations": [
                "라우팅은 이 환경에서 후보를 거의 못 줄인다(전 문서가 같은 분야) — "
                "라우팅 이득 측정용이 아니라 dense 열화 곡선 측정용",
                "paper-mixed의 추출 규칙 한계 상속",
            ],
        }, f, ensure_ascii=False, indent=2)
    print("완료.")


if __name__ == "__main__":
    main()
