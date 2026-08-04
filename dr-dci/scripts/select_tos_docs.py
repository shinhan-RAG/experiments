"""
parsed_md(9,417건) → 실험2(문서→청크·엘리먼트) QA용 대표 문서 선정.

- 파일명 기반 유형 분류(약관/사업방법서/공시·요약/코드형/기타) + 파일 크기 기반 소/중/대 버킷
- 유형×크기 매트릭스에서 대표 문서 선정 (문서 크기별 성능 열화 측정을 위해 크기 분산 확보)
- 359 QA 원본 계열(통합건강보장보험 원(ONE) 판매약관)은 증강 트랙 앵커로 반드시 포함
출력: data/raw/shinhan-tos/docs_selected.json
"""
import json
import os
import re
import unicodedata
from pathlib import Path


def winpath(p) -> str:
    """Windows 260자 경로 제한 회피용 확장 경로."""
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + os.path.abspath(s)
    return s

BASE = Path(__file__).resolve().parents[1]
SRC = BASE.parents[0] / "parsed_md"
OUT_DIR = BASE / "data" / "raw" / "shinhan-tos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODE_RE = re.compile(r"^(SHL\d+|\d{5,}|[A-Z]{2,4}\d{4,})")

# 유형 내 상대 백분위로 소/중/대 대표를 뽑는다 (유형별 크기 스케일이 크게 달라
# 절대 기준으로는 사업방법서 전체가 '소'로 뭉개짐 — 사업방법서 max 149KB vs 약관 max 5.8MB)
PCTL = {"소": 0.10, "중": 0.50, "대": 0.95}


def classify(name: str) -> str:
    n = unicodedata.normalize("NFC", name)
    if "판매약관" in n or ("약관" in n and "사업방법서" not in n):
        return "약관"
    if "사업방법서" in n:
        return "사업방법서"
    if "상품요약서" in n or "공시" in n or "요약서" in n:
        return "공시·요약"
    if CODE_RE.match(n):
        return "코드형"
    return "기타"


def main():
    docs = []
    for p in SRC.rglob("*.md"):
        st = os.stat(winpath(p))
        docs.append({
            "path": str(p.relative_to(SRC)),
            "name": unicodedata.normalize("NFC", p.stem),
            "type": classify(p.name),
            "bytes": st.st_size,
        })
    print(f"총 md: {len(docs)}")
    from collections import Counter
    print("유형 분포:", dict(Counter(d["type"] for d in docs)))

    selected = []
    seen_paths = set()

    def pick(d, bucket, reason):
        if d["path"] in seen_paths:
            return
        seen_paths.add(d["path"])
        selected.append({**d, "bucket": bucket, "reason": reason})

    # 1) 앵커: 359 QA 원본 계열 — 통합건강보장보험 원(ONE) 판매약관
    anchors = [d for d in docs
               if "통합건강보장보험" in d["name"] and "원(ONE)" in d["name"]
               and d["type"] == "약관"]
    anchors.sort(key=lambda d: -d["bytes"])
    for d in anchors[:2]:
        pick(d, "대", "앵커: 359 QA 원본 계열(원 ONE 판매약관) — 증강 트랙 기준 문서")

    # 2) 유형×크기(유형 내 백분위) 매트릭스 대표 선정 — 난수 미사용, 재현 가능
    quota = {"약관": 2, "사업방법서": 2, "공시·요약": 1, "코드형": 1, "기타": 1}
    for typ, per_bucket in quota.items():
        pool = sorted((d for d in docs if d["type"] == typ), key=lambda d: d["bytes"])
        if not pool:
            continue
        for bk, q in PCTL.items():
            base = int(len(pool) * q)
            picked = 0
            for idx in range(base, len(pool)):
                if pool[idx]["path"] not in seen_paths:
                    pick(pool[idx], bk, f"{typ}×{bk} 대표 (유형 내 p{int(q*100)} 부근)")
                    picked += 1
                    if picked >= per_bucket:
                        break

    # 3) 극단 크기 1개: 최대 약관 (문서 크기 열화 측정 상한)
    biggest = max((d for d in docs if d["type"] == "약관"), key=lambda d: d["bytes"])
    pick(biggest, "대", "최대 크기 약관 — 성능 열화 상한 측정용")

    print(f"\n선정: {len(selected)}건")
    for d in selected:
        print(f"  [{d['type']}/{d['bucket']}] {d['bytes']:>9,}B  {d['name'][:60]}")

    out = OUT_DIR / "docs_selected.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"src_root": str(SRC), "count": len(selected), "docs": selected},
                  f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
