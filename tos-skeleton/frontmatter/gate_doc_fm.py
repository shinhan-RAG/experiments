#!/usr/bin/env python3
"""문서별 frontmatter 품질 게이트 — 규칙 추출물의 오탐·누락·오염 검사.

검사:
 G1 특약명 신뢰도: 코퍼스 교차 빈도(여러 문서에서 재등장=표준 명칭) + 절단 의심(조사/어미로 시작)
 G2 조 번호 연속성: 제1조..제N조 결번율 (세그먼트 누락 신호)
 G3 채움 이상치: A유형인데 특약·조 빈약, B유형인데 내용 없음
 G4 오염 플래그: 헤딩·항목의 비한글(환각) 비율
출력: out/gate_doc_fm.json + 콘솔 리포트
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")

BAD_START = re.compile(r"^(은|는|이|가|을|를|의|에|로|와|과|한|할|및|또는|경우|따라|위한)")
NONKO = re.compile(r"[一-鿿]|[A-Za-z]{6,}")


def main():
    rows = [json.loads(l) for l in open(OUT / "doc_frontmatter.jsonl")]

    # ── G1: 특약명 교차 빈도
    rider_freq = Counter()
    for r in rows:
        if r["ftype"] == "A":
            for t in r["fm"]["특약구성"]:
                rider_freq[t["name"]] += 1
    uniq_riders = len(rider_freq)
    singleton = [n for n, c in rider_freq.items() if c == 1]
    truncated = [n for n in rider_freq if BAD_START.match(n)]
    print(f"G1 특약명: 고유 {uniq_riders:,}종 (연인원 {sum(rider_freq.values()):,})")
    print(f"   1회 등장(오탐 후보) {len(singleton):,}종 ({len(singleton)/uniq_riders*100:.0f}%)")
    print(f"   절단 의심(조사 시작) {len(truncated):,}종  예: {truncated[:5]}")
    print(f"   최다 등장: {[n for n,_ in rider_freq.most_common(3)]}")

    # ── G2: 조 번호 연속성 (A유형)
    gap_rates = []
    for r in rows:
        if r["ftype"] != "A":
            continue
        nums = set()
        for j in r["fm"]["조"]:
            m = re.match(r"제(\d+)조", j)
            if m:
                nums.add(int(m.group(1)))
        if len(nums) >= 10:
            expected = max(nums) - min(nums) + 1
            gap_rates.append(1 - len(nums) / expected)
    import statistics
    bad_gap = sum(1 for g in gap_rates if g > 0.3)
    print(f"\nG2 조 연속성 (A {len(gap_rates):,}건): 결번율 중앙 {statistics.median(gap_rates)*100:.0f}%"
          f" | 결번 30%+ 문서 {bad_gap:,}건 ({bad_gap/len(gap_rates)*100:.0f}%)")

    # ── G3: 채움 이상치
    a_no_rider = [r["file"] for r in rows if r["ftype"] == "A"
                  and not r["fm"]["특약구성"] and r["n_lines"] > 2000]
    b_empty = [r["file"] for r in rows if r["ftype"] == "B"
               and not (r["fm"]["항목"] or r["fm"]["서식"] or r["fm"]["헤딩"])]
    print(f"\nG3 이상치: 대형 약관(2천줄+)인데 특약 0건 = {len(a_no_rider):,}건"
          f" | B인데 내용 전무 = {len(b_empty):,}건")

    # ── G4: 오염 (비한글/중국어)
    contaminated = []
    for r in rows:
        texts = []
        if r["ftype"] == "A":
            texts = r["fm"]["조"][:20] + [t["name"] for t in r["fm"]["특약구성"][:10]]
        elif r["ftype"] == "B":
            texts = r["fm"]["항목"] + r["fm"]["헤딩"]
        n_bad = sum(1 for t in texts if NONKO.search(t))
        if texts and n_bad / len(texts) > 0.3:
            contaminated.append(r["file"])
    print(f"\nG4 오염 플래그 (환각 의심 30%+): {len(contaminated):,}건"
          f" ({len(contaminated)/len(rows)*100:.1f}%)")

    json.dump({"singleton_riders": singleton[:200], "truncated_riders": truncated,
               "a_no_rider": a_no_rider[:100], "b_empty": b_empty[:100],
               "contaminated": contaminated},
              open(OUT / "gate_doc_fm.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n상세 목록 → out/gate_doc_fm.json")


if __name__ == "__main__":
    sys.exit(main())
