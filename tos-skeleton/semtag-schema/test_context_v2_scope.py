#!/usr/bin/env python3
"""scope 과캡처 수리 회귀 테스트 — 원ONE 실측 문장형 6종 기각 + 정상 헤더 보존.

수리 대상: RIDER2가 '특약'으로 끝나는 문장 조각을 특약 헤더로 오인
(대표: "제1-10조(특약의 무효) … 경우에는 이 특약" 26회 → 653 element 오염).
"""
from context_v2 import RIDER2, RIDER2_PROSE


def is_rider_header(line):
    s = line.strip()
    return bool("특약" in s and RIDER2.match(s) and not RIDER2_PROSE.search(s))


# 원ONE 실측 문장형 전수(6종) — 전부 기각되어야 함
PROSE_LINES = [
    "제1-10조(특약의 무효) 제1항 이외에 다음에 해당되는 경우에는 이 특약",
    "- 갱신시 이 특약과 갱신계약의 보험기간이 동일하지 않은 주계약 및 주계약에 부가된 특약",
    "소멸되는 주계약 및 부가특약",
    "발생 시 소멸되는 주계약 및 부가특약",
    "의 부활(효력회복)]에 따라 이 특약을 부활(효력회복)하는 경우 이 특약",
]

# 정상 헤더 대표(원ONE 실측) — 전부 통과해야 함. 특히 '및' 포함 정상명 반례 주의.
HEADER_LINES = [
    "(간편)보험료납입면제특약",
    "특정신체부위·질병보장제한부 인수특약",
    "# 선지급서비스특약",
    "## 장애인전용보험전환특약",
    "(간편)[기본]암진단특약",
    "(간편)상급종합병원(국립암센터 및 원자력병원 포함) 암주요치료비특약",
    "사후 사망보험금 신속지급특약",
    "표준하체인수특약",
]


def main():
    fails = []
    for l in PROSE_LINES:
        if is_rider_header(l):
            fails.append(f"문장형 미기각: {l[:50]}")
    for l in HEADER_LINES:
        if not is_rider_header(l):
            fails.append(f"정상 헤더 오기각: {l[:50]}")
    if fails:
        for f in fails:
            print("FAIL", f)
        raise SystemExit(1)
    print(f"OK — 문장형 {len(PROSE_LINES)}종 기각, 정상 헤더 {len(HEADER_LINES)}종 보존")


if __name__ == "__main__":
    main()
