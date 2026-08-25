#!/usr/bin/env python3
"""문서 프로필 — 상품/문서 고유 표기를 코드에서 분리 (G1 일반화 리팩터).

원칙: 파이프라인 로직은 문서 무관, 문서 고유 표기는 여기(기본 프로필) 또는
`<doc>.profile.json` 사이드카에만 둔다. 프로필이 문서와 안 맞으면 generic
유도 규칙이 폴백으로 작동한다(주계약 헤더를 문서에서 직접 추론).
"""
import json
import re
from pathlib import Path

# 기존 build_universe.py 의 리터럴을 그대로 옮긴 기본 프로필 — 산출물 byte 동일 보장.
DEFAULT_PROFILE = {
    "main_header_patterns": [
        r"^신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^)]*\)\s*$",      # 260507판 표기
        r"^\(간편\)신한통합건강보장보험\s*원\(ONE\)\(무배당[^)]*\)\s*$",    # 250212판 표기
    ],
    "brand_noise_patterns": [
        r"^#?\s*SHINHAN LIFE$", r"^S$", r"^S 신한라\s?이프$",
    ],
}

# 문서 무관 구조 노이즈(구분선·페이지번호·좌표쌍) — 프로필이 아니라 로직 소속.
STRUCTURAL_NOISE_PATTERNS = [
    r"^_{3,}$", r"^-{3}$", r"^\d{10,}$", r"^\d{1,4}$",
    r"^\(\d+,\d+\),\(\d+,\d+\)$",
]


def load_profile(doc_path=None):
    """<doc>.profile.json 사이드카가 있으면 그것, 없으면 기본 프로필."""
    if doc_path:
        side = Path(str(doc_path) + ".profile.json")
        if side.exists():
            data = json.loads(side.read_text(encoding="utf-8"))
            return {**DEFAULT_PROFILE, **data}
    return dict(DEFAULT_PROFILE)


def compile_mains(profile):
    return [re.compile(p) for p in profile["main_header_patterns"]]


def compile_noise(profile):
    return [re.compile(p) for p in
            STRUCTURAL_NOISE_PATTERNS + profile["brand_noise_patterns"]]


_RIDER_LINE = re.compile(r"^(?!.*\|).{1,100}특약\s*\(\s*무배당[^)]*\)\s*$")
_TITLE_LINE = re.compile(r"^(?!.*\|)[^|]{4,80}\(\s*무배당[^)]*\)\s*$")


def derive_main_patterns(lines):
    """generic 폴백: 첫 특약 헤더 이전에 나오는 '상품명(무배당…)' 꼴 제목 줄에서
    주계약 헤더 패턴을 유도한다. 프로필 패턴이 문서와 안 맞을 때만 사용."""
    seen = []
    for line in lines:
        s = line.strip()
        if _RIDER_LINE.match(s):
            break
        if _TITLE_LINE.match(s) and "특약" not in s and s not in seen:
            seen.append(s)
    return [re.compile("^" + re.escape(s).replace(r"\ ", r"\s*") + r"\s*$")
            for s in seen]
