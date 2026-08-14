#!/usr/bin/env python3
"""특약 > 절(관) > 조 경계 기반 조립(粗粒) 엘리먼트 코퍼스 빌더.

배경
----
기존 `out/elements_repaired_v1.jsonl`(49,929건, 중앙값 24자)은 과분절 상태다.
등록된 최고성능 구성은 훨씬 굵은 단위였다 — 5,924건 / 중앙값 309자,
특약 > 절 > 조 경계 분할, MAXC 4000 / MAXSPAN 50.

이 스크립트는 그 굵기를 재현한다. 하나의 **조(제N조 / 제N-N조)** 와 그에 종속된
항/호/표를 **하나의 엘리먼트**로 묶는다. 그래서 중앙값이 24자가 아니라 ~300자가 된다.

좌표계(중요)
------------
`char_start`/`char_end`는 문서를 아래처럼 읽은 **raw 오프셋**이다.

    raw = NFC(path.read_text(encoding="utf-8").replace("\\r\\n", "\\n"))

정답셋 `out/gold_train.jsonl`(coord_space:"raw")이 이 좌표로 스팬 겹침 채점을 하므로
좌표계가 어긋나면 모든 지표가 조용히 0이 된다. 시작 시 문서 길이를 assert한다.

구조 검출(표/산식/제목/단락), 4000자 분절, contract_scope 추적은 build_elements.py에서
가져왔다(import하지 않는다 — 거기엔 macOS 하드코딩 경로가 있다).

사용법
    python build_elements_psection.py
"""
from __future__ import annotations

import bisect
import collections
import io
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- Windows stdout 인코딩
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"
DOC = (BASE / "noah_qaset"
       / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md")

EXPECTED_DOC_LEN = 2_757_704   # raw(NFC, LF) 문자 수 — 정답셋 좌표계의 기준
MAXC = 4000                    # 엘리먼트 최대 길이(하드 캡)
MAXSPAN = 50                   # 엘리먼트가 걸칠 수 있는 최대 원문 줄 수
MIN_SEG_CHARS = 60             # 이보다 짧은 세그먼트(머리글만 있는 조각)는 다음 것과 합친다

# ---------------------------------------------------------------- 경계 패턴
# 조: "제1조 목적", "제3조(보험금의 지급사유)", "제2조의2 ...", "제1-13조 ..."
ART_RE = re.compile(r"^제\s*\d+(?:\s*-\s*\d+|\s*의\s*\d+)?\s*조")
# 절/관: "제1관 목적 및 용어의 정의", "제2절 ..."
GWAN_RE = re.compile(r"^제\s*\d+\s*[관절](?:\s|$|[(:])")
# 특약명 머리글(문서상 각 페이지 머리/꼬리로 반복 등장한다 — 특약 스코프의 근거)
RIDER_RE = re.compile(r"^\(간편\).{0,60}특약\(무배당[^)]*\)\s*$")
# 주계약명
MAIN_RE = re.compile(
    r"^(?:\(간편\))?신한(?:\(간편가입\))?통합건강보장보험\s*원\(ONE\)\(무배당[^)]*\)\s*$")

TABLE_ROW_RE = re.compile(r"^\s*\|")
MD_HEADING_RE = re.compile(r"^#{1,6}\s")


def load_doc() -> tuple[str, list[str], list[int]]:
    """문서를 raw 좌표계로 읽고, 줄 목록과 줄 시작 오프셋을 반환한다."""
    raw = unicodedata.normalize(
        "NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    assert len(raw) == EXPECTED_DOC_LEN, (
        f"문서 길이가 {len(raw):,}자 — 기대값 {EXPECTED_DOC_LEN:,}자와 다르다. "
        "좌표계가 어긋나면 gold 스팬 채점이 전부 0이 된다.")
    lines = raw.split("\n")
    offs: list[int] = []
    pos = 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    return raw, lines, offs


def find_boundaries(lines: list[str]) -> tuple[list[int], list[tuple[int, str]]]:
    """분할 경계 줄 번호와, 특약 스코프 경계((줄번호, 스코프명))를 찾는다."""
    bounds: list[int] = []
    scopes: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or "|" in s:      # 목차/표 안의 "제N조ㅇㅇ | 87" 은 경계가 아니다
            continue
        if RIDER_RE.match(s):
            bounds.append(i)
            scopes.append((i, s))
        elif MAIN_RE.match(s):
            bounds.append(i)
            scopes.append((i, "주계약(" + s + ")"))
        elif ART_RE.match(s) or GWAN_RE.match(s):
            bounds.append(i)
    return bounds, scopes


def segment(lines: list[str], bounds: list[int]) -> list[tuple[int, int]]:
    """경계로 1차 분할한 뒤, 머리글만 남은 짧은 조각을 뒤쪽과 병합한다.

    병합이 있어야 "제2관 보험금의 지급" 같은 절 머리글이 바로 뒤 조와 한 덩어리가 되고,
    조 본문이 조각나지 않는다.
    """
    segs: list[tuple[int, int]] = []
    prev = 0
    for i in bounds:
        if i > prev:
            segs.append((prev, i))
        prev = i
    segs.append((prev, len(lines)))
    segs = [(a, b) for a, b in segs if "\n".join(lines[a:b]).strip()]

    merged: list[tuple[int, int]] = []
    k = 0
    while k < len(segs):
        a, b = segs[k]
        while len("\n".join(lines[a:b]).strip()) < MIN_SEG_CHARS and k + 1 < len(segs):
            k += 1
            b = segs[k][1]
        merged.append((a, b))
        k += 1
    return merged


def enforce_caps(a: int, b: int, lines: list[str]) -> list[tuple[int, int]]:
    """MAXC / MAXSPAN을 넘는 세그먼트를 줄 단위로 쪼갠다."""
    parts: list[tuple[int, int]] = []
    st, cur = a, 0
    for k in range(a, b):
        cur += len(lines[k]) + 1
        if cur >= MAXC or (k - st + 1) >= MAXSPAN:
            parts.append((st, k + 1))
            st, cur = k + 1, 0
    if st < b:
        parts.append((st, b))
    return parts or [(a, b)]


def classify(text: str) -> str:
    """표 / 산식 / 제목 / 단락."""
    body = [ln for ln in text.split("\n") if ln.strip()]
    if not body:
        return "paragraph"
    n_tbl = sum(1 for ln in body if TABLE_ROW_RE.match(ln))
    if n_tbl * 2 >= len(body):
        return "table"
    if any(ln.lstrip().startswith("$$") for ln in body):
        return "formula"
    if len(body) == 1 and (MD_HEADING_RE.match(body[0]) or len(body[0]) < 60):
        return "heading"
    return "paragraph"


def main() -> None:
    raw, lines, offs = load_doc()
    n_lines = len(lines)
    print(f"[문서] {DOC.name}")
    print(f"[문서] raw 길이 {len(raw):,}자 / {n_lines:,}줄 — 좌표계 검증 통과")

    bounds, scopes = find_boundaries(lines)
    scope_lines = [s[0] for s in scopes]

    def scope_of(line_idx: int) -> str:
        j = bisect.bisect_right(scope_lines, line_idx) - 1
        return scopes[j][1] if j >= 0 else ""

    segs = segment(lines, bounds)
    print(f"[경계] 분할 경계 {len(bounds):,}개 / 특약 스코프 경계 {len(scopes):,}개 "
          f"-> 병합 후 세그먼트 {len(segs):,}개")

    rows: list[dict] = []
    split_parts = 0
    for idx, (a, b) in enumerate(segs):
        parts = enforce_caps(a, b, lines)
        parts = [(p, q) for p, q in parts if "\n".join(lines[p:q]).strip()]
        if not parts:
            continue
        if len(parts) > 1:
            split_parts += len(parts)
        # 줄 하나가 MAXC를 넘는 초대형 표 행은 문자 단위로 한 번 더 자른다(하드 캡).
        spans: list[tuple[int, int, int, int]] = []   # (c0, c1, line_start, line_end)
        for p, q in parts:
            c0 = offs[p]
            c1 = offs[q - 1] + len(lines[q - 1])
            if c1 - c0 <= MAXC:
                spans.append((c0, c1, p, q))
                continue
            for s0 in range(c0, c1, MAXC):
                spans.append((s0, min(s0 + MAXC, c1), p, q))
        if len(spans) > len(parts):
            split_parts += len(spans) - len(parts)

        base = f"e{idx:05d}"
        for pi, (c0, c1, p, q) in enumerate(spans):
            text = raw[c0:c1]                     # 스팬이 곧 텍스트 — 정의상 일치한다
            eid = base if len(spans) == 1 else f"{base}__r{pi + 1:03d}"
            rows.append({
                "element_id": eid,
                "element_type": classify(text),
                "text": text,
                "char_start": c0,
                "char_end": c1,
                "line_start": p + 1,
                "line_end": q,
                "contract_scope": scope_of(p),
            })

    out_path = OUT / "elements_psection.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = sorted(len(r["text"]) for r in rows)
    stats = {
        "n": len(rows),
        "by_type": dict(collections.Counter(r["element_type"] for r in rows)),
        "median_len": int(statistics.median(lens)),
        "p90_len": lens[int(len(lens) * 0.9) - 1],
        "max_len": lens[-1],
        "split_parts": split_parts,
        "maxc": MAXC,
        "maxspan": MAXSPAN,
        "doc_len": len(raw),
        "coord_space": "raw",
    }
    (OUT / "element_psection_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ 스팬 검증
    step = max(len(rows) // 20, 1)
    sample = [rows[i] for i in range(0, len(rows), step)][:20]
    ok = sum(1 for r in sample if raw[r["char_start"]:r["char_end"]] == r["text"])
    all_ok = sum(1 for r in rows if raw[r["char_start"]:r["char_end"]] == r["text"])

    print()
    print("=" * 62)
    print(f"  엘리먼트 수      : {stats['n']:,}      (목표 ~5,900)")
    print(f"  텍스트 중앙값    : {stats['median_len']:,}자   (목표 250~400)")
    print("=" * 62)
    print(f"  p90 {stats['p90_len']:,}자 / 최대 {stats['max_len']:,}자 / "
          f"cap 분절 파트 {stats['split_parts']:,}개")
    print(f"  타입 분포: {stats['by_type']}")
    print(f"  스팬 검증(표본 20): {ok}/20 일치")
    print(f"  스팬 검증(전수)   : {all_ok:,}/{len(rows):,} 일치")
    print(f"  출력: {out_path}")
    print(f"  통계: {OUT / 'element_psection_stats.json'}")

    if ok != 20 or all_ok != len(rows):
        print("[치명] char_start/char_end가 텍스트와 어긋난다.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
