"""분할 보강판 v2 — 구세대 약관 분할 정책 실험(exp9) 전용. 결정적, LLM 불사용.

기존 semtag_experiment.split_elements 는 수정하지 않는다(회귀 기준 유지).
본 모듈은 동일 로직을 복제한 뒤, 실측 진단(exp9_split_diag.py)에서 수치로
확인된 구세대 패턴만 보수적으로 추가한다.

추가 규칙 (각각 원ONE 회귀 게이트 대상):
R1. 대시 런(파서 아티팩트): 모든 줄이 공백 또는 '-' 단독인 연속 블록이
    (대시 줄 >= 4개) AND (블록 길이 >= 8줄)이면 노이즈 런으로 판정.
    - 런의 시작/종료 지점에서 경계 발동 (표 전환과 동일한 상태 전이 방식)
    - 런 내부만으로 구성된 element 는 제거(drop) — 검색 대상에서 배제
    원ONE 에는 '-' 단독 줄이 5개뿐이며 전부 고립(런 미형성) → 미발동.
R2. 각괄호 조 제목: `제N조 [제목]` — 조와 '[' 사이 공백 필수, 줄이 ']'로
    끝나야 함(뒤 공백 허용). 원ONE 의 줄머리 `제1-13조[...] 제1항...` 류
    교차참조는 '[' 앞 공백이 없거나 ']' 뒤에 본문이 이어져 미발동.
R3. '의' 탈락 변형: `제N조 M 【제목】` (OCR 로 '의'가 소실된 조의M) —
    조 뒤 공백+숫자+괄호(여는 괄호는 기존과 동일한 ( （ 【 만).

사용: split_elements_v2(lines) — 반환 형식은 split_elements 와 동일
      (eid/text/start/end/type), 단 노이즈 런 element 는 제거됨.
"""
import re

import semtag_experiment as SE

# R2: 각괄호 조 제목 (공백 + '[' ... ']' 로 줄 종료)
JO_SQ = re.compile(
    r"^#{0,4}\s*제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?\s+\[[^\]]{1,45}\]\s*$")
# R3: '의' 탈락 — 조 + 공백 + 숫자 + (기존과 동일한 여는 괄호)
JO_UI = re.compile(
    r"^#{0,4}\s*제\s?\d+(?:-\d+)?\s?조\s\d{1,2}\s?[\(（【]")

# R1: 대시 단독 줄
DASH_ONLY = re.compile(r"^\s*-\s*$")
RUN_MIN_LEN = 8     # 블록 최소 길이(줄)
RUN_MIN_DASH = 4    # 블록 내 대시 줄 최소 개수


def noise_run_mask(lines):
    """lines(0-based) 각 줄이 노이즈 런(R1) 소속인지 bool 리스트로 반환."""
    n = len(lines)
    mask = [False] * n
    i = 0
    while i < n:
        s = lines[i].strip()
        if s == "" or DASH_ONLY.match(lines[i]):
            j = i
            dash = 0
            while j < n:
                t = lines[j].strip()
                if t == "":
                    j += 1
                elif DASH_ONLY.match(lines[j]):
                    dash += 1
                    j += 1
                else:
                    break
            if (j - i) >= RUN_MIN_LEN and dash >= RUN_MIN_DASH:
                for k in range(i, j):
                    mask[k] = True
            i = j
        else:
            i += 1
    return mask


def split_elements_v2(lines):
    """split_elements 동일 로직 + R1/R2/R3. 반환 형식 동일."""
    noise = noise_run_mask(lines)
    els, cur, start = [], [], 1

    def flush(end):
        nonlocal cur, start
        if not cur or not any(l.strip() for l in cur):
            cur = []
            return
        text = "\n".join(cur)
        n_table = sum(1 for l in cur if l.lstrip().startswith("|"))
        if n_table / max(1, len(cur)) > 0.5:
            et = "table"
        elif SE.FORMULA.search(text):
            et = "formula"
        elif len(cur) <= 2 and SE.HEAD.match(cur[0]):
            et = "heading"
        else:
            et = "text"
        for off in range(0, len(cur), SE.MAX_LINES):
            part = cur[off:off + SE.MAX_LINES]
            els.append({"lines": part, "start": start + off,
                        "end": start + off + len(part) - 1, "type": et})
        cur = []

    in_table = False
    in_noise = False
    for i, line in enumerate(lines, 1):
        is_tbl = line.lstrip().startswith("|")
        is_noise = noise[i - 1]
        boundary = (SE.JO.match(line) or SE.HEAD.match(line)
                    or (is_tbl != in_table)
                    or JO_SQ.match(line) or JO_UI.match(line)
                    or (is_noise != in_noise))
        if boundary and cur:
            flush(i - 1)
            start = i
        cur.append(line)
        in_table = is_tbl
        in_noise = is_noise
    flush(len(lines))
    # R1: 노이즈 런 내부만으로 구성된 element 제거
    els = [e for e in els
           if not all(noise[k] for k in range(e["start"] - 1, e["end"]))]
    for j, e in enumerate(els):
        e["eid"] = f"e{j:05d}"
        e["text"] = "\n".join(e.pop("lines"))
    return els
