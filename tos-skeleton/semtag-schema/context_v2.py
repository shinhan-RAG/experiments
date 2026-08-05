"""보강판 컨텍스트 추출기 — 이 문서 포맷 실측 기반.

조 헤딩: `# 제N(-N)조(의N) 제목...` (괄호 없음, 헤딩/일반줄 모두) — TOC 표 줄(|)은 제외
특약 헤딩: `...특약`으로 끝나는 짧은 단독 줄
"""
import re
import semtag_experiment as SE

JO2 = re.compile(r"^#{0,4}\s*(제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?)\s+(.{2,50})$")
JO2P = re.compile(r"^#{0,4}\s*(제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?)\s*[\(（【]\s*([^\)）】]{1,40})")
RIDER2 = re.compile(r"^#{0,2}\s*[\(\[]?[가-힣A-Za-z0-9\(\)\[\]%·\-\s]{1,55}특약\s*(?:약관)?\s*$")
# 특약으로 끝나되 특약명이 아닌 문장 조각 가드 — 조·항 참조, "경우", 자기참조("이 특약"),
# 용언 수식("~되는/하는 "), 불릿·조사 시작. 정상명 반례("… 및 원자력병원 포함) …특약")는
# 통과해야 하므로 "및" 단독은 걸지 않는다.
RIDER2_PROSE = re.compile(
    r"제\s?\d+(?:-\d+)?\s?[조항]|경우|이\s?특약|않은|[되하]는\s|^\s*[-·]|^\s*의\s")
PYEON2 = re.compile(r"^#{0,4}\s*(제\s?\d+\s?편)\b")


def annotate_v2(els, lines):
    """라인 단위 스캔으로 (scope, pyeon, jo) 상태열을 만든 뒤 element 시작 라인에 배정."""
    n = len(lines)
    ctx = [None] * (n + 2)
    scope, pyeon, jo = "주계약", "", ""
    for i, raw in enumerate(lines, 1):
        first = raw.strip()
        if first and not first.startswith("|"):
            if ("특약" in first and RIDER2.match(first)
                    and not RIDER2_PROSE.search(first)):
                scope = re.sub(r"^#+\s*", "", first)[:34]
                jo = ""
            m = PYEON2.match(first)
            if m:
                pyeon = SE.ns(m.group(1))[:10]
            m = JO2P.match(first) or JO2.match(first)
            if m:
                jo = SE.ns(m.group(1)) + "(" + m.group(2).strip().rstrip(")）】")[:22] + ")"
        ctx[i] = (scope, pyeon, jo)
    for e in els:
        e["scope"], e["pyeon"], e["jo"] = ctx[min(e["start"], n)]
    return els
