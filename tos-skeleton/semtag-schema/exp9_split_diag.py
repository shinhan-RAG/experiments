#!/usr/bin/env python3
"""실험 9 — 구세대 약관 분할 정책 진단 + 보강판(split_v2) 검증. 결정적, LLM 불사용.

절차:
1) 진단 — 구세대 표본 7문서 + 회귀 기준(원ONE):
   (a) element 크기 분포(p50/p90/max, MAX_LINES=120 도달 비율)
   (b) 조(條) 헤더 줄 census: strict(현행 JO 매치) vs 미매치 변형 분류
       (각괄호 제목 / '의' 탈락 / 무괄호 제목형 / 표 행 내부 / 광폭 공백 / 참조문)
   (c) 헤더 줄이 element 경계(시작 줄)로 실제 사용됐는지 대조
   (d) 노이즈 런(대시 단독 줄 연속 블록) 규모
2) 검증 — split_v2.split_elements_v2 적용 전후:
   element 수·크기 분포·[역할] 커버리지(roles_of_v2 yakgwan)·[섹션] 커버리지
   (context_v2.annotate_v2 의 jo 비어있지 않음 비율)
3) 회귀 게이트 — 원ONE 에서 v1/v2 분할 결과 완전 동일(개수·경계·유형·본문)
4) 결정성 — 전 문서 2회 실행 해시 일치

출력: out/exp9/results.json
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
import split_v2 as SV
from context_v2 import annotate_v2
from role_vocab import roles_of_v2

HERE = Path(__file__).parent
SRC = Path(
    "/private/tmp/claude-501/-Users-donggyu-Documents----PageIndex/"
    "ef37eba4-cc2f-4769-9601-98adf8b2cab7/scratchpad/source/parsed_md")
ONE = Path(
    "/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
    "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")

# 구세대 표본: 지정 2 + 확대 5 (사업방법서 제외, 6자리 날짜 앞 2자리 92~05 확인)
OLD_DOCS = [
    ("건강지기암(2000)", SRC / "건강지기암보험/GunGangJiGi_000401_000430_P.md"),
    ("개인연금저축(1999)",
     SRC / "개인연금저축프리스타일/개인연금저축프리스타일_1999.02.01~2000.03.21.md"),
    ("노후복지연금1종(1992)",
     SRC / "노후복지연금보험 1종/(2100_2104_2110_2114)노후복지연금보험_1종(920812_921109).md"),
    ("노후복지연금(1997)",
     SRC / "노후복지연금보험 1종/[3]노후복지연금보험_970401~970930_2270~2284_약관.md"),
    ("네덜란드두배로종신(1994)", SRC / "네덜란드두배로종신보험/Double_WL_941215_970331.md"),
    ("군인연금(1997)", SRC / "군인연금보험/[3]군인연금보험_970317~990131_2946~7949_약관.md"),
    ("신한종신(2000)", SRC / "(무)신한종신보험/약관_(무)신한종신보험_000401_000704_6600_6605.md"),
]

# ── 조 헤더 census 분류기 ──────────────────────────────────────────────
LOOSE = re.compile(r"^[#*\s]*제\s{0,3}\d+(?:-\d+)?\s{0,3}조(?:\s{0,3}의\s{0,3}\d+)?")
JO_TBL = re.compile(r"^\s*\|\s*제\s{0,3}\d+(?:-\d+)?\s{0,3}조")
SENT = re.compile(r"(니다|한다|규정|따라|의하여|준용|말합니|봅니)")


def classify_line(line):
    """조 헤더 후보 줄 분류. None = 후보 아님."""
    if SE.JO.match(line):
        return "strict"
    if JO_TBL.match(line):
        return "table_row"
    if line.lstrip().startswith("|"):
        return None
    m = LOOSE.match(line)
    if not m:
        return None
    rest = line[m.end():]
    if SV.JO_SQ.match(line):
        return "sq_bracket_header"
    if SV.JO_UI.match(line):
        return "missing_ui_header"
    if re.match(r"\s*\[", rest):
        return "sq_bracket_ref"        # '[' 이후가 제목형이 아님(교차참조 등)
    if re.match(r"\s*[\(（【]", rest):
        return "paren_nonstrict"       # 괄호는 있으나 strict 미매치(공백 변형 등)
    r = rest.strip()
    if r and len(r) <= 30 and not SENT.search(r):
        return "no_bracket_header"     # 무괄호 제목형(보수적 후보)
    return "ref_or_prose"              # 줄머리 교차참조·본문 줄바꿈


def el_sizes(els):
    sz = sorted(e["end"] - e["start"] + 1 for e in els)
    n = len(sz)
    return {"n": n, "p50": sz[n // 2], "p90": sz[int(n * 0.9)], "max": sz[-1],
            "at_cap": sum(1 for s in sz if s == SE.MAX_LINES),
            "at_cap_pct": round(sum(1 for s in sz if s == SE.MAX_LINES) / n * 100, 1),
            "mean": round(sum(sz) / n, 1)}


def coverage(els, lines):
    els = annotate_v2(els, lines)
    n = max(1, len(els))
    role = sum(1 for e in els if roles_of_v2(e["text"], "yakgwan")) / n
    sect = sum(1 for e in els if e.get("jo")) / n
    return round(role, 3), round(sect, 3)


def els_hash(els):
    return hashlib.sha256(json.dumps(
        [(e["start"], e["end"], e["type"], e["text"]) for e in els],
        ensure_ascii=False).encode()).hexdigest()


def diagnose(name, lines):
    els = SE.split_elements(lines)
    starts = {e["start"] for e in els}
    census = Counter()
    miss_not_boundary = Counter()
    examples = {}
    for i, line in enumerate(lines, 1):
        c = classify_line(line)
        if not c:
            continue
        census[c] += 1
        if c not in ("strict", "table_row", "ref_or_prose", "sq_bracket_ref") \
                and i not in starts:
            miss_not_boundary[c] += 1
        if c != "strict" and len(examples.get(c, [])) < 3:
            examples.setdefault(c, []).append(line.strip()[:70])
    strict_used = sum(1 for i, l in enumerate(lines, 1)
                      if SE.JO.match(l) and i in starts)
    noise = SV.noise_run_mask(lines)
    runs = 0
    prev = False
    for b in noise:
        if b and not prev:
            runs += 1
        prev = b
    return {
        "doc": name, "lines": len(lines),
        "baseline": el_sizes(els),
        "type_dist": dict(Counter(e["type"] for e in els)),
        "jo_census": dict(census),
        "strict_total": census["strict"],
        "strict_used_as_boundary": strict_used,
        "miss_not_boundary": dict(miss_not_boundary),
        "examples": examples,
        "noise_runs": {"n_runs": runs, "lines": sum(noise),
                       "pct_of_doc": round(sum(noise) / len(lines) * 100, 1)},
    }


def compare(name, lines):
    v1 = SE.split_elements(lines)
    v2 = SV.split_elements_v2(lines)
    r1, s1 = coverage(SE.split_elements(lines), lines)
    r2, s2 = coverage(SV.split_elements_v2(lines), lines)
    return {"doc": name,
            "v1": {**el_sizes(v1), "role_cov": r1, "sect_cov": s1},
            "v2": {**el_sizes(v2), "role_cov": r2, "sect_cov": s2}}


def main():
    out = {"diag": [], "compare": [], "regression": {}, "determinism": {}}
    docs = OLD_DOCS + [("원ONE(2026)·회귀기준", ONE)]
    cache = {}
    for name, p in docs:
        lines = SE.nfc(p.read_text(encoding="utf-8", errors="ignore")).splitlines()
        cache[name] = lines
        out["diag"].append(diagnose(name, lines))
        print(f"[diag] {name}: {out['diag'][-1]['baseline']}", flush=True)

    for name, p in OLD_DOCS:
        out["compare"].append(compare(name, cache[name]))
        c = out["compare"][-1]
        print(f"[cmp ] {name}: n {c['v1']['n']}->{c['v2']['n']} "
              f"role {c['v1']['role_cov']}->{c['v2']['role_cov']}", flush=True)

    # 회귀 게이트: 원ONE v1 == v2 (개수·경계·유형·본문 완전 동일)
    lines = cache["원ONE(2026)·회귀기준"]
    v1 = SE.split_elements(lines)
    v2 = SV.split_elements_v2(lines)
    same = (len(v1) == len(v2) and
            all(a["start"] == b["start"] and a["end"] == b["end"]
                and a["type"] == b["type"] and a["text"] == b["text"]
                and a["eid"] == b["eid"] for a, b in zip(v1, v2)))
    diff = None
    if not same:
        diff = {"v1_n": len(v1), "v2_n": len(v2),
                "first_diff": next((i for i, (a, b) in enumerate(zip(v1, v2))
                                    if (a["start"], a["end"]) != (b["start"], b["end"])),
                                   None)}
    out["regression"] = {"one_v1_elements": len(v1), "one_v2_elements": len(v2),
                         "identical": same, "diff": diff}
    print(f"[gate] 원ONE identical={same} n={len(v1)}->{len(v2)}", flush=True)

    # 결정성: 전 문서 v1/v2 2회 해시 일치
    det = True
    for name, _ in docs:
        l = cache[name]
        det &= els_hash(SE.split_elements(l)) == els_hash(SE.split_elements(l))
        det &= els_hash(SV.split_elements_v2(l)) == els_hash(SV.split_elements_v2(l))
    out["determinism"] = {"two_runs_identical": det}
    print(f"[det ] two_runs_identical={det}", flush=True)

    d = HERE / "out/exp9"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved:", d / "results.json", flush=True)


if __name__ == "__main__":
    main()
