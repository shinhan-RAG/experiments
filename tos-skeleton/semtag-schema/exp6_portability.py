#!/usr/bin/env python3
"""실험 6 — 스키마 이식성(범용성) 측정. 결정적, gold 불요(커버리지만).

키 4종([유형][소속][섹션][역할])이 문서 유형·시대·종목을 넘어 유지되는지,
어느 필드가 value 추출기 교체를 요구하는지 측정한다.
표본: 약관 4(2026 통합건강 기준/2000 암/1999 연금/2016 종신) + 사업방법서 3(2025/2012/2019).
"""
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).parent
SRC = Path(
    "/private/tmp/claude-501/-Users-donggyu-Documents----PageIndex/"
    "ef37eba4-cc2f-4769-9601-98adf8b2cab7/scratchpad/source/parsed_md")
ONE = Path(
    "/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
    "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")

DOCS = [
    ("기준: 원ONE 약관(건강,2026)", ONE, "jo"),
    ("약관: 건강지기암보험(2000)", SRC / "건강지기암보험/GunGangJiGi_000401_000430_P.md", "jo"),
    ("약관: 개인연금저축(1999)",
     SRC / "개인연금저축프리스타일/개인연금저축프리스타일_1999.02.01~2000.03.21.md", "jo"),
    ("약관: 유니버설종신(2016)",
     SRC / "(무)신한유니버설Plus종신보험/(무)신한유니버설Plus종신보험_판매약관_20160816_.md", "jo"),
    ("사업방법서: 큐브(2025)",
     SRC / "신한(간편가입)통합건강보험 큐브(무배당, 해약환급금 미지급형)/"
           "사업방법서_신한(간편가입)통합건강보험 큐브(무배당, 해약환급금 미지급형)_250401.md", "item"),
    ("사업방법서: 세이프업변액(2012)",
     SRC / "(무)세이프업(Safe-Up)변액연금보험Ⅱ/(무)세이프업_사업방법서_20120402.md", "item"),
    ("사업방법서: 치매보험(2019)",
     SRC / "신한간병비받는간편한치매보험(무배당, 무해지환급형)/"
           "사업방법서_신한간병비받는간편한치매보험(무배당_무해지환급형)_20190225.md", "item"),
]

# 사업방법서(항목형) [섹션] value 추출기 플러그인 — "N. …에 관한 사항" 문법
ITEM = re.compile(
    r"^#{0,4}\s*(\d{1,2})\s*\.\s*"
    r"(.{2,40}?에 관한 사항|보험종목의 명칭.{0,20}|.{2,40}?(사항|명칭|기간|나이|주기))\s*$")


def annotate_item_sections(els, lines):
    marks = {}
    for i, l in enumerate(lines):
        m = ITEM.match(l.strip())
        if m:
            marks[i] = f"{m.group(1)}. {m.group(2).strip()}"
    for e in els:
        best = None
        for i in sorted(marks):
            if i <= e["start"]:
                best = marks[i]
            else:
                break
        e["section_item"] = best
    return els


def main():
    rows = []
    for name, p, kind in DOCS:
        lines = SE.nfc(p.read_text(encoding="utf-8", errors="ignore")).splitlines()
        els = annotate_v2(SE.split_elements(lines), lines)
        if kind == "item":
            els = annotate_item_sections(els, lines)
        n = len(els)
        sect = [e.get("jo") if kind == "jo" else e.get("section_item") for e in els]
        scope_vals = Counter(e.get("scope") for e in els)
        tags = []
        for e, s in zip(els, sect):
            parts = [f"[유형]{e['type']}"]
            if e.get("scope"):
                parts.append(f"[소속]{e['scope']}")
            if s:
                parts.append(f"[섹션]{s}")
            r = SE.roles_of(e["text"])
            if r:
                parts.append(f"[역할]{r[0]}")
            tags.append(" ".join(parts))
        rows.append({
            "doc": name, "extractor": kind, "lines": len(lines), "elements": n,
            "section_cov": round(sum(1 for s in sect if s) / n, 3),
            "scope_cov": round(sum(1 for e in els if e.get("scope")) / n, 3),
            "scope_top": scope_vals.most_common(2),
            "scope_distinct": len(scope_vals),
            "role_cov": round(sum(1 for e in els if SE.roles_of(e["text"])) / n, 3),
            "tag_p50": int(statistics.median(len(t) for t in tags)),
            "over200": sum(1 for t in tags if len(t) > 200),
        })
    out = HERE / "out/exp6"
    out.mkdir(exist_ok=True)
    (out / "portability.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
