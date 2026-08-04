#!/usr/bin/env python3
"""9,417건 문서 EDA — 구조 시그니처 추출 → 유형 군집.

가설: 문서 유형별 뼈대 구조(조·관·표 패턴)가 있고 값만 다르다.
특징: doc_type(파일명) × 내부 구조(조 체계·헤딩·표 비율·길이·특약 패턴)
출력: out/eda_features.jsonl, 콘솔 유형 분포 리포트
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")

JO = re.compile(r"^#{0,3}\s*제\s?\d+(-\d+)?\s?조")
KWAN = re.compile(r"^#{0,3}\s*제\s?\d+\s?관")
PYEON = re.compile(r"^#{0,3}\s*제\s?\d+\s?편")
HEAD = re.compile(r"^(#{1,3})\s+(.+)")
TEUKYAK = re.compile(r"특약\s*$|특약\)")
BUPYO = re.compile(r"부표|별표|지급기준표")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def doc_type(fn):
    f = fn.replace(" ", "")
    if "공시" in f:
        return "공시약관"
    if "약관" in f:
        return "판매약관"
    if "사업방법서" in f or "사방서" in f:
        return "사업방법서"
    if "요약서" in f:
        return "상품요약서"
    return "미분류"


def features(path):
    n_lines = n_head = n_jo = n_kwan = n_pyeon = n_table = n_teuk = n_bupyo = 0
    heads = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                n_lines += 1
                s = line.strip()
                if not s:
                    continue
                if s.startswith("|"):
                    n_table += 1
                m = HEAD.match(s)
                if m:
                    n_head += 1
                    if len(heads) < 40:
                        heads.append(m.group(2)[:40])
                if JO.match(s):
                    n_jo += 1
                if KWAN.match(s):
                    n_kwan += 1
                if PYEON.match(s):
                    n_pyeon += 1
                if m and TEUKYAK.search(m.group(2)):
                    n_teuk += 1
                if BUPYO.search(s[:30]):
                    n_bupyo += 1
    except OSError:
        return None
    return {"n_lines": n_lines, "n_head": n_head, "n_jo": n_jo, "n_kwan": n_kwan,
            "n_pyeon": n_pyeon, "n_table": n_table, "n_teuk": n_teuk,
            "n_bupyo": n_bupyo, "heads": heads}


def structure_class(ft):
    """내부 구조 시그니처 → 구조 클래스."""
    if ft["n_lines"] < 30:
        return "빈약/스텁"
    jo_density = ft["n_jo"] / max(1, ft["n_lines"] / 100)
    table_ratio = ft["n_table"] / max(1, ft["n_lines"])
    if ft["n_jo"] >= 30 and ft["n_pyeon"] + ft["n_kwan"] >= 3:
        return "조문형(편·관·조 체계)"
    if ft["n_jo"] >= 10:
        return "조문형(단순)"
    if table_ratio > 0.5:
        return "표 중심"
    if table_ratio > 0.2:
        return "표+서술 혼합"
    if ft["n_head"] >= 10:
        return "헤딩 목록형"
    return "서술형(구조 빈약)"


def main():
    rows = []
    dist = Counter()
    for i, p in enumerate(sorted(ROOT.rglob("*.md"))):
        ft = features(p)
        if ft is None:
            continue
        dt = doc_type(nfc(p.name))
        sc = structure_class(ft)
        rows.append({"file": nfc(str(p.relative_to(ROOT))), "doc_type": dt,
                     "structure": sc, **{k: v for k, v in ft.items() if k != "heads"},
                     "heads": ft["heads"][:12]})
        dist[(dt, sc)] += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}...", flush=True)

    with open(OUT / "eda_features.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(rows):,}건 | 유형(파일명 doc_type × 내부 구조) 분포:\n")
    print(f"{'doc_type':10s} {'구조 클래스':22s} {'건수':>7s}")
    for (dt, sc), c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"{dt:10s} {sc:22s} {c:7,}")


if __name__ == "__main__":
    sys.exit(main())
