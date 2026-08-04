#!/usr/bin/env python3
"""컬렉션(상품) frontmatter — 규칙 기반 생성 + 품질 게이트. LLM 미사용.

출처: 폴더명(상품), 파일명(종류·날짜·버전), 대표 문서의 md 헤딩(h1~h2).
출력: out/collection_fm.jsonl (컬렉션당 1행), 콘솔 품질 리포트
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")

FLAGS = {"무배당": ["무배당", "(무)"], "갱신형": ["갱신형"], "변액": ["변액"],
         "저해지": ["저해지"], "간편": ["간편"], "유니버설": ["유니버설", "유니버셜"],
         "달러/외화": ["달러", "외화"]}
KINDS = ["암", "연금", "종신", "저축", "어린이", "정기", "상해", "건강", "치아",
         "간병", "치매", "교육", "여성", "CI", "실손", "즉시연금", "퇴직"]
DATE = re.compile(r"(20\d{6}|19\d{6}|\d{6})")
HEAD = re.compile(r"^#{1,2}\s+(.{2,60})$")


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


def parse_date(fn):
    m = DATE.search(fn)
    if not m:
        return None
    d = m.group(1)
    if len(d) == 6:  # YYMMDD 추정
        d = ("20" if d[0] in "012" else "19") + d
    return d


def headings(path, limit=25):
    try:
        out = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = HEAD.match(line.strip())
                if m:
                    out.append(m.group(1).strip())
                    if len(out) >= limit:
                        break
        return out
    except OSError:
        return []


def main():
    rows = []
    stats = Counter()
    for coll_dir in sorted(ROOT.iterdir()):
        if not coll_dir.is_dir():
            continue
        name = nfc(coll_dir.name)
        files = sorted(f for f in coll_dir.rglob("*.md"))
        stats["collections"] += 1
        if not files:
            stats["empty"] += 1
        identity = {"flags": [k for k, pats in FLAGS.items()
                              if any(p in name for p in pats)],
                    "kind_hints": [k for k in KINDS if k in name]}
        inventory = []
        for f in files:
            fn = nfc(f.name)
            inventory.append({"file": fn, "doc_type": doc_type(fn),
                              "date": parse_date(fn)})
        rep = {}
        for t in ("판매약관", "사업방법서", "공시약관", "상품요약서"):
            cands = [iv for iv in inventory if iv["doc_type"] == t]
            if cands:
                rep[t] = max(cands, key=lambda iv: iv["date"] or "")["file"]
        structure = []
        rep_file = rep.get("판매약관") or rep.get("공시약관") or rep.get("사업방법서")
        if rep_file:
            structure = headings(coll_dir / rep_file)
        rows.append({"collection": name, "identity": identity,
                     "inventory": inventory, "representative": rep,
                     "structure": structure})
        stats["docs"] += len(inventory)
        stats["dated"] += sum(1 for iv in inventory if iv["date"])
        stats["typed"] += sum(1 for iv in inventory if iv["doc_type"] != "미분류")
        if structure:
            stats["with_structure"] += 1

    OUT.mkdir(exist_ok=True)
    with open(OUT / "collection_fm.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 품질 게이트 (기계 대조)
    print(f"컬렉션 {stats['collections']} (빈 폴더 {stats['empty']}) | 문서 {stats['docs']:,}")
    print(f"날짜 파싱   {stats['dated']:,}/{stats['docs']:,} = {stats['dated']/stats['docs']*100:.0f}%")
    print(f"종류 분류   {stats['typed']:,}/{stats['docs']:,} = {stats['typed']/stats['docs']*100:.0f}%")
    print(f"대표문서 지정 컬렉션 {sum(1 for r in rows if r['representative'])}/{stats['collections']}")
    print(f"목차 요약 확보 컬렉션 {stats['with_structure']}/{stats['collections']}")


if __name__ == "__main__":
    sys.exit(main())
