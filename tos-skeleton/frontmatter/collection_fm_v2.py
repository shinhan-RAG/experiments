#!/usr/bin/env python3
"""컬렉션 frontmatter v2 — codex 리뷰 반영.

개선: 날짜 파서(문서번호 제외·연월 구분·검증·신뢰도), structure 정제(로고·반복·조각 제거),
      스키마 정정(collection 메타 + products[].product), 질의별 재현 가능한 원천 보존.
출력: out/collection_frontmatter_v2.json
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
HEAD = re.compile(r"^#{1,2}\s+(.{2,60})$")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def parse_date_v2(fn):
    """파일명 → {raw, date, granularity, confidence} | None.
    문서번호 대괄호 구간 제외, 구분자형/8자리/6자리(YYMMDD·YYYYMM) 구분, 월·일 검증."""
    s = re.sub(r"\[[^\]]*\]", " ", fn)          # [1000464] 문서번호 제거
    s = re.sub(r"\.md$", "", s)

    def valid_md(m, d):
        return 1 <= m <= 12 and 1 <= d <= 31

    # 1) 구분자형 2008.05.01 / 2008-05-01
    m = re.search(r"(19|20)(\d{2})[.\-_/](\d{1,2})[.\-_/](\d{1,2})", s)
    if m and valid_md(int(m.group(3)), int(m.group(4))):
        return {"raw": m.group(0), "date": f"{m.group(1)}{m.group(2)}{int(m.group(3)):02d}{int(m.group(4)):02d}",
                "granularity": "day", "confidence": "high"}
    # 2) 8자리 YYYYMMDD
    for m in re.finditer(r"(?<!\d)((?:19|20)\d{6})(?!\d)", s):
        y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
        if 1990 <= y <= 2030 and valid_md(mo, d):
            return {"raw": m.group(1), "date": m.group(1),
                    "granularity": "day", "confidence": "high"}
    # 3) 6자리 YYMMDD
    for m in re.finditer(r"(?<!\d)(\d{6})(?!\d)", s):
        t = m.group(1)
        yy, mo, d = int(t[:2]), int(t[2:4]), int(t[4:6])
        if valid_md(mo, d):
            y = 2000 + yy if yy <= 30 else 1900 + yy
            conf = "ambiguous" if t[:2] in ("19", "20") else "high"  # 200805류
            return {"raw": t, "date": f"{y}{t[2:]}", "granularity": "day",
                    "confidence": conf}
        # YYYYMM (앞 4자리가 연도)
        y4, mo2 = int(t[:4]), int(t[4:6])
        if 1990 <= y4 <= 2030 and 1 <= mo2 <= 12:
            return {"raw": t, "date": f"{t}00", "granularity": "month",
                    "confidence": "medium"}
    return None


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


def clean_structure(headings):
    out, seen = [], set()
    for h in headings:
        s = re.sub(r"\s+", " ", h).strip()
        if len(s) < 3 or s in seen:
            continue
        if re.fullmatch(r"[A-Za-z\s]+", s):          # SHINHAN LIFE 등 로마자 장식
            continue
        if re.search(r"·{3,}|\.{4,}", s):            # 목차 점선 줄
            continue
        if not re.search(r"[가-힣]{2}", s):           # 한글 실질어 없는 조각
            continue
        seen.add(s)
        out.append(s)
    return out[:25]


def headings(path, limit=80):
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
    products = []
    stats = Counter()
    for coll_dir in sorted(ROOT.iterdir()):
        if not coll_dir.is_dir():
            continue
        name = nfc(coll_dir.name)
        files = sorted(coll_dir.rglob("*.md"))
        identity = {"flags": [k for k, p in FLAGS.items() if any(x in name for x in p)],
                    "kind_hints": [k for k in KINDS if k in name]}
        inventory = []
        for f in files:
            fn = nfc(f.name)
            d = parse_date_v2(fn)
            inventory.append({"file": fn, "doc_type": doc_type(fn), "date": d})
            stats["docs"] += 1
            if d:
                stats["dated"] += 1
                stats[f"conf_{d['confidence']}"] += 1
        rep = {}
        for t in ("판매약관", "사업방법서", "공시약관", "상품요약서"):
            cands = [iv for iv in inventory if iv["doc_type"] == t and iv["date"]
                     and iv["date"]["confidence"] == "high"]  # 신뢰 날짜만 정본 판정
            if cands:
                rep[t] = max(cands, key=lambda iv: iv["date"]["date"])["file"]
        structure = []
        rf = rep.get("판매약관") or rep.get("공시약관") or rep.get("사업방법서")
        if rf:
            structure = clean_structure(headings(coll_dir / rf))
        products.append({"product": name, "identity": identity, "inventory": inventory,
                         "representative": rep, "structure": structure})
        if structure:
            stats["with_structure"] += 1

    fm = {"collection": {"name": "신한 상품문서 컬렉션 (parsed_md)", "version": "v2",
                         "n_products": len(products), "n_documents": stats["docs"]},
          "products": products}
    (OUT / "collection_frontmatter_v2.json").write_text(
        json.dumps(fm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"상품 {len(products)} | 문서 {stats['docs']:,}")
    print(f"날짜 파싱 {stats['dated']:,} (high {stats['conf_high']:,} / "
          f"medium {stats['conf_medium']:,} / ambiguous {stats['conf_ambiguous']:,})")
    print(f"정본 지정 상품 {sum(1 for p in products if p['representative'])} | "
          f"목차(정제) 확보 {stats['with_structure']}")


if __name__ == "__main__":
    sys.exit(main())
