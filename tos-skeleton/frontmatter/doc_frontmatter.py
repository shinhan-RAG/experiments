#!/usr/bin/env python3
"""문서별 frontmatter 추출기 — 유형 A/B/C/D 전부, 규칙만(LLM 0회).

유형 판정: EDA 구조 시그니처 (파일명 doc_type가 아니라 내부 구조 기준 — 미분류 재분류 포함)
  A 약관군(조문형)     : 특약 구성 + 조 제목 + 부표 목록 + 편·관 골격
  B 사업방법서군(서술형) : 번호 항목 + 서식 목록 + 정제 헤딩
  C 표 중심/혼합       : 표 캡션 + 표 컬럼 헤더 + 정제 헤딩
  D 스텁/빈약          : 첫 줄 표본만 (frontmatter 대상 아님 표시)
출력: out/doc_frontmatter.jsonl (문서당 1행) + 콘솔 품질 리포트
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")

HEAD = re.compile(r"^(#{1,3})\s+(.+)")
JO = re.compile(r"^#{0,3}\s*(제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?)\s*[.\)]?\s*(.{0,50})")
KWAN = re.compile(r"^#{0,3}\s*(제\s?\d+\s?[관편])\s+(.{2,40})")
TEUKYAK = re.compile(r"([가-힣A-Za-z0-9\(\)\[\]·%\s]{2,40}특약)\s*(?:\(|$|약관)")
BUPYO = re.compile(r"^#{0,3}\s*[\[\(]?\s*(부표\s?\d*|별표\s?\d*|[가-힣\s]{0,20}분류표|보험금\s?지급기준표)\s*[\]\)]?\s*(.{0,30})")
NUM_ITEM = re.compile(r"^(\d{1,2})\.\s+(.{2,50})$")
FORM = re.compile(r"(.{2,30}서식)")
TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
DECO = re.compile(r"^[A-Za-z\s]+$|·{3,}|\.{4,}")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def fn_doc_type(fn):
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


def clean(s, seen):
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 3 or s in seen or DECO.search(s) or not re.search(r"[가-힣]{2}", s):
        return None
    seen.add(s)
    return s


def parse_rider_params(name):
    p = {}
    if "(간편)" in name or "간편" in name[:10]:
        p["간편"] = True
    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        p["변형"] = m.group(1)
    if "갱신형" in name:
        p["갱신"] = True
    if "미지급" in name:
        p["환급"] = "미지급형"
    elif "일부지급" in name:
        p["환급"] = "일부지급형"
    return p


def scan(path):
    """1패스 스캔으로 유형 판정용 카운트 + 유형별 원재료를 동시 수집."""
    st = {"n_lines": 0, "n_jo": 0, "n_kwan_pyeon": 0, "n_table": 0, "n_head": 0}
    jo_titles, kwan_titles, riders, bupyos = [], [], [], []
    items, forms, heads = [], [], []
    tbl_headers, tbl_captions = [], []
    first_lines = []
    prev_nonblank = ""
    in_table = False
    seen = {"jo": set(), "kw": set(), "rd": set(), "bp": set(),
            "it": set(), "fm": set(), "hd": set(), "th": set(), "tc": set()}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                st["n_lines"] += 1
                s = line.strip()
                if not s:
                    in_table = False
                    continue
                if len(first_lines) < 5:
                    first_lines.append(s[:60])
                # 표
                tr = TABLE_ROW.match(s)
                if tr:
                    st["n_table"] += 1
                    if not in_table:  # 표 시작
                        if (c := clean(prev_nonblank[:50], seen["tc"])) and len(tbl_captions) < 20:
                            tbl_captions.append(c)
                        hdr = " | ".join(x.strip() for x in tr.group(1).split("|")[:6] if x.strip())
                        if hdr and hdr not in seen["th"] and len(tbl_headers) < 20:
                            seen["th"].add(hdr)
                            tbl_headers.append(hdr[:80])
                        in_table = True
                else:
                    in_table = False
                # 헤딩
                hm = HEAD.match(s)
                if hm:
                    st["n_head"] += 1
                    if (c := clean(hm.group(2)[:50], seen["hd"])) and len(heads) < 30:
                        heads.append(c)
                # 조·관·편
                jm = JO.match(s)
                if jm and len(jm.group(2)) < 45:
                    st["n_jo"] += 1
                    key = jm.group(1).replace(" ", "")
                    if key not in seen["jo"] and len(jo_titles) < 400:
                        seen["jo"].add(key)
                        jo_titles.append(f"{key} {jm.group(2).strip()}".strip())
                km = KWAN.match(s)
                if km:
                    st["n_kwan_pyeon"] += 1
                    if (c := clean(km.group(1) + " " + km.group(2), seen["kw"])) and len(kwan_titles) < 60:
                        kwan_titles.append(c)
                # 특약 (헤딩·캡션 줄에서만 — 본문 문장 오탐 방지)
                if hm or (len(s) < 60 and not s[0].isdigit()):
                    tm = TEUKYAK.search(s)
                    if tm:
                        name = re.sub(r"\s+", "", tm.group(1))[-40:]
                        if len(name) >= 4 and name not in seen["rd"] and len(riders) < 250:
                            seen["rd"].add(name)
                            riders.append(name)
                # 부표
                bm = BUPYO.match(s)
                if bm:
                    if (c := clean((bm.group(1) + " " + bm.group(2)).strip(), seen["bp"])) and len(bupyos) < 40:
                        bupyos.append(c)
                # 번호 항목 / 서식
                nm = NUM_ITEM.match(s)
                if nm:
                    if (c := clean(nm.group(2), seen["it"])) and len(items) < 40:
                        items.append(f"{nm.group(1)}. {c}")
                fm = FORM.search(s)
                if fm and len(s) < 60:
                    if (c := clean(fm.group(1), seen["fm"])) and len(forms) < 20:
                        forms.append(c)
                prev_nonblank = s
    except OSError:
        return None
    return st, {"jo_titles": jo_titles, "kwan_titles": kwan_titles, "riders": riders,
                "bupyos": bupyos, "items": items, "forms": forms, "heads": heads,
                "tbl_headers": tbl_headers, "tbl_captions": tbl_captions,
                "first_lines": first_lines}


def assign_type(st):
    if st["n_lines"] < 30:
        return "D"
    ratio = st["n_table"] / max(1, st["n_lines"])
    if st["n_jo"] >= 10:
        return "A"
    if ratio > 0.2:
        return "C"
    return "B"


def build_fm(ftype, raw):
    if ftype == "A":
        return {"골격": raw["kwan_titles"][:30],
                "조": raw["jo_titles"],
                "특약구성": [{"name": r, **parse_rider_params(r)} for r in raw["riders"]],
                "부표": raw["bupyos"]}
    if ftype == "B":
        return {"항목": raw["items"], "서식": raw["forms"], "헤딩": raw["heads"][:20]}
    if ftype == "C":
        return {"표캡션": raw["tbl_captions"], "표헤더": raw["tbl_headers"],
                "헤딩": raw["heads"][:20]}
    return {"첫줄": raw["first_lines"], "비고": "스텁 — frontmatter 부적합"}


def main():
    n = 0
    tdist = Counter()
    fill = Counter()
    with open(OUT / "doc_frontmatter.jsonl", "w") as out:
        for p in sorted(ROOT.rglob("*.md")):
            r = scan(p)
            if r is None:
                continue
            st, raw = r
            ftype = assign_type(st)
            fm = build_fm(ftype, raw)
            rec = {"file": nfc(str(p.relative_to(ROOT))),
                   "product": nfc(p.parent.name),
                   "fn_doc_type": fn_doc_type(nfc(p.name)),
                   "ftype": ftype, "n_lines": st["n_lines"], "fm": fm}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            tdist[ftype] += 1
            if ftype == "A" and fm["특약구성"]:
                fill["A_특약"] += 1
            if ftype == "A" and fm["조"]:
                fill["A_조"] += 1
            if ftype == "B" and (fm["항목"] or fm["서식"] or fm["헤딩"]):
                fill["B_내용"] += 1
            if ftype == "C" and (fm["표헤더"] or fm["표캡션"]):
                fill["C_표"] += 1
            if n % 2000 == 0:
                print(f"  {n}...", flush=True)

    print(f"\n총 {n:,}건 → out/doc_frontmatter.jsonl")
    print(f"유형 분포: A(약관군) {tdist['A']:,} | B(서술형) {tdist['B']:,} | "
          f"C(표중심) {tdist['C']:,} | D(스텁) {tdist['D']:,}")
    print(f"채움률: A 특약 {fill['A_특약']}/{tdist['A']} · A 조 {fill['A_조']}/{tdist['A']} | "
          f"B 내용 {fill['B_내용']}/{tdist['B']} | C 표 {fill['C_표']}/{tdist['C']}")


if __name__ == "__main__":
    sys.exit(main())
