#!/usr/bin/env python3
"""정규화 텍스트 → 계층 분해: 약관 단위(주계약/특약) → 조(條).

페이지 헤더 감지(2패스):
  1) `<15자리+ 문서코드> <특약명> [페이지]` 형태
  2) 같은 접두어 + 서로 다른 페이지번호가 5회 이상 반복되는 줄 (예: `상품명_대면 66`)
헤더의 이름이 현재 약관 단위가 되고, 본문 조 헤딩(`제2-2조 제목`)으로 조를 나눈다.
출력: data/segmented/{vid}.json = [{unit, articles: [{no, title, text}]}]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton")
RAW = BASE / "data" / "raw_text"
OUT = BASE / "data" / "segmented"

CODED = re.compile(r"^\d{15,}\s+(.+?)(?:\s+\d{1,4})?$")
TRAIL_NUM = re.compile(r"^(.{4,90}?)\s+(\d{1,4})$")
FOOTER = re.compile(r"^SHINHAN LIFE(\s+\d{1,4})?$")
ARTICLE = re.compile(r"^(제\d+(?:-\d+)?조(?:의\d+)?)\s+([^()]{1,50}?)(?<!\d)$")
CHANNEL = re.compile(r"_(대면|TM)$")


def find_header_prefixes(lines):
    pages = defaultdict(set)
    for line in lines:
        m = TRAIL_NUM.match(line)
        if m and not CODED.match(line):
            pages[m.group(1)].add(m.group(2))
    return {p for p, nums in pages.items() if len(nums) >= 5}


def segment(lines):
    prefixes = find_header_prefixes(lines)
    units = []
    cur_unit = "_전문(front matter)"
    cur_art = {"no": "_서두", "title": "", "text": []}
    arts = [cur_art]

    def flush_unit():
        nonlocal arts, cur_art
        if any(a["text"] or a["no"] != "_서두" for a in arts):
            units.append({"unit": cur_unit, "articles": arts})
        cur_art = {"no": "_서두", "title": "", "text": []}
        arts = [cur_art]

    def set_unit(name):
        nonlocal cur_unit
        name = CHANNEL.sub("", name.strip())
        if name != cur_unit:
            flush_unit()
            cur_unit = name

    for line in lines:
        if FOOTER.match(line):
            continue
        m = CODED.match(line)
        if m:
            name = m.group(1).strip()
            if name != "SHINHAN LIFE":
                set_unit(name)
            continue
        t = TRAIL_NUM.match(line)
        if t and t.group(1) in prefixes:
            set_unit(t.group(1))
            continue
        a = ARTICLE.match(line)
        if a:
            cur_art = {"no": a.group(1), "title": a.group(2).strip(), "text": []}
            arts.append(cur_art)
            continue
        cur_art["text"].append(line)

    flush_unit()
    merged, order = {}, []
    for u in units:
        if u["unit"] not in merged:
            merged[u["unit"]] = u
            order.append(u["unit"])
        else:
            merged[u["unit"]]["articles"].extend(u["articles"])
    return [merged[k] for k in order]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for txt in sorted(RAW.glob("*.txt")):
        lines = txt.read_text(encoding="utf-8").splitlines()
        units = segment(lines)
        n_art = sum(len(u["articles"]) for u in units)
        (OUT / f"{txt.stem}.json").write_text(
            json.dumps(units, ensure_ascii=False), encoding="utf-8")
        print(f"{txt.stem:24s} units={len(units):3d} articles={n_art:5d}")


if __name__ == "__main__":
    sys.exit(main())
