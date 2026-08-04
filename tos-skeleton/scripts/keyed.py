#!/usr/bin/env python3
"""위치 슬롯 → 이름 붙은 key-value 변환.

1) 레이아웃 슬롯 제거: 전 버전 값이 공백 제거 후 동일하면 뼈대로 취급
2) 값 타입 규칙으로 키 부여 (차수/고시번호/날짜/특약참조/기타)
3) 같은 조에서 같은 키에 같은 값이 반복되면 병합 (치환 지점 수만 기록)
4) 문맥 시그니처(슬롯 앞뒤 뼈대 토큰)로 규칙 미적용 슬롯에 안정적 키 생성
출력: results/{tag}_kv.json  = {version: {article: {key: value}}}
      results/{tag}_kv_schema.json = 키별 타입·출현 조·치환 지점 수
"""
import json
import re
import sys
from collections import defaultdict

BASE = "/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/results"

RULES = [
    ("kcd_edition",        re.compile(r"^제(\d+)차$")),
    ("edition_notice",     re.compile(r"제\d{4}-\d+호")),
    ("date",               re.compile(r"(\d{4})\s*[.년]\s*(\d{1,2})\s*[.월]\s*(\d{1,2})")),
    ("rider_ref",          re.compile(r"특약|보장보험|주계약")),
    ("percent",            re.compile(r"\d+(\.\d+)?\s*%")),
    ("amount",             re.compile(r"[\d,]+\s*(원|만원)")),
]


def ns(s):
    return re.sub(r"\s+", "", s)


def type_of(vals_across):
    joined = [" ".join(v) for v in vals_across if v]
    if not joined:
        return "text"
    for name, rx in RULES:
        if all(rx.search(ns(j)) or rx.search(j) for j in joined):
            return name
    return "text"


def main(tag):
    sk = json.load(open(f"{BASE}/{tag}_skeleton.json"))
    vals = json.load(open(f"{BASE}/{tag}_values.json"))
    vids = list(vals)

    kv = {v: defaultdict(dict) for v in vids}
    schema = {}

    for art, s in sk.items():
        if s.get("fixed") or s.get("align_fail"):
            continue
        skel = s["tokens"]
        positions = sorted(int(p) for p in vals[vids[0]][art])
        # 1) 레이아웃 슬롯 걸러내기
        semantic = []
        for pos in positions:
            values = [vals[v][art][str(pos)] for v in vids]
            if len({ns(" ".join(x)) for x in values}) > 1:
                semantic.append((pos, values))
        if not semantic:
            continue
        # 2) 키 부여
        counters = defaultdict(int)
        for pos, values in semantic:
            t = type_of(values)
            if t == "text":  # 문맥 시그니처로 구분
                ctx = ns("".join(skel[max(0, pos - 2):pos]))[-12:]
                key = f"text@{ctx}" if ctx else f"text@{pos}"
            else:
                key = t
            # 3) 같은 키·같은 값 분포면 병합, 다른 값 분포면 번호 증가
            base_key, k = key, key
            while True:
                prev = schema.get((art, k))
                cur_sig = tuple(ns(" ".join(v)) for v in values)
                if prev is None:
                    schema[(art, k)] = {"type": t, "sig": cur_sig, "sites": 1}
                    break
                if prev["sig"] == cur_sig:
                    prev["sites"] += 1
                    break
                counters[base_key] += 1
                k = f"{base_key}_{counters[base_key] + 1}"
            for v, val in zip(vids, values):
                kv[v][art][k] = " ".join(val)

    out_kv = {v: {a: dict(d) for a, d in kv[v].items()} for v in vids}
    json.dump(out_kv, open(f"{BASE}/{tag}_kv.json", "w"), ensure_ascii=False, indent=1)
    out_schema = [
        {"article": a, "key": k, "type": m["type"], "sites": m["sites"]}
        for (a, k), m in sorted(schema.items())
    ]
    json.dump(out_schema, open(f"{BASE}/{tag}_kv_schema.json", "w"),
              ensure_ascii=False, indent=1)

    n_art = len({a for a, _ in schema})
    from collections import Counter
    types = Counter(m["type"] for m in schema.values())
    print(f"조 {n_art}개에서 키 {len(schema)}개 (치환 지점 {sum(m['sites'] for m in schema.values())}곳)")
    for t, c in types.most_common():
        print(f"  {t:16s} {c}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "전체17")
