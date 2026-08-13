#!/usr/bin/env python3
"""특약 제목 라인 패턴으로 contract_scope 재부여 후 chunks.jsonl 갱신."""
import json, re, unicodedata, os
DOC = "/Users/seyoung/Documents/02 Braincrew/Shinhan Life/QA_set/판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
lines = raw.split("\n")
RIDER = re.compile(r"^\(간편\).{0,60}특약\(무배당[^)]*\)\s*$")
MAIN = re.compile(r"^신한\(간편가입\)통합건강보험 원\(ONE\)\(무배당[^)]*\)\s*$")
bounds = []  # (line_no_1based, scope)
for i, l in enumerate(lines):
    s = l.strip()
    if "|" in s: continue
    if RIDER.match(s): bounds.append((i + 1, s))
    elif MAIN.match(s) and i + 1 > 200: bounds.append((i + 1, "주계약(" + s + ")"))
print("boundaries:", len(bounds))
chunks = [json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"))]
import bisect
bl = [b[0] for b in bounds]
n_set = 0
for c in chunks:
    j = bisect.bisect_right(bl, c["line_start"]) - 1
    if j >= 0:
        c["contract_scope"] = bounds[j][1]
        n_set += 1
    elif not c["contract_scope"]:
        c["contract_scope"] = ""
with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
import collections
print("scope 부여:", n_set, "/", len(chunks), "| distinct:", len(set(c["contract_scope"] for c in chunks)))
