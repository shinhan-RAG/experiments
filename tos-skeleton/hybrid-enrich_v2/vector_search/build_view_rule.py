#!/usr/bin/env python3
"""view_RULE 생성 — LLM 0회, 규칙 소개문(U4 태그) + 원문.

각 청크에 겹치는 element들의 U4 태그(contract_key/subject_key/role/article_title)를
집계해 [특약]/[주제]/[역할]/[조항] 소개문을 만들어 청크 원문 앞에 붙인다.
V9(LLM 소개문)의 무-LLM 대체 view. 결정론 — 같은 입력이면 byte 동일 출력.

Usage:
    python build_view_rule.py            # out/view_RULE.jsonl
    이후 임베딩: python embed_chunks.py --view RULE
"""
import bisect
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FS = HERE.parent / "filesearch"


def clean_subjects(keys):
    """v2: 절단 조각 정제 — 괄호 불균형·과길이·공백 낀 장문 배제, 부분문자열 병합."""
    ok = []
    for k in keys:
        if k.count("(") != k.count(")") or k.count("[") != k.count("]"):
            continue
        if len(k) > 20 or (" " in k and len(k) > 12):
            continue
        ok.append(k)
    ok.sort(key=len, reverse=True)
    merged = []
    for k in ok:
        if not any(k in m for m in merged):
            merged.append(k)
    return merged


import re as _re


def annex_start(els):
    """본문 [별첨1] 표제(참조 문구 아님) 최소 위치."""
    doc_end = max(e["char_end"] for e in els.values())
    starts = [e["char_start"] for e in els.values()
              if e["char_start"] > 0.6 * doc_end
              and e["text"].strip().startswith("[별첨1]")]
    return min(starts) if starts else None


def table_titles(text):
    """별표 존 청크에서 표 제목 추출 — 표NN(다음 비어있지 않은 줄이 제목) 또는 …분류표/기준표 단독 줄."""
    out = []
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    for i, ln in enumerate(lines):
        if _re.fullmatch(r"표\s*\d+(?:-\d+)?\.?", ln) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if 2 <= len(nxt) <= 50 and "|" not in nxt:
                item = f"{ln} {nxt}"
                if item not in out:
                    out.append(item)
        else:
            m = _re.fullmatch(r"((?:표\s*\d+(?:-\d+)?\.?\s*)?[^|]{2,50}?(?:분류표|기준표|한도표))", ln)
            if m and m.group(1) not in out:
                out.append(m.group(1))
    return out[:3]


def body_table_refs(els, ann):
    """본문(별첨 이전)에서 표N 참조 → 참조 특약명 집계."""
    import collections as _c
    refs = _c.defaultdict(_c.Counter)
    for e in els.values():
        if ann and e["char_start"] < ann and e.get("contract_scope"):
            for num in _re.findall(r"(?:별첨\s*2\]?\s*)?[<(]?\s*표\s*(\d+(?:-\d+)?)", e["text"]):
                refs[num][e["contract_scope"]] += 1
    return refs


def main(v2=False, mode="both"):
    chunks = [json.loads(l) for l in open(OUT / "chunks.jsonl", encoding="utf-8")]
    tags = [json.loads(l) for l in open(FS / "out/tags_u4_fact_rules.jsonl", encoding="utf-8")]
    els = {json.loads(l)["element_id"]: json.loads(l)
           for l in open(FS / "out/elements_u3.jsonl", encoding="utf-8")}
    tag_rows = sorted((els[t["element_id"]]["char_start"],
                       els[t["element_id"]]["char_end"], t)
                      for t in tags if t["element_id"] in els)
    starts = [r[0] for r in tag_rows]

    ann = annex_start(els) if v2 else None
    refs = body_table_refs(els, ann) if v2 else {}
    suffix = {"both": "RULE2", "annex": "RULE2A", "subject": "RULE2B"}[mode]
    out_name = f"view_{suffix}.jsonl" if v2 else "view_RULE.jsonl"
    with open(OUT / out_name, "w", encoding="utf-8") as f:
        for c in chunks:
            i = bisect.bisect_left(starts, c["char_start"]) - 3
            subs, roles, contracts = (collections.Counter(),
                                      collections.Counter(),
                                      collections.Counter())
            arts = []
            for s, e, t in tag_rows[max(0, i):]:
                if s >= c["char_end"]:
                    break
                if e <= c["char_start"]:
                    continue
                for x in t.get("subject_key") or []:
                    subs[x] += 1
                for x in t.get("role") or []:
                    roles[x] += 1
                if t.get("contract_key"):
                    contracts[t["contract_key"]] += 1
                a = (t.get("locator") or {}).get("article_title") or ""
                if a and a not in arts:
                    arts.append(a)
            parts = []
            in_annex = (v2 and mode in ("both", "annex")
                        and ann is not None and c["char_start"] >= ann)
            if in_annex:
                titles = table_titles(c["text"])
                if titles:
                    parts.append("[별표] " + " · ".join(titles))
                    ref_contracts = []
                    for ti in titles:
                        m = _re.match(r"표\s*(\d+(?:-\d+)?)", ti)
                        if m:
                            for ct, _n in refs.get(m.group(1), {}).most_common(3):
                                if ct not in ref_contracts:
                                    ref_contracts.append(ct)
                    if ref_contracts:
                        parts.append("[참조특약] " + " · ".join(ref_contracts[:3]))
                else:
                    parts.append("[별첨] 약관 권말 별첨(법령 인용·분류표) 구간")
                subj = clean_subjects([k for k, _ in subs.most_common(12)])
                if subj:
                    parts.append("[주제] " + " · ".join(subj[:8]))
            else:
                if contracts:
                    parts.append("[특약] " + " · ".join(k for k, _ in contracts.most_common(2)))
                keys = [k for k, _ in subs.most_common(12)]
                keys = (clean_subjects(keys)
                        if (v2 and mode in ("both", "subject")) else keys[:8])
                if keys:
                    parts.append("[주제] " + " · ".join(keys[:8]))
                if roles:
                    parts.append("[역할] " + " · ".join(k for k, _ in roles.most_common(4)))
                if arts:
                    parts.append("[조항] " + " · ".join(arts[:4]))
            prefix = ("\n".join(parts) + "\n") if parts else ""
            f.write(json.dumps({"chunk_id": c["chunk_id"], "text": prefix + c["text"]},
                               ensure_ascii=False) + "\n")
    print(json.dumps({"chunks": len(chunks), "out": str(OUT / out_name),
                      "annex_start": ann}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="RULE2: 별표 존 오귀속 교정 + subject_key 정제")
    ap.add_argument("--mode", default="both", choices=("both", "annex", "subject"))
    args = ap.parse_args()
    main(v2=args.v2, mode=args.mode)
