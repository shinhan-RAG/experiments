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


def main():
    chunks = [json.loads(l) for l in open(OUT / "chunks.jsonl", encoding="utf-8")]
    tags = [json.loads(l) for l in open(FS / "out/tags_u4_fact_rules.jsonl", encoding="utf-8")]
    els = {json.loads(l)["element_id"]: json.loads(l)
           for l in open(FS / "out/elements_u3.jsonl", encoding="utf-8")}
    tag_rows = sorted((els[t["element_id"]]["char_start"],
                       els[t["element_id"]]["char_end"], t)
                      for t in tags if t["element_id"] in els)
    starts = [r[0] for r in tag_rows]

    with open(OUT / "view_RULE.jsonl", "w", encoding="utf-8") as f:
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
            if contracts:
                parts.append("[특약] " + " · ".join(k for k, _ in contracts.most_common(2)))
            if subs:
                parts.append("[주제] " + " · ".join(k for k, _ in subs.most_common(8)))
            if roles:
                parts.append("[역할] " + " · ".join(k for k, _ in roles.most_common(4)))
            if arts:
                parts.append("[조항] " + " · ".join(arts[:4]))
            prefix = ("\n".join(parts) + "\n") if parts else ""
            f.write(json.dumps({"chunk_id": c["chunk_id"], "text": prefix + c["text"]},
                               ensure_ascii=False) + "\n")
    print(json.dumps({"chunks": len(chunks), "out": str(OUT / "view_RULE.jsonl")}))


if __name__ == "__main__":
    main()
