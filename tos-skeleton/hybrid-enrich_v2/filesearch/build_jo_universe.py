#!/usr/bin/env python3
"""u2(항 단위) → 조(條) 단위 우주(u2jo). P-section(특약>절>조, ≤4,000자)에 대응하는 굵은 입도.

규칙: contract_scope 가 바뀌거나 heading 이 조·관·편·장·절 제목이면 새 단위 시작.
단위가 4,000자를 넘으면 element 경계에서 분절(part). 각 조 단위는 구성 element id 목록을 보존해
"세밀 색인 → 조 map-back" 채점에 쓴다.
"""
import argparse, hashlib, json, os, re

JO = re.compile(r"^제\d+(?:-\d+)?(?:조|관|편|장|절)")
MAXC = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    units, cur, cur_scope = [], [], None

    def close():
        if not cur:
            return
        # 4,000자 분절
        part, plen = [], 0
        for e in cur:
            if part and plen + len(e["text"]) > MAXC:
                units.append(part); part, plen = [], 0
            part.append(e); plen += len(e["text"]) + 1
        if part:
            units.append(part)
        cur.clear()

    for e in E:
        new = (e["contract_scope"] != cur_scope) or (e["element_type"] == "heading" and JO.match(e["text"].strip()))
        if new:
            close()
            cur_scope = e["contract_scope"]
        cur.append(e)
    close()
    with open(a.out, "w", encoding="utf-8") as f:
        for k, part in enumerate(units):
            title = next((x["text"].strip() for x in part if x["element_type"] == "heading"), "")
            f.write(json.dumps({
                "element_id": f"j{k:05d}", "element_type": "jo", "title": title[:80],
                "text": "\n".join(x["text"] for x in part),
                "char_start": part[0]["char_start"], "char_end": part[-1]["char_end"],
                "line_start": part[0]["line_start"], "line_end": part[-1]["line_end"],
                "contract_scope": part[0]["contract_scope"], "members": [x["element_id"] for x in part],
            }, ensure_ascii=False) + "\n")
    lens = sorted(sum(len(x["text"]) for x in p) for p in units)
    stats = {"n_units": len(units), "median_chars": lens[len(lens) // 2], "p90": lens[int(len(lens) * .9)],
             "max": lens[-1], "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}
    json.dump(stats, open(a.out.replace(".jsonl", "_stats.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
