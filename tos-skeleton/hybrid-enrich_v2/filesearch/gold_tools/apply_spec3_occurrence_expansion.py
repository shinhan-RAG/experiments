#!/usr/bin/env python3
"""정답기준 명세 v1 §3 의 '동일 문구 출현 확장'을 결정론으로 적용해 새 overlay 를 만든다.

규칙(명세 그대로):
- 각 gold member 의 정규화 텍스트(공백 제거 앞 160자, 40자 미만은 확장 안 함)와 정확히 일치하는
  문서 내 모든 위치를 같은 group 의 OR member 로 추가한다.
- 질문이 특약을 지정하면(질문 문자열에 특약명 표면형 존재) 그 특약 scope 안 출현만 인정.
- 일반 질문(특약 미지정)은 문서 전체 출현 인정.
QA/검색 결과는 사용하지 않는다(질문 문자열 + 문서 + 기존 gold 만). 추가 member 는 src=occurrence_spec3.
"""
import argparse, bisect, hashlib, json, re, sys, unicodedata
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent  # filesearch 루트 (gold_tools 하위로 이동)
sys.path.insert(0, str(HERE))
from map_gold_spans import norm_map
from textmatch import compact, contract_core

DOC = "/Users/ralph/Desktop/신한라이프/data_set/noah_qaset/판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
KEY, MIN, CAP = 160, 40, 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--elements", default=str(HERE / "out/elements_u3.jsonl"))
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    nd, idx = norm_map(raw)
    E = [json.loads(l) for l in open(a.elements, encoding="utf-8")]
    starts = [e["char_start"] for e in E]
    J = [json.loads(l) for l in open(str(HERE / "out/elements_u3jo.jsonl"), encoding="utf-8")]
    jstarts = [u["char_start"] for u in J]
    def jo_at(c):
        i = bisect.bisect_right(jstarts, c) - 1
        return J[i]["element_id"] if i >= 0 else ""
    def scope_at(c):
        i = bisect.bisect_right(starts, c) - 1
        return E[i]["contract_scope"] if i >= 0 else ""
    contracts = sorted({e["contract_scope"] for e in E if e["contract_scope"]}, key=len, reverse=True)
    cores = [(c, contract_core(re.sub(r"^주계약\((.*)\)$", r"\1", c))) for c in contracts]
    rows = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    n_exp = n_general = n_scoped = 0
    for row in rows:
        q = compact(row.get("q", ""))
        named = {c for c, core in cores if core and len(core) >= 3 and core in q}
        if named:
            n_scoped += 1
        else:
            n_general += 1
        for gr in row["groups"]:
            keys = set()
            for m in list(gr["members"]):
                k = re.sub(r"\s", "", raw[m["c0"]:m["c1"]])[:KEY]
                if len(k) < MIN or k in keys:
                    continue
                keys.add(k)
                added, p = 0, nd.find(k)
                while p >= 0 and added < CAP:
                    c0, c1 = idx[p], idx[p + len(k) - 1] + 1
                    if not any(c0 < x["c1"] and c1 > x["c0"] for x in gr["members"]):
                        sc = scope_at(c0)
                        ok = (not named) or (sc in named) or (re.sub(r"^주계약\((.*)\)$", r"\1", sc) in named)
                        if ok:
                            gr["members"].append({"c0": c0, "c1": c1, "src": "occurrence_spec3", "jo": jo_at(c0)})
                            added += 1; n_exp += 1
                    p = nd.find(k, p + 1)
            gr["c0"] = min(m["c0"] for m in gr["members"]); gr["c1"] = max(m["c1"] for m in gr["members"])
    with open(a.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    man = {"policy": "spec v1 §3 identical-text occurrence expansion (deterministic; question-string scope gate; no retrieval/QA input)",
           "params": {"key": KEY, "min": MIN, "cap": CAP},
           "counts": {"rows": len(rows), "general_q": n_general, "scoped_q": n_scoped, "added_members": n_exp},
           "base": {"path": str(Path(a.gold).resolve()), "sha256": hashlib.sha256(open(a.gold,'rb').read()).hexdigest()},
           "output_sha256": hashlib.sha256(open(a.output,'rb').read()).hexdigest()}
    json.dump(man, open(a.output + ".manifest.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(man["counts"] | {"sha": man["output_sha256"][:16]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
