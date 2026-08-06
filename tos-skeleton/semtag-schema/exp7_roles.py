#!/usr/bin/env python3
"""실험 7 — 역할 사전 유형별 보강(v2)의 커버리지·안정성 측정. 결정적, gold 불요.

측정:
1. exp6 표본 7문서: [역할] 커버리지 v1(기존 16종) vs v2(유형별 보강)
2. 광역 표본(약관 40 + 사업방법서 40, 결정적 등간격): 집계 커버리지 v1 vs v2
3. 원ONE 안정성: 기존 role1 배정이 v2에서 바뀐 element 비율(재현성 — 낮아야 함)
4. 태그 길이(role1만 태그행) ≤200자 준수
"""
import json
import statistics
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2
from role_vocab import roles_of_v2

HERE = Path(__file__).parent
SRC = Path(
    "/private/tmp/claude-501/-Users-donggyu-Documents----PageIndex/"
    "ef37eba4-cc2f-4769-9601-98adf8b2cab7/scratchpad/source/parsed_md")
ONE = Path(
    "/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
    "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")

EXP6_DOCS = [
    ("원ONE 약관(2026)", ONE, "yakgwan"),
    ("건강지기암 약관(2000)", SRC / "건강지기암보험/GunGangJiGi_000401_000430_P.md", "yakgwan"),
    ("개인연금저축 약관(1999)",
     SRC / "개인연금저축프리스타일/개인연금저축프리스타일_1999.02.01~2000.03.21.md", "yakgwan"),
    ("유니버설종신 약관(2016)",
     SRC / "(무)신한유니버설Plus종신보험/(무)신한유니버설Plus종신보험_판매약관_20160816_.md", "yakgwan"),
    ("사업방법서 큐브(2025)",
     SRC / "신한(간편가입)통합건강보험 큐브(무배당, 해약환급금 미지급형)/"
           "사업방법서_신한(간편가입)통합건강보험 큐브(무배당, 해약환급금 미지급형)_250401.md", "bizmethod"),
    ("사업방법서 세이프업변액(2012)",
     SRC / "(무)세이프업(Safe-Up)변액연금보험Ⅱ/(무)세이프업_사업방법서_20120402.md", "bizmethod"),
    ("사업방법서 치매보험(2019)",
     SRC / "신한간병비받는간편한치매보험(무배당, 무해지환급형)/"
           "사업방법서_신한간병비받는간편한치매보험(무배당_무해지환급형)_20190225.md", "bizmethod"),
]


def load_els(p):
    lines = SE.nfc(p.read_text(encoding="utf-8", errors="ignore")).splitlines()
    return annotate_v2(SE.split_elements(lines), lines)


def cov(els, fn):
    return sum(1 for e in els if fn(e["text"])) / max(1, len(els))


def sample(lst, k=40):
    step = max(1, len(lst) // k)
    return lst[::step][:k]


def main():
    out = {"per_doc": [], "broad": {}, "stability": {}, "tags": {}}

    for name, p, dt in EXP6_DOCS:
        els = load_els(p)
        v1 = cov(els, SE.roles_of)
        v2 = cov(els, lambda t: roles_of_v2(t, dt))
        out["per_doc"].append({"doc": name, "doctype": dt, "n": len(els),
                               "v1": round(v1, 3), "v2": round(v2, 3)})

    all_md = sorted(q for q in SRC.rglob("*.md") if ".chunks" not in str(q))
    yak = sample([q for q in all_md if "약관" in q.name and "사업방법서" not in q.name], 40)
    biz = sample([q for q in all_md if "사업방법서" in q.name], 40)
    for label, docs, dt in [("yakgwan_40", yak, "yakgwan"), ("bizmethod_40", biz, "bizmethod")]:
        tot = h1 = h2 = 0
        used = 0
        for q in docs:
            try:
                els = load_els(q)
            except Exception:
                continue
            used += 1
            tot += len(els)
            h1 += sum(1 for e in els if SE.roles_of(e["text"]))
            h2 += sum(1 for e in els if roles_of_v2(e["text"], dt))
        out["broad"][label] = {"docs": used, "elements": tot,
                               "v1": round(h1 / max(1, tot), 3),
                               "v2": round(h2 / max(1, tot), 3)}

    # 원ONE 안정성 + 태그 길이
    els = load_els(ONE)
    changed = 0
    tags = []
    for e in els:
        r1 = SE.roles_of(e["text"])
        r2 = roles_of_v2(e["text"], "yakgwan")
        if r1 and r2 and r1[0] != r2[0]:
            changed += 1
        parts = [f"[유형]{e['type']}"]
        if e.get("scope"):
            parts.append(f"[소속]{e['scope']}")
        if e.get("jo"):
            parts.append(f"[섹션]{e['jo']}")
        if r2:
            parts.append(f"[역할]{r2[0]}")
        tags.append(" ".join(parts))
    out["stability"] = {"one_role1_changed": changed, "one_elements": len(els),
                        "changed_pct": round(changed / len(els) * 100, 2)}
    out["tags"] = {"p50": int(statistics.median(len(t) for t in tags)),
                   "over200": sum(1 for t in tags if len(t) > 200)}

    (HERE / "out/exp7").mkdir(exist_ok=True)
    (HERE / "out/exp7/roles_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
