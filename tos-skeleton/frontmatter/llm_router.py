#!/usr/bin/env python3
"""LLM 라우터: 명함첩(특약명+태그+급부명)을 haiku에 주고 관련 특약 선택.

문항별 결과를 out/llm_routes.jsonl에 증분 저장 (재실행 시 이어서).
완료 후 라우터 정확도 + B_soft(LLM) recall 평가.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import ns, BASE
from gate2b import load_all, evaluate

ROUTES = BASE / "frontmatter" / "out" / "llm_routes.jsonl"

DISEASE = {'암':'암','뇌혈관':'뇌혈관질환','심장':'허혈심장질환','치매':'치매','당뇨':'당뇨',
 '치아':'치아','장해':'장해','골절':'골절','수술':'수술','입원':'입원','간병':'간병',
 '유방':'여성암','전립선':'남성암','갑상선':'갑상선암','재해':'재해','생활비':'생활비'}


def build_card(fm):
    cov = {c["rider"]: c for c in fm["coverage"]}
    lines = []
    for i, c in enumerate(fm["composition"]):
        tags = sorted({v for k, v in DISEASE.items() if k in c["rider"]})
        bens = ", ".join(b["name"] for b in cov.get(c["rider"], {}).get("benefits", []))
        lines.append(f"{i}| {c['rider']} [{','.join(tags)}] :: {bens}")
    return "\n".join(lines)


def ask(card, question):
    prompt = f"""아래는 보험상품에 부가된 특약 목록이다 (번호| 특약명 [태그] :: 급부명들).

질문에 답하기 위해 찾아봐야 할 특약을 관련도 순으로 최대 5개 골라, 번호만 JSON 배열로 출력하라.
다른 텍스트 금지. 관련 특약이 없으면 [] 출력.

질문: {question}

특약 목록:
{card}"""
    r = subprocess.run(["claude", "-p", "--model", "haiku"], input=prompt,
                       capture_output=True, text=True, timeout=180)
    m = re.search(r"\[[\d,\s]*\]", r.stdout)
    return json.loads(m.group(0)) if m else None


def main():
    units, chunks, bm, rider_names, rider_bm, ALWAYS, mapped = load_all()
    fm = json.loads((BASE / "frontmatter" / "out" / "260507.frontmatter.json").read_text())
    card = build_card(fm)
    comp_names = [c["rider"] for c in fm["composition"]]

    done = {}
    if ROUTES.exists():
        for l in open(ROUTES):
            d = json.loads(l)
            done[d["q"]] = d["riders"]
    todo = [m for m in mapped if m["q"] not in done]
    print(f"기존 {len(done)} 재사용, 신규 {len(todo)} 라우팅", flush=True)

    with open(ROUTES, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(ask, card, m["q"]): m for m in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            m = futs[fu]
            try:
                idxs = fu.result()
            except Exception as e:
                idxs = None
            riders = [comp_names[j] for j in (idxs or []) if 0 <= j < len(comp_names)]
            done[m["q"]] = riders
            f.write(json.dumps({"q": m["q"], "riders": riders}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # ── 평가: 라우터 정확도 + recall
    rider_gold = routed_hit = 0
    for item in mapped:
        gu = {chunks[i]["unit"] for i in item["gold"]}
        if not (gu - ALWAYS):
            continue
        rider_gold += 1
        if gu & set(done.get(item["q"], [])):
            routed_hit += 1
    print(f"\nLLM 라우터 정확도: {routed_hit}/{rider_gold} = {routed_hit/max(1,rider_gold)*100:.0f}% (BM25 라우터: 69%)")

    all_idx = list(range(len(chunks)))
    res = []
    for item in mapped:
        q = item["q"]
        base = [bm.score(q, i) for i in all_idx]
        sel = ALWAYS | set(done.get(q, []))
        sc = [base[i] * (1.3 if chunks[i]["unit"] in sel else 1.0) for i in all_idx]
        res.append(sorted(all_idx, key=lambda i: -sc[i])[:20])
    evaluate("B soft LLM", res, mapped)


if __name__ == "__main__":
    main()
