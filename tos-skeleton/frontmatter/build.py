#!/usr/bin/env python3
"""약관 frontmatter v1 생성 파이프라인 (프로토타입).

구성(composition): 세그먼테이션에서 결정적 추출 + 특약명 룰 파싱
보장요약(coverage): 특약별 지급사유 조 → claude CLI(LLM)로 구조화 추출
사용: python3 build.py <version> [--riders N]
출력: out/<version>.frontmatter.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton")
OUT = BASE / "frontmatter" / "out"
PRODUCT = "신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)"

PROMPT = """다음은 보험 특약 약관의 '보험금의 지급사유' 조항 원문이다.
급부(지급 항목)별로 추출해 JSON 배열만 출력하라. 다른 텍스트 금지.

각 원소: {{"name": "급부명(원문 표기)", "trigger": "지급 조건 한 줄 요약",
"source": "{art_no}", "key_terms": ["원문에 글자 그대로 등장하는 핵심 용어 2~4개"]}}

key_terms는 반드시 아래 원문에 실제로 등장하는 문자열이어야 한다.

특약명: {rider}
조항({art_no} {art_title}):
{text}
"""


def parse_rider_name(name):
    p = {"simplified": name.startswith("(간편)"), "variant": None,
         "renewal": None, "refund": None}
    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        p["variant"] = f"[{m.group(1)}]"
    if "갱신형" in name:
        p["renewal"] = "갱신형"
    if "미지급형" in name:
        p["refund"] = "해약환급금 미지급형"
    elif "일부지급형" in name:
        p["refund"] = "해약환급금 일부지급형"
    return p


def merged_articles(unit):
    """목차 빈 항목과 본문 중복 조 병합."""
    m = {}
    order = []
    for a in unit["articles"]:
        k = (a["no"], a["title"])
        if k not in m:
            m[k] = list(a["text"])
            order.append(k)
        else:
            m[k].extend(a["text"])
    return [(no, title, m[(no, title)]) for no, title in order]


def llm_extract(rider, art_no, art_title, text, model):
    prompt = PROMPT.format(rider=rider, art_no=art_no, art_title=art_title,
                           text=text[:6000])
    r = subprocess.run(["claude", "-p", "--model", model],
                       input=prompt, capture_output=True, text=True, timeout=180)
    out = r.stdout.strip()
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return None, f"no-json: {out[:120]}"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"bad-json: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--riders", type=int, default=None, help="특약 수 제한(프로토)")
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()

    units = json.loads((BASE / "data" / "segmented" / f"{args.version}.json")
                       .read_text(encoding="utf-8"))
    riders = [u for u in units if "특약" in u["unit"] and "일반사항" not in u["unit"]]

    composition = [{"rider": u["unit"], "params": parse_rider_name(u["unit"])}
                   for u in riders]

    # 증분: 기존 산출물의 coverage는 재사용
    OUT.mkdir(exist_ok=True)
    dest = OUT / f"{args.version}.frontmatter.json"
    done = {}
    if dest.exists():
        done = {c["rider"]: c for c in json.loads(dest.read_text())["coverage"]}

    coverage, errors = list(done.values()), []
    targets = [u for u in (riders[: args.riders] if args.riders else riders)
               if u["unit"] not in done]
    print(f"기존 {len(done)}건 재사용, 신규 {len(targets)}건 추출", flush=True)

    def work(u):
        arts = merged_articles(u)
        pay = [(no, t, tx) for no, t, tx in arts
               if "지급사유" in t and "않는" not in t and tx]
        if not pay:
            return u["unit"], None, "지급사유 조 없음"
        no, title, tx = pay[0]
        benefits, err = llm_extract(u["unit"], no, title, " ".join(tx), args.model)
        return u["unit"], benefits, err

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, u): u for u in targets}
        for i, f in enumerate(as_completed(futs), 1):
            rider, benefits, err = f.result()
            if err:
                errors.append({"rider": rider, "error": err})
                print(f"  ERR {rider[:44]} {err[:60]}", flush=True)
            else:
                coverage.append({"rider": rider, "benefits": benefits})
                print(f"  ok [{i}/{len(targets)}] {rider[:44]:46s} 급부 {len(benefits)}", flush=True)
            if i % 10 == 0:  # 증분 저장
                dest.write_text(json.dumps(
                    {"document": {"product": PRODUCT, "version": args.version, "doc_type": "판매약관"},
                     "composition": composition, "coverage": coverage,
                     "extract_errors": errors}, ensure_ascii=False, indent=1), encoding="utf-8")

    fm = {"document": {"product": PRODUCT, "version": args.version, "doc_type": "판매약관"},
          "composition": composition, "coverage": coverage,
          "extract_errors": errors}
    dest.write_text(json.dumps(fm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncomposition {len(composition)} riders | coverage {len(coverage)} | errors {len(errors)}")
    print(f"-> {dest}")


if __name__ == "__main__":
    sys.exit(main())
