#!/usr/bin/env python3
"""3-arm 실험 준비: 엘리먼트 분할 → 정답 재매핑 → 코퍼스 3벌 → frontmatter 2종.

엘리먼트 = 조(條) 내부를 항(①②…)·번호 항목·부표 경계로 자른 블록.
출력:
  out/elements.json            [{id, unit, no, title, text}]
  out/element_gold.json        {질문: [element_id]}
  corpus3/A | corpus3/B1 | corpus3/B3   (B1/B3만 FRONTMATTER.md 포함)
"""
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import ns, BASE, QA

C3 = BASE / "frontmatter" / "corpus3"
OUT = BASE / "frontmatter" / "out"

MARK = re.compile(r"^(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|부표\d|※|\d{1,2}\.\s)")


def safe(name, maxlen=70):
    return re.sub(r"[/\\:*?\"<>|]", "_", name).strip()[:maxlen]


def split_elements(lines):
    """조 텍스트 줄들 → 마커 경계로 블록 분할 (없으면 ~500자 단위)."""
    blocks, cur = [], []
    for l in lines:
        if MARK.match(l) and cur:
            blocks.append(cur)
            cur = [l]
        else:
            cur.append(l)
            if sum(len(x) for x in cur) > 900:  # 마커 없는 장문 방지
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    return [" ".join(b) for b in blocks if len(ns(" ".join(b))) > 20]


def main():
    units = json.loads((BASE / "data" / "segmented" / "260507.json").read_text())
    elements = []
    for u in units:
        merged, order = {}, []
        for a in u["articles"]:
            k = (a["no"], a["title"])
            if k not in merged:
                merged[k] = []
                order.append(k)
            merged[k].extend(a["text"])
        for no, title in order:
            for txt in split_elements(merged[(no, title)]):
                elements.append({"id": f"e{len(elements):05d}", "unit": u["unit"],
                                 "no": no, "title": title, "text": txt})
    print(f"엘리먼트 {len(elements)}개 (조 대비 {len(elements)/3293:.1f}배)")
    (OUT / "elements.json").write_text(json.dumps(elements, ensure_ascii=False))

    # ── QA 정답 → 엘리먼트 재매핑
    ns_el = [ns(e["text"]) for e in elements]
    qa = [json.loads(l) for l in open(QA)]
    gold_map, unmapped = {}, 0
    for q in qa:
        gold = set()
        for src in q.get("출처", []):
            quote = ns(src.get("quote", ""))
            if len(quote) < 15:
                continue
            hit = [e["id"] for e, t in zip(elements, ns_el) if quote in t]
            if not hit:
                # 인용문이 엘리먼트 경계에 걸친 경우: 20자 윈도 다수결
                wins = [quote[:20], quote[len(quote)//2:len(quote)//2+20], quote[-20:]]
                sc = Counter()
                for w in wins:
                    for e, t in zip(elements, ns_el):
                        if w in t:
                            sc[e["id"]] += 1
                hit = [i for i, c in sc.items() if c >= 1 and sc.most_common(1)[0][1] >= 2
                       ] if sc else []
                hit = [i for i, c in sc.items() if c >= 1] if hit else hit
            gold |= set(hit)
        if gold:
            gold_map[q["질문"]] = sorted(gold)
        else:
            unmapped += 1
    print(f"정답 매핑 {len(gold_map)}문항 (실패 {unmapped})")
    (OUT / "element_gold.json").write_text(json.dumps(gold_map, ensure_ascii=False))

    # ── 코퍼스 3벌
    if C3.exists():
        shutil.rmtree(C3)
    for arm in ("A", "B1", "B3"):
        d = C3 / arm
        d.mkdir(parents=True)
    for e in elements:
        rel = Path(safe(e["unit"])) / f"{e['id']}_{safe(e['no'], 20)}.md"
        body = f"# [{e['id']}] {e['unit']} | {e['no']} {e['title']}\n\n{e['text']}\n"
        for arm in ("A", "B1", "B3"):
            p = C3 / arm / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")

    # ── frontmatter 2종 (B1: 연결 없음 / B3: elements 연결)
    fm = json.loads((OUT / "260507.frontmatter.json").read_text())
    cov = {c["rider"]: c for c in fm["coverage"]}
    main_cov = json.loads((OUT / "main_coverage.json").read_text()) \
        if (OUT / "main_coverage.json").exists() else []
    el_by_unit_no = {}
    for e in elements:
        el_by_unit_no.setdefault((e["unit"], e["no"]), []).append(e["id"])

    el_by_unit = {}
    for e in elements:
        el_by_unit.setdefault(e["unit"], {}).setdefault(f"{e['no']} {e['title']}", []).append(e["id"])

    def article_index(unit):
        rows = []
        for art, ids in el_by_unit.get(unit, {}).items():
            rows.append(f"{art}={','.join(ids[:3])}")
        return "; ".join(rows)

    def fm_lines(with_elements):
        L = ["# frontmatter — 신한(간편가입)통합건강보험 원(ONE) 판매약관 260507판",
             "", "## document",
             "상품: 신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)",
             "버전: 260507 / 문서종류: 판매약관", "", "## 주계약"]
        mu = next(u["unit"] for u in units if "원(ONE)" in u["unit"])
        for b in main_cov:
            row = f"- 급부: {b['name']} | 조건: {b['trigger']} | 근거조: {b['source']}"
            if with_elements:
                ids = el_by_unit_no.get((mu, b["source"]), [])[:6]
                row += f" | elements: {','.join(ids)}"
            L.append(row)
        if with_elements:
            L.append(f"- 조색인: {article_index(mu)}")
        L.append("")
        L.append("## 특약 (구성 + 급부)")
        for c in fm["composition"]:
            p = c["params"]
            pt = ", ".join(x for x in [
                "간편심사" if p["simplified"] else None, p["variant"],
                p["renewal"], p["refund"]] if x)
            L.append(f"### {c['rider']}  ({pt})")
            for b in cov.get(c["rider"], {}).get("benefits", []):
                row = f"- 급부: {b['name']} | 조건: {b['trigger']} | 근거조: {b['source']}"
                if with_elements:
                    ids = el_by_unit_no.get((c["rider"], b["source"]), [])[:6]
                    row += f" | elements: {','.join(ids)}"
                L.append(row)
            if with_elements:
                L.append(f"- 조색인: {article_index(c['rider'])}")
        return "\n".join(L)

    (C3 / "B1" / "FRONTMATTER.md").write_text(fm_lines(False), encoding="utf-8")
    (C3 / "B3" / "FRONTMATTER.md").write_text(fm_lines(True), encoding="utf-8")
    n = sum(1 for _ in (C3 / "A").rglob("*.md"))
    print(f"코퍼스 3벌 (각 {n}파일) + FRONTMATTER.md (B1/B3)")


if __name__ == "__main__":
    main()
