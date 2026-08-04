#!/usr/bin/env python3
"""본실험: 검색형 170문항 × A/B1/B3. 홉 하드캡 --max-turns 12.

에이전트 출력: 근거 엘리먼트 순위 목록(최대 5) + 답변 텍스트.
결과: out/agentic3_full.jsonl (증분, 재실행 시 이어서)
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BASE

C3 = BASE / "frontmatter" / "corpus3"
OUT = BASE / "frontmatter" / "out" / "agentic3_full.jsonl"

PROMPT = """너는 보험 판매약관 검색 에이전트다. 현재 폴더에 약관이 엘리먼트 단위 md 파일로 있다
(폴더=주계약/특약, 파일명=[엘리먼트ID]_조번호.md, 파일 첫 줄에 [ID]와 소속 조가 표기됨).
{fm_hint}
Glob/Grep/Read 도구로 탐색해서 아래 질문의 근거 엘리먼트를 찾고, 그 내용으로 질문에 답하라.
반드시 후보 엘리먼트 파일을 Read로 열어 내용을 확인한 뒤 답하라.
마지막 줄에 반드시 다음 JSON만 출력하라 (그 외 설명 금지):
{{"elements": ["관련도 순 엘리먼트ID 최대 5개"], "answer": "질문에 대한 답변 2~3문장"}}

질문: {q}"""

FM_HINT = "루트에 FRONTMATTER.md(이 약관의 구성·급부·조 색인)가 있다. 먼저 참고하면 탐색을 줄일 수 있다."


def run_one(arm, q):
    cwd = C3 / arm
    hint = FM_HINT if (cwd / "FRONTMATTER.md").exists() else ""
    r = subprocess.run(
        ["claude", "-p", "--model", "haiku", "--output-format", "json",
         "--max-turns", "30",
         "--allowedTools", "Glob", "Grep", "Read",
         "--disallowedTools", "Bash", "Write", "Edit", "WebSearch", "WebFetch"],
        input=PROMPT.format(q=q, fm_hint=hint), capture_output=True, text=True,
        timeout=420, cwd=str(cwd))
    try:
        out = json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout or r.stderr)[:200]}
    result = str(out.get("result", ""))
    els, answer = [], ""
    m = re.search(r'\{[^{}]*"elements"[^{}]*\}', result, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            els = [e for e in d.get("elements", []) if re.fullmatch(r"e\d{5}", str(e))]
            answer = str(d.get("answer", ""))
        except json.JSONDecodeError:
            pass
    if not els:
        els = list(dict.fromkeys(re.findall(r"e\d{5}", result[-600:])))[:5]
    return {"elements": els, "answer": answer, "turns": out.get("num_turns"),
            "cost": out.get("total_cost_usd"), "dur_ms": out.get("duration_ms")}


def main():
    gold_map = json.loads((BASE / "frontmatter" / "out" / "element_gold.json").read_text())
    qs = json.loads((BASE / "frontmatter" / "out" / "search_questions.json").read_text())
    items = [(q, gold_map[q]) for q in qs if q in gold_map]

    done = set()
    if OUT.exists():
        for l in open(OUT):
            d = json.loads(l)
            if "error" not in d:
                done.add((d["arm"], d["q"]))
    jobs = [(arm, q, g) for q, g in items for arm in ("A", "B1", "B3")
            if (arm, q) not in done]
    print(f"실행 {len(jobs)}건 (기존 {len(done)} 재사용)", flush=True)

    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_one, arm, q): (arm, q, g) for arm, q, g in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            arm, q, g = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = {"error": str(e)[:200]}
            hit = bool(set(res.get("elements", [])) & set(g))
            rec = {"arm": arm, "q": q, "hit": hit, "gold": g[:5], **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)}] {arm:3s} {'HIT ' if hit else 'miss'} {q[:30]}", flush=True)
    print("DONE_ALL", flush=True)


if __name__ == "__main__":
    main()
