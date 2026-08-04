#!/usr/bin/env python3
"""3-arm 에이전틱 서치 평가 (A / B1 / B3).

각 문항을 claude -p(haiku, Glob/Grep/Read만) 에이전트로 실행. cwd=corpus3/{arm}.
에이전트는 근거 엘리먼트 ID를 지목 → 정답 엘리먼트와 대조.
결과: out/agentic3_results.jsonl (증분). 사용: python3 agentic_eval3.py <문항수>
"""
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import BASE, QA

C3 = BASE / "frontmatter" / "corpus3"
OUT = BASE / "frontmatter" / "out" / "agentic3_results.jsonl"

PROMPT = """너는 보험 판매약관 검색 에이전트다. 현재 폴더에 약관이 엘리먼트 단위 md 파일로 있다
(폴더=주계약/특약, 파일명=[엘리먼트ID]_조번호.md, 파일 첫 줄에 [ID]와 소속 조가 표기됨).
{fm_hint}
Glob/Grep/Read 도구로 탐색해서, 아래 질문의 근거가 되는 엘리먼트를 찾아라.
반드시 후보 엘리먼트 파일을 Read로 열어 내용을 확인한 뒤 답하라.
탐색은 도구 호출 최대 8회 안에 끝내라.
마지막 줄에 반드시 다음 JSON만 출력하라 (설명 금지):
{{"elements": ["e00123", ...]}}

질문: {q}"""

FM_HINT = "루트에 FRONTMATTER.md(이 약관의 구성·급부 색인)가 있다. 먼저 참고하면 탐색을 줄일 수 있다."


def run_one(arm, q):
    cwd = C3 / arm
    hint = FM_HINT if (cwd / "FRONTMATTER.md").exists() else ""
    r = subprocess.run(
        ["claude", "-p", "--model", "haiku", "--output-format", "json",
         "--allowedTools", "Glob", "Grep", "Read",
         "--disallowedTools", "Bash", "Write", "Edit", "WebSearch", "WebFetch"],
        input=PROMPT.format(q=q, fm_hint=hint), capture_output=True, text=True,
        timeout=300, cwd=str(cwd))
    try:
        out = json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout or r.stderr)[:200]}
    m = re.findall(r"e\d{5}", str(out.get("result", ""))[-500:])
    return {"elements": sorted(set(m)), "turns": out.get("num_turns"),
            "cost": out.get("total_cost_usd"), "dur_ms": out.get("duration_ms")}


def main(n):
    gold_map = json.loads((BASE / "frontmatter" / "out" / "element_gold.json").read_text())
    qa = [json.loads(l) for l in open(QA)]
    items = [(q["질문"], gold_map[q["질문"]]) for q in qa if q["질문"] in gold_map][:n]

    done = set()
    if OUT.exists():
        for l in open(OUT):
            d = json.loads(l)
            done.add((d["arm"], d["q"]))
    jobs = [(arm, q, g) for q, g in items for arm in ("A", "B1", "B3")
            if (arm, q) not in done]
    print(f"실행 {len(jobs)}건 (기존 {len(done)} 재사용)", flush=True)

    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
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
            print(f"  [{i}/{len(jobs)}] {arm:3s} {'HIT ' if hit else 'miss'} "
                  f"turns={res.get('turns')} {q[:34]}", flush=True)

    rows = [json.loads(l) for l in open(OUT)]
    print()
    for arm in ("A", "B1", "B3"):
        rs = [r for r in rows if r["arm"] == arm and "error" not in r]
        if not rs:
            continue
        hits = sum(r["hit"] for r in rs)
        turns = [r["turns"] for r in rs if r.get("turns")]
        durs = [r["dur_ms"] for r in rs if r.get("dur_ms")]
        print(f"{arm:3s}: 도달률 {hits}/{len(rs)} = {hits/len(rs)*100:.0f}% | "
              f"평균 홉 {sum(turns)/max(1,len(turns)):.1f} | "
              f"평균 {sum(durs)/max(1,len(durs))/1000:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
