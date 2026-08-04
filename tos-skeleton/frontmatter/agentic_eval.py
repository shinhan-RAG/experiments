#!/usr/bin/env python3
"""에이전틱 서치 A/B 평가.

각 문항을 claude -p(haiku, Grep/Glob/Read만 허용) 에이전트로 실행:
  A안 cwd=corpus_plain (INDEX 없음) / B안 cwd=corpus_indexed (INDEX.md 있음)
에이전트가 근거 조항 파일 경로를 답하면 gold와 대조. num_turns=홉 수.
결과: out/agentic_results.jsonl (증분)
사용: python3 agentic_eval.py <문항수>
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import ns, BASE
from gate2b import load_all

CORPUS = BASE / "frontmatter" / "corpus"
OUT = BASE / "frontmatter" / "out" / "agentic_results.jsonl"

PROMPT = """너는 보험 판매약관 검색 에이전트다. 현재 폴더에 약관이 조(條) 단위 md 파일로 있다
(폴더=특약/주계약, 파일=조). INDEX.md가 있으면 먼저 읽으면 특약 구성을 빠르게 파악할 수 있다.
Glob/Grep/Read 도구로 탐색해서, 아래 질문의 근거가 되는 조항 파일을 찾아라.

탐색은 최대 8번의 도구 호출 안에 끝내라.
마지막 줄에 반드시 다음 형식의 JSON만 출력하라 (설명 금지):
{{"files": ["<찾은 근거 파일의 상대경로>", ...]}}

질문: {q}"""


def run_one(arm, q):
    cwd = CORPUS / f"corpus_{arm}"
    r = subprocess.run(
        ["claude", "-p", "--model", "haiku", "--output-format", "json",
         "--allowedTools", "Glob", "Grep", "Read",
         "--disallowedTools", "Bash", "Write", "Edit", "WebSearch", "WebFetch"],
        input=PROMPT.format(q=q), capture_output=True, text=True,
        timeout=300, cwd=str(cwd))
    try:
        out = json.loads(r.stdout)
        text = out.get("result", "")
        turns = out.get("num_turns")
        cost = out.get("total_cost_usd")
        dur = out.get("duration_ms")
    except Exception:
        return {"error": r.stdout[:200] or r.stderr[:200]}
    m = re.search(r'\{"files".*\}', text, re.S)
    files = []
    if m:
        try:
            files = json.loads(m.group(0)).get("files", [])
        except json.JSONDecodeError:
            pass
    return {"files": files, "turns": turns, "cost": cost, "dur_ms": dur}


def main(n):
    units, chunks, bm, rider_names, rider_bm, ALWAYS, mapped = load_all()
    cmap = json.loads((BASE / "frontmatter" / "out" / "corpus_map.json").read_text())
    # 문항 선정: 매핑된 것 중 앞에서부터 n개 (특약/주계약 자연 비율)
    items = mapped[:n]

    done = set()
    if OUT.exists():
        for l in open(OUT):
            d = json.loads(l)
            done.add((d["arm"], d["q"]))

    jobs = [(arm, it) for it in items for arm in ("plain", "indexed")
            if (arm, it["q"]) not in done]
    print(f"실행 {len(jobs)}건 (기존 {len(done)} 재사용)", flush=True)

    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, arm, it["q"]): (arm, it) for arm, it in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            arm, it = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = {"error": str(e)[:200]}
            gold_rel = {cmap[str(g)]["rel"] for g in it["gold"] if str(g) in cmap}
            hit = any(any(ns(fp).endswith(ns(Path(g).name)) or ns(g) in ns(fp)
                          for g in gold_rel) for fp in res.get("files", []))
            rec = {"arm": arm, "q": it["q"], "hit": hit, **res,
                   "gold": sorted(gold_rel)[:3]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(jobs)}] {arm:8s} {'HIT ' if hit else 'miss'} "
                  f"turns={res.get('turns')} {it['q'][:36]}", flush=True)

    # 집계
    rows = [json.loads(l) for l in open(OUT)]
    for arm in ("plain", "indexed"):
        rs = [r for r in rows if r["arm"] == arm and "error" not in r]
        if not rs:
            continue
        hits = sum(r["hit"] for r in rs)
        turns = [r["turns"] for r in rs if r.get("turns")]
        durs = [r["dur_ms"] for r in rs if r.get("dur_ms")]
        label = "A plain(색인無)" if arm == "plain" else "B indexed(색인有)"
        print(f"\n{label}: 도달률 {hits}/{len(rs)} = {hits/len(rs)*100:.0f}% | "
              f"평균 홉 {sum(turns)/len(turns):.1f} | 평균 {sum(durs)/len(durs)/1000:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
