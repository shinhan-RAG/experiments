#!/usr/bin/env python3
"""에이전틱 문서 찾기 — frontmatter 유/무 A/B (컬렉션 = parsed_md 전체).

질의: v2 인벤토리(high 신뢰 날짜)에서 합성 — 최신정본/연도종류/종류
에이전트: claude -p haiku, Glob/Grep/Read, --max-turns 30, cwd=parsed_md
  A: 폴더·파일명만 / B: + COLLECTION_FRONTMATTER.md (절대경로, --add-dir)
출력: out/agentic_docfind.jsonl (증분)
"""
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
FM_MD = OUT / "COLLECTION_FRONTMATTER.md"
DEST = OUT / "agentic_docfind.jsonl"
rng = random.Random(11)


def clean_name(name):
    s = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    s = s.replace("무배당", " ").replace("(무)", " ")
    return re.sub(r"\s+", " ", s).strip()


def render_fm(fm):
    L = [f"# {fm['collection']['name']} — 컬렉션 frontmatter",
         f"상품 {fm['collection']['n_products']} / 문서 {fm['collection']['n_documents']}",
         "형식: 폴더명 아래에 '종류 | 날짜 | 파일명' (★=해당 종류 최신 정본)", ""]
    for p in fm["products"]:
        if not p["inventory"]:
            continue
        flags = ",".join(p["identity"]["flags"] + p["identity"]["kind_hints"])
        L.append(f"## {p['product']}  [{flags}]")
        reps = set(p["representative"].values())
        for iv in p["inventory"]:
            d = iv["date"]["date"] if iv["date"] else "????"
            star = "★" if iv["file"] in reps else " "
            L.append(f"- {star} {iv['doc_type']} | {d} | {iv['file']}")
    return "\n".join(L)


def make_queries(fm, n_per=20):
    qs = []
    prods = [p for p in fm["products"] if len(p["inventory"]) >= 3]
    rng.shuffle(prods)
    for p in prods:
        name = clean_name(p["product"])
        if len(name) < 4:
            continue
        inv = p["inventory"]
        hi = [iv for iv in inv if iv["date"] and iv["date"]["confidence"] == "high"
              and iv["doc_type"] != "미분류"]
        yak = [iv for iv in hi if iv["doc_type"] == "판매약관"]
        if len(yak) >= 2 and len([q for q in qs if q["type"] == "최신정본"]) < n_per:
            mx = max(iv["date"]["date"] for iv in yak)
            gold = [p["product"] + "/" + iv["file"] for iv in yak if iv["date"]["date"] == mx]
            qs.append({"type": "최신정본", "q": f"{name} 최신 판매약관 파일 찾아줘",
                       "gold": gold})
        if hi and len([q for q in qs if q["type"] == "연도종류"]) < n_per:
            iv = rng.choice(hi)
            y, t = iv["date"]["date"][:4], iv["doc_type"]
            gold = [p["product"] + "/" + x["file"] for x in hi
                    if x["date"]["date"][:4] == y and x["doc_type"] == t]
            qs.append({"type": "연도종류", "q": f"{name} {y}년 {t} 파일 찾아줘", "gold": gold})
        types = {iv["doc_type"] for iv in inv} - {"미분류"}
        if types and len([q for q in qs if q["type"] == "종류"]) < n_per:
            t = rng.choice(sorted(types))
            gold = [p["product"] + "/" + iv["file"] for iv in inv if iv["doc_type"] == t]
            qs.append({"type": "종류", "q": f"{name} {t} 아무거나 하나 찾아줘", "gold": gold})
        if all(len([q for q in qs if q["type"] == t]) >= n_per
               for t in ("최신정본", "연도종류", "종류")):
            break
    return qs


PROMPT = """너는 보험 문서 검색 에이전트다. 현재 폴더에 보험 상품 문서가
'상품폴더/파일.md' 구조로 9,417개 있다 (폴더=상품명).
{fm_hint}
Glob/Grep/Read 도구로 탐색해서 아래 요청에 맞는 파일을 찾아라.
날짜 표기는 6자리(YYMMDD)와 8자리(YYYYMMDD)가 섞여 있으니 주의하라.
마지막 줄에 반드시 다음 JSON만 출력하라:
{{"file": "상품폴더명/파일명.md"}}

요청: {q}"""

FM_HINT = ("색인 파일이 있다: {fm_path}\n"
           "(상품별 문서 목록 — 종류·정규화 날짜·★최신정본 표시. 먼저 Grep하면 빠르다)")


def run_one(arm, q):
    hint = FM_HINT.format(fm_path=FM_MD) if arm == "B" else ""
    cmd = ["claude", "-p", "--model", "haiku", "--output-format", "json",
           "--max-turns", "30",
           "--allowedTools", "Glob", "Grep", "Read",
           "--disallowedTools", "Bash", "Write", "Edit", "WebSearch", "WebFetch"]
    if arm == "B":
        cmd += ["--add-dir", str(OUT)]
    r = subprocess.run(cmd, input=PROMPT.format(q=q, fm_hint=hint),
                       capture_output=True, text=True, timeout=420, cwd=str(ROOT))
    try:
        out = json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout or r.stderr)[:200]}
    m = re.search(r'\{[^{}]*"file"[^{}]*\}', str(out.get("result", "")), re.S)
    f = ""
    if m:
        try:
            f = json.loads(m.group(0)).get("file", "")
        except json.JSONDecodeError:
            pass
    return {"file": unicodedata.normalize("NFC", f), "turns": out.get("num_turns"),
            "dur_ms": out.get("duration_ms"), "cost": out.get("total_cost_usd")}


def main():
    fm = json.loads((OUT / "collection_frontmatter_v2.json").read_text())
    if not FM_MD.exists():
        FM_MD.write_text(render_fm(fm), encoding="utf-8")
        print(f"FRONTMATTER.md 렌더링 {FM_MD.stat().st_size//1024}KB")
    queries = make_queries(fm)
    print(f"질의 {len(queries)}: {Counter(q['type'] for q in queries)}", flush=True)

    done = set()
    if DEST.exists():
        for l in open(DEST):
            d = json.loads(l)
            if "error" not in d:
                done.add((d["arm"], d["q"]))
    jobs = [(arm, item) for item in queries for arm in ("A", "B")
            if (arm, item["q"]) not in done]
    print(f"실행 {len(jobs)}건", flush=True)

    with open(DEST, "a") as f, ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_one, arm, it["q"]): (arm, it) for arm, it in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            arm, it = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = {"error": str(e)[:200]}
            gold_ns = {unicodedata.normalize("NFC", g) for g in it["gold"]}
            got = res.get("file", "")
            hit = any(got.endswith(Path(g).name) or got == g for g in gold_ns)
            rec = {"arm": arm, "type": it["type"], "q": it["q"], "hit": hit,
                   "gold": it["gold"][:3], **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)}] {arm} {'HIT ' if hit else 'miss'} "
                  f"t={res.get('turns')} {it['q'][:30]}", flush=True)

    rows = [json.loads(l) for l in open(DEST)]
    print()
    for arm in ("A", "B"):
        for t in ("최신정본", "연도종류", "종류", None):
            rs = [r for r in rows if r["arm"] == arm and "error" not in r
                  and (t is None or r["type"] == t)]
            if not rs:
                continue
            label = t or "전체"
            hits = sum(r["hit"] for r in rs)
            turns = [r["turns"] for r in rs if r.get("turns")]
            print(f"{arm} {label:6s}: {hits}/{len(rs)} = {hits/len(rs)*100:3.0f}% | "
                  f"홉 {sum(turns)/max(1,len(turns)):.1f}")
    print("DOCFIND_DONE", flush=True)


if __name__ == "__main__":
    main()
