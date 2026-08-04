#!/usr/bin/env python3
"""내용 QA v2 — 자연어 질문(LLM 생성) + 기계 검증 정답(ripgrep 문서군).

v1 문제(무작위 구절 → 비자연 질의·널뛰기) 교정:
  본문의 온전한 대목 → haiku가 현업스러운 질문 생성 → 판별 구절 rg로 gold 검증
출력: out/content_qa_v2.jsonl (증분)
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
DEST = OUT / "content_qa_v2.jsonl"
rng = random.Random(123)
QUOTA = {"A": 90, "B": 45, "C": 15}

PROMPT = """다음은 보험 문서의 한 대목이다. 이 대목의 내용을 찾으려는 현업 직원이
물을 법한 자연스러운 검색 질문을 1개만 만들어라.

규칙:
- 상품명·회사명을 질문에 넣지 마라 (내용만으로 찾는 상황)
- 대목의 핵심 내용(보장·조건·기준·절차)을 묻는 실무형 질문일 것
- 질문 한 문장만 출력. 다른 텍스트 금지.

대목:
{passage}"""


def nfc(s):
    return unicodedata.normalize("NFC", s)


def pick_passage(path):
    try:
        lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except OSError:
        return None, None
    # 온전한 대목: 연속 본문 줄 3~6개, 표·헤딩 제외, 한글 위주
    cands = []
    buf = []
    for l in lines[10:]:
        if l and not l.startswith(("|", "#")) and len(re.findall(r"[가-힣]", l)) > len(l) * 0.4:
            buf.append(l)
            if 2 <= len(buf) and 100 <= sum(map(len, buf)) <= 450:
                cands.append(" ".join(buf))
                buf = []
        else:
            buf = []
    if not cands:
        return None, None
    passage = rng.choice(cands)
    # 판별 앵커: 대목 안의 특징적 연속 토큰 (rg용)
    toks = [t for t in passage.split() if 4 <= len(t) <= 14]
    anchors = [" ".join(toks[i:i + 2]) for i in range(0, max(1, len(toks) - 2), 3)][:6]
    return passage, anchors


def rg_gold(term):
    r = subprocess.run(["rg", "-l", "--fixed-strings", "--max-count", "1", term, str(ROOT)],
                       capture_output=True, text=True, timeout=60)
    return [nfc(str(Path(x).relative_to(ROOT))) for x in r.stdout.splitlines() if x]


def gen_question(passage):
    r = subprocess.run(["claude", "-p", "--model", "haiku"],
                       input=PROMPT.format(passage=passage),
                       capture_output=True, text=True, timeout=90)
    q = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
    return q if 8 <= len(q) <= 120 and "?" in q or q.endswith(("요", "까", "지")) else ""


def one(ftype, f):
    passage, anchors = pick_passage(ROOT / f)
    if not passage:
        return None
    gold = None
    for a in anchors:
        try:
            g = rg_gold(a)
        except Exception:
            continue
        if 1 <= len(g) <= 50 and f in g:
            gold = (a, g)
            break
    if not gold:
        return None
    q = gen_question(passage)
    if not q:
        return None
    return {"type": ftype, "q": q, "anchor": gold[0], "src": f,
            "gold": gold[1], "n_gold": len(gold[1]), "passage": passage[:200]}


def main():
    docs = [json.loads(l) for l in open(OUT / "doc_frontmatter.jsonl")]
    by_type = {}
    for d in docs:
        by_type.setdefault(d["ftype"], []).append(d["file"])
    for v in by_type.values():
        rng.shuffle(v)

    done = Counter()
    if DEST.exists():
        for l in open(DEST):
            done[json.loads(l)["type"]] += 1
    jobs = []
    for t, q in QUOTA.items():
        need = max(0, (q - done[t]) * 8)  # 성공률 감안 3배 시도
        jobs += [(t, f) for f in by_type.get(t, [])[:need]]
    print(f"생성 시도 {len(jobs)}건 (기존 {sum(done.values())})", flush=True)

    counts = Counter(done)
    with open(DEST, "a") as out, ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(one, t, f): t for t, f in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            t = futs[fu]
            if counts[t] >= QUOTA[t]:
                continue
            try:
                rec = fu.result()
            except Exception:
                rec = None
            if rec:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                counts[t] += 1
            if i % 30 == 0:
                print(f"  {i}/{len(jobs)} | 확보 {dict(counts)}", flush=True)
            if all(counts[k] >= v for k, v in QUOTA.items()):
                break
    print(f"CONTENT_QA_V2_DONE {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
