#!/usr/bin/env python3
"""내용 질의 QA셋 생성 — 본문 유래 용어, 정답=ripgrep 전수 검증(문서군).

설계: 층화 표본(A/B/C) → 본문 중간부에서 판별력 있는 구절 추출(frontmatter 필드 안 봄)
      → rg로 해당 구절이 실존하는 문서 전수 확인 → 정답군 1~50건이면 채택
출력: out/content_qa.jsonl
"""
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/seyoung/Downloads/parsed_md")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/frontmatter/out")
rng = random.Random(99)

QUOTA = {"A": 200, "B": 80, "C": 20}
BOILER = re.compile(r"회사는|계약자는|피보험자가|합니다|됩니다|경우에는|있습니다|바랍니다")
KO = re.compile(r"[가-힣]")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def candidate_terms(path):
    """본문 중간부에서 구절 후보 추출 (한 줄에서 연속 토큰 2~3개)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    body = [l.strip() for l in lines[40:] if 20 <= len(l.strip()) <= 90
            and not l.strip().startswith(("|", "#"))]
    rng.shuffle(body)
    out = []
    for l in body[:30]:
        toks = [t for t in l.split() if 3 <= len(t) <= 14 and KO.search(t)]
        if len(toks) < 3:
            continue
        i = rng.randrange(len(toks) - 2)
        phrase = " ".join(toks[i:i + rng.choice((2, 3))])
        if len(phrase) < 8 or BOILER.search(phrase):
            continue
        out.append(phrase)
        if len(out) >= 6:
            break
    return out


def rg_gold(term):
    r = subprocess.run(["rg", "-l", "--fixed-strings", "--max-count", "1", term, str(ROOT)],
                       capture_output=True, text=True, timeout=60)
    files = [nfc(str(Path(x).relative_to(ROOT))) for x in r.stdout.splitlines() if x]
    return files


def main():
    docs = [json.loads(l) for l in open(OUT / "doc_frontmatter.jsonl")]
    by_type = {}
    for d in docs:
        by_type.setdefault(d["ftype"], []).append(d["file"])
    for v in by_type.values():
        rng.shuffle(v)

    qa, tried = [], 0
    counts = Counter()
    idx = {t: 0 for t in QUOTA}
    while any(counts[t] < q for t, q in QUOTA.items()) and tried < 3000:
        for t, q in QUOTA.items():
            if counts[t] >= q or idx[t] >= len(by_type.get(t, [])):
                continue
            f = by_type[t][idx[t]]
            idx[t] += 1
            for term in candidate_terms(ROOT / f):
                tried += 1
                try:
                    gold = rg_gold(term)
                except Exception:
                    continue
                if 1 <= len(gold) <= 50 and f in gold:
                    qa.append({"type": t, "q": f"{term} 내용이 있는 문서 찾아줘",
                               "term": term, "src": f, "gold": gold,
                               "n_gold": len(gold)})
                    counts[t] += 1
                    break
        if all(idx[t] >= len(by_type.get(t, [])) for t in QUOTA):
            break

    with open(OUT / "content_qa.jsonl", "w") as f:
        for x in qa:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import statistics
    sizes = [x["n_gold"] for x in qa]
    print(f"QA {len(qa)}문항 (시도 {tried}): " + str(dict(counts)))
    print(f"정답군 크기: 중앙 {statistics.median(sizes):.0f}, 최대 {max(sizes)}, 1건짜리 {sum(1 for s in sizes if s==1)}")
    for x in qa[:5]:
        print(f"  예: [{x['type']}] {x['q'][:50]}  (정답 {x['n_gold']}건)")


if __name__ == "__main__":
    sys.exit(main())
