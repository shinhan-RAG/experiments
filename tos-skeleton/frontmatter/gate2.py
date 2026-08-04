#!/usr/bin/env python3
"""게이트 2 — frontmatter 라우팅이 검색을 개선하는지 QA셋 359로 측정.

A(baseline): 전체 조 청크에 BM25
B(routing) : 질문 ↔ frontmatter(특약명·급부·key_terms) 매칭으로 특약 top-R 선별
             → (주계약 + 전문 + 선별 특약) 청크로 제한한 BM25
정답: 문항의 출처 quote가 포함된 청크 (공백 무시 부분일치, 실패 시 20자 윈도)
지표: recall@k — top-k 안에 정답 청크가 있는 문항 비율
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton")
QA = Path("/Users/seyoung/workspace/braincrew/project/shinhan/05_RAG_QA/QA_set/정답셋_359_최종_v4.jsonl")
VERSION = "260507"
TOP_R = 5


def ns(s):
    return re.sub(r"\s+", "", s)


def bigrams(s):
    s = ns(s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs):
        self.N = len(docs)
        self.tf, self.dl, self.df = [], [], Counter()
        for d in docs:
            c = Counter(bigrams(d))
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            for t in c:
                self.df[t] += 1
        self.avgdl = sum(self.dl) / max(1, self.N)

    def score(self, query, idx):
        q = Counter(bigrams(query))
        s = 0.0
        for t, qc in q.items():
            if t not in self.tf[idx]:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            f = self.tf[idx][t]
            s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.dl[idx] / self.avgdl))
        return s

    def rank(self, query, candidates):
        return sorted(candidates, key=lambda i: -self.score(query, i))


def main():
    # ── 청크: (unit, 조) 단위, 중복 조 병합
    units = json.loads((BASE / "data" / "segmented" / f"{VERSION}.json").read_text())
    chunks = []          # {unit, no, title, text}
    for u in units:
        merged = {}
        order = []
        for a in u["articles"]:
            k = (a["no"], a["title"])
            if k not in merged:
                merged[k] = []
                order.append(k)
            merged[k].extend(a["text"])
        for no, title in order:
            txt = " ".join(merged[(no, title)])
            if len(ns(txt)) > 30:
                chunks.append({"unit": u["unit"], "no": no, "title": title, "text": txt})
    print(f"청크 {len(chunks)}개 (조 단위)")
    ns_chunks = [ns(c["text"]) for c in chunks]
    bm = BM25([c["title"] + " " + c["text"] for c in chunks])

    # ── frontmatter 라우팅 인덱스: 특약별 프로필 텍스트
    fm = json.loads((BASE / "frontmatter" / "out" / f"{VERSION}.frontmatter.json").read_text())
    cov = {c["rider"]: c for c in fm["coverage"]}
    profiles = {}
    for c in fm["composition"]:
        parts = [c["rider"]]
        for b in cov.get(c["rider"], {}).get("benefits", []):
            parts += [b.get("name", ""), b.get("trigger", "")] + b.get("key_terms", [])
        profiles[c["rider"]] = " ".join(parts)
    rider_names = list(profiles)
    rider_bm = BM25([profiles[r] for r in rider_names])
    ALWAYS = {u["unit"] for u in units if "특약" not in u["unit"]}  # 주계약·전문 등

    # ── QA 로드 + 정답 청크 매핑
    qa = [json.loads(l) for l in open(QA)]
    mapped, unmapped = [], 0
    for q in qa:
        gold = set()
        for src in q.get("출처", []):
            quote = ns(src.get("quote", ""))
            if len(quote) < 15:
                continue
            hit = [i for i, t in enumerate(ns_chunks) if quote in t]
            if not hit:  # 파서 표기 차이 대비: 20자 윈도 3개로 재시도
                wins = [quote[:20], quote[len(quote)//2:len(quote)//2+20], quote[-20:]]
                score = Counter()
                for w in wins:
                    for i, t in enumerate(ns_chunks):
                        if w in t:
                            score[i] += 1
                hit = [i for i, c in score.items() if c >= 2]
            gold |= set(hit)
        if gold:
            mapped.append({"q": q["질문"], "gold": gold})
        else:
            unmapped += 1
    print(f"정답 매핑: {len(mapped)}문항 사용, {unmapped}문항 매핑 실패(제외)")

    # ── A vs B
    KS = [1, 3, 5, 10]
    hits = {"A": Counter(), "B": Counter()}
    all_idx = list(range(len(chunks)))
    routed_sizes = []
    for item in mapped:
        q, gold = item["q"], item["gold"]
        # A: 전체 BM25
        ra = bm.rank(q, all_idx)
        # B: frontmatter 라우팅 → 후보 축소 → BM25
        top_riders = [rider_names[i] for i in rider_bm.rank(q, range(len(rider_names)))[:TOP_R]]
        allowed = ALWAYS | set(top_riders)
        cand = [i for i in all_idx if chunks[i]["unit"] in allowed]
        routed_sizes.append(len(cand))
        rb = bm.rank(q, cand)
        for k in KS:
            if gold & set(ra[:k]):
                hits["A"][k] += 1
            if gold & set(rb[:k]):
                hits["B"][k] += 1

    n = len(mapped)
    print(f"\n라우팅 후 평균 후보 청크: {sum(routed_sizes)//n} / {len(chunks)}")
    print(f"\n{'':8s}" + "".join(f"recall@{k:<4d}" for k in KS))
    for m in ("A", "B"):
        label = "A 전체검색" if m == "A" else "B 라우팅  "
        print(f"{label}  " + "".join(f"{hits[m][k]/n:8.3f}  " for k in KS))
    print(f"\nΔ(B−A)    " + "".join(f"{(hits['B'][k]-hits['A'][k])/n:+8.3f}  " for k in KS))

    out = {"n": n, "unmapped": unmapped,
           "recall": {m: {k: hits[m][k] / n for k in KS} for m in ("A", "B")}}
    (BASE / "frontmatter" / "out" / "gate2_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
