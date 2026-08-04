"""
5단계 스모크: shinhan-tos 데이터셋 포맷 검증 + 자체 BM25 베이스라인.

외부 인프라(임베딩 서버·API 키) 없이 돌 수 있는 최소 검증:
1. run_experiment.py의 로더(load_corpus/load_queries/load_supporting_spans)로 로드 확인
2. 순수 파이썬 BM25(문자 bigram 토큰)로 Recall@k 산출
   - within-doc: 정답 문서의 청크만 후보 (실험2 조건 — 문서가 이미 선택된 상태)
   - full-corpus: 전체 4천여 청크 후보 (참고 — 기존 평면 검색 방식)
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "data" / "raw" / "shinhan-tos"


def tokens(s: str):
    s = "".join(ch if not ch.isspace() else " " for ch in s.lower())
    out = []
    for w in s.split():
        if len(w) == 1:
            out.append(w)
        else:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        self.df = Counter()
        for c in self.tf:
            self.df.update(c.keys())
        self.N = len(docs)

    def score(self, q, i):
        s = 0.0
        for t in set(tokens(q)):
            if t not in self.tf[i]:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            f = self.tf[i][t]
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s

    def rank(self, q, cand_idx, topk=10):
        scored = sorted(((self.score(q, i), i) for i in cand_idx), reverse=True)
        return [i for _, i in scored[:topk]]


def main():
    # 1) 하네스 로더로 포맷 검증
    from run_experiment import load_corpus, load_queries, load_supporting_spans
    corpus = load_corpus("shinhan-tos")
    queries, qrels = load_queries("shinhan-tos")
    spans = load_supporting_spans("shinhan-tos")
    print(f"로더 OK: corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)} spans={len(spans)}")
    assert len(queries) == len(qrels) == len(spans), "질의·정답·span 수 불일치"

    # 2) BM25 베이스라인
    idx_of = {c["_id"]: i for i, c in enumerate(corpus)}
    by_doc = defaultdict(list)
    for i, c in enumerate(corpus):
        by_doc[c["doc"]].append(i)
    gold = {str(r["query-id"]): r["corpus-id"] for r in qrels}
    qtext = {q["_id"]: q["text"] for q in queries}

    bm25 = BM25([c["text"] for c in corpus])
    hits = {"within@1": 0, "within@5": 0, "full@1": 0, "full@5": 0, "full@10": 0}
    n = 0
    for qid, cid in gold.items():
        if cid not in idx_of:
            continue
        gi = idx_of[cid]
        doc = corpus[gi]["doc"]
        n += 1
        r = bm25.rank(qtext[qid], by_doc[doc], topk=5)
        hits["within@1"] += gi == r[0]
        hits["within@5"] += gi in r
        r = bm25.rank(qtext[qid], range(len(corpus)), topk=10)
        hits["full@1"] += gi == r[0]
        hits["full@5"] += gi in r[:5]
        hits["full@10"] += gi in r
    print(f"\nBM25 베이스라인 (n={n})")
    print(f"  실험2 조건(정답 문서 내 검색): Recall@1={hits['within@1']/n:.3f}  Recall@5={hits['within@5']/n:.3f}")
    print(f"  참고(전체 청크 평면 검색):    Recall@1={hits['full@1']/n:.3f}  Recall@5={hits['full@5']/n:.3f}  Recall@10={hits['full@10']/n:.3f}")

    out = {"n": n, **{k: v / n for k, v in hits.items()}}
    with open(DATA_DIR / "smoke_baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {DATA_DIR / 'smoke_baseline.json'}")


if __name__ == "__main__":
    main()
