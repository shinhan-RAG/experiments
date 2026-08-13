#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""multi-gold로 새로 구성한 156건(중 132건 gold있음)에 대해 OR방식 R@20 측정.
기존 341건과는 별도로 리포트 -- "근거 발견율" 개념으로 분리."""
import json, re, math
from collections import Counter

OUT = "/home/work/source/embed_exp/dr-dci-lsh/tos-skeleton/hybrid-enrich/out"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def ns(s):
    return re.sub(r"\s+", "", s)


def bigrams(s):
    s = ns(s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs, k1=1.8, b=0.3):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tf, self.dl, self.df = [], [], Counter()
        for d in docs:
            c = Counter(bigrams(d))
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            for t in c:
                self.df[t] += 1
        self.avgdl = sum(self.dl) / max(1, self.N)

    def rank(self, query, id_list, top_n=100):
        q = list(Counter(bigrams(query)))
        scores = []
        for i in range(self.N):
            s = 0.0
            tfi = self.tf[i]
            for t in q:
                f = tfi.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            scores.append((s, id_list[i]))
        scores.sort(key=lambda x: -x[0])
        return [eid for _, eid in scores[:top_n]]


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("dragonkue/BGE-m3-ko", device="cuda:1")

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8)


def cosine_rank(qv, id_list, mat, top_n=100):
    import numpy as np
    sims = [(float(np.dot(qv, mat[i])), id_list[i]) for i in range(len(id_list))]
    sims.sort(key=lambda x: -x[0])
    return [e for _, e in sims[:top_n]]


def rrf_fuse(channels, top_n=100, k=60, threshold=6, boost=0.05):
    rrf = {}
    for rl, w in channels:
        for r, eid in enumerate(rl):
            rrf[eid] = rrf.get(eid, 0.0) + w * (1.0 / (k + r + 1))
            if r < threshold:
                rrf[eid] += w * boost
    ranked = sorted(rrf.items(), key=lambda x: -x[1])
    return [eid for eid, _ in ranked[:top_n]]


def metrics(hit, n):
    return {f"R@{k}": round(hit[k] / n, 4) for k in (1, 5, 10, 20, 50, 100)}


def main():
    elements = load_jsonl(f"{OUT}/elements_semtag.jsonl")
    meta_rows = {r["element_id"]: r for r in load_jsonl(f"{OUT}/element_meta_v2_250212.jsonl")}
    id_list = [e["element_id"] for e in elements]

    tag_reps = [e.get("contract_scope", "") + " ||| " + e["text"].replace("\n", " ") + " ||| " + (meta_rows[e["element_id"]]["embedding_text"] or "") for e in elements]
    bm_tag = BM25(tag_reps, k1=1.8, b=0.3)
    embedder = Embedder()
    meta_texts = [meta_rows[e["element_id"]]["embedding_text"] or e.get("contract_scope", "") for e in elements]
    meta_matrix = embedder.encode(meta_texts)
    print("인덱스 준비 완료", flush=True)

    all_items = load_jsonl(f"{OUT}/gold_mapped_noah_v3_multigold.jsonl")
    # 156건(원래 실패했던 것)만 추림
    target = [it for it in all_items if it["core_retrieval"] == "False"]
    with_gold = [it for it in target if it["gold"]]
    print(f"156건 중 multi-gold 있는 것: {len(with_gold)}건 (없는 24건은 채점불가)", flush=True)

    avg_gold_size = sum(len(it["gold"]) for it in with_gold) / len(with_gold)
    print(f"평균 gold 개수: {avg_gold_size:.1f}개", flush=True)

    hit = {k: 0 for k in (1, 5, 10, 20, 50, 100)}
    for it in with_gold:
        gold = set(it["gold"])
        tag_rank = bm_tag.rank(it["q"], id_list, top_n=100)
        qv = embedder.encode([it["q"]])[0]
        meta_rank = cosine_rank(qv, id_list, meta_matrix, top_n=100)
        fused = rrf_fuse([(tag_rank, 1.0), (meta_rank, 1.3)], threshold=6, boost=0.05)
        rk = next((i + 1 for i, x in enumerate(fused) if x in gold), None)
        for k in (1, 5, 10, 20, 50, 100):
            if rk and rk <= k:
                hit[k] += 1

    n = len(with_gold)
    print(f"\n=== multi-gold OR방식 결과 ({n}건, 평균 gold {avg_gold_size:.1f}개) ===")
    print(json.dumps(metrics(hit, n), ensure_ascii=False))

    # 비교: 341건 R@20도 다시 표시(참고용)
    n_full_156 = 156
    print(f"\n비교용: 156건 전체 분모 기준 (gold없는 24건 포함) R@20 = {hit[20]}/{n_full_156} = {round(hit[20]/n_full_156,4)}")


if __name__ == "__main__":
    main()
