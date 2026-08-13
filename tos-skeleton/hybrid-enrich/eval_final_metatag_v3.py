#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최종 확정: 점수융합(RRF) - META+TAG 재료, v3(최종) gold, 497건 완전일치 통일."""
import json, re, math
from collections import Counter

OUT = "out"


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


def main():
    elements = load_jsonl(f"{OUT}/elements_semtag.jsonl")
    meta_rows = {r["element_id"]: r for r in load_jsonl(f"{OUT}/element_meta_v2_250212.jsonl")}
    id_list = [e["element_id"] for e in elements]
    tag_reps = [e.get("contract_scope", "") + " ||| " + e["text"].replace("\n", " ") + " ||| " + (meta_rows[e["element_id"]]["embedding_text"] or "") for e in elements]
    bm_tag = BM25(tag_reps, k1=1.8, b=0.3)
    embedder = Embedder()

    # META+TAG 재료 (META 요약 + 원문 결합) -- 결합방법론 재검증에서 1위였던 재료
    el_by_id = {e["element_id"]: e for e in elements}
    meta_tag_texts = [(meta_rows[eid]["embedding_text"] or "") + "\n[원문]\n" + el_by_id[eid]["text"][:2000] for eid in id_list]
    meta_tag_mat = embedder.encode(meta_tag_texts)
    print("META+TAG 인덱스 준비 완료", flush=True)

    ks = (5, 10, 20, 50, 100)
    rows_v3 = load_jsonl(f"{OUT}/gold_mapped_noah_v3_multigold_v3.jsonl")
    n_total = len(rows_v3)
    exact = {k: 0 for k in ks}
    for it in rows_v3:
        if not it.get("gold"):
            continue
        gold = set(it["gold"])
        n_gold = len(gold)
        tag_rank = bm_tag.rank(it["q"], id_list, top_n=100)
        qv = embedder.encode([it["q"]])[0]
        vec_rank = cosine_rank(qv, id_list, meta_tag_mat, top_n=100)
        fused = rrf_fuse([(tag_rank, 1.0), (vec_rank, 1.3)])
        for k in ks:
            found = len(set(fused[:k]) & gold)
            if found == n_gold:
                exact[k] += 1
    print("\n=== 최종: RRF-META+TAG, v3 gold, 497건 완전일치 통일 ===")
    print({f"R@{k}": round(100 * exact[k] / n_total, 1) for k in ks})


if __name__ == "__main__":
    main()
