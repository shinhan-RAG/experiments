#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""noah v3 497건 전체, multi-gold(OR매칭) 기준으로 TAG/META단독/융합 계산."""
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

    all_items = load_jsonl(f"{OUT}/gold_mapped_noah_v3_multigold.jsonl")  # 497건 전부, multi-gold 포함
    n_total = len(all_items)
    n_with_gold = sum(1 for it in all_items if it["gold"])
    print(f"전체 문항수: {n_total} / gold 있는 문항: {n_with_gold}", flush=True)

    hit_tag = {k: 0 for k in (1, 5, 10, 20, 50, 100)}
    hit_meta = {k: 0 for k in (1, 5, 10, 20, 50, 100)}
    hit_fuse = {k: 0 for k in (1, 5, 10, 20, 50, 100)}

    for it in all_items:
        if not it["gold"]:
            continue
        gold = set(it["gold"])
        tag_rank = bm_tag.rank(it["q"], id_list, top_n=100)
        qv = embedder.encode([it["q"]])[0]
        meta_rank = cosine_rank(qv, id_list, meta_matrix, top_n=100)
        fused = rrf_fuse([(tag_rank, 1.0), (meta_rank, 1.3)], threshold=6, boost=0.05)

        tr = next((i + 1 for i, x in enumerate(tag_rank) if x in gold), None)
        mr = next((i + 1 for i, x in enumerate(meta_rank) if x in gold), None)
        fr = next((i + 1 for i, x in enumerate(fused) if x in gold), None)
        for k in (1, 5, 10, 20, 50, 100):
            if tr and tr <= k: hit_tag[k] += 1
            if mr and mr <= k: hit_meta[k] += 1
            if fr and fr <= k: hit_fuse[k] += 1

    print(f"497건 분모 기준 (multi-gold OR매칭)", flush=True)
    print("TAG단독 :", json.dumps(metrics(hit_tag, n_total), ensure_ascii=False))
    print("META단독:", json.dumps(metrics(hit_meta, n_total), ensure_ascii=False))
    print("TAG+META:", json.dumps(metrics(hit_fuse, n_total), ensure_ascii=False))


if __name__ == "__main__":
    main()
