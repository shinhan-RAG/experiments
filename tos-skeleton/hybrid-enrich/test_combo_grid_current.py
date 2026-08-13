#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""초반 결합방법론 그리드(run_meta_tag_grid_lsh.py)를 현재 TAG/META/gold 기준으로 재실행."""
import json, re, math
from collections import Counter

OUT = "/home/work/source/embed_exp/dr-dci-lsh/tos-skeleton/hybrid-enrich/out"
RRF_K = 60


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def ns(s):
    return re.sub(r"\s+", "", s)


def bigrams(s):
    s = ns(s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


JO_QUERY_RE = re.compile(r"제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?")


def scope_core(s):
    s = re.sub(r"\(무배당[^)]*\)", "", s or "")
    s = re.sub(r"^\(간편\)", "", s)
    return s.strip()


def build_scope_index(elements):
    cores = {}
    for e in elements:
        s = e["contract_scope"]
        if s and s != "주계약":
            c = scope_core(s)
            if len(c) >= 4:
                cores.setdefault(c, s)
    return sorted(cores.items(), key=lambda x: -len(x[0]))


def extract_entities(query, scope_cores):
    q_scopes = [full for core, full in scope_cores if core in query]
    q_jo = list(dict.fromkeys(m.replace(" ", "") for m in JO_QUERY_RE.findall(query.replace(" ", ""))))
    return q_scopes[:3], q_jo[:3]


def structured_candidates(query, elements, scope_cores):
    q_scopes, q_jo = extract_entities(query, scope_cores)
    if not q_scopes and not q_jo:
        return None
    cand = []
    for e in elements:
        hit = False
        if q_scopes and e["contract_scope"] in q_scopes:
            hit = True
        if q_jo and e["jo"]:
            jo_num = e["jo"].split("(")[0].replace(" ", "")
            if any(j.replace(" ", "") == jo_num for j in q_jo):
                hit = True
        if hit:
            cand.append(e)
    return cand if cand else None


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

    def scores(self, query):
        q = list(Counter(bigrams(query)))
        out = []
        for i in range(self.N):
            s = 0.0
            tfi = self.tf[i]
            for t in q:
                f = tfi.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                B = 1 - self.b + self.b * (self.dl[i] / self.avgdl)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * B)
            out.append(s)
        return out

    def rank(self, query, id_list, top_n=50, allowed_ids=None):
        scores = self.scores(query)
        order = sorted(range(self.N), key=lambda i: -scores[i])
        ids = [id_list[i] for i in order]
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            ids = [x for x in ids if x in allowed]
        return ids[:top_n]


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("dragonkue/BGE-m3-ko", device="cuda:1")

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8)


def cosine_rank(q_vec, id_list, matrix, top_n=50, allowed_idx=None):
    import numpy as np
    idxs = allowed_idx if allowed_idx is not None else range(len(id_list))
    sims = [(float(np.dot(q_vec, matrix[i])), id_list[i]) for i in idxs]
    sims.sort(key=lambda x: -x[0])
    return [eid for _, eid in sims[:top_n]]


def arch_concat(q_vec, id_list, matrix):
    return cosine_rank(q_vec, id_list, matrix, top_n=100)


def arch_rrf(bm_tag, query, id_list, q_vec, matrix):
    tag_rank = bm_tag.rank(query, id_list, top_n=100)
    vec_rank = cosine_rank(q_vec, id_list, matrix, top_n=100)
    rrf = {}
    for r, eid in enumerate(tag_rank):
        rrf[eid] = rrf.get(eid, 0.0) + 1.0 * (1.0 / (RRF_K + r + 1))
        if r < 6:
            rrf[eid] += 1.0 * 0.05
    for r, eid in enumerate(vec_rank):
        rrf[eid] = rrf.get(eid, 0.0) + 1.3 * (1.0 / (RRF_K + r + 1))
        if r < 6:
            rrf[eid] += 1.3 * 0.05
    ranked = sorted(rrf.items(), key=lambda x: -x[1])
    return [eid for eid, _ in ranked[:100]]


def arch_cascade(query, elements, scope_cores, id_list, id_to_idx, q_vec, matrix):
    cand = structured_candidates(query, elements, scope_cores)
    allowed_idx = [id_to_idx[e["element_id"]] for e in cand] if cand else None
    return cosine_rank(q_vec, id_list, matrix, top_n=100, allowed_idx=allowed_idx)


def arch_routing(bm_tag, query, elements, scope_cores, id_list, q_vec, matrix):
    cand = structured_candidates(query, elements, scope_cores)
    if cand:
        allowed_ids = {e["element_id"] for e in cand}
        return bm_tag.rank(query, id_list, top_n=100, allowed_ids=allowed_ids)
    return cosine_rank(q_vec, id_list, matrix, top_n=100)


def eval_method(name, rank_fn, items):
    hit = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    n = len(items)
    for it in items:
        ranked = rank_fn(it["q"])
        gold = set(it["gold"])
        rk = next((k + 1 for k, x in enumerate(ranked) if x in gold), None)
        for k in (5, 10, 20, 50, 100):
            if rk and rk <= k:
                hit[k] += 1
    return {"method": name, "n": n, **{f"R@{k}": round(hit[k] / n, 4) for k in (5, 10, 20, 50, 100)}}


def main():
    elements = load_jsonl(f"{OUT}/elements_semtag.jsonl")
    gold_items = [it for it in load_jsonl(f"{OUT}/gold_mapped_noah_v3.jsonl") if it.get("gold")]

    id_list = [e["element_id"] for e in elements]
    id_to_idx = {eid: i for i, eid in enumerate(id_list)}
    scope_cores = build_scope_index(elements)
    el_by_id = {e["element_id"]: e for e in elements}

    print(f"elements={len(elements)} gold_items={len(gold_items)}", flush=True)

    meta_rows = {r["element_id"]: r for r in load_jsonl(f"{OUT}/element_meta_v2_250212.jsonl")}

    tag_reps = [el_by_id[eid].get("contract_scope", "") + " ||| " + el_by_id[eid]["text"].replace("\n", " ") + " ||| " + (meta_rows[eid]["embedding_text"] or "") for eid in id_list]
    bm_tag = BM25(tag_reps, k1=1.8, b=0.3)
    print("TAG BM25 index built", flush=True)

    embedder = Embedder()

    raw_texts = [el_by_id[eid]["text"][:2000] for eid in id_list]
    meta_only_texts = [meta_rows[eid]["embedding_text"] or el_by_id[eid].get("contract_scope", "") for eid in id_list]
    meta_concat_texts = [(meta_rows[eid]["embedding_text"] or "") + "\n[원문]\n" + el_by_id[eid]["text"][:2000] for eid in id_list]
    concat_all_texts = [
        el_by_id[eid].get("contract_scope", "") + "\n" + (meta_rows[eid]["embedding_text"] or "") + "\n[원문]\n" + el_by_id[eid]["text"][:2000]
        for eid in id_list
    ]

    materials = {"원문만": raw_texts, "META만": meta_only_texts, "META+원문": meta_concat_texts}
    mats = {}
    for name, texts in materials.items():
        mats[name] = embedder.encode(texts)
        print(f"임베딩 완료: {name}", flush=True)
    concat_mat = embedder.encode(concat_all_texts)
    print("결합임베딩 완료", flush=True)

    q_cache = {}

    def qv(query):
        if query not in q_cache:
            q_cache[query] = embedder.encode([query])[0]
        return q_cache[query]

    results = []
    results.append(eval_method("결합임베딩(concat) - TAG+META+원문",
        lambda q: arch_concat(qv(q), id_list, concat_mat), gold_items))

    for mat_name, mat in mats.items():
        results.append(eval_method(f"점수융합(RRF) - {mat_name}",
            lambda q, mat=mat: arch_rrf(bm_tag, q, id_list, qv(q), mat), gold_items))
        results.append(eval_method(f"계층재순위(cascade) - {mat_name}",
            lambda q, mat=mat: arch_cascade(q, elements, scope_cores, id_list, id_to_idx, qv(q), mat), gold_items))
        results.append(eval_method(f"질의유형라우팅 - {mat_name}",
            lambda q, mat=mat: arch_routing(bm_tag, q, elements, scope_cores, id_list, qv(q), mat), gold_items))

    results.sort(key=lambda r: -r["R@20"])
    print()
    for r in results:
        print(f"{r['method']:40s} n={r['n']:4d} R@5={r['R@5']:.4f} R@10={r['R@10']:.4f} R@20={r['R@20']:.4f} R@50={r['R@50']:.4f} R@100={r['R@100']:.4f}")

    json.dump(results, open(f"{OUT}/combo_grid_current.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("저장: out/combo_grid_current.json")


if __name__ == "__main__":
    main()
