# -*- coding: utf-8 -*-
"""
채널별 **원점수**를 조 단위로 저장 — 점수 기반 융합 실험용.

현재 융합은 RRF 라 순위만 쓰고 점수를 버린다. dense 유사도 .92 인 조와 .31 인 조가
인접 순위면 기여가 거의 같아진다. 채널별 점수를 정규화해 합치면 확신도가 살아난다.

한 번 실행해 4채널 점수를 저장하고, 정규화·가중은 fuse_scores.py 에서 오프라인 스윕한다.

채널
  clm     태그 슬롯 구조 매칭      (tag_hybrid provenance.clm_score)
  sparse  태그 어휘 희소 검색      (tag_hybrid provenance.sparse_score)
  bm25    메타 청크 BM25          (bm25s 원점수)
  dense   메타 청크 dense 유사도   (BGE-m3-ko 코사인)

element/chunk -> 조 접기는 **채널별 최댓값**을 쓴다(가장 좋은 근거가 그 조를 대표).

출력: out/scores/scores.jsonl  {qid, jo: {조: [clm, sparse, bm25, dense]}}
"""
import argparse, json, sys, io, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
FS, VS = R / "filesearch", R / "vector_search"
A0823 = VS / "noah" / "0823"
for p in (str(FS), str(VS), str(A0823)):
    sys.path.insert(0, p)

from units import Units                                    # noqa: E402
from tag_hybrid import TagHybridSearch, default_paths      # noqa: E402


def qtag_slots(row):
    return {k: (row or {}).get(k) or [] for k in
            ("contract", "role", "subject", "qualifier", "schema")}


def bm25_scored(hs, query, depth):
    """
    bm25_search 는 점수를 버리므로 retriever 를 직접 호출한다.
    색인이 _ko_stemmer(조사 제거)로 만들어졌으므로 질의도 **같은 stemmer** 로 토큰화해야 한다.
    """
    import bm25s
    from hybrid_search import _ko_stemmer
    retriever, ids = hs._load_bm25()
    qtok = bm25s.tokenize(query, stemmer=_ko_stemmer, stopwords=[],
                          token_pattern=r"[가-힣a-zA-Z0-9]+", show_progress=False)
    k = min(depth, len(ids))
    idx, sc = retriever.retrieve(qtok, k=k, show_progress=False)
    return [(ids[int(idx[0, i])], float(sc[0, i]))
            for i in range(sc.shape[1]) if float(sc[0, i]) > 0]


def main():
    ap = argparse.ArgumentParser()
    el, tg, jo, idx = default_paths()
    ap.add_argument("--gold", default=str(HERE / "out" / "gold_v7_train_strict.jsonl"))
    ap.add_argument("--qtags", default=str(HERE / "out" / "qtags_luna.jsonl"))
    ap.add_argument("--tags", default=str(tg))
    ap.add_argument("--elements", default=str(el))
    ap.add_argument("--jo", default=str(jo))
    ap.add_argument("--index-dir", default=str(idx))
    ap.add_argument("--limit", type=int, default=2000, help="태그 채널 후보 수")
    ap.add_argument("--meta-depth", type=int, default=2000, help="메타 채널 후보 수")
    ap.add_argument("--keep", type=int, default=1200, help="조당 저장 상한(채널 합집합)")
    ap.add_argument("--out", default=str(HERE / "out" / "scores" / "scores.jsonl"))
    ap.add_argument("--n", type=int, default=-1)
    a = ap.parse_args()

    out_p = Path(a.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    engine = TagHybridSearch(Path(a.elements), Path(a.tags), Path(a.jo), Path(a.index_dir))
    units = Units(a.jo)
    qtags = {}
    for l in Path(a.qtags).open(encoding="utf-8"):
        d = json.loads(l); qtags[d["qid"]] = d

    from hybrid_search import ChunkHybridSearch
    hs = ChunkHybridSearch(view="V9")

    G = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    if a.n > 0:
        G = G[:a.n]

    fh = out_p.open("w", encoding="utf-8")
    t0 = time.time()
    for i, g in enumerate(G, 1):
        q = g["q"]
        jo_sc = {}          # 조 -> [clm, sparse, bm25, dense]  (채널별 최댓값)

        def put(j, ch, v):
            if not j:
                return
            cur = jo_sc.setdefault(j, [0.0, 0.0, 0.0, 0.0])
            if v > cur[ch]:
                cur[ch] = v

        # --- 태그 채널: collapse_jo=False 로 항 단위 점수를 전부 받아 조로 접는다
        rows, _slots, _ = engine.search(
            q, explicit=qtag_slots(qtags.get(g["qid"])),
            weights={"clm": 1.0, "sparse": 1.0},
            channel_top_k=a.limit, limit=a.limit, collapse_jo=False)
        for r in rows:
            pv = r.get("provenance") or {}
            put(r.get("jo"), 0, float(pv.get("clm_score") or 0.0))
            put(r.get("jo"), 1, float(pv.get("sparse_score") or 0.0))

        # --- 메타 채널: chunk 점수를 조로 접는다
        for uid, s in bm25_scored(hs, q, a.meta_depth):
            c = hs.chunks.get(uid) or {}
            if c.get("char_start") is None:
                continue
            u = units.jo_of_span(int(c["char_start"]), int(c["char_end"]))
            if u:
                put(u["element_id"], 2, s)
        for uid, s in hs.dense_search(q, a.meta_depth):
            c = hs.chunks.get(uid) or {}
            if c.get("char_start") is None:
                continue
            u = units.jo_of_span(int(c["char_start"]), int(c["char_end"]))
            if u:
                put(u["element_id"], 3, float(s))

        # 저장량 제한: 각 채널 정규화 순위 합으로 상위 keep 개만
        if len(jo_sc) > a.keep:
            mx = [max((v[c] for v in jo_sc.values()), default=1.0) or 1.0 for c in range(4)]
            top = sorted(jo_sc, key=lambda j: -sum(jo_sc[j][c] / mx[c] for c in range(4)))[:a.keep]
            jo_sc = {j: jo_sc[j] for j in top}

        fh.write(json.dumps({"qid": g["qid"], "n_jo": len(jo_sc),
                             "jo": {j: [round(x, 6) for x in v] for j, v in jo_sc.items()}},
                            ensure_ascii=False) + "\n")
        fh.flush()
        if i % 20 == 0 or i == len(G):
            print("  %d/%d  %.0fs  (조 %d)" % (i, len(G), time.time() - t0, len(jo_sc)), flush=True)
    fh.close()
    print("saved ->", out_p)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
