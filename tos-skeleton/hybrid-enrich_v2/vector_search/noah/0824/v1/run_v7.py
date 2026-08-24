# -*- coding: utf-8 -*-
"""
gold v7 전용 검색 재실행 — 메타(V9) + 태그 양방향, RRF 융합, 깊이 무제한.

0823 `eval_det.py` 의 host_retrieve 경로를 그대로 재현하되:
  - gold 는 v7 (required 그룹만 채점)
  - qtags 는 교체 가능 (기본 out/qtags_luna.jsonl, gpt-5.6-luna 생성)
  - **깊이 무제한** — 아카이브 예측이 top-40 에서 잘려 있어 재랭킹 여유를 못 재던 문제를 없앤다
  - 채널별(tag / meta) + 융합(union) R@k 를 한 번에 산출

사용:
  python run_v7.py                       # train, strict+lenient
  python run_v7.py --meta-top-k 400 --limit 400
"""
import argparse, json, sys, io, collections, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
FS, VS = R / "filesearch", R / "vector_search"
A0823 = VS / "noah" / "0823"
for p in (str(FS), str(VS), str(A0823)):
    sys.path.insert(0, p)

from units import Units                                    # noqa: E402
from tag_hybrid import TagHybridSearch, default_paths      # noqa: E402
sys.path.insert(0, str(HERE))
from scoring_v6 import required_groups                     # noqa: E402

KS = (1, 3, 5, 10, 20, 40, 80, 200, 400, 1000)


def qtag_slots(row):
    return {k: (row or {}).get(k) or [] for k in
            ("contract", "role", "subject", "qualifier", "schema")}


def jo_rank(rows, jid):
    out, seen = [], set()
    for row in rows:
        jo = row.get("jo")
        if jo and jo not in seen and jo in jid:
            seen.add(jo); out.append(jid[jo])
    return out


def union_rank(tag_units, meta_units, jid, top_k, rrf_k=60, w_tag=1.0, w_meta=1.0):
    """
    RRF 융합. **입력은 조 단위로 이미 접힌(dedup 된) 순위 목록**이다.

    element 순위를 그대로 넣으면 한 조에 여러 element 가 매핑될 때 그 조의 순위가 희석된다.
    (2026-08-24 실측: 조 단위로 접고 융합하면 R@5 .3738 -> .4068)
    """
    sc = {}
    for rank, u in enumerate(tag_units, 1):
        j = u["element_id"]
        if j in jid:
            sc[j] = sc.get(j, 0) + w_tag / (rrf_k + rank)
    for rank, u in enumerate(meta_units, 1):
        j = u["element_id"]
        if j in jid:
            sc[j] = sc.get(j, 0) + w_meta / (rrf_k + rank)
    order = sorted(sc, key=lambda j: (-sc[j], j))
    return [jid[j] for j in order[:top_k]]


def build_jo_index(jo_path):
    """조 구간 목록 (정렬)."""
    iv = []
    for line in open(jo_path, encoding="utf-8"):
        d = json.loads(line)
        iv.append((int(d["char_start"]), int(d["char_end"]), d["element_id"]))
    iv.sort()
    return iv


def jos_overlapping(iv, a, b, cap):
    """
    chunk span [a,b) 과 겹치는 **모든** 조를 겹침 길이 내림차순으로.

    units.py 의 jo_of_span 은 겹침 최대인 조 하나로만 접는데, 그러면 5,797 chunk 가
    도달하는 조가 7,599 중 2,390(31.5%) 로 줄어 메타 채널의 기여 상한이 묶인다.
    """
    out = []
    lo, hi = 0, len(iv)
    while lo < hi:
        mid = (lo + hi) // 2
        if iv[mid][1] <= a:
            lo = mid + 1
        else:
            hi = mid
    for s, e, j in iv[lo:]:
        if s >= b:
            break
        ov = min(e, b) - max(s, a)
        if ov > 0:
            out.append((ov, j))
    out.sort(key=lambda x: -x[0])
    return [j for _, j in out[:cap]]


def meta_search_deep(hs, query, depth, rrf_k=60):
    """
    메타(V9) 채널을 **깊이 지정**으로 실행.

    hybrid_search.py 의 hybrid_search() 는 BM25/dense 를 CHANNEL_TOPK=50 으로 하드코딩해
    top_k 인자가 후보 생성에 영향을 주지 못한다(조 65개까지밖에 안 나옴). 여기서는
    bm25_search / dense_search 를 직접 depth 로 호출하고 같은 RRF(k=60)로 합친다.

    반환: (ordered_chunks, bm25_ids, dense_ids)
    """
    bm = hs.bm25_search(query, depth)
    dn = [u for u, _ in hs.dense_search(query, depth)]
    fused = {}
    for rank, uid in enumerate(bm, 1):
        fused[uid] = fused.get(uid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, uid in enumerate(dn, 1):
        fused[uid] = fused.get(uid, 0.0) + 1.0 / (rrf_k + rank)
    order = sorted(fused, key=lambda u: (-fused[u], u))
    out = []
    for uid in order:
        c = hs.chunks.get(uid) or {}
        if c.get("char_start") is not None:
            out.append({"id": uid, "char_start": int(c["char_start"]),
                        "char_end": int(c["char_end"])})
    return out, bm, dn


def overlaps(u, g):
    a, b = u["char_start"], u["char_end"]
    return any(a < m["c1"] and b > m["c0"] for m in g["members"])


def rk(ranked, groups):
    """gi -> 최초 1-based 순위"""
    h = {}
    for i, u in enumerate(ranked, 1):
        for gi, g in enumerate(groups):
            if gi not in h and overlaps(u, g):
                h[gi] = i
    return h


def metrics(h, n, ks=KS):
    out = {}
    for k in ks:
        c = sum(1 for gi in h if h[gi] <= k)
        out["R@%d" % k] = c / n
        out["suff@%d" % k] = 1.0 if len(h) == n and max(h.values(), default=10 ** 9) <= k else 0.0
    out["R@inf"] = len(h) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    el, tg, jo, idx = default_paths()
    ap.add_argument("--gold-strict", default=str(HERE / "out" / "gold_v7_train_strict.jsonl"))
    ap.add_argument("--gold-lenient", default=str(HERE / "out" / "gold_v7_train_lenient.jsonl"))
    ap.add_argument("--qtags", default=str(HERE / "out" / "qtags_luna.jsonl"))
    ap.add_argument("--elements", default=str(el))
    ap.add_argument("--tags", default=str(tg))
    ap.add_argument("--jo", default=str(jo))
    ap.add_argument("--index-dir", default=str(idx))
    ap.add_argument("--limit", type=int, default=2000, help="태그 채널 후보 수")
    ap.add_argument("--meta-top-k", type=int, default=2000, help="메타 V9 채널 후보 수")
    ap.add_argument("--union-top-k", type=int, default=100000, help="융합 깊이(기본 무제한)")
    ap.add_argument("--w-clm", type=float, default=1.0, help="태그 채널 clm 가중")
    ap.add_argument("--w-sparse", type=float, default=2.0, help="태그 채널 sparse 가중")
    ap.add_argument("--meta-depth", type=int, default=2000,
                    help="메타 BM25/dense 각 채널 후보 깊이 (원본 CHANNEL_TOPK=50 우회)")
    ap.add_argument("--meta-map", choices=("1to1", "1toN"), default="1to1",
                    help="chunk->조 매핑. 1to1=겹침최대 1개(기존), 1toN=겹치는 전부")
    ap.add_argument("--meta-jo-cap", type=int, default=8, help="1toN 일 때 chunk 당 조 상한")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--w-tag", type=float, default=1.0)
    ap.add_argument("--w-meta", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(HERE / "out" / "run_v7"))
    ap.add_argument("--n", type=int, default=-1)
    a = ap.parse_args()

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    engine = TagHybridSearch(Path(a.elements), Path(a.tags), Path(a.jo), Path(a.index_dir))
    units = Units(a.jo)
    qtags = {}
    if Path(a.qtags).exists():
        for l in Path(a.qtags).open(encoding="utf-8"):
            d = json.loads(l); qtags[d["qid"]] = d
    print("qtags %d개 로드 (%s)" % (len(qtags), Path(a.qtags).name), flush=True)

    JOIV = build_jo_index(a.jo)
    from hybrid_search import ChunkHybridSearch
    hs = ChunkHybridSearch(view="V9")

    G = {"strict": [json.loads(l) for l in open(a.gold_strict, encoding="utf-8")],
         "lenient": [json.loads(l) for l in open(a.gold_lenient, encoding="utf-8")]}
    if a.n > 0:
        G = {k: v[:a.n] for k, v in G.items()}
    len_by_qid = {d["qid"]: d for d in G["lenient"]}

    agg = {m: {ch: collections.defaultdict(list) for ch in ("tag", "meta", "union")}
           for m in ("strict", "lenient")}
    fh = (out_dir / "pred_v7.jsonl").open("w", encoding="utf-8")
    t0, miss_q = time.time(), 0

    for i, g in enumerate(G["strict"], 1):
        slots = qtag_slots(qtags.get(g["qid"]))
        if g["qid"] not in qtags:
            miss_q += 1
        rows, used_slots, _ = engine.search(
            g["q"], explicit=slots, weights={"clm": a.w_clm, "sparse": a.w_sparse},
            channel_top_k=a.limit, limit=a.limit, collapse_jo=True)
        tag_ranked = jo_rank(rows, engine.jid)

        res, bm_ids, dn_ids = meta_search_deep(hs, g["q"], a.meta_depth)
        meta_items = []
        for r in res:
            if r.get("char_start") is None:
                continue
            cs, ce = int(r["char_start"]), int(r["char_end"])
            if a.meta_map == "1toN":
                for j in jos_overlapping(JOIV, cs, ce, a.meta_jo_cap):
                    meta_items.append({"id": r["id"], "jo": j})
            else:
                u = units.jo_of_span(cs, ce)
                meta_items.append({"id": r["id"], "jo": u["element_id"] if u else ""})
        meta_ranked = jo_rank(meta_items, engine.jid)
        uni = union_rank(tag_ranked, meta_ranked, engine.jid, a.union_top_k,
                         rrf_k=a.rrf_k, w_tag=a.w_tag, w_meta=a.w_meta)

        rec = {"qid": g["qid"], "q": g["q"], "slots": used_slots,
               "n_tag": len(tag_ranked), "n_meta": len(meta_ranked), "n_union": len(uni),
               "ranked_jo": [x["element_id"] for x in uni[:400]],
               "tag_jo": [x["element_id"] for x in tag_ranked[:400]],
               "meta_jo": [x["element_id"] for x in meta_ranked[:400]],
               "n_chunk": len(res),
               "meta_bm25": bm_ids[:600], "meta_dense": dn_ids[:600]}
        for mode, gold_rec in (("strict", g), ("lenient", len_by_qid[g["qid"]])):
            req = required_groups(gold_rec)
            for ch, ranked in (("tag", tag_ranked), ("meta", meta_ranked), ("union", uni)):
                mm = metrics(rk(ranked, req), len(req))
                for k, v in mm.items():
                    agg[mode][ch][k].append(v)
                if mode == "strict":
                    rec["%s_R@5" % ch] = round(mm["R@5"], 4)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(G["strict"]):
            print("  %d/%d  %.0fs" % (i, len(G["strict"]), time.time() - t0), flush=True)
    fh.close()

    report = {"n": len(G["strict"]), "qtags": Path(a.qtags).name,
              "qtags_missing": miss_q, "tags": Path(a.tags).name,
              "limit": a.limit, "meta_top_k": a.meta_top_k,
              "union_top_k": a.union_top_k,
              "fusion": "jo-collapsed RRF", "rrf_k": a.rrf_k,
              "meta_map": a.meta_map, "meta_jo_cap": a.meta_jo_cap,
              "meta_depth": a.meta_depth, "w_clm": a.w_clm, "w_sparse": a.w_sparse,
              "w_tag": a.w_tag, "w_meta": a.w_meta, "results": {}}
    for mode in ("strict", "lenient"):
        report["results"][mode] = {
            ch: {k: round(sum(v) / len(v), 4) for k, v in agg[mode][ch].items()}
            for ch in ("tag", "meta", "union")}
    (out_dir / "summary_v7.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    for mode in ("strict", "lenient"):
        print("== %s (n=%d)" % (mode, report["n"]))
        for ch in ("tag", "meta", "union"):
            r = report["results"][mode][ch]
            print("   %-6s R@5 %.4f  R@10 %.4f  R@40 %.4f  R@200 %.4f  R@inf %.4f"
                  % (ch, r["R@5"], r["R@10"], r["R@40"], r["R@200"], r["R@inf"]))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
