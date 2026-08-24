# -*- coding: utf-8 -*-
"""
점수 기반 융합 오프라인 스윕 — run_scores.py 가 저장한 4채널 원점수를 정규화·가중해 합친다.

RRF 는 순위만 쓰고 점수를 버린다(dense .92 와 .31 이 인접 순위면 기여가 같아짐).
여기서는 채널별로 **질의 안에서** 정규화한 뒤 가중합해 확신도를 살린다.

정규화
  minmax  (x - min) / (max - min)          채널 내 상대 위치
  zscore  (x - mean) / std                  분포 대비 이례성
  rank    1 / (60 + 순위)                    RRF 와 동일(비교 기준선)
  maxnorm x / max                            0 을 0 으로 유지 (희소 채널에 안전)

사용:
  python fuse_scores.py                      # 기본 스윕
  python fuse_scores.py --top 20             # 상위 조합 20개
"""
import argparse, json, sys, io, itertools, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
CH = ("clm", "sparse", "bm25", "dense")


def load(gold_p, scores_p, jo_p):
    jo = {}
    for l in open(jo_p, encoding="utf-8"):
        d = json.loads(l)
        jo[d["element_id"]] = (int(d["char_start"]), int(d["char_end"]))
    gold = {}
    for l in open(gold_p, encoding="utf-8"):
        d = json.loads(l)
        g = [x for x in d["groups"] if x["grade"] == "required"]
        if g:
            gold[d["qid"]] = g
    S = {}
    for l in open(scores_p, encoding="utf-8"):
        d = json.loads(l)
        S[d["qid"]] = d["jo"]
    return jo, gold, S


def normalize(vals, mode):
    """vals: list[float] (한 질의·한 채널). 반환 동형 list."""
    if mode == "maxnorm":
        m = max(vals) or 1.0
        return [v / m for v in vals]
    if mode == "minmax":
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return [(v - lo) / rng for v in vals]
    if mode == "zscore":
        n = len(vals)
        mu = sum(vals) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / n) or 1.0
        return [(v - mu) / sd for v in vals]
    if mode == "rank":
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        out = [0.0] * len(vals)
        for r, i in enumerate(order, 1):
            out[i] = 0.0 if vals[i] <= 0 else 1.0 / (60 + r)
        return out
    raise ValueError(mode)


def evaluate(jo, gold, S, mode, w, k=5):
    def ov(u, g):
        return any(u[0] < m["c1"] and u[1] > m["c0"] for m in g["members"])
    tot = 0.0
    n = 0
    for qid, req in gold.items():
        sc = S.get(qid)
        if not sc:
            continue
        js = list(sc.keys())
        cols = [[sc[j][c] for j in js] for c in range(4)]
        norm = [normalize(col, mode) for col in cols]
        fused = [sum(w[c] * norm[c][i] for c in range(4)) for i in range(len(js))]
        top = sorted(range(len(js)), key=lambda i: -fused[i])[:k]
        units = [jo[js[i]] for i in top if js[i] in jo]
        hit = sum(1 for g in req if any(ov(u, g) for u in units))
        tot += hit / len(req)
        n += 1
    return (tot / n if n else 0.0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE / "out" / "gold_v7_train_strict.jsonl"))
    ap.add_argument("--scores", default=str(HERE / "out" / "scores" / "scores.jsonl"))
    ap.add_argument("--jo", default=str(R / "filesearch" / "out" / "elements_u2jo.jsonl"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    jo, gold, S = load(a.gold, a.scores, a.jo)
    print("gold %d | scores %d | 조 인덱스 %d" % (len(gold), len(S), len(jo)), flush=True)

    grid = [0, 1, 2]
    combos = [w for w in itertools.product(grid, repeat=4) if sum(w) > 0]
    res = []
    for mode in ("rank", "maxnorm", "minmax", "zscore"):
        for w in combos:
            r5, n = evaluate(jo, gold, S, mode, w, a.k)
            res.append((r5, mode, w, n))
    res.sort(reverse=True)

    base = [x for x in res if x[1] == "rank" and x[2] == (1, 1, 1, 1)]
    print()
    print("기준선 rank(=RRF) 1:1:1:1  R@%d %.4f" % (a.k, base[0][0] if base else float("nan")))
    print()
    print("%-8s %-16s %8s" % ("정규화", "가중(clm,sp,bm,dn)", "R@%d" % a.k))
    for r5, mode, w, n in res[:a.top]:
        print("%-8s %-16s %8.4f" % (mode, str(w), r5))
    print()
    best = res[0]
    print("최고: %s %s  R@%d %.4f  (n=%d)" % (best[1], best[2], a.k, best[0], best[3]))
    print("기준선 대비 %+.4f (%+.1fpp)" % (best[0] - base[0][0], 100 * (best[0] - base[0][0])))

    by_mode = {}
    for r5, mode, w, n in res:
        if mode not in by_mode or r5 > by_mode[mode][0]:
            by_mode[mode] = (r5, w)
    print()
    print("정규화별 최고:")
    for m in ("rank", "maxnorm", "minmax", "zscore"):
        if m in by_mode:
            print("  %-8s %8.4f  %s" % (m, by_mode[m][0], by_mode[m][1]))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
