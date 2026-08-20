#!/usr/bin/env python3
"""대표 표본 추출 — train 60 / test 30 (사용자 요청: 전수 대신 난이도 분포를 보존한 표본).

train 60: 채점 가능한 323문항을 난이도(0819 밤샘 7 arm 적중 수: 7=쉬움, 1~6=중간, 0=어려움)
  × n_groups(1 / ≥2) × core 로 층화하고 비례 배분. 표본의 base_c3 R@5/suff@10 이 전체와
  ±3%p 이내인지 대표성 게이트를 출력한다.
test 30: gold 비어있지 않은 138문항을 task_type × core × gold 수(1/≥2)로 층화.
  ※ test 는 채점용 span 파일이 아직 없음(cowork spanmap 대기) — qid 목록만 확정해 둔다.

출력: out/gold_spans_train60.jsonl(채점 파일 서브셋), out/train60_qids.json, out/test30_qids.json
"""
import collections, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[2]
FS = V2 / "filesearch"
SEED = 20260820
ARMS = ["base_c3", "c4_alias", "c4_scope", "c4_ref", "c4_fbp", "c4_fb", "c4_all"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def stratified(items, key, n, rnd):
    """층화 비례 배분(최대 잔여법). 각 층에서 최소 1개 보장 시도."""
    strata = collections.defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    total = len(items)
    quota = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rem = n - sum(alloc.values())
    for k in sorted(quota, key=lambda k: -(quota[k] - alloc[k]))[:rem]:
        alloc[k] += 1
    out = []
    for k, v in strata.items():
        rnd.shuffle(v)
        out.extend(v[: alloc[k]])
    return out, {str(k): (alloc[k], len(strata[k])) for k in strata}


def main():
    rnd = random.Random(SEED)
    (HERE / "out").mkdir(exist_ok=True)

    # ---------- train 60 ----------
    gold = [g for g in load_jsonl(FS / "out/gold_spans_lsh_train.jsonl") if g["groups"] and g.get("status", "ok") == "ok"]
    hits = collections.Counter()
    base_rows = {}
    for arm in ARMS:
        for r in load_jsonl(V2 / f"results/vector_search/0820_night_{arm}_train323.results.jsonl"):
            hits[r["qid"]] += 1 if r["suff@10"] == 1.0 else 0
            if arm == "base_c3":
                base_rows[r["qid"]] = r

    def diff(q):
        h = hits.get(q, 0)
        return "easy" if h == 7 else ("hard" if h == 0 else "mid")

    def key(g):
        return (diff(g["qid"]), "multi" if len(g["groups"]) >= 2 else "single", g["core_retrieval"] == "True")

    sub, alloc = stratified(gold, key, 60, rnd)
    sub_qids = {g["qid"] for g in sub}
    with open(HERE / "out/gold_spans_train60.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            if g["qid"] in sub_qids:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    json.dump(sorted(sub_qids), open(HERE / "out/train60_qids.json", "w"), indent=1)

    # 대표성 게이트: 표본의 0819 base_c3 성적 vs 전체
    def mean(rows, k):
        rows = list(rows)
        return sum(r[k] for r in rows) / len(rows)

    full = base_rows.values()
    samp = [base_rows[q] for q in sub_qids if q in base_rows]
    print(f"[train60] n={len(sub_qids)} 층화={{난이도×근거수×core}}")
    for k in ("R@5", "suff@10"):
        d = mean(samp, k) - mean(full, k)
        flag = "OK" if abs(d) <= 0.03 else "!! 3%p 초과"
        print(f"  base_c3 {k}: full={mean(full,k):.3f} sample={mean(samp,k):.3f} Δ={d:+.3f} [{flag}]")
    dist = collections.Counter(diff(q) for q in sub_qids)
    print(f"  난이도 분포: {dict(dist)} (전체: {dict(collections.Counter(diff(g['qid']) for g in gold))})")
    for k, (a_, t_) in sorted(alloc.items()):
        print(f"    {k}: {a_}/{t_}")

    # ---------- test 30 ----------
    test = [r for r in load_jsonl(V2 / "out/gold_mapped_noah_v3_149_test.jsonl") if r.get("gold")]
    kt = lambda r: (r["task_type"], r["core_retrieval"] == "True", "multi" if len(r["gold"]) >= 2 else "single")
    tsub, talloc = stratified(test, kt, 30, rnd)
    json.dump(sorted(r["qid"] for r in tsub), open(HERE / "out/test30_qids.json", "w"), indent=1)
    print(f"[test30] n={len(tsub)} (모집단 {len(test)}, gold 빈 문항 {149-len(test)} 제외) — 채점은 cowork spanmap 수령 후 가능")
    for k, (a_, t_) in sorted(talloc.items()):
        print(f"    {k}: {a_}/{t_}")


if __name__ == "__main__":
    main()
