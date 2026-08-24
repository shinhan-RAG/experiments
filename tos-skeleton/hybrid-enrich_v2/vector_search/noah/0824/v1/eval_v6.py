# -*- coding: utf-8 -*-
"""
gold v6 평가 러너.

예측 파일(ranked_jo / ranked_element / u2_top / jo_top)을 gold v6 로 채점한다.
ID 는 반드시 검색기가 실제로 색인한 element 파일로 푼다 — 유니버스를 잘못 잡으면
R@5 가 .587 -> .178 로 무너진다(0824 실측).

사용:
  python eval_v6.py --pred <예측.jsonl> [--gold out/gold_v6_train_strict.jsonl]
                    [--field ranked_jo] [--depth 40]
  python eval_v6.py --sweep <예측.jsonl>       # k 스윕 + 층위 병기
"""
import argparse, json, io, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_v6 import score, aggregate, required_groups, PRIMARY_K  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
U2JO = R / "filesearch" / "out" / "elements_u2jo.jsonl"
U2 = R / "filesearch" / "out" / "elements_u2.jsonl"


def load_units(path):
    m = {}
    for line in Path(path).open(encoding="utf-8"):
        d = json.loads(line)
        m[d["element_id"]] = {"char_start": int(d["char_start"]),
                              "char_end": int(d["char_end"])}
    return m


def resolve(ids, jo, el):
    out = []
    for x in ids:
        u = jo.get(x) or el.get(x)
        if u:
            out.append(u)
    return out


def pick_field(rec, want):
    if want and want in rec:
        return want
    for f in ("ranked_jo", "jo_top", "ranked_element", "u2_top", "submitted"):
        if f in rec and isinstance(rec[f], list):
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default=str(HERE / "out" / "gold_v6_train_strict.jsonl"))
    ap.add_argument("--field", default=None)
    ap.add_argument("--depth", type=int, default=0, help="0 = 전체")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    jo, el = load_units(U2JO), load_units(U2)
    gold = {}
    for line in Path(a.gold).open(encoding="utf-8"):
        d = json.loads(line)
        if required_groups(d):
            gold[d["qid"]] = d

    preds = {}
    field_used = collections.Counter()
    for line in Path(a.pred).open(encoding="utf-8"):
        d = json.loads(line)
        f = pick_field(d, a.field)
        if not f:
            continue
        field_used[f] += 1
        preds[d["qid"]] = d[f]

    shared = [q for q in gold if q in preds]
    ks = (1, 5, 10, 20, 40, 100, 200, 400) if a.sweep else (1, 5, 10, 20, 40)

    per, strat = [], {"clean": [], "partial": []}
    for q in shared:
        ids = preds[q]
        if a.depth:
            ids = ids[:a.depth]
        units = resolve(ids, jo, el)
        s = score(units, gold[q], ks=ks)
        if not s:
            continue
        per.append(s)
        strat["partial" if gold[q].get("partial") else "clean"].append(s)

    agg = aggregate(per)
    out = {
        "gold": Path(a.gold).name,
        "pred": Path(a.pred).name,
        "field": field_used.most_common(1)[0][0] if field_used else None,
        "depth": a.depth or "full",
        "gold_rows": len(gold),
        "pred_rows": len(preds),
        "matched_n": len(per),
        "unmatched_gold": sorted(set(gold) - set(preds))[:10],
        "primary_metric": "sufficient@%d" % PRIMARY_K,
        "overall": agg,
        "strata": {k: aggregate(v) for k, v in strat.items() if v},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
