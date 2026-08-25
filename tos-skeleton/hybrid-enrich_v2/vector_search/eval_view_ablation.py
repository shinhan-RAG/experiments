#!/usr/bin/env python3
"""메타 채널 view ablation — V9(LLM 소개문) vs RAW(원문만) vs RULE(규칙 꼬리표 소개문).

메타 채널 단독의 gold JO 노출을 213문항에서 결정론으로 잰다 (에이전트 0회, LLM 0회).
채점: 각 질문의 gold member 와 char span 이 겹치는 JO 를 정답 JO 집합으로 보고,
hybrid top-K 청크의 span → JO 사상 후 노출@K 및 첫 정답 순위를 기록한다.
"""
import argparse
import bisect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hybrid_search import ChunkHybridSearch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(HERE.parent / "filesearch/out/gold_train_scoped_u3_reviewed_overlay_v23.jsonl"))
    ap.add_argument("--jo", default=str(HERE.parent / "filesearch/out/elements_u3jo.jsonl"))
    ap.add_argument("--views", nargs="+", default=["V9", "RAW", "RULE"])
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", default=str(HERE / "out/view_ablation_goldv23.json"))
    a = ap.parse_args()

    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    jos = [json.loads(l) for l in open(a.jo, encoding="utf-8")]
    jo_starts = [u["char_start"] for u in jos]

    def jo_of_span(c0, c1):
        i = bisect.bisect_right(jo_starts, c0) - 1
        best, ov = None, 0
        for u in jos[max(0, i - 2):min(len(jos), i + 6)]:
            o = min(c1, u["char_end"]) - max(c0, u["char_start"])
            if o > ov:
                best, ov = u, o
        return best

    gold_jos = {}
    for g in gold:
        s = set()
        for gr in g["groups"]:
            for m in gr["members"]:
                for u in jos:
                    if u["char_start"] < m["c1"] and u["char_end"] > m["c0"]:
                        s.add(u["element_id"])
        gold_jos[g["qid"]] = s

    report = {}
    for view in a.views:
        hs = ChunkHybridSearch(view=view)
        rows = []
        for g in gold:
            res = hs.search(g["q"], strategy="hybrid", top_k=a.top_k)
            seen = []
            for r in res:
                u = jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
                jid = u["element_id"] if u else ""
                if jid and jid not in seen:
                    seen.append(jid)
            gj = gold_jos[g["qid"]]
            ranks = [i + 1 for i, jid in enumerate(seen) if jid in gj]
            rows.append({"qid": g["qid"], "first_rank": ranks[0] if ranks else None})
        n = len(rows)
        summ = {f"exposed@{k}": sum(1 for r in rows
                                    if r["first_rank"] and r["first_rank"] <= k) / n
                for k in (5, 10, 20, 40)}
        summ["mrr@40"] = sum(1.0 / r["first_rank"] for r in rows if r["first_rank"]) / n
        report[view] = {"summary": summ, "rows": rows}
        print(view, json.dumps(summ, ensure_ascii=False))

    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
