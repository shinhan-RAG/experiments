#!/usr/bin/env python3
"""호스트측 결정론 제출 재조합(R10) — 에이전트 제출 상위 5 보존, 첫 검색 top-N 을 6위부터 보충.

에이전트 재실행 없이 기존 run 의 host_trace 에서 첫 search 반환 순서를 읽어
submitted 리스트를 재조합해 재채점한다. 상위 5는 절대 재배열하지 않으므로(append-only)
R@5 는 top5 미달(<5 gold 커버) 문항에서만 변할 수 있다 — 제로섬 강등이 구조적으로 불가능.

사용:
  python3 recompose_submissions.py --run out/host_agent/<run>/ \
      --gold ../../../filesearch/out/gold_....jsonl [--anchor-n 2] \
      --out results_recomposed.jsonl
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score  # noqa: E402
from units import Units  # noqa: E402


def first_search_ids(trace):
    """host_trace 에서 최초 search 응답의 노출 id 순서."""
    for turn in trace:
        resp = turn.get("response") or {}
        results = resp.get("results")
        if results:
            order = []
            for item in results:
                i = item.get("id")
                if i and i not in order:
                    order.append(i)
            return order
    return []


def recompose(submitted, fs_ids, anchor_n):
    head = submitted[:5]
    anchors = [x for x in fs_ids[:anchor_n] if x not in head]
    tail = [x for x in submitted[5:] if x not in anchors]
    return (head + anchors + tail)[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--jo", default=str(FS / "out/elements_u3jo.jsonl"))
    ap.add_argument("--anchor-n", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(a.gold, encoding="utf-8")}
    units = Units(a.jo)
    rows = []
    for line in open(run / "results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        sessions = sorted(run.glob(f"sessions/{r['qid']}_*_r{r['rep']}"))
        trace_path = next((s / "host_trace.json" for s in sessions
                           if (s / "host_trace.json").exists()), None)
        fs_ids = (first_search_ids(json.loads(trace_path.read_text(encoding="utf-8")))
                  if trace_path else [])
        sub2 = recompose(list(r["submitted"]), fs_ids, a.anchor_n)
        sc = score(units.resolve(sub2), gold[r["qid"]]["groups"], ks=(1, 5, 10, 20))
        rows.append({**r, "submitted": sub2, "recomposed": sub2 != list(r["submitted"])[:10],
                     **{k: sc[k] for k in ("R@1", "R@5", "R@10", "suff@5")}})
    out = Path(a.out)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                   encoding="utf-8")
    byq = {}
    for r in rows:
        byq.setdefault(r["qid"], []).append(r["R@5"])
    macro = sum(sum(v) / len(v) for v in byq.values()) / len(byq)
    print(json.dumps({"rows": len(rows), "qids": len(byq), "anchor_n": a.anchor_n,
                      "R@5_macro": round(macro, 4),
                      "recomposed_rows": sum(r["recomposed"] for r in rows)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
