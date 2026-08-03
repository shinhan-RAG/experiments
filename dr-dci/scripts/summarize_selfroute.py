"""무추론 라우팅 probe 결과 요약 — stdout 복사 회수용 (KT는 다운로드 불가).

results/part5_pull_backend/ 최신 N개(기본 3 = 20k/50k/110k)의 결과 JSON에서
백엔드별 R@5/R@20/nDCG@10/latency와 라우팅 정확도를 표로 출력한다.

사용:  python scripts/summarize_selfroute.py [N]
"""
import glob
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
METRICS = ("recall_at_5", "recall_at_20", "ndcg_at_10")


def summarize(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    m = d["manifest"]
    print(f"\n=== {Path(path).name} | dataset={m.get('dataset')} "
          f"subset={m.get('subset_size')} commit={str(m.get('git_commit'))[:8]} ===")
    acc = m.get("self_routing_accuracy") or {}
    for backend, a in acc.items():
        print(f"  [routing] {backend}: top1={a['accuracy_top1']} "
              f"top2={a['accuracy_top2']} fallback={a['fallback_rate']} "
              f"legal_top1={a['legal_top1_share']} (n={a['evaluated_queries']})")
    header = f"  {'backend':22s} " + " ".join(f"{k:>12s}" for k in METRICS) + f" {'latency_s':>10s}"
    print(header)
    for backend, val in d["full_results"].items():
        rows = val.get("probe_rows") or []
        if not rows:
            continue
        n = len(rows)
        agg = [sum(r[k] for r in rows) / n for k in METRICS]
        lat = sum(r["probe_latency_seconds"] for r in rows) / n
        print(f"  {backend:22s} " + " ".join(f"{v:12.4f}" for v in agg)
              + f" {lat:10.3f}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    paths = sorted(glob.glob(str(BASE / "results" / "part5_pull_backend" / "*.json")))
    if not paths:
        sys.exit("no results under results/part5_pull_backend/")
    for path in paths[-n:]:
        summarize(path)


if __name__ == "__main__":
    main()
