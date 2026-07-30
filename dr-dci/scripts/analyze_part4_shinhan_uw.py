"""Part 4 (shinhan-uw) 결과 심층 분석 — 리포트용 근거 표 생성.

로그의 집계값만으로는 리포트를 못 쓴다. 필요한 것은 (a) 어떤 질의가 실패했는지,
(b) 표/텍스트 층으로 갈리는지, (c) 문서별로 갈리는지, (d) 단일 retriever 대비
융합·리랭킹이 얼마나 기여했는지다. 이 스크립트는 결과 JSON 의 질의별 행을 열어
그 네 가지를 뽑는다.

BM25 기준선은 여기서 직접 계산한다 — 결정론적이고 GPU·API 가 필요 없으므로
"그냥 색인해서 어휘 검색만 했을 때"의 바닥값을 같은 질의·같은 코퍼스에서 얻는다.

해석 금지 지표는 출력하지 않는다:
  - density@k / span_f1@k : 분모가 top-k 총 길이라 이론상 최대 ≈0.002
    (src/eval/span_metrics.py:75). coverage@k 만 쓴다.
  - efficiency : gold_recall / pull_count 라 1-pull hybrid 가 구조적으로 이긴다
    (src/eval/judge.py:83-87).

  PYTHONPATH=. python scripts/analyze_part4_shinhan_uw.py
  PYTHONPATH=. python scripts/analyze_part4_shinhan_uw.py --result results/part4_generalization/20260730_135514.json
  PYTHONPATH=. python scripts/analyze_part4_shinhan_uw.py --no-bm25      # BM25 기준선 생략
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW = DATA_DIR / "raw" / "shinhan-uw"
RESULTS_DIR = BASE_DIR / "results" / "part4_generalization"
DATASET = "shinhan-uw"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_result() -> Path:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"결과 파일이 없다: {RESULTS_DIR}")
    return files[-1]


def ranked_ids(row: dict) -> list[str]:
    """질의별 검색 결과 순위 목록. hybrid 는 retrieved_docs, 에이전트는 workspace_docs."""
    for key in ("retrieved_docs", "workspace_docs"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value]
    return []


def hr(char: str = "─", n: int = 78) -> None:
    print(char * n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, help="결과 JSON (기본: 가장 최근 파일)")
    ap.add_argument("--no-bm25", action="store_true")
    args = ap.parse_args()

    path = args.result or latest_result()
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = payload.get("full_results", payload)
    meta = {m["qid"]: m for m in load_jsonl(RAW / "qa_meta.jsonl")}
    queries = {str(q["_id"]): q for q in load_jsonl(RAW / "queries.jsonl")}
    qrels = load_jsonl(RAW / "qrels.jsonl")
    gold_of = {str(r["query-id"]): str(r["corpus-id"]) for r in qrels}

    print(f"결과 파일 : {path}")
    manifest = payload.get("manifest", {})
    if manifest:
        print(f"git commit: {manifest.get('git_commit', '?')}")
    arms = [k for k in full if isinstance(full[k], dict) and "results" in full[k]]
    print(f"arm       : {arms}")

    # ---------------------------------------------------------------- 1. 요약
    hr("=")
    print("1. arm 요약 (해석 가능한 지표만)")
    hr("=")
    header = (f"{'arm':<22}{'n':>4}{'recall@5':>10}{'recall@20':>11}"
              f"{'nDCG@10':>9}{'cover@20':>10}{'acc(judged)':>13}{'acc(전체)':>11}")
    print(header)
    hr()
    summary = {}
    for arm in arms:
        rows = full[arm]["results"]
        m = full[arm]["metrics"]
        cm = m.get("chunk_metrics", {})
        judged = m.get("judged_n") or m.get("n_judged") or 0
        acc_j = m.get("accuracy")
        # 미판정 질의를 오답으로 세는 동일 분모 정확도.
        # judged_n 이 arm 마다 다르면 accuracy 를 그대로 비교할 수 없다.
        n_correct = round((acc_j or 0) * judged)
        acc_all = n_correct / len(rows) if rows else 0.0
        summary[arm] = {
            "n": len(rows), "judged": judged, "n_correct": n_correct,
            "acc_judged": acc_j, "acc_all": acc_all,
            "recall@5": cm.get("recall@5"), "recall@20": cm.get("recall@20"),
            "ndcg@10": cm.get("ndcg@10"), "coverage@20": cm.get("coverage@20"),
            "avg_pulls": m.get("avg_pulls"), "avg_turns": m.get("avg_turns"),
            "avg_candidates": m.get("avg_retrieved_candidates"),
            "recall_ci95": m.get("recall_ci95"),
        }
        s = summary[arm]
        print(f"{arm:<22}{s['n']:>4}{s['recall@5']:>10.4f}{s['recall@20']:>11.4f}"
              f"{s['ndcg@10']:>9.4f}{s['coverage@20']:>10.4f}"
              f"{(s['acc_judged'] or 0):>9.4f}({s['judged']:>2}){s['acc_all']:>11.4f}")
    print()
    for arm in arms:
        s = summary[arm]
        ci = s["recall_ci95"]
        print(f"  {arm}: recall CI95={ci}  avg_pulls={s['avg_pulls']}  "
              f"avg_turns={s['avg_turns']}  avg_candidates={s['avg_candidates']}")
        if s["judged"] != s["n"]:
            print(f"    [!] judged {s['judged']}/{s['n']} — 답변 생성 실패 "
                  f"{s['n'] - s['judged']}건. acc(judged) 는 성공한 질의만의 정확도이므로 "
                  f"arm 간 비교는 acc(전체) 로 한다.")

    # ------------------------------------------------------- 2. BM25 기준선
    if not args.no_bm25:
        hr("=")
        print("2. BM25 단독 기준선 (어휘 검색만 — 융합·리랭킹·dense 없음)")
        hr("=")
        sys.path.insert(0, str(BASE_DIR))
        from src.retrieval.bm25 import BM25
        from src.eval.retrieval_metrics import rank_metrics

        corpus = load_jsonl(RAW / "corpus.jsonl")
        bm = BM25()
        bm.fit(corpus)
        rows, ranks = [], []
        for qid, q in queries.items():
            ranked = [r["doc_id"] for r in bm.search(q["text"], top_k=100)]
            g = gold_of[qid]
            rows.append(rank_metrics(ranked[:20], {g}, gains={g: 2.0}))
            ranks.append(ranked.index(g) + 1 if g in ranked else None)
        agg = lambda k: sum(r[k] for r in rows) / len(rows)
        print(f"  recall@5={agg('recall_at_5'):.4f}  recall@20={agg('recall_at_20'):.4f}  "
              f"nDCG@10={agg('ndcg_at_10'):.4f}")
        print(f"  gold 1위 {sum(1 for r in ranks if r == 1)}/{len(ranks)}  "
              f"top-20 밖 {sum(1 for r in ranks if r is None or r > 20)}/{len(ranks)}")
        print("\n  → 단일 retriever 대비 파이프라인 기여도:")
        for arm in arms:
            s = summary[arm]
            print(f"     {arm:<22} recall@5 {agg('recall_at_5'):.2f} → {s['recall@5']:.2f}"
                  f"   recall@20 {agg('recall_at_20'):.2f} → {s['recall@20']:.2f}")

    # --------------------------------------------------------- 3. 층별 분해
    hr("=")
    print("3. 층별 분해 (element_type / 문서)")
    hr("=")
    for arm in arms:
        rows = full[arm]["results"]
        print(f"\n  [{arm}]")
        for key, label in (("element_type", "element_type"), ("doc", "문서")):
            buckets: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                m = meta.get(str(row["query_id"]))
                if m:
                    buckets[str(m[key])].append(row)
            print(f"    {label:<12}{'n':>4}{'recall':>9}{'정답률':>9}   (미판정)")
            for name in sorted(buckets, key=lambda k: -len(buckets[k])):
                bucket = buckets[name]
                rec = sum(r.get("gold_recall", 0) for r in bucket) / len(bucket)
                judged = [r for r in bucket if r.get("judgment") in ("correct", "incorrect")]
                acc = (sum(r["judgment"] == "correct" for r in judged) / len(judged)
                       if judged else float("nan"))
                print(f"      {name[:34]:<34}{len(bucket):>4}{rec:>9.3f}{acc:>9.3f}"
                      f"   {len(bucket) - len(judged)}")

    # ------------------------------------------------------- 4. 실패 질의
    hr("=")
    print("4. 실패 질의 — 정답 청크를 못 찾은 건 (리포트의 핵심 근거)")
    hr("=")
    for arm in arms:
        rows = full[arm]["results"]
        missed = [r for r in rows if r.get("gold_recall", 0) < 1.0]
        print(f"\n  [{arm}] 검색 실패 {len(missed)}/{len(rows)} "
              f"({len(missed) / len(rows):.0%})")
        for row in sorted(missed, key=lambda r: int(r["query_id"])):
            qid = str(row["query_id"])
            m = meta.get(qid, {})
            g = gold_of.get(qid, "")
            rank = None
            ids = ranked_ids(row)
            if g in ids:
                rank = ids.index(g) + 1
            print(f"    q{qid:<3} [{m.get('element_type','?'):<6}] "
                  f"judgment={str(row.get('judgment')):<9} "
                  f"gold순위={rank if rank else '검색 안 됨'}")
            print(f"         Q: {row.get('query_text','')[:88]}")
            print(f"         문서: {m.get('doc','?')[:60]}")

    # ------------------------------- 5. gold 순위 분포 (검색된 건에 한해)
    hr("=")
    print("5. gold 순위 분포 (검색된 질의만)")
    hr("=")
    for arm in arms:
        rows = full[arm]["results"]
        buckets = Counter()
        for row in rows:
            ids = ranked_ids(row)
            g = gold_of.get(str(row["query_id"]), "")
            if g not in ids:
                buckets["미검색"] += 1
                continue
            r = ids.index(g) + 1
            buckets["1위" if r == 1 else "2-3위" if r <= 3 else
                    "4-5위" if r <= 5 else "6-10위" if r <= 10 else "11위 이하"] += 1
        order = ["1위", "2-3위", "4-5위", "6-10위", "11위 이하", "미검색"]
        print(f"  {arm:<22}" + "".join(f"{k}:{buckets.get(k,0):<7}" for k in order))

    hr("=")
    print("보고 시 제외할 것: density@k · span_f1@k (분모가 top-k 총 길이, 최대 ≈0.002)")
    print("                  efficiency (= gold_recall/pull_count, 1-pull hybrid 가 구조적 우위)")
    print("교란 요인 명시   : hybrid 만 리랭킹한다. 에이전트 pull 경로엔 리랭커가 없다")
    print("                  (run_experiment.py:562-563). 격차 일부는 방법 차이가 아니다.")
    print("                  Part 4 는 augment:false 라 에이전트의 grep(tag_filter)/find() 가")
    print("                  동작하지 않는다 — DCI 도구 4개 중 2개가 죽은 상태의 성능이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
