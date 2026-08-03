"""단계 분리 funnel 사후 분석 (측정 설계 교정 B).

Gold R@W는 "pull → 워크스페이스 → agent 필터/read → 최종"을 합산한 지표라
어느 단계에서 gold가 사라졌는지 구분할 수 없다. 이 스크립트는 결과 JSON의
질의별 계측(pull_stats.result_doc_ids, trace의 grep/find doc_ids)에서 4단계
funnel을 복원한다:

  ① pull_recall     : pull이 가져온 문서(전체 pull 합집합)에 gold가 있었나
                      — agent의 재작성 질의 기준 (probe와 다른 축)
  ② filter_exposure : grep/find 결과에 gold가 노출됐나 — 이 저장소의 grep/find는
                      문서를 제거하지 않으므로 "제거"가 아니라 "agent의 주의에서
                      gold가 탈락했는가"로 해석해야 한다
  ③ read_recall     : agent가 실제 read한 문서에 gold가 있었나 (기존 필드)
  ④ gold_recall     : 최종 워크스페이스 생존율 (기존 지표, Gold R@W)

계측 이전(구버전) 결과 JSON은 ①②를 "n/a (pre-instrumentation)"로 표기하고
③④만 산출한다 — 재실험 없이 신·구 결과 모두에 실행 가능.

사용:
  python scripts/analyze_stage_funnel.py results/part1_stacking/20260803_120000.json
  python scripts/analyze_stage_funnel.py <결과.json> --dataset trec-covid  # manifest에 dataset 없을 때
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from run_experiment import (  # noqa: E402
    dataset_dir, positive_gold_by_query, to_parent_ids,
)


def load_parent_map(dataset: str) -> dict:
    """corpus.jsonl에서 {chunk_id → parent_id}만 추출 (본문 미보관)."""
    path = dataset_dir(dataset) / "corpus.jsonl"
    if not path.exists():
        return {}
    parent_map = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            pid = doc.get("parent_id")
            if pid and pid != doc["_id"]:
                parent_map[str(doc["_id"])] = str(pid)
    return parent_map


def load_qrels(dataset: str) -> list:
    with open(dataset_dir(dataset) / "qrels.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def recall(ids: list, gold: set, parent_map: dict) -> float:
    mapped = set(to_parent_ids(ids, parent_map))
    return len(mapped & gold) / len(gold)


def analyze_arm(rows: list, query_gold: dict, parent_map: dict) -> dict:
    n = 0
    pull_recalls = []
    pull_instrumented = True
    filter_used = 0
    filter_gold_exposed = 0
    filter_instrumented = True
    read_recalls = []
    final_recalls = []

    for r in rows:
        if r.get("failed"):
            continue
        gold = query_gold.get(str(r["query_id"]))
        if not gold:
            continue
        n += 1
        final_recalls.append(float(r.get("gold_recall", 0.0)))
        read_recalls.append(float(r.get("read_recall", 0.0)))

        # ① pull 단계
        pulled_ids = []
        for stats in r.get("pull_stats", []):
            ids = stats.get("result_doc_ids")
            if ids is None:
                pull_instrumented = False
                break
            pulled_ids.extend(ids)
        else:
            pull_recalls.append(recall(pulled_ids, gold, parent_map))

        # ② 필터(grep/find) 노출 단계 — 필터를 쓴 질의만 분모
        used_filter = False
        exposed = False
        for ev in r.get("trace", []):
            if ev.get("tool") not in ("grep", "find"):
                continue
            summary = ev.get("result_summary", {})
            ids = summary.get("match_doc_ids") or summary.get("doc_ids")
            if ids is None:
                # 구버전 trace(개수만 기록) — 판별 불가
                if summary.get("matches") or summary.get("matched"):
                    filter_instrumented = False
                continue
            used_filter = True
            if set(to_parent_ids(ids, parent_map)) & gold:
                exposed = True
        if used_filter:
            filter_used += 1
            if exposed:
                filter_gold_exposed += 1

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": n,
        "pull_recall": (avg(pull_recalls) if pull_instrumented and pull_recalls
                        else "n/a (pre-instrumentation)"),
        "filter_used_queries": filter_used,
        "filter_gold_exposure_rate": (
            round(filter_gold_exposed / filter_used, 4) if filter_used
            else ("n/a (pre-instrumentation)" if not filter_instrumented
                  else "n/a (no filter use)")),
        "read_recall": avg(read_recalls),
        "final_gold_recall": avg(final_recalls),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result_json", nargs="+")
    ap.add_argument("--dataset", default="",
                    help="manifest에 dataset이 없는 구버전 결과용 오버라이드")
    ap.add_argument("--out", default="",
                    help="funnel JSON 출력 경로 (기본: <입력>_funnel.json)")
    args = ap.parse_args()

    for path_str in args.result_json:
        path = Path(path_str)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        manifest = data.get("manifest", {})
        dataset = args.dataset or manifest.get("dataset")
        if not dataset:
            datasets = manifest.get("datasets")
            if datasets and len(datasets) == 1:
                dataset = datasets[0]
        if not dataset:
            raise SystemExit(
                f"{path}: manifest에 dataset이 없음 — --dataset으로 지정할 것"
            )

        query_gold = positive_gold_by_query(load_qrels(dataset))
        parent_map = load_parent_map(dataset)

        report = {"source": str(path), "dataset": dataset, "arms": {}}
        print(f"\n=== {path.name} (dataset={dataset}) ===")
        header = (f"{'arm':30s} {'n':>4s} {'①pull':>10s} {'②filter노출':>12s} "
                  f"{'③read':>8s} {'④final':>8s}")
        print(header)
        print("-" * len(header))
        for arm, block in data.get("full_results", {}).items():
            rows = block.get("results")
            if not rows:
                continue
            funnel = analyze_arm(rows, query_gold, parent_map)
            report["arms"][arm] = funnel
            print(f"{arm:30s} {funnel['n']:>4d} "
                  f"{str(funnel['pull_recall']):>10s} "
                  f"{str(funnel['filter_gold_exposure_rate']):>12s} "
                  f"{str(funnel['read_recall']):>8s} "
                  f"{str(funnel['final_gold_recall']):>8s}")

        out_path = Path(args.out) if args.out else path.with_name(
            path.stem + "_funnel.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nfunnel saved: {out_path}")
        print("해석 주의: ②는 '필터가 gold를 제거했다'가 아니라 '필터 결과에 "
              "gold가 보였다'는 노출 지표다 (grep/find는 조회 전용).")


if __name__ == "__main__":
    main()
