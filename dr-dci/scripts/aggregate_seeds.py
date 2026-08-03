"""replicate(--seeds) 결과 집계 (측정 설계 교정 D).

같은 part를 seed만 바꿔 반복 실행한 결과 파일들을 모아 arm별 지표의
replicate 간 분산(mean/std/min/max)과 paired delta의 부호 일관성을 보고한다.

분산의 해석에 대한 정직성: agent 경로는 temperature=0이므로 run 간 분산의
주원인은 seed가 아니라 OpenAI API 비결정성(모델 스냅샷)과 임베딩 서버의 수치
비결정성이다. seed는 bootstrap/샘플링 재현성용이다. 각 파일의
system_fingerprints 합집합을 병기해 "분산이 API 스냅샷 차이인지" 판별 근거를
남긴다. probe-only 결과는 결정적이므로 std=0이어야 정상이다 — 0이 아니면
배선(캐시/엔드포인트)이 다른 것.

사용:
  python scripts/aggregate_seeds.py results/part1_stacking/20260803_*.json
  python scripts/aggregate_seeds.py <파일1> <파일2> <파일3>
"""
import argparse
import json
import math
from pathlib import Path


def mean_std(vals: list) -> tuple:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result_json", nargs="+")
    ap.add_argument("--metric", default="avg_gold_recall",
                    help="arm summary에서 집계할 지표 (기본 avg_gold_recall)")
    args = ap.parse_args()

    runs = []
    for p in args.result_json:
        path = Path(p)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs.append((path, data))
    if len(runs) < 2:
        raise SystemExit("replicate 집계에는 결과 파일이 2개 이상 필요하다")

    # 같은 실험인지 최소 검증: arms 구성이 같아야 한다
    arm_sets = [tuple(sorted(d.get("summary", {}).keys())) for _, d in runs]
    if len(set(arm_sets)) > 1:
        print("[warn] 파일 간 arm 구성이 다르다 — 같은 part/config의 replicate가 "
              "맞는지 확인할 것")

    seeds = [d.get("manifest", {}).get("seed") for _, d in runs]
    print(f"files={len(runs)} seeds={seeds}")

    # arm별 지표 집계
    print(f"\n== arm별 {args.metric} (replicate 간) ==")
    header = f"{'arm':35s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'n_rep':>5s}"
    print(header)
    print("-" * len(header))
    all_arms = sorted({a for s in arm_sets for a in s})
    for arm in all_arms:
        vals = []
        for _, d in runs:
            metrics = d.get("summary", {}).get(arm)
            if isinstance(metrics, dict) and args.metric in metrics:
                v = metrics[args.metric]
                if v is not None:
                    vals.append(float(v))
        if not vals:
            continue
        m, s = mean_std(vals)
        print(f"{arm:35s} {m:>8.4f} {s:>8.4f} {min(vals):>8.4f} "
              f"{max(vals):>8.4f} {len(vals):>5d}")

    # analysis delta 부호 일관성 (gold_recall 또는 probe recall_at_20 우선)
    print("\n== paired delta 부호 일관성 (replicate 간) ==")
    delta_keys = sorted({k for _, d in runs for k in d.get("analysis", {})})
    for key in delta_keys:
        deltas = []
        for _, d in runs:
            block = d.get("analysis", {}).get(key)
            if not isinstance(block, dict):
                continue
            inner = (block.get("gold_recall") or block.get("recall_at_20")
                     or block.get("treatment_minus_control") or block)
            md = inner.get("mean_delta") if isinstance(inner, dict) else None
            if md is not None:
                deltas.append(float(md))
        if not deltas:
            continue
        signs = {1 if v > 0 else (-1 if v < 0 else 0) for v in deltas}
        consistent = "일관" if len(signs) == 1 else "**불일치**"
        print(f"  {key}: deltas={[round(v, 4) for v in deltas]} → {consistent}")

    # system fingerprint 합집합 (API 스냅샷 차이 판별 근거)
    print("\n== system_fingerprints (분산 원인 판별용) ==")
    for path, d in runs:
        fps = set()
        for block in d.get("full_results", {}).values():
            for row in block.get("results", []) or []:
                fps.update(row.get("system_fingerprints", []))
        seed = d.get("manifest", {}).get("seed")
        print(f"  {path.name} (seed={seed}): "
              f"{sorted(fps) if fps else '(없음/probe-only)'}")


if __name__ == "__main__":
    main()
