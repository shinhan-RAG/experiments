#!/usr/bin/env python3
"""서버에서 내려받은 results/ 아카이브가 '제대로' 있는지 검사한다.

파일 존재만으로는 부족하다. arm 이 다 돌았는지, 증강이 실제로 먹었는지,
judge 가 실행됐는지는 내용을 봐야 안다. 어제 Part 2 가 죽으면서 완료된 arm 결과까지
날아간 사고와, arm 8개가 라벨만 남고 전부 baseline 으로 붕괴하는 버그
(subset null -> load_augmentations 가 조용히 None 을 반환)를 잡는 것이 목적이다.

    python scripts/verify_results.py results/
    python scripts/verify_results.py results/ --baseline old_results/
    python scripts/verify_results.py results/part1_stacking/20260730_101010.json

종료코드: FAIL 이 하나라도 있으면 1, 아니면 0 (WARN 은 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 콘솔 기본 코덱(cp949)은 이 파일의 한국어/기호를 못 찍는다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Part 1 의 8 arm. experiment_legal.yaml / experiment_shinhan.yaml 의
# part1_stacking.steps 와 일치해야 한다.
PART1_ARMS = [
    "baseline", "taxonomy_only", "tags_only", "prefix_only", "metadata_only",
    "stack_tax_tags", "stack_tax_tags_prefix", "stack_all",
]
PART3_ARMS = ["approach_A", "approach_B", "approach_C"]

# 보고에서 제외해야 하는 지표. 이유는 아래 print_caveats 참조.
EXCLUDED_METRICS = ["avg_efficiency", "density@20", "span_f1@20"]

PART_DIRS = ["part1_stacking", "part2_scaling", "part3_tags",
             "part4_generalization", "part5_pull_backend"]


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"    [OK]   {msg}")

    def warn(self, msg: str) -> None:
        print(f"    [WARN] {msg}")
        self.warns.append(msg)

    def fail(self, msg: str) -> None:
        print(f"    [FAIL] {msg}")
        self.fails.append(msg)


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_per_part(root: Path) -> dict[str, Path]:
    """각 파트 디렉터리에서 가장 최근 JSON 하나씩."""
    out = {}
    for part in PART_DIRS:
        d = root / part
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.json"))
        if files:
            out[part] = files[-1]      # 파일명이 타임스탬프라 사전순 = 시간순
    return out


def get(metrics: dict, key: str):
    """summary 항목은 지표를 평평하게 담는다. chunk_metrics 안도 함께 본다."""
    if key in metrics:
        return metrics[key]
    return (metrics.get("chunk_metrics") or {}).get(key)


# ---------------------------------------------------------------- 파트별 기대 arm

def expected_keys(part: str, data: dict) -> list[str]:
    man = data.get("manifest") or {}
    summary = data.get("summary") or {}

    if part == "part1_stacking":
        return man.get("arms") or PART1_ARMS

    if part == "part3_tags":
        approaches = man.get("approaches")
        if approaches:
            return [f"approach_{a}" for a in approaches]
        return PART3_ARMS

    if part == "part2_scaling":
        sizes = man.get("subsets") or []
        arms = man.get("arms") or []
        if not sizes:
            return []                                   # manifest 없으면 검사 생략
        keys = []
        for size in sizes:
            sk = f"{size // 1000}k"
            for arm in arms:
                keys.append(f"{arm}_{sk}")
                if man.get("include_single_pull"):
                    keys.append(f"single-pull_{arm}_{sk}")
            keys.append(f"hybrid_{sk}")
        return keys

    # part4 / part5 는 config 에서 데이터셋·백엔드를 읽어야 알 수 있다.
    # 실행된 키를 그대로 보고하고 개수만 확인한다.
    return list(summary)


# ---------------------------------------------------------------- 검사 본체

def check_file(part: str, path: Path, rep: Report) -> dict:
    print(f"\n{'=' * 78}\n{part}  <-  {path}")
    data = load(path)
    summary = data.get("summary") or {}
    man = data.get("manifest") or {}

    if man.get("git_commit"):
        print(f"  git_commit: {man['git_commit'][:12]}  dataset: {man.get('dataset')}"
              f"  subset: {man.get('subset_size') or man.get('subsets')}")
    else:
        rep.warn("manifest 가 비어 있다 — 어떤 커밋/config 로 돌았는지 추적 불가")

    if not summary:
        rep.fail("summary 가 비어 있다 — 파트가 완주하지 않았다")
        return {}

    # 1) 필수 키 완비
    want = expected_keys(part, data)
    missing = [k for k in want if k not in summary]
    extra = [k for k in summary if k not in want]
    print(f"  keys({len(summary)}): {sorted(summary)}")
    if missing:
        rep.fail(f"누락: {missing}")
    else:
        rep.ok(f"기대 키 {len(want)}개 완비")
    if extra:
        rep.warn(f"기대 목록에 없는 키: {extra}")

    # 2) 지표 표
    print(f"\n  {'key':34s} {'recall':>8s} {'acc':>8s} {'judged':>7s} {'n':>4s} {'nDCG@10':>8s}")
    recalls = {}
    for key in sorted(summary):
        m = summary[key]
        if not isinstance(m, dict):
            rep.fail(f"{key}: summary 항목이 dict 가 아니다 ({type(m).__name__})")
            continue
        r = m.get("avg_gold_recall")
        recalls[key] = r
        acc = m.get("accuracy")
        print(f"  {key:34s} {_f(r):>8s} {_f(acc):>8s} "
              f"{str(m.get('judged_n')):>7s} {str(m.get('n')):>4s} "
              f"{_f(get(m, 'ndcg@10')):>8s}")

    # 3) 증강 붕괴 — arm 이 여러 개인데 recall 이 전부 같으면 증강이 안 먹었다
    if part in ("part1_stacking", "part3_tags"):
        vals = {v for v in recalls.values() if v is not None}
        if len(recalls) > 1 and len(vals) <= 1:
            rep.fail("모든 arm 의 avg_gold_recall 이 동일하다 → 증강 미적용. "
                     "config 의 subset 이 non-null 인지 확인할 것 "
                     "(null 이면 load_augmentations 가 조용히 None 을 반환한다)")
        elif vals:
            rep.ok(f"arm 간 recall 이 서로 다름 ({len(vals)}개 고유값)")

    # 4) judge 축. 전 arm 공통 문제는 한 줄로 묶어 경고한다.
    no_latency, zero_chunk = [], []
    for key, m in summary.items():
        if not isinstance(m, dict):
            continue
        n, judged = m.get("n"), m.get("judged_n")
        if m.get("accuracy") is None or not judged:
            rep.warn(f"{key}: accuracy 미측정 (judged_n={judged}) - "
                     "data/reference_answers/<dataset>.json 존재 여부 확인")
        elif n and judged != n:
            corrected = round(m["accuracy"] * judged / n, 4)
            rep.warn(f"{key}: 분모 불일치 judged_n={judged} != n={n}. "
                     f"보고 시 n 기준 {corrected} 를 병기할 것")
        if m.get("judge_error_n"):
            rep.warn(f"{key}: judge 오류 {m['judge_error_n']}건")
        if m.get("avg_latency_seconds") == 0:
            no_latency.append(key)
        cm = m.get("chunk_metrics") or {}
        if cm and all(v == 0 for v in cm.values()):
            zero_chunk.append(key)
    if no_latency:
        rep.warn(f"avg_latency_seconds=0 인 키 {len(no_latency)}/{len(summary)}개 "
                 "- 지연 비교 불가")
    if zero_chunk:
        rep.warn(f"chunk_metrics 전부 0 인 키 {len(zero_chunk)}/{len(summary)}개 "
                 "- parent 단위 평가라 청크 지표가 계산되지 않은 것일 수 있다")

    # 5) part2 규모 곡선 - hybrid recall 이 규모와 무관하게 천장에 붙어 있는지
    if part == "part2_scaling":
        hyb = {k: v for k, v in recalls.items()
               if k.startswith("hybrid_") and v is not None}
        if len(hyb) >= 2 and min(hyb.values()) >= 0.95:
            rep.warn(f"hybrid recall 이 전 규모에서 천장이다 "
                     f"({', '.join(f'{k}={v:.2f}' for k, v in sorted(hyb.items()))}) -> "
                     "판시사항=질의 / 본문=gold 구성의 어휘 중복으로 BM25 가 거의 확실히 "
                     "맞히는 산물일 수 있다. 데이터셋 구성 한계로 명시해 보고할 것")

    return recalls


def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------- 기준선 비교

def compare(part: str, new: dict, old: dict) -> None:
    common = sorted(set(new) & set(old))
    if not common:
        print(f"  (공통 키 없음 — {part} 비교 생략)")
        return
    print(f"\n  --- {part}: 기준선 대비 avg_gold_recall ---")
    print(f"  {'key':34s} {'baseline':>10s} {'new':>10s} {'delta':>10s}")
    for k in common:
        o, n = old[k], new[k]
        d = f"{n - o:+.4f}" if (isinstance(o, (int, float))
                                and isinstance(n, (int, float))) else "-"
        print(f"  {k:34s} {_f(o):>10s} {_f(n):>10s} {d:>10s}")
    only_new = sorted(set(new) - set(old))
    only_old = sorted(set(old) - set(new))
    if only_new:
        print(f"  새 결과에만: {only_new}")
    if only_old:
        print(f"  기준선에만: {only_old}")


def print_caveats() -> None:
    print(f"\n{'=' * 78}\n보고 시 제외/주의할 지표")
    print("  - density@20 / span_f1@20 : 분모가 top-k 청크 전체 길이라 이론적 최대치가")
    print("      약 0.002 다 (src/eval/span_metrics.py:75). coverage 만 해석한다.")
    print("  - avg_efficiency          : gold_recall / pull_count 정의라 1회 pull 하는")
    print("      hybrid 가 구조적으로 이긴다 (src/eval/judge.py:83-87). 아키텍처 간")
    print("      비교 지표로 부적절하다.")
    print("  - recall_ci95             : 비대응(marginal) CI 다 (src/eval/judge.py:174).")
    print("      같은 질의 집합을 돌았으므로 겹침을 '차이 없음'으로 읽지 말 것.")
    print("      paired 비교는 src/eval/comparison.py 의 compare_probe_rows 를 쓴다.")
    print("  - 신한 vs 법률            : 신한은 청크 단위, 법률은 parent 단위 평가다.")
    print("      절대 점수를 직접 비교하지 말 것.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path,
                    help="results/ 디렉터리 또는 단일 결과 JSON")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="비교할 기준선 results/ 디렉터리 또는 JSON (예: 배치8 결과)")
    args = ap.parse_args()

    if not args.target.exists():
        print(f"경로 없음: {args.target}", file=sys.stderr)
        return 2

    if args.target.is_file():
        part = args.target.parent.name
        targets = {part if part in PART_DIRS else "unknown": args.target}
    else:
        targets = latest_per_part(args.target)
        if not targets:
            print(f"결과 JSON 을 찾지 못했다: {args.target}\n"
                  f"  기대 구조: {args.target}/part1_stacking/*.json", file=sys.stderr)
            return 2

    rep = Report()
    new_recalls = {}
    for part, path in targets.items():
        new_recalls[part] = check_file(part, path, rep)

    if args.baseline:
        print(f"\n{'=' * 78}\n기준선 비교: {args.baseline}")
        if args.baseline.is_file():
            bparts = {args.baseline.parent.name: args.baseline}
        else:
            bparts = latest_per_part(args.baseline)
        for part, path in bparts.items():
            if part not in new_recalls:
                continue
            old = load(path).get("summary") or {}
            old_recalls = {k: v.get("avg_gold_recall")
                           for k, v in old.items() if isinstance(v, dict)}
            compare(part, new_recalls[part], old_recalls)
        print("\n  임베딩 백엔드(batch 8 -> 128)만 바뀌었다면 recall 차이는 작아야 한다.")
        print("  큰 차이가 나면 캐시가 섞였는지, 또는 배치 크기가 정규화에 영향을 줬는지 확인한다.")

    print_caveats()

    print(f"\n{'=' * 78}")
    print(f"검사한 파트: {len(targets)}  FAIL {len(rep.fails)}  WARN {len(rep.warns)}")
    for m in rep.fails:
        print(f"  FAIL: {m}")
    if not rep.fails:
        print("  치명적 문제 없음. WARN 은 위 표와 함께 보고서에 반영할 것.")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
