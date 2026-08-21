#!/usr/bin/env python3
"""0820 에이전틱 제출 로그를 LSH train v2 span gold로 재채점한다.

검색은 다시 실행하지 않는다. 저장된 submitted ID를 현재의 조/청크 index로
resolve한 뒤, filesearch/scoring.py의 기존 span-overlap 규칙을 그대로 적용한다.

분모는 두 가지를 함께 보고한다.
- intersection: 기존 제출 로그와 새 유효 gold가 모두 있는 문항
- itt: 새 유효 gold 전체(기존 제출이 없는 문항은 0점)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V2 = HERE.parents[2]
FS = V2 / "filesearch"
sys.path.insert(0, str(FS))

from scoring import score  # noqa: E402
from units import Units  # noqa: E402


METRICS = ("R@1", "R@5", "R@10", "R@20", "S@5", "suff@5", "suff@10", "RR@10")
ARM_RUNS = {
    "base_c3": ("full_base_a", "full_base_b"),
    "fs_slot": ("full_fs_slot",),
    "dual_active": ("full_dual_active",),
}


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_metrics(rows, denominator=None):
    denominator = len(rows) if denominator is None else denominator
    if denominator == 0:
        return {key: 0.0 for key in METRICS}
    return {key: round(sum(row[key] for row in rows) / denominator, 4) for key in METRICS}


def load_arm_rows(agent_dir, run_names):
    rows = []
    sources = []
    for run_name in run_names:
        path = agent_dir / run_name / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"결과 파일 없음: {path}")
        part = load_jsonl(path)
        rows.extend(part)
        sources.append({"run": run_name, "path": str(path), "n_rows": len(part), "sha256": sha256(path)})
    by_qid = {}
    duplicates = []
    for row in rows:
        qid = row["qid"]
        if qid in by_qid:
            duplicates.append(qid)
        by_qid[qid] = row
    if duplicates:
        raise ValueError(f"arm 내부 QID 중복: {sorted(set(duplicates))}")
    return by_qid, sources


def unresolved_submissions(units, rows):
    unresolved = []
    for qid, row in rows.items():
        for rank, element_id in enumerate(row.get("submitted", []), 1):
            if not units.resolve([element_id]):
                unresolved.append({"qid": qid, "rank": rank, "id": element_id})
    return unresolved


def zero_score():
    return {key: 0.0 for key in METRICS}


def rescore_arm(units, gold, rows):
    gold_qids = set(gold)
    result_qids = set(rows)
    common_qids = sorted(gold_qids & result_qids)
    missing_qids = sorted(gold_qids - result_qids)
    ignored_qids = sorted(result_qids - gold_qids)

    rescored = []
    old_same_q = []
    per_q = []
    for qid in common_qids:
        old = rows[qid]
        ranked = units.resolve(old.get("submitted", []))
        new_score = score(ranked, gold[qid]["groups"], ks=(1, 5, 10, 20))
        old_score = {key: float(old.get(key, 0.0)) for key in METRICS}
        rescored.append(new_score)
        old_same_q.append(old_score)
        per_q.append({
            "qid": qid,
            "submitted": old.get("submitted", []),
            "resolved_jo": [unit["element_id"] for unit in ranked],
            "old": old_score,
            "v2": {key: new_score[key] for key in METRICS},
        })

    old_reported = [
        {key: float(row.get(key, 0.0)) for key in METRICS}
        for row in rows.values()
    ]
    old_common_metrics = mean_metrics(old_same_q)
    intersection_metrics = mean_metrics(rescored)
    itt_metrics = mean_metrics(rescored, denominator=len(gold))

    return {
        "n_result": len(rows),
        "n_intersection": len(common_qids),
        "n_v2_itt": len(gold),
        "missing_result_qids": missing_qids,
        "ignored_result_qids": ignored_qids,
        "unresolved_submissions": unresolved_submissions(units, rows),
        "metrics": {
            "old_reported": mean_metrics(old_reported),
            "old_same_q": old_common_metrics,
            "v2_intersection": intersection_metrics,
            "v2_itt_missing_zero": itt_metrics,
            "delta_v2_minus_old_same_q": {
                key: round(intersection_metrics[key] - old_common_metrics[key], 4)
                for key in METRICS
            },
        },
        "per_q": per_q + [
            {"qid": qid, "submitted": [], "resolved_jo": [], "old": None, "v2": zero_score(), "missing_result": True}
            for qid in missing_qids
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=FS / "out/gold_spans_lsh_train_v2.jsonl",
        help="채점용 v2 span gold JSONL",
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=HERE / "out/agent",
        help="0820 agent run 디렉터리",
    )
    parser.add_argument(
        "--jo",
        type=Path,
        default=FS / "out/elements_u2jo.jsonl",
        help="element/jo ID resolve용 조 index",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=V2 / "vector_search/out/chunks.jsonl",
        help="c* 제출 ID resolve용 chunk index",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "out/rescore_gold_v2.json",
        help="재채점 보고서 JSON 경로",
    )
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    eligible = {
        row["qid"]: row
        for row in gold_rows
        if row.get("groups") and row.get("status", "ok") == "ok"
    }
    excluded = [
        {
            "qid": row["qid"],
            "status": row.get("status", "ok"),
            "n_groups": len(row.get("groups", [])),
            "unmapped": row.get("unmapped", []),
            "reason": (
                "empty_groups+status_not_ok"
                if not row.get("groups") and row.get("status", "ok") != "ok"
                else "empty_groups"
                if not row.get("groups")
                else "status_not_ok"
            ),
        }
        for row in gold_rows
        if row["qid"] not in eligible
    ]

    units = Units(args.jo, args.chunks)
    report = {
        "method": "saved submitted IDs -> current Units.resolve -> existing span-overlap scoring",
        "sources": {
            "gold": {"path": str(args.gold), "sha256": sha256(args.gold)},
            "jo": {"path": str(args.jo), "sha256": sha256(args.jo)},
            "chunks": {"path": str(args.chunks), "sha256": sha256(args.chunks)},
        },
        "denominators": {
            "n_gold_rows": len(gold_rows),
            "n_eligible": len(eligible),
            "n_excluded": len(excluded),
            "excluded": excluded,
            "policy": {
                "intersection": "기존 결과와 유효 v2 gold가 모두 있는 QID만 집계",
                "itt": "유효 v2 gold 전체를 분모로 사용하고 기존 결과가 없는 QID는 0점",
            },
        },
        "arms": {},
    }

    for arm, run_names in ARM_RUNS.items():
        rows, sources = load_arm_rows(args.agent_dir, run_names)
        result = rescore_arm(units, eligible, rows)
        result["source_runs"] = sources
        report["arms"][arm] = result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"gold={len(gold_rows)} eligible={len(eligible)} excluded={len(excluded)}")
    print(f"output={args.out}")
    print("| arm | n(common/ITT) | R@5 old->v2 | R@10 old->v2 | suff@10 old->v2 | ITT R@5/R@10/suff@10 |")
    print("|---|---:|---:|---:|---:|---:|")
    for arm, result in report["arms"].items():
        m = result["metrics"]
        old = m["old_same_q"]
        new = m["v2_intersection"]
        itt = m["v2_itt_missing_zero"]
        print(
            f"| {arm} | {result['n_intersection']}/{result['n_v2_itt']} "
            f"| {old['R@5']:.4f}->{new['R@5']:.4f} "
            f"| {old['R@10']:.4f}->{new['R@10']:.4f} "
            f"| {old['suff@10']:.4f}->{new['suff@10']:.4f} "
            f"| {itt['R@5']:.4f}/{itt['R@10']:.4f}/{itt['suff@10']:.4f} |"
        )
        print(
            f"  missing={result['missing_result_qids']} "
            f"ignored={len(result['ignored_result_qids'])} "
            f"unresolved={result['unresolved_submissions']}"
        )


if __name__ == "__main__":
    main()
