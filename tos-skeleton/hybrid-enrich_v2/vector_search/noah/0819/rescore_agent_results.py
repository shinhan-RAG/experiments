#!/usr/bin/env python3
"""저장된 agent submitted ID를 지정한 버전 Gold로 재채점한다(LLM 재실행 없음)."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score
from units import Units


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("gold")
    parser.add_argument("--arms", default=str(HERE / "arms.json"))
    parser.add_argument("--output")
    parser.add_argument(
        "--exclude-missing-gold", action="store_true",
        help="Gold에 없는 qid를 명시적으로 제외하고 manifest에 기록한다.")
    args = parser.parse_args()
    rows = load_jsonl(args.results)
    gold = {row["qid"]: row for row in load_jsonl(args.gold)}
    arms_path = Path(args.arms).resolve()
    arms = json.loads(arms_path.read_text(encoding="utf-8"))
    arm_names = {row["arm"] for row in rows}
    missing_arms = sorted(arm_names - set(arms))
    if missing_arms:
        raise ValueError(f"unknown arms: {missing_arms}")
    units = {
        arm: Units(FS / "out" / arms[arm].get("jo", "elements_u2jo.jsonl"))
        for arm in arm_names
    }
    missing_gold_qids = sorted({row["qid"] for row in rows if row["qid"] not in gold})
    if missing_gold_qids and not args.exclude_missing_gold:
        raise ValueError(
            "results contain qids absent from Gold; pass --exclude-missing-gold only "
            f"for an adjudicated exclusion: {missing_gold_qids}")
    included_rows = [row for row in rows if row["qid"] in gold]
    rescored = []
    for row in included_rows:
        metrics = score(units[row["arm"]].resolve(row.get("submitted", [])), gold[row["qid"]]["groups"],
                        ks=(1, 5, 10, 20))
        rescored.append({**row, **metrics, "gold_sha256": hashlib.sha256(
            Path(args.gold).read_bytes()).hexdigest()})
    output = Path(args.output) if args.output else Path(args.results).with_name("results_rescored.jsonl")
    with output.open("w", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {}
    for arm in sorted({row["arm"] for row in rescored}):
        subset = [row for row in rescored if row["arm"] == arm]
        summary[arm] = {key: sum(row[key] for row in subset) / len(subset)
                        for key in ("R@1", "R@5", "R@10", "suff@5", "suff@10")}
        summary[arm].update(errors=sum(row.get("errors", 0) for row in subset),
                            protocol_errors=sum(row.get("protocol_errors", 0) for row in subset))
    manifest = {
        "results": str(Path(args.results).resolve()),
        "results_sha256": hashlib.sha256(Path(args.results).read_bytes()).hexdigest(),
        "gold": str(Path(args.gold).resolve()),
        "gold_sha256": hashlib.sha256(Path(args.gold).read_bytes()).hexdigest(),
        "arms": str(arms_path),
        "arms_sha256": hashlib.sha256(arms_path.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "output_rows": len(rescored),
        "excluded_missing_gold_qids": missing_gold_qids,
        "excluded_missing_gold_rows": len(rows) - len(included_rows),
        "arm_jo": {
            arm: {
                "path": str((FS / "out" / arms[arm].get("jo", "elements_u2jo.jsonl")).resolve()),
                "sha256": hashlib.sha256((FS / "out" / arms[arm].get("jo", "elements_u2jo.jsonl")).read_bytes()).hexdigest(),
            }
            for arm in sorted(arm_names)
        },
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "summary": summary,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gold": str(args.gold), "arms": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
