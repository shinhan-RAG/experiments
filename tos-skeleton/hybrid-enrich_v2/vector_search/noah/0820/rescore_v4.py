#!/usr/bin/env python3
"""기존 에이전틱 런(제출 id 로그)을 v4 교정 gold 로 재채점. 재실행·추가 비용 0.

분모 2종 병기(0817 교훈 "분모가 결론을 뒤집는다"):
- 원본: train60 전체 60문항, v3 gold (기존 보고치 재현)
- 교정: v4 regold 28 + v3 유지 32 − 제외 2 = 58문항

제외(문항 내재적 결함만, 성적 무관 기준):
- v3-offline-0413, v3-offline-0508: v4 최종 정답셋에서 삭제된 문항 + v3 gold 감사 결과
  질문과 무관한 span(특약 소멸 문단 / 법정상속인·산출기초율 문단)으로 오매핑 확인.
"""
import json, sys, glob, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parent.parent.parent / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score
from units import Units

EXCLUDE = {
    "v3-offline-0413": "v4 삭제 + v3 gold 오매핑(진단 조건 질문 ↔ 특약 소멸 문단)",
    "v3-offline-0508": "v4 삭제 + v3 gold 오매핑(항암약물치료 질문 ↔ 법정상속인·산출기초율 문단)",
}

gold_v4 = {g["qid"]: g for g in (json.loads(l) for l in open(HERE / "out/gold_v4_train60.jsonl", encoding="utf-8"))}
gold_v3 = {g["qid"]: g for g in (json.loads(l) for l in open(FS / "out/gold_spans_lsh_train.jsonl", encoding="utf-8"))}
U = Units(FS / "out/elements_u2jo.jsonl")

def rescore(results_path, golds, exclude=()):
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    per, out = [], {}
    for r in rows:
        if r["qid"] in exclude:
            continue
        g = golds[r["qid"]]
        ranked = U.resolve(r["submitted"])
        sc = score(ranked, g["groups"], ks=(1, 5, 10, 20))
        per.append({"qid": r["qid"], "core": r["core"], **sc})
    n = len(per)
    agg = {k: round(sum(p[k] for p in per) / n, 4) for k in ("R@1", "R@5", "R@10", "S@5", "suff@5", "suff@10", "RR@10")}
    agg["n"] = n
    return agg, per

def main():
    print(f"{'arm':26s} {'v3전체60':>10s} {'v4교정58':>10s} {'suff@10(v4)':>12s}")
    detail = {}
    for d in sorted(glob.glob(str(HERE / "out/agent/local_*60"))):
        rf = os.path.join(d, "results.jsonl")
        if not os.path.exists(rf):
            continue
        name = os.path.basename(d)
        a_v3, _ = rescore(rf, gold_v3)
        a_v4, per = rescore(rf, gold_v4, exclude=EXCLUDE)
        detail[name] = {"v3_full": a_v3, "v4_clean": a_v4, "per_q": per}
        print(f"{name:26s} {a_v3['R@5']:>10.4f} {a_v4['R@5']:>10.4f} {a_v4['suff@10']:>12.4f}")
    json.dump({"exclude": EXCLUDE, "arms": detail}, open(HERE / "out/rescore_v4.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 최고 arm 의 잔여 실패 문항
    best = max(detail, key=lambda k: detail[k]["v4_clean"]["R@5"])
    print("\n최고 arm:", best, detail[best]["v4_clean"])
    print("잔여 실패(R@5<1):")
    for p in detail[best]["per_q"]:
        if p["R@5"] < 1.0:
            print(" ", p["qid"], round(p["R@5"], 2))

if __name__ == "__main__":
    main()
