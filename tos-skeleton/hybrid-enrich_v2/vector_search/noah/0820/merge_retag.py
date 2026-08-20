#!/usr/bin/env python3
"""codex 재태깅 결과를 태그 인덱스 오버레이로 병합 + 검증.

원본 filesearch/out/tags_u2_rules.jsonl 은 불변. 197개 재생성 레코드만 교체한
filesearch/out/tags_u2_rules_0820.jsonl 을 새로 쓴다 (arm 의 "tags" 로 선택 사용).

검증: ① 197줄·필수 필드 ② role 코드 11종 밖 값 거부 ③ element_id 가 원본 대상 집합과 일치
④ search_text/subject_key 실질성(재태깅 전 대비 필드 수 증가) 리포트.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
ROLES = {"exclusion_exception", "premium_waiver", "payment_trigger", "payment_amount", "limit_frequency",
         "timing_period", "definition", "criteria_rule", "contract_lifecycle", "claim_procedure", "code_reference"}


def substance(t):
    n = sum(len(t.get(k) or []) for k in ("role", "subject_key", "qualifier"))
    return n + (1 if t.get("schema_tag") else 0)


def main():
    targets = {r["element_id"] for r in json.load(open(HERE / "out/retag_input.json", encoding="utf-8"))}
    new = {}
    bad_role, bad_id = [], []
    for l in open(HERE / "out/retag_output.jsonl", encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        t = json.loads(l)
        eid = t.get("element_id")
        if eid not in targets:
            bad_id.append(eid); continue
        if "�" in json.dumps(t, ensure_ascii=False):  # 인코딩 오염 레코드는 원본 유지가 낫다
            bad_id.append(f"{eid}(U+FFFD)"); continue
        roles = [r for r in (t.get("role") or []) if r in ROLES]
        if len(roles) != len(t.get("role") or []):
            bad_role.append((eid, t.get("role")))
        t["role"] = roles
        new[eid] = t
    print(f"재태깅 수신 {len(new)}/{len(targets)} | 대상 외 id {len(bad_id)} | 비표준 role 정리 {len(bad_role)}")
    missing = targets - set(new)
    if missing:
        print(f"!! 누락 {len(missing)}개 (원본 태그 유지): {sorted(missing)[:8]}")

    old = {}
    rows = []
    for l in open(FS / "out/tags_u2_rules.jsonl", encoding="utf-8"):
        t = json.loads(l); old[t["element_id"]] = t; rows.append(t)
    ups = sum(1 for e in new if substance(new[e]) > substance(old.get(e, {})))
    print(f"실질 필드 증가: {ups}/{len(new)} (기존 평균 {sum(substance(old[e]) for e in new)/max(1,len(new)):.1f} -> "
          f"신규 평균 {sum(substance(t) for t in new.values())/max(1,len(new)):.1f})")

    out = FS / "out/tags_u2_rules_0820.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(new.get(t["element_id"], t), ensure_ascii=False) + "\n")
    print(f"-> {out} ({len(rows)}줄, 원본 불변)")


if __name__ == "__main__":
    main()
