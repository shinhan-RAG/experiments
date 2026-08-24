# -*- coding: utf-8 -*-
"""
채점기 v6 — 정답기준 명세 v1 (2026-08-18) 구현.

§2  채점은 grade == "required" 그룹만 사용. supporting 은 부지표로만 집계.
§3  히트 = 제출 유닛의 문자 구간이 그룹의 인정 위치(members) 중 하나와 겹침.
    인정 위치 = 등록 위치 + 동일문구 출현 확장(lenient gold 에 미리 반영됨).
    퍼지 매칭 없음.
§4  주지표 sufficient@10 — 모든 required 그룹이 top-10 안에서 회수(OR 은 1개면 충족).
    부지표 S@5, 진단용 required 회수율(R@k), supporting 커버리지.
§7  hit / score 의사코드 그대로.

기존 filesearch/scoring.py 와 호환: groups[].members[].c0/c1 스키마 동일.
"""
from __future__ import annotations

PRIMARY_K = 10          # 명세 §4 — top-k 고정
DEFAULT_KS = (1, 5, 10, 20, 40)


def overlaps(unit, group):
    """명세 §3/§7 — unit(char_start,char_end) 이 group 의 인정 위치와 겹치는가."""
    a, b = unit["char_start"], unit["char_end"]
    for m in (group.get("members") or [group]):
        if a < m["c1"] and b > m["c0"]:
            return True
    return False


def required_groups(rec):
    return [g for g in rec.get("groups", []) if g.get("grade", "required") == "required"]


def supporting_groups(rec):
    return [g for g in rec.get("groups", []) if g.get("grade") == "supporting"]


def score(ranked_units, rec, ks=DEFAULT_KS, primary_k=PRIMARY_K):
    """
    ranked_units: 순위 순서의 [{"char_start":int,"char_end":int}, ...]
    rec:          gold 레코드 1건 (groups 포함)
    반환:         지표 dict. required 그룹이 0개면 None (채점 모집단에서 제외).
    """
    req = required_groups(rec)
    if not req:
        return None

    first_hit = {}                       # gi -> 1-based rank
    for r, u in enumerate(ranked_units):
        for gi, g in enumerate(req):
            if gi not in first_hit and overlaps(u, g):
                first_hit[gi] = r + 1

    out = {"n_required": len(req)}
    for k in ks:
        n = sum(1 for gi in first_hit if first_hit[gi] <= k)
        out["R@%d" % k] = n / len(req)                       # 명세: 비율 부분점수
        out["S@%d" % k] = 1.0 if n else 0.0                  # 첫 히트가 top-k 안
        out["suff@%d" % k] = 1.0 if len(first_hit) == len(req) and max(
            first_hit.values()) <= k else 0.0

    # 주지표
    out["sufficient@%d" % primary_k] = out["suff@%d" % primary_k]
    out["primary"] = out["suff@%d" % primary_k]

    fr = min(first_hit.values()) if first_hit else None
    out["RR@10"] = 1.0 / fr if fr and fr <= 10 else 0.0

    sup = supporting_groups(rec)
    if sup:
        cov = sum(1 for g in sup
                  if any(overlaps(u, g) for u in ranked_units[:primary_k]))
        out["supporting_cov"] = cov / len(sup)
    out["reached"] = 1.0 if len(first_hit) == len(req) else 0.0   # 순위 무한대 도달
    return out


def aggregate(per_q):
    """문항별 dict 리스트 -> 매크로 평균. n 을 반드시 함께 반환한다(명세 §5)."""
    per_q = [d for d in per_q if d]
    if not per_q:
        return {"n": 0}
    keys = [k for k in per_q[0] if k not in ("n_required",)]
    agg = {"n": len(per_q)}
    for k in keys:
        vals = [d[k] for d in per_q if k in d]
        if vals:
            agg[k] = sum(vals) / len(vals)
    agg["n_required_total"] = sum(d["n_required"] for d in per_q)
    return agg
