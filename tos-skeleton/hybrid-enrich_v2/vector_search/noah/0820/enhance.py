#!/usr/bin/env python3
"""0819 검색 툴 고도화 공통 로직 — agent_tools.py 와 eval_det.py 가 같이 쓴다.

1) alias 질의 확장: filesearch/aliases.json 양방향 사전 → 어휘(lex) 채널 토큰에만 union 추가.
   슬롯 매칭은 원문 유지(태그 채널 강화는 단조 손해 — 0818).
2) scope soft-descent: --scope "<특약>[/<관>[/<조>]]" 매치 element 에 가산 부스트(필터 아님).
   hard descent(필터) 는 0818 전례에서 패배 — 반드시 부스트로만.
3) browse: --q 없이 --scope 만 주어진 search 의 축퇴 모드. 계층 하위 목록 반환.
4) 참조 그래프: 조 본문의 "제N조" 참조 → 같은 특약의 조 id (build_refs.py 가 사전계산).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from textmatch import compact, fuzzy_contains  # noqa: E402

REFS_PATH = HERE / "out" / "refs_jo.json"

# ---------------------------------------------------------------- alias 확장
_ALIAS = None


def alias_pairs() -> dict:
    """{compact(어): [동치어 원문…]} 양방향 사전."""
    global _ALIAS
    if _ALIAS is None:
        d = json.load(open(FS / "aliases.json", encoding="utf-8"))
        m: dict = {}
        for head, alts in d.items():
            if head.startswith("_"):
                continue
            group = [head] + list(alts)
            for g in group:
                key = compact(g)
                if key:
                    m.setdefault(key, [])
                    m[key].extend(x for x in group if x != g)
        _ALIAS = {k: list(dict.fromkeys(v)) for k, v in m.items()}
    return _ALIAS


def expand_query(q: str, toks: list) -> tuple[list, dict]:
    """질문에 등장한 사전 표제어의 동치어를 lex 토큰으로 추가. (추가 토큰, 로그) 반환."""
    cq = compact(q)
    have = {compact(t) for t in toks}
    extra, logm = [], {}
    for key, alts in alias_pairs().items():
        if key and key in cq:
            add = [a for a in alts if compact(a) not in cq and compact(a) not in have]
            if add:
                logm[key] = add
                extra.extend(add)
    return list(dict.fromkeys(extra)), logm


# ---------------------------------------------------------------- scope soft-descent
def parse_scope(scope: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in (scope or "").split("/") if p.strip()]
    parts += ["", "", ""]
    return parts[0], parts[1], parts[2]


def _mid_match(mid: str, section: str) -> bool:
    if not section:
        return False
    cm, cs = compact(mid), compact(section)
    return bool(cm) and bool(cs) and (cm in cs or cs in cm)


def scope_match(S, i: int, big: str, mid: str, small: str) -> bool:
    art = S.rows[i].get("article") or ["", "", ""]
    if big and not fuzzy_contains(big, S.E[i].get("contract_scope", ""), "contract"):
        return False
    if mid and not _mid_match(mid, art[2] if len(art) > 2 else ""):
        return False
    if small:
        t = (art[0] or "") + " " + (art[1] or "")
        if compact(small) not in compact(t):
            return False
    return True


def apply_scope_boost(res, S, eidx: dict, scope: str, boost: float):
    """rank() 결과 [(element, score)] 에 scope 매치 가산 부스트 후 재정렬."""
    big, mid, small = parse_scope(scope)
    if not big:
        return res
    out = []
    for e, sc in res:
        if scope_match(S, eidx[e["element_id"]], big, mid, small):
            sc = sc + boost
        out.append((e, sc))
    out.sort(key=lambda x: (-x[1], x[0]["line_start"], x[0]["line_end"], x[0]["element_id"]))
    return out


# ---------------------------------------------------------------- browse
def browse(S, scope: str, cap: int = 40) -> dict:
    """--q 없는 search: 계층 하위 목록. 특약(대) → 관(중) → 조(소)."""
    big, mid, _ = parse_scope(scope)
    if not big:
        cnt, order = {}, []
        for e in S.E:
            c = e["contract_scope"]
            if c not in cnt:
                order.append(c)
            cnt[c] = cnt.get(c, 0) + 1
        return {"level": "특약", "items": [{"name": c, "n_elements": cnt[c]} for c in order[:cap]]}
    idxs = [i for i, e in enumerate(S.E) if fuzzy_contains(big, e["contract_scope"], "contract")]
    if not idxs:
        return {"level": "특약", "error": "일치하는 특약이 없습니다. --scope 없이 browse 해 정확한 특약명을 확인하십시오.", "items": []}
    contract = S.E[idxs[0]]["contract_scope"]
    if not mid:
        cnt, order = {}, []
        for i in idxs:
            art = S.rows[i].get("article") or ["", "", ""]
            s = (art[2] if len(art) > 2 else "") or "(관 없음)"
            if s not in cnt:
                order.append(s)
            cnt[s] = cnt.get(s, 0) + 1
        return {"level": "관", "contract": contract,
                "items": [{"name": s, "n_elements": cnt[s]} for s in order[:cap]]}
    cnt, order, titles = {}, [], {}
    for i in idxs:
        art = S.rows[i].get("article") or ["", "", ""]
        if not _mid_match(mid, art[2] if len(art) > 2 else ""):
            continue
        a = art[0] or "(조 없음)"
        if a not in cnt:
            order.append(a)
            titles[a] = art[1] if len(art) > 1 else ""
        cnt[a] = cnt.get(a, 0) + 1
    return {"level": "조", "contract": contract,
            "items": [{"article": a, "title": titles[a], "n_elements": cnt[a]} for a in order[:cap]]}


# ---------------------------------------------------------------- 참조 그래프
_JO_REF = re.compile(r"제\s?(\d+(?:-\d+)?)\s?조(?:\s?의\s?(\d+))?")
_refs_cache = None


def build_refs(J: list, cap: int = 4) -> dict:
    """조 본문에서 같은 특약의 다른 조 참조를 추출. {jo_id: [jo_id…]}"""
    idx = {}
    for u in J:
        m = _JO_REF.search(u.get("title") or "")
        if m:
            idx.setdefault((u.get("contract_scope", ""), m.group(1), m.group(2) or ""), u["element_id"])
    refs = {}
    for u in J:
        me = _JO_REF.search(u.get("title") or "")
        mekey = (me.group(1), me.group(2) or "") if me else None
        seen, out = set(), []
        for m in _JO_REF.finditer(u.get("text") or ""):
            key = (m.group(1), m.group(2) or "")
            if key == mekey:
                continue
            jid = idx.get((u.get("contract_scope", ""), key[0], key[1]))
            if jid and jid != u["element_id"] and jid not in seen:
                seen.add(jid)
                out.append(jid)
            if len(out) >= cap:
                break
        if out:
            refs[u["element_id"]] = out
    return refs


def load_refs() -> dict:
    global _refs_cache
    if _refs_cache is None:
        _refs_cache = json.load(open(REFS_PATH, encoding="utf-8")) if REFS_PATH.exists() else {}
    return _refs_cache
