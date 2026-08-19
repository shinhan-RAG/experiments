#!/usr/bin/env python3
"""Slot FileSearch 재구현 — v2ds 최종 스펙(MANIFEST 2026-08-12 "검색_3_3") + 기존 AND 대조.

CLM(coordination-level matching):
  후보 = 슬롯 합집합(어느 한 슬롯이라도 매치) ∪ 어휘 채널(질문 토큰이 본문에 존재)
  점수 = 맞은 조건 수 (슬롯당 1, 어휘 채널은 --lex binary(1) 또는 count(토큰 수))
  정렬 = 점수 내림차순 → 문서순(line_start, line_end, element_id)
AND(기존 slot_filesearch): 요청 슬롯 전부 매치 → 문서순.
금지: BM25 · IDF · 임베딩 · reranker · 학습 정렬. 매치는 어떤 항이든 정확히 1점.

질의 라우터(규칙, LLM 0회): contract(특약명 매치) / role(ROLE_RULES) / subject(질문 명사구·도메인어) / qualifier(값) / schema(표·산식 언급).
LLM qtags 는 --qtags 로 덧씌움(qid → slots). 필드 매치 규칙(fuzzy_contains)은 hybrid-enrich/slot_filesearch.py 를 그대로 재사용.
"""
import argparse, collections, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from slot_filesearch import fuzzy_contains, compact, contract_core, ROLE_ALIASES  # noqa: E402
from build_element_fields_v4 import (QUOTED_RE, SUBJECT_RE, DEFINITION_RE, VALUE_RE, ROLE_RULES,  # noqa: E402
                                     clean_subjects, unique)

FIELDS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")
ACTIVE = ("contract", "subject", "role", "qualifier", "schema")
TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?")
STOP = set("알려줘 알려주세요 어떻게 무엇 무슨 어떤 경우 대해 대한 있나요 있어 있는지 되나요 되는지 인가요 인지 뭐야 뭔가요 해줘 해주세요 궁금 관련 그리고 또는 정리 설명 확인 부탁 조건 내용 기준 방법 여부 가능 해당 이후 이전 때 것 수 등 및 의 이 가 을 를 은 는 에 에서 로 으로 와 과 도 만 까지 부터 처럼 이란 라는 하는 되는 있는 없는 통합건강 통합건강보장 원 one ONE 보험 특약 주계약 약관 상품".split())


def canonical(t):
    if "semantic_role" in t:  # OLD 스키마 (slot_filesearch.canonical_old 와 동일 사상 + 역할코드 동치어 병기)
        OLD_ROLE_CODES = {"면책사유(보장 제외)": "exclusion_exception", "보험금 지급사유(보장 내용)": "payment_trigger",
                          "보험금 지급기준·지급한도": "limit_frequency payment_amount", "지급 세부규정": "criteria_rule",
                          "용어 정의": "definition", "갱신 조건": "contract_lifecycle", "보험료 납입면제": "premium_waiver",
                          "보험기간·보장개시": "timing_period", "해지·해약환급금": "contract_lifecycle",
                          "청약·철회": "contract_lifecycle", "계약 성립·무효": "contract_lifecycle", "보험료 납입·부활": "contract_lifecycle"}
        r = t.get("semantic_role", "")
        return {
            "contract": [t.get("contract_scope", ""), t.get("topic", "")],
            "subject": (t.get("aliases") or []) + [t.get("table_title", ""), t.get("formula_subject", "")],
            "role": [r, OLD_ROLE_CODES.get(r, "")],
            "article": [t.get("article", "")],
            "table": t.get("table_headers") or [],
            "qualifier": (t.get("values") or []) + (t.get("conditions") or []),
            "reference": [],
            "schema": [t.get("element_type", "")],
        }
    loc = t.get("locator") or {}
    roles = t.get("role") or []
    return {
        "contract": [t.get("contract_key", "")],
        "subject": t.get("subject_key") or [],
        "role": roles + [ROLE_ALIASES.get(r, r) for r in roles],
        "article": [loc.get("article", ""), loc.get("article_title", ""), loc.get("section", "")],
        "table": (loc.get("table_headers") or []) + (loc.get("row_keys") or []),
        "qualifier": t.get("qualifier") or [],
        "reference": t.get("reference") or [],
        "schema": [t.get("schema_tag", "")],
    }


class Router:
    """규칙 질의 라우터. 특약 사전은 태그의 contract_key 집합에서 만든다(QA 미참조)."""

    def __init__(self, contracts):
        self.contracts = sorted({c for c in contracts if c}, key=len, reverse=True)
        # core = 정규화 핵심명, core_nb = 대괄호 수식어([기본]·[3~100%장해형]…) 제거 후 정규화
        self.cores = [(c, contract_core(c), contract_core(re.sub(r"\[.*?\]", "", c))) for c in self.contracts]
        self.partial = True  # 부분 일치(공통 부분문자열) 허용

    @staticmethod
    def _lcs(a, b):
        best = 0
        for i in range(len(a)):
            for j in range(len(b)):
                k = 0
                while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                    k += 1
                if k > best:
                    best = k
        return best

    def route(self, q):
        cq = compact(q)
        slots = collections.defaultdict(list)
        partial_hits = []
        for c, core, core_nb in self.cores:
            if not core:
                continue
            if (len(core) >= 3 and core in cq) or (len(core_nb) >= 3 and core_nb in cq):
                slots["contract"].append(c)
            elif self.partial and len(core_nb) >= 4:
                l = self._lcs(core_nb, cq)
                if l >= 6 or (l >= 4 and l / len(core_nb) >= 0.6):
                    partial_hits.append((l / len(core_nb), c))
        if not slots["contract"] and partial_hits:
            top = max(x[0] for x in partial_hits)
            slots["contract"] = [c for r, c in partial_hits if r >= top - 1e-9]
            slots["_conf"] = {"contract": 0.5}  # 부분 일치 신뢰도(가중 절반)
        if "주계약" in q or "주보험" in q:
            slots["contract"] += [c for c in self.contracts if c.startswith("(간편)신한통합건강보장보험") or c.startswith("신한(간편가입)")]
        for pat, role in ROLE_RULES:
            if re.search(pat, q):
                slots["role"].append(role)
        subj = clean_subjects(QUOTED_RE.findall(q) + DEFINITION_RE.findall(q) + SUBJECT_RE.findall(q), 10)
        slots["subject"] = subj
        slots["qualifier"] = unique(VALUE_RE.findall(q), 10)
        if re.search(r"(?<![가-힣])표(?![가-힣적준])|분류표|별표|부표|테이블", q):
            slots["schema"].append("table")
        if re.search(r"산식|계산식|수식", q):
            slots["schema"].append("formula")
        toks = [t for t in TOKEN_RE.findall(q) if t not in STOP and not re.fullmatch(r"\d+", t)]
        return {k: v for k, v in slots.items() if v}, unique(toks, 30)


class SlotSearch:
    def __init__(self, elements_path, tags_path):
        self.E = [json.loads(l) for l in open(elements_path, encoding="utf-8")]
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(tags_path, encoding="utf-8")}
        self.rows = [canonical(T[e["element_id"]]) for e in self.E]
        self.body = [compact(e["text"]) for e in self.E]
        self.router = Router(r["contract"][0] for r in self.rows)

    @staticmethod
    def field_match(row, field, requested):
        vals = [str(v) for v in row.get(field) or [] if str(v).strip()]
        return any(fuzzy_contains(q, v, field) for q in requested for v in vals)

    def match_table(self, slots, tokens):
        """질의 1건에 대한 element별 슬롯 매치·어휘 토큰 수를 한 번만 계산(arm 간 재사용)."""
        req = {f: v for f, v in slots.items() if f in FIELDS and v}
        M = [{f: self.field_match(row, f, v) for f, v in req.items()} for row in self.rows]
        L = [sum(1 for t in tokens if compact(t) in b) for b in self.body] if tokens else [0] * len(self.rows)
        return M, L

    def rank(self, M, L, mode="clm", lex="binary", weights=None, limit=50, scope_filter=None):
        weights = weights or {}
        scored = []
        for i, m in enumerate(M):
            if scope_filter is not None and self.E[i]["contract_scope"] not in scope_filter:
                continue
            if mode == "and":
                if m and not all(m.values()):
                    continue
                scored.append((0.0, i)); continue
            score = sum(weights.get(f, 1.0) for f, hit in m.items() if hit)
            if L[i]:
                score += 1.0 if lex == "binary" else L[i]
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], self.E[x[1]]["line_start"], self.E[x[1]]["line_end"], self.E[x[1]]["element_id"]))
        return [(self.E[i], s) for s, i in scored[:limit]]

    def search(self, slots, tokens, mode="clm", lex="binary", weights=None, limit=50):
        M, L = self.match_table(slots, tokens)
        return self.rank(M, L, mode, lex, weights, limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default=str(HERE / "out" / "elements_u2.jsonl"))
    ap.add_argument("--tags", default=str(HERE / "out" / "tags_u2_rules.jsonl"))
    ap.add_argument("--q", required=True)
    ap.add_argument("--mode", default="clm", choices=("clm", "and"))
    a = ap.parse_args()
    S = SlotSearch(a.elements, a.tags)
    slots, toks = S.router.route(a.q)
    print(json.dumps({"slots": slots, "tokens": toks}, ensure_ascii=False))
    for e, s in S.search(slots, toks, a.mode)[:10]:
        print(s, e["element_id"], e["line_start"], e["contract_scope"][:30], e["text"][:80].replace("\n", " "))
