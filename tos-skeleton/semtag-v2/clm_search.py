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
sys.path.insert(0, str(HERE.parent / "hybrid-enrich"))
from slot_filesearch import fuzzy_contains, compact, contract_core, ROLE_ALIASES  # noqa: E402
from build_element_fields_v4 import (QUOTED_RE, SUBJECT_RE, DEFINITION_RE, VALUE_RE, ROLE_RULES,  # noqa: E402
                                     clean_subjects, unique)

FIELDS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")
ACTIVE = ("contract", "subject", "role", "qualifier", "schema")
TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?")
STOP = set("알려줘 알려주세요 어떻게 무엇 무슨 어떤 경우 대해 대한 있나요 있어 있는지 되나요 되는지 인가요 인지 뭐야 뭔가요 해줘 해주세요 궁금 관련 그리고 또는 정리 설명 확인 부탁 조건 내용 기준 방법 여부 가능 해당 이후 이전 때 것 수 등 및 의 이 가 을 를 은 는 에 에서 로 으로 와 과 도 만 까지 부터 처럼 이란 라는 하는 되는 있는 없는 통합건강 통합건강보장 원 one ONE 보험 특약 주계약 약관 상품".split())


def canonical(t):
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
        self.cores = [(c, contract_core(c)) for c in self.contracts]

    def route(self, q):
        cq = compact(q)
        slots = collections.defaultdict(list)
        for c, core in self.cores:
            if not core:
                continue
            core_nb = re.sub(r"\[.*?\]", "", core)  # [기본]·[3~100%장해형] 같은 대괄호 수식어 제거본
            if (len(core) >= 3 and core in cq) or (len(core_nb) >= 3 and core_nb in cq):
                slots["contract"].append(c)
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

    def search(self, slots, tokens, mode="clm", lex="binary", weights=None, limit=50):
        weights = weights or {}
        scored = []
        req = {f: v for f, v in slots.items() if f in FIELDS and v}
        for i, row in enumerate(self.rows):
            m = {f: self.field_match(row, f, v) for f, v in req.items()}
            if mode == "and":
                if req and not all(m.values()):
                    continue
                scored.append((0.0, i))
                continue
            score = sum(weights.get(f, 1.0) for f, hit in m.items() if hit)
            if tokens:
                lt = sum(1 for t in tokens if compact(t) in self.body[i])
                score += (1.0 if lt else 0.0) if lex == "binary" else lt
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], self.E[x[1]]["line_start"], self.E[x[1]]["line_end"], self.E[x[1]]["element_id"]))
        return [(self.E[i], s) for s, i in scored[:limit]]


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
