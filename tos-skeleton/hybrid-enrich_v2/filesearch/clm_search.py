#!/usr/bin/env python3
"""Slot FileSearch 재구현 — v2ds 최종 스펙(MANIFEST 2026-08-12 "검색_3_3") + 기존 AND 대조.

CLM(coordination-level matching):
  후보 = 슬롯 합집합(어느 한 슬롯이라도 매치) ∪ 어휘 채널(질문 토큰이 본문에 존재)
  점수 = 맞은 조건 수 (슬롯당 1, 어휘 채널은 --lex binary(1) 또는 count(토큰 수))
  정렬 = 점수 내림차순 → 문서순(line_start, line_end, element_id)
AND(기존 slot_filesearch): 요청 슬롯 전부 매치 → 문서순.
금지: BM25 · IDF · 임베딩 · reranker · 학습 정렬. 매치는 어떤 항이든 정확히 1점.
위 금지는 CLM 재현 경로의 불변식이다. 실험 arm ``ranker=bm25f``는 별도 모듈에서만
지연 로드되며 CLM 기본 동작과 점수에는 영향을 주지 않는다.

질의 라우터(규칙, LLM 0회): contract(특약명 매치) / role(ROLE_RULES) / subject(질문 명사구·도메인어) / qualifier(값) / schema(표·산식 언급).
LLM qtags 는 --qtags 로 덧씌움(qid → slots). 필드 매치 규칙(fuzzy_contains)은 hybrid-enrich/slot_filesearch.py 를 그대로 재사용.
"""
import argparse, collections, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textmatch import fuzzy_contains, compact, contract_core, ROLE_ALIASES  # noqa: E402
from patterns import (QUOTED_RE, SUBJECT_RE, DEFINITION_RE, VALUE_RE, ROLE_RULES,  # noqa: E402
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
        self.rev = {}        # 대상어(급여금명·질병명 등) → 특약 집합. SlotSearch 가 태그에서 채운다
        self.rev_max = 3     # 이 수 이하의 특약에만 나오는 대상어만 특약 추론에 사용

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
        # 별칭 정규화: 질문의 구어("보상제외기간")를 문서 표기("면책기간")로 치환한 사본도 매칭에 사용
        if getattr(self, "alias_map", None):
            for alt, canon in self.alias_map.items():
                if alt in cq:
                    cq = cq + canon  # 치환이 아니라 병기(원 표현 보존)
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
        if not slots["contract"] and self.rev:
            # 급여금명·질병명 → 특약 역색인(태그 subject 에서 생성). 질문에 그 대상어가 있고 특약 후보가 rev_max 이하이면 특약 슬롯 추론
            hits = collections.Counter()
            for term, cs in self.rev.items():
                if len(cs) <= self.rev_max and term in cq:
                    for c in cs:
                        hits[c] += 1
            if hits:
                top = max(hits.values())
                slots["contract"] = [c for c, n in hits.items() if n == top]
                slots["_conf"] = {"contract": 0.75}
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

    @staticmethod
    def context_backtrack(q, slots, tokens):
        """Return an identity-free intent query only when the routed identity is contextual.

        The gate uses document relations and query form, not product or disease names.  It is
        intentionally conservative because unconditional identity relaxation already failed the
        agent Recall gate.
        """
        identities = list(slots.get("contract", [])) + list(slots.get("identity", []))
        if not identities:
            return None
        reasons = []
        if len(identities) >= 3:
            reasons.append("identity_overload")
        if re.search(r"주계약.*특약|특약.*주계약", q):
            reasons.append("cross_document_relation")
        if re.search(r"(?:보험|상품)(?:으로|에서|에\s*가입|을\s*가입|를\s*가입)", q):
            reasons.append("collection_context")
        if "등" in q and len(re.findall(r"[,/·]", q)) >= 1:
            reasons.append("enumerated_topics")
        if not reasons:
            return None

        identity_text = [compact(value) for value in identities]
        intent_tokens = [token for token in tokens
                         if not any(compact(token) and compact(token) in value
                                    for value in identity_text)]
        intent_slots = {key: list(value) for key, value in slots.items()
                        if key not in {"contract", "identity", "role", "function", "_conf"}}
        for key in ("subject", "topic"):
            intent_slots[key] = [value for value in intent_slots.get(key, [])
                                 if not any(compact(value) in identity for identity in identity_text)]
            if not intent_slots[key]:
                intent_slots.pop(key, None)
        intent_text = " ".join(dict.fromkeys(intent_tokens + [
            str(value) for key in ("subject", "topic", "qualifier", "constraint", "schema", "structure")
            for value in intent_slots.get(key, [])
        ]))
        roles = []
        for pattern, role in ROLE_RULES:
            if re.search(pattern, intent_text):
                # Colloquial "refund" is often a benefit-coverage question.  Lifecycle intent
                # requires an explicit lifecycle noun once product-form text has been removed.
                if role == "contract_lifecycle" and not re.search(
                        r"갱신|해지|소멸|무효|해약|환급금", intent_text):
                    continue
                roles.append(role)
        if roles:
            intent_slots["role"] = unique(roles)
        return {"query": intent_text or q, "slots": intent_slots,
                "tokens": unique(intent_tokens, 30), "reasons": reasons}


class SlotSearch:
    def __init__(self, elements_path, tags_path):
        self.E = [json.loads(l) for l in open(elements_path, encoding="utf-8")]
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(tags_path, encoding="utf-8")}
        self.tags = [T[e["element_id"]] for e in self.E]
        self.rows = [canonical(t) for t in self.tags]
        self.body = [compact(e["text"]) for e in self.E]
        self.router = Router(r["contract"][0] for r in self.rows)
        try:
            _al = json.load(open(Path(__file__).resolve().parent / "aliases.json", encoding="utf-8")); _al.pop("_comment", None)
            self.router.alias_map = {compact(a): compact(c) for c, alts in _al.items() for a in alts if len(compact(a)) >= 3}
        except Exception:
            self.router.alias_map = {}
        rev = collections.defaultdict(set)
        for row in self.rows:
            c = row["contract"][0]
            for v in row["subject"]:
                k = compact(v)
                if len(k) >= 4:
                    rev[k].add(c)
        self.router.rev = {k: cs for k, cs in rev.items() if len(cs) <= self.router.rev_max}

    def ensure_structured(self):
        """BM25F arm에서만 구조 인덱스를 지연 생성한다. 기존 CLM 경로는 무변경."""
        if not hasattr(self, "_structured"):
            from structured_search import StructuredTagIndex
            self._structured = StructuredTagIndex(
                self.E, self.tags, Path(__file__).resolve().parent / "aliases.json")
        return self._structured

    def rank_structured(self, slots, tokens, raw_query, weights=None, profile="full", limit=200,
                        lexical_counts=None):
        """Semantic Tag 필드 단일 단계 BM25F 검색. 반환 형식은 rank()와 동일."""
        if lexical_counts is None:
            _, lexical_counts = self.match_table(slots, tokens)
        idx = self.ensure_structured()
        return [(self.E[i], score) for i, score in idx.rank(
            raw_query, slots, lexical_counts, weights=weights, profile=profile, limit=limit)]

    def rank_structured_portfolio(self, slots, tokens, raw_query, weights=None, profile="core",
                                  mode="all", limit=400, window=40, seed_quota=20,
                                  lexical_counts=None, process_query=None, rrf_k=60,
                                  coverage_seed=3, coverage_per_role=1):
        """기존 순위 일부를 보존하면서 규칙 후속질의를 첫 페이지에 교차 노출한다."""
        if lexical_counts is None:
            lexical_counts = self.lexical_counts(tokens)
        idx = self.ensure_structured()
        if mode == "context_backtrack":
            seed = idx.rank(raw_query, slots, lexical_counts, weights=weights,
                            profile=profile, limit=max(limit, 100))
            plan = self.router.context_backtrack(process_query or raw_query, slots, tokens)
            if not plan:
                self._last_portfolio = {"mode": mode, "window": window,
                                        "seed_quota": seed_quota, "specs": [], "trace": []}
                return [(self.E[i], score) for i, score in seed[:limit]]
            relaxed = idx.rank(plan["query"], plan["slots"],
                               self.lexical_counts(plan["tokens"]), weights=weights,
                               profile=profile, limit=max(window, 100))
            merged, seen, trace = [], set(), []

            def push(item, phase):
                i, score_ = item
                if i in seen:
                    return False
                seen.add(i); merged.append(item)
                trace.append({"index": i, "phase": phase,
                              "reason": ",".join(plan["reasons"])})
                return True

            for item in seed[:min(seed_quota, window, limit)]:
                push(item, "seed")
            for item in relaxed:
                if len(merged) >= min(window, limit):
                    break
                push(item, "context_backtrack")
            for item in seed:
                if len(merged) >= limit:
                    break
                push(item, "seed_tail")
            self._last_portfolio = {
                "mode": mode, "window": window, "seed_quota": seed_quota,
                "specs": [{"phase": "context_backtrack", "query": plan["query"],
                           "reason": ",".join(plan["reasons"])}], "trace": trace}
            return [(self.E[i], score) for i, score in merged]
        ranked = idx.rank_portfolio(raw_query, slots, tokens, lexical_counts, weights=weights,
                                    profile=profile, mode=mode, limit=limit,
                                    window=window, seed_quota=seed_quota,
                                    process_query=process_query, rrf_k=rrf_k,
                                    coverage_seed=coverage_seed,
                                    coverage_per_role=coverage_per_role)
        self._last_portfolio = idx.last_portfolio
        return [(self.E[i], score) for i, score in ranked]

    @staticmethod
    def field_match(row, field, requested):
        vals = [str(v) for v in row.get(field) or [] if str(v).strip()]
        return any(fuzzy_contains(q, v, field) for q in requested for v in vals)

    def match_table(self, slots, tokens):
        """질의 1건에 대한 element별 슬롯 매치·어휘 토큰 수를 한 번만 계산(arm 간 재사용)."""
        req = {f: v for f, v in slots.items() if f in FIELDS and v}
        M = [{f: self.field_match(row, f, v) for f, v in req.items()} for row in self.rows]
        L = self.lexical_counts(tokens)
        return M, L

    def lexical_counts(self, tokens):
        """원문 어휘 채널만 계산한다(BM25F-only 평가에서 fuzzy 슬롯 표 생략용)."""
        return [sum(1 for t in tokens if compact(t) in b) for b in self.body] if tokens else [0] * len(self.rows)

    def hybrid_fallback_gate(self, slots, ranked, top_k=5, min_axis_coverage=0.5):
        """후보 *개수*가 아니라 상위 결과의 구조 축 충족도로 메타 폴백을 판정한다.

        질의별 BM25F 원점수 임계값은 질의 길이·필드 수에 민감하고, 결과 개수는 상한
        400으로 포화된다. 그래서 라우터가 실제 요청한 범용 축이 top-k에서 한 번이라도
        충족됐는지만 사용한다. 상품/질병 전용 어휘나 Gold label은 사용하지 않는다.
        """
        requested = {
            "identity": list(slots.get("contract", [])) + list(slots.get("identity", [])),
            "topic": list(slots.get("subject", [])) + list(slots.get("topic", [])),
            "function": list(slots.get("role", [])) + list(slots.get("function", [])),
            "constraint": list(slots.get("qualifier", [])) + list(slots.get("constraint", [])),
            "structure": list(slots.get("schema", [])) + list(slots.get("structure", [])),
        }
        requested = {axis: list(dict.fromkeys(str(v) for v in values if str(v).strip()))
                     for axis, values in requested.items() if values}
        top_rows = [self.rows[self._eidx[e["element_id"]]] for e, _ in ranked[:top_k]]
        field_map = {"identity": "contract", "topic": "subject", "function": "role",
                     "constraint": "qualifier", "structure": "schema"}
        hits = {
            axis: any(self.field_match(row, field_map[axis], values) for row in top_rows)
            for axis, values in requested.items()
        }
        coverage = sum(hits.values()) / max(1, len(hits))
        reasons = []
        if not ranked:
            reasons.append("empty")
        if requested.get("identity") and not hits.get("identity", False):
            reasons.append("identity_miss")
        if len(requested) >= 2 and coverage < min_axis_coverage:
            reasons.append("axis_coverage")
        if not requested and ranked:
            reasons.append("no_routed_axis")
        return {"fallback": bool(reasons), "reasons": reasons,
                "requested_axes": sorted(requested),
                "matched_axes": sorted(axis for axis, hit in hits.items() if hit),
                "coverage": round(coverage, 4), "top_k": top_k}

    def rank(self, M, L, mode="clm", lex="binary", weights=None, limit=50, scope_filter=None, rare=False, n_tokens=None, rare_cap=None):
        """점수 = Σ_슬롯 hit·w_f·(rare 이면 희소성 계수) + 어휘항.
        rare: 슬롯 f 의 희소성 계수 = 1 + log(N / df_f), df_f = 이 질의에서 슬롯 f 가 매치한 element 수 (설계서 2.5 희소성 검사의 질의 시점 구현).
        lex: binary(있으면 1) / count(매치 토큰 수) / cov(매치 토큰 수 / 질의 토큰 수, 0~1)
        동점 해소: (점수, 어휘 커버리지, 문서순)"""
        import math
        weights = weights or {}
        N = len(M)
        rf = {}
        if rare and M and M[0]:
            for f in M[0]:
                df = sum(1 for m in M if m.get(f))
                rf[f] = (1.0 + math.log(N / df)) if df else 0.0
                if rare_cap:
                    rf[f] = min(rf[f], rare_cap)
        scored = []
        for i, m in enumerate(M):
            if scope_filter is not None and self.E[i]["contract_scope"] not in scope_filter:
                continue
            if mode == "and":
                if m and not all(m.values()):
                    continue
                scored.append((0.0, 0.0, i)); continue
            score = sum(weights.get(f, 1.0) * (rf.get(f, 1.0) if rare else 1.0) for f, hit in m.items() if hit)
            cov = (L[i] / n_tokens) if n_tokens else 0.0
            if L[i]:
                score += 1.0 if lex == "binary" else (L[i] if lex == "count" else cov * weights.get("_lex", 1.0))
            if score > 0:
                scored.append((score, cov, i))
        scored.sort(key=lambda x: (-x[0], -x[1], self.E[x[2]]["line_start"], self.E[x[2]]["line_end"], self.E[x[2]]["element_id"]))
        if getattr(self, "variant_rr", False):
            # 동점(점수·cov 동일) 블록 안에서만 base_contract(판본 제외 특약명) 라운드로빈 — 전역 다양화 아님
            import itertools, re as _re
            def base(i):
                return _re.sub(r"\(무배당[^)]*\)", "", self.E[i]["contract_scope"]).strip()
            out = []
            for _, grp in itertools.groupby(scored, key=lambda x: (x[0], x[1])):
                grp = list(grp)
                buckets = {}
                for it in grp:
                    buckets.setdefault(base(it[2]), []).append(it)
                order = sorted(buckets)
                while any(buckets[b] for b in order):
                    for b in order:
                        if buckets[b]:
                            out.append(buckets[b].pop(0))
                if len(out) >= limit:
                    break
            scored = out
        return [(self.E[i], s) for s, _, i in scored[:limit]]

    def search(self, slots, tokens, mode="clm", lex="binary", weights=None, limit=50, rare=False):
        M, L = self.match_table(slots, tokens)
        return self.rank(M, L, mode, lex, weights, limit, rare=rare, n_tokens=len(tokens))


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
