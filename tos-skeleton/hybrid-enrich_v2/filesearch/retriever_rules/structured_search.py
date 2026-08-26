"""결정론적 Semantic Tag 1차 검색기.

별도 LLM·임베딩·reranker 없이 구조 필드 자체를 BM25F로 점수화한다.
기존 CLM은 그대로 두고 arm의 ``ranker=bm25f``에서만 사용한다.

핵심 불변식:
- 특약은 base/full/variant identity를 분리한다.
- article/table/reference처럼 생성돼 있던 휴면 필드를 검색에 포함한다.
- 필드 TF는 합산한 뒤 포화시킨다(Robertson et al., CIKM 2004 BM25F).
- 계층/판본 신호는 hard filter가 아니라 유한한 가산점만 준다.
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from schema_adapter import adapt_tag
from patterns import ROLE_KO
from textmatch import compact, contract_core


FIELD_NAMES = (
    "identity_container", "identity_full", "identity_base", "variant", "topic", "function",
    "locator", "table", "constraint", "relation", "structure", "evidence", "evidence_unit",
    "extra",
)

DEFAULT_WEIGHTS = {
    # U3와 기존 arm의 순위를 완전히 보존한다. C25처럼 새 태그를 쓰는 arm만
    # sfw에서 두 축의 유한 가중치를 명시적으로 켠다.
    "identity_container": 0.0,
    "identity_full": 2.4,
    "identity_base": 2.0,
    "variant": 2.2,
    "topic": 1.8,
    "function": 1.2,
    "locator": 1.7,
    "table": 1.6,
    "constraint": 1.6,
    "relation": 1.1,
    "structure": 0.35,
    "evidence": 0.0,
    # C28 opt-in: 긴 표/조문 전체가 아니라 한 행·문장 안에서 질의 근거가
    # 함께 나타나는지를 점수화한다. 기존 arm은 0이라 순위가 바뀌지 않는다.
    "evidence_unit": 0.0,
    "extra": 0.15,
}

DEFAULT_B = {f: (0.2 if f in {"identity_container", "identity_full", "identity_base", "variant", "structure"} else 0.7)
             for f in FIELD_NAMES}


def profile_weights(profile: str = "full") -> dict[str, float]:
    """전수 평가용 1변수 ablation. full 이외는 해당 축만 0으로 만든다."""
    weights = dict(DEFAULT_WEIGHTS)
    disabled = {
        "core": {"variant", "locator", "table", "relation"},
        "no_identity_split": {"identity_base", "variant"},
        "no_variant": {"variant"},
        "no_locator": {"locator", "table", "relation"},
        "no_extra": {"extra"},
    }.get(profile, set())
    if profile not in {"full", "core", "no_identity_split", "no_variant", "no_locator", "no_extra"}:
        raise ValueError(f"unknown structured-search profile: {profile}")
    for field in disabled:
        weights[field] = 0.0
    return weights

JO_RE = re.compile(r"제\s?(\d+(?:-\d+)?)\s?조(?:\s?의\s?(\d+))?")
CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{2}(?:\.\d{1,2})?)(?![A-Za-z])", re.I)
BENEFIT_QUERY_RE = re.compile(
    r"([가-힣A-Za-z0-9·]{1,40}?(?:진단|수술|치료|입원|통원|장해|사망))"
    r"(?:급여금|보험금|진단비|치료비|수술비|지원비|금)"
)
BRACKET_RE = re.compile(r"\[([^\]]{2,40})\]")
KNOWN_VARIANTS = (
    "해약환급금 미지급형", "해지환급금 미지급형", "갱신형", "일반형",
    "삭감없음용", "기본", "간편심사형",
)

# 질문이 요구하는 서로 다른 근거 역할을 찾는 범용 규칙. 상품명·질병명은 넣지 않는다.
# Router의 보수적 ROLE_RULES는 기준선 불변을 위해 건드리지 않고,
# evidence portfolio와 명시적 role-coverage arm에서만 쓴다.
BASE_EVIDENCE_ROLE_RULES = (
    (r"지급|보장|급여금|진단비|보험금|나오", "payment_trigger"),
    (r"한도|횟수|몇\s*(?:번|일)|여러\s*번|반복|재입원|재수술|연간|최초\s*1회", "limit_frequency"),
    (r"지급금액|지급률|금액|얼마|산정|계산", "payment_amount"),
    (r"진단|판정|요건|기준|대상|해당|포함|가능", "criteria_rule"),
    (r"정의|뜻|의미|무엇|뭐야|이라\s*함은|라\s*함은", "definition"),
    (r"보장개시|책임개시|감액|대기기간|보험기간|가입연령|언제부터|이전|이후", "timing_period"),
    (r"지급하지\s*않|보상하지\s*않|면책|제외|제한|안\s*되", "exclusion_exception"),
    (r"납입.?면제", "premium_waiver"),
    (r"청구|구비서류|제출서류|신청", "claim_procedure"),
    (r"갱신|해지|소멸|무효|환급|유지|변경", "contract_lifecycle"),
    (r"분류표|분류코드|질병코드|수가코드|코드", "code_reference"),
)

# C24에서만 켜는 확장 규칙. C23의 실험 처치를 보존하기 위해 전역 기본값에
# 합치지 않는다.
EXPANDED_EVIDENCE_ROLE_RULES = (
    (r"몇\s*회", "limit_frequency"),
    # "X도 포함/보장되나요"는 범주 구성원 여부를 묻는 일반적인 표면형이다.
    # 특정 질병명을 열거하지 않고 정의/분류 근거를 별도 탐색한다.
    (r"(?:도|이|가|은|는)?\s*(?:포함|해당|보장)\s*(?:되|되는지|되나|되나요)",
     "code_reference"),
)


def requested_evidence_roles(raw_query: str, slots: dict, limit: int = 3,
                             expanded: bool = False) -> list[str]:
    """질문이 동시에 요구하는 근거 역할을 범용 표면형으로 추출한다.

    단일 역할 질문은 기존 순위를 보존하기 위해 빈 목록을 반환한다.
    """
    roles = list(dict.fromkeys(list(slots.get("role", [])) +
                               list(slots.get("function", []))))
    rules = BASE_EVIDENCE_ROLE_RULES + (EXPANDED_EVIDENCE_ROLE_RULES if expanded else ())
    for pattern, role in rules:
        if re.search(pattern, raw_query, re.I) and role not in roles:
            roles.append(role)
    return roles[:limit] if len(roles) >= 2 else []


def char_bigrams(value: str) -> list[str]:
    """띄어쓰기/기호 차이에 강한 결정론적 문자 bigram."""
    value = compact(value)
    if not value:
        return []
    return [value[i:i + 2] for i in range(len(value) - 1)] if len(value) > 1 else [value]


def fact_query_expansion(raw_query: str, slots: dict) -> str:
    """fact-card arm에서만 쓰는 상품/질병 독립 표면형 확장."""
    parts = []
    for base in BENEFIT_QUERY_RE.findall(raw_query):
        base = re.sub(r"\s+", "", base)
        if len(base) >= 2:
            parts.extend((base + "금", base + "급여금", base + "보험금"))
    for role in requested_evidence_roles(raw_query, slots, expanded=True):
        parts.extend((role, ROLE_KO.get(role, role)))
    return " ".join(dict.fromkeys(parts))


def variants_of(value: str) -> list[str]:
    """표시용 수식어와 판본을 base identity와 분리해 보존한다."""
    s = str(value or "")
    out = [v for v in KNOWN_VARIANTS if compact(v) in compact(s)]
    out.extend(x.strip() for x in BRACKET_RE.findall(s) if x.strip())
    return list(dict.fromkeys(out))


def jo_keys(value: str) -> list[str]:
    out = []
    for a, b in JO_RE.findall(value or ""):
        out.append(f"제{a}조" + (f"의{b}" if b else ""))
    return list(dict.fromkeys(out))


def explicit_identity_stem(value: str) -> str:
    """질문에 직접 쓴 identity만 인정하기 위한 범용 표면형.

    문서명 끝의 법적/판본 주석만 제거하고 본문 identity와 구조 표시는
    보존한다.  이로써 짧은 core가 더 긴 문서명 안에 우연히 포함된 경우를
    exact identity로 오인하지 않는다.
    """
    return compact(re.sub(r"\(\s*무배당[^)]*\)\s*$", "", str(value or "")))


def explicit_identity_name(value: str) -> str:
    """질문에 ``…특약``으로 직접 적힌 base 계약명.

    가입 유형과 대괄호형 플랜 수식어는 표시 축이므로 제거하되 ``특약``
    접미사는 남긴다. 따라서 질문의 일반 기능어인 ``보험료 납입면제``는
    계약명으로 승격되지 않고, ``6대질병장해특약``처럼 직접 쓴 이름만
    보너스 후보가 된다.
    """
    text = re.sub(r"\(\s*무배당[^)]*\)\s*$", "", str(value or ""))
    text = re.sub(r"^\s*\(간편\)\s*", "", text)
    text = re.sub(r"^\s*(?:\[[^]]+\]\s*)+", "", text)
    return compact(text)


class StructuredTagIndex:
    """Sparse posting 기반 BM25F 인덱스. 문서 원문 대신 Semantic Tag 필드만 색인한다."""

    def __init__(self, elements: list[dict], tags: list[dict], alias_path: str | Path | None = None):
        self.N = len(elements)
        self.fields = []
        self.length = {f: [0] * self.N for f in FIELD_NAMES}
        self.avglen = {}
        self.postings = {f: collections.defaultdict(list) for f in FIELD_NAMES}
        self.evidence_unit_postings = collections.defaultdict(list)
        self.evidence_unit_count = 0
        self.evidence_unit_total_length = 0
        self.df = collections.Counter()
        self.alias_groups = self._load_aliases(alias_path)
        self.adapter_stats = collections.Counter()

        totals = collections.Counter()
        for i, (e, tag) in enumerate(zip(elements, tags)):
            row = adapt_tag(tag, e)
            evidence_units = [str(value).strip() for value in tag.get("evidence_anchor", [])
                              if str(value).strip()]
            identities = [str(x) for x in row["identity"] if str(x).strip()]
            full = " ".join(identities)
            vals = {
                "identity_container": " ".join(row["container"]),
                "identity_full": full,
                "identity_base": " ".join(dict.fromkeys(contract_core(x) for x in identities if contract_core(x))),
                "variant": " ".join(dict.fromkeys(v for x in identities for v in variants_of(x))),
                "topic": " ".join(row["topic"]),
                "function": " ".join(row["function"]),
                "locator": " ".join(row["locator"]),
                "table": " ".join(row["table"]),
                "constraint": " ".join(row["constraint"]),
                "relation": " ".join(row["relation"]),
                "structure": " ".join(row["structure"]),
                "evidence": " ".join(row["evidence"]),
                "evidence_unit": " ".join(evidence_units),
                "extra": " ".join(row["extra"]),
            }
            self.adapter_stats["documents"] += 1
            self.adapter_stats["recognized_paths"] += len(row["_recognized_paths"])
            self.adapter_stats["unknown_paths"] += len(row["_unknown_paths"])
            self.adapter_stats["documents_with_unknown"] += bool(row["_unknown_paths"])
            self.fields.append(vals)
            # 새 opt-in 축이 weight=0인 기존 arm의 IDF를 바꾸지 않도록 legacy
            # document frequency는 기존 필드에서만 계산한다.
            seen = set()
            for field, text in vals.items():
                if field == "evidence_unit":
                    continue
                tf = collections.Counter(char_bigrams(text))
                length = sum(tf.values())
                self.length[field][i] = length
                totals[field] += length
                for term, n in tf.items():
                    self.postings[field][term].append((i, n))
                    if field not in {"identity_container", "evidence"}:
                        seen.add(term)
            for term in seen:
                self.df[term] += 1
            for unit_index, unit in enumerate(evidence_units):
                tf = collections.Counter(char_bigrams(unit))
                unit_length = sum(tf.values())
                if not unit_length:
                    continue
                self.evidence_unit_count += 1
                self.evidence_unit_total_length += unit_length
                for term, n in tf.items():
                    self.evidence_unit_postings[term].append((i, unit_index, n, unit_length))
        self.avglen = {f: totals[f] / max(1, self.N) for f in FIELD_NAMES}
        self.avg_evidence_unit_length = (
            self.evidence_unit_total_length / max(1, self.evidence_unit_count)
        )

    @staticmethod
    def _load_aliases(path):
        if not path or not Path(path).exists():
            return []
        data = json.load(open(path, encoding="utf-8"))
        return [[head] + list(alts) for head, alts in data.items() if not head.startswith("_")]

    def expand_aliases(self, raw_query: str) -> str:
        cq = compact(raw_query)
        extra = []
        for group in self.alias_groups:
            if any(compact(x) and compact(x) in cq for x in group):
                extra.extend(group)
        return raw_query + (" " + " ".join(dict.fromkeys(extra)) if extra else "")

    @staticmethod
    def query_text(raw_query: str, slots: dict) -> str:
        parts = [raw_query]
        for field, values in slots.items():
            if field.startswith("_"):
                continue
            if field == "contract":
                # 라우터가 같은 base의 모든 판본을 열거해도 질의에 없던 판본은 주입하지 않는다.
                parts.extend(contract_core(v) for v in values if contract_core(v))
            else:
                parts.extend(map(str, values))
        return " ".join(parts)

    def score(self, raw_query: str, slots: dict, lexical_counts: list[int] | None = None,
              weights: dict | None = None, b: dict | None = None, k1: float = 1.2,
              lexical_weight: float = 0.2) -> list[float]:
        supplied_weights = dict(weights or {})
        axis_bonus = float(supplied_weights.pop("_axis", 0.0))
        coord_bonus = float(supplied_weights.pop("_coord", 0.0))
        exact_identity_bonus = float(supplied_weights.pop("_exact_identity", 0.0))
        locator_coverage_bonus = float(supplied_weights.pop("_locator_coverage", 0.0))
        evidence_role_bonus = float(supplied_weights.pop("_evidence_role", 0.0))
        hierarchy_bonus = float(supplied_weights.pop("_hierarchy", 0.0))
        evidence_unit_rare_n = int(supplied_weights.pop("_evidence_unit_rare", 0))
        weights = {**DEFAULT_WEIGHTS, **supplied_weights}
        b = {**DEFAULT_B, **(b or {})}
        qtext = self.expand_aliases(self.query_text(raw_query, slots))
        if weights.get("evidence", 0.0) or weights.get("evidence_unit", 0.0):
            qtext += " " + fact_query_expansion(raw_query, slots)
        qterms = list(dict.fromkeys(char_bigrams(qtext)))
        raw_qterm_set = set(char_bigrams(raw_query))
        scores = [0.0] * self.N

        def term_idf(term):
            df = self.df.get(term, 0)
            # legacy 필드에 없던 새 fact/container 용어만 해당 opt-in posting의
            # document frequency를 사용한다. 기존 용어의 IDF는 U3와 동일하다.
            if not df:
                new_docs = set()
                for field in ("identity_container", "evidence"):
                    if weights.get(field, 0.0):
                        new_docs.update(i for i, _ in self.postings[field].get(term, ()))
                if weights.get("evidence_unit", 0.0):
                    new_docs.update(i for i, _, _, _ in self.evidence_unit_postings.get(term, ()))
                df = len(new_docs)
            return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        for term in qterms:
            accum = collections.defaultdict(float)
            for field in FIELD_NAMES:
                if field == "evidence_unit":
                    continue
                avg = self.avglen[field]
                for i, tf in self.postings[field].get(term, ()):
                    norm = (1.0 - b[field]) + b[field] * (self.length[field][i] / avg if avg else 0.0)
                    accum[i] += weights[field] * tf / max(norm, 1e-9)
            if not accum:
                continue
            df = self.df[term]
            idf = term_idf(term)
            for i, x in accum.items():
                scores[i] += idf * x / (k1 + x)

        # 한 표/조문 안의 여러 행을 합쳐 길이 패널티를 주지 않는다. 질의의
        # 여러 term이 같은 행·문장에 함께 나타날 때만 해당 unit 점수가 커진다.
        evidence_unit_weight = float(weights.get("evidence_unit", 0.0))
        matched_unit_documents = 0
        selected_unit_terms = qterms
        if evidence_unit_weight:
            if evidence_unit_rare_n:
                candidates = [term for term in dict.fromkeys(char_bigrams(raw_query))
                              if self.evidence_unit_postings.get(term)]
                selected_unit_terms = sorted(
                    candidates, key=lambda term: (-term_idf(term), raw_query.find(term), term)
                )[:evidence_unit_rare_n]
            unit_scores = collections.defaultdict(float)
            unit_b = b.get("evidence_unit", 0.7)
            for term in selected_unit_terms:
                idf = term_idf(term)
                for i, unit_index, tf, unit_length in self.evidence_unit_postings.get(term, ()):
                    norm = ((1.0 - unit_b) + unit_b *
                            (unit_length / max(self.avg_evidence_unit_length, 1e-9)))
                    x = tf / max(norm, 1e-9)
                    unit_scores[(i, unit_index)] += idf * x / (k1 + x)
            best_by_document = {}
            for (i, _), unit_score in unit_scores.items():
                best_by_document[i] = max(best_by_document.get(i, 0.0), unit_score)
            for i, unit_score in best_by_document.items():
                scores[i] += evidence_unit_weight * unit_score
            matched_unit_documents = len(best_by_document)

        if lexical_counts:
            for i, n in enumerate(lexical_counts):
                if n:
                    scores[i] += lexical_weight * n

        # 계층/판본은 불완전 신호이므로 제거 조건이 아니라 유한 가산점만 준다.
        identity_values = list(slots.get("identity", [])) + list(slots.get("contract", []))
        contract_cores = {compact(contract_core(v)) for v in identity_values if contract_core(v)}
        # Router가 상위 문서/상품을 contract로 반환한 경우 그것을 특정 section의
        # hard preference로 오해하지 않는다. 문서 하위의 모든 section에 동일한
        # 유한 가산점을 주고, section identity bonus에서는 제외한다.
        known_container_cores = {
            compact(contract_core(vals["identity_container"]))
            for vals in self.fields if vals["identity_container"]
        }
        container_query_cores = contract_cores.intersection(known_container_cores)
        compact_query = compact(raw_query)
        identity_confidence = float((slots.get("_conf") or {}).get("contract", 1.0))
        explicit_candidates = []
        if identity_confidence >= 1.0:
            for value in identity_values:
                core = compact(contract_core(value))
                full_stem = explicit_identity_stem(value)
                name_stem = explicit_identity_name(value)
                stem = (full_stem if len(full_stem) >= 6 and full_stem in compact_query
                        else name_stem)
                if len(core) >= 6 and len(stem) >= 6 and stem in compact_query:
                    explicit_candidates.append((core, stem))
        # 동시에 명시된 중첩 identity는 가장 구체적인 표면형만 보너스를 받는다.
        exact_identity_cores = {
            core for core, stem in explicit_candidates
            if not any(stem != other_stem and stem in other_stem
                       for _, other_stem in explicit_candidates)
        }
        boosted_documents = 0
        coverage_documents = 0
        max_locator_coverage = 0.0
        query_variants = set(variants_of(raw_query))
        query_jo = set(jo_keys(raw_query))
        query_codes = {compact(x) for x in CODE_RE.findall(raw_query)}
        evidence_roles = requested_evidence_roles(raw_query, slots)
        role_coverage_documents = 0
        max_role_coverage = 0
        axis_requests = (
            ((list(slots.get("identity", [])) + list(slots.get("contract", []))),
             ("identity_full", "identity_base")),
            ((list(slots.get("topic", [])) + list(slots.get("subject", []))), ("topic",)),
            ((list(slots.get("function", [])) + list(slots.get("role", []))), ("function",)),
            ((list(slots.get("constraint", [])) + list(slots.get("qualifier", []))),
             ("constraint", "table")),
            ((list(slots.get("structure", [])) + list(slots.get("schema", []))),
             ("structure", "table")),
        )
        for i, vals in enumerate(self.fields):
            section_cores = contract_cores - container_query_cores if hierarchy_bonus else contract_cores
            if section_cores and any(core in compact(vals["identity_base"]) for core in section_cores):
                scores[i] += 2.0
            if hierarchy_bonus and container_query_cores:
                document_container = compact(contract_core(vals["identity_container"]))
                if document_container in container_query_cores:
                    scores[i] += hierarchy_bonus
            if exact_identity_bonus and exact_identity_cores:
                document_core = compact(contract_core(vals["identity_full"]))
                if document_core in exact_identity_cores:
                    scores[i] += exact_identity_bonus
                    boosted_documents += 1
            if query_variants and query_variants.intersection(variants_of(vals["variant"])):
                scores[i] += 2.5
            if query_jo and query_jo.intersection(jo_keys(vals["locator"])):
                scores[i] += 2.0
            if query_codes:
                searchable = compact(" ".join(vals[f] for f in ("topic", "table", "constraint", "locator", "extra")))
                scores[i] += 2.5 * sum(1 for code in query_codes if code and code in searchable)
            if evidence_role_bonus and evidence_roles:
                document_roles = set(vals["function"].split())
                matched_roles = len(document_roles.intersection(evidence_roles))
                if matched_roles:
                    scores[i] += evidence_role_bonus * matched_roles
                    role_coverage_documents += 1
                    max_role_coverage = max(max_role_coverage, matched_roles)
            if locator_coverage_bonus:
                # 질문이 조/절 제목(locator)의 대부분을 직접 말한 경우만 보너스를 준다.
                # body에서 뽑힌 topic은 "준용" 같은 간접 언급도 포함할 수 있어 제외한다.
                field_terms = set(char_bigrams(vals["locator"]))
                matched = raw_qterm_set & field_terms
                coverage = 0.0
                if len(field_terms) >= 4 and len(matched) >= 3:
                    matched_weight = sum(term_idf(term) for term in matched)
                    denominator = (sum(term_idf(term) for term in field_terms)
                                   + sum(term_idf(term) for term in raw_qterm_set))
                    coverage = 2.0 * matched_weight / max(denominator, 1e-9)
                if coverage >= 0.20:
                    scores[i] += locator_coverage_bonus * coverage * coverage
                    coverage_documents += 1
                    max_locator_coverage = max(max_locator_coverage, coverage)
            if axis_bonus or coord_bonus:
                matched_axes = 0
                for requested, fields in axis_requests:
                    haystack = compact(" ".join(vals[f] for f in fields))
                    if haystack and any(len(compact(v)) >= 2 and compact(v) in haystack
                                        for v in requested):
                        matched_axes += 1
                scores[i] += axis_bonus * matched_axes
                scores[i] += coord_bonus * max(0, matched_axes - 1)
        self.last_exact_identity = {
            "enabled": bool(exact_identity_bonus),
            "confidence": identity_confidence,
            "matched_cores": sorted(exact_identity_cores),
            "boosted_documents": boosted_documents,
        }
        self.last_locator_coverage = {
            "enabled": bool(locator_coverage_bonus),
            "threshold": 0.20,
            "min_matched_bigrams": 3,
            "boosted_documents": coverage_documents,
            "max_coverage": round(max_locator_coverage, 6),
        }
        self.last_evidence_role = {
            "enabled": bool(evidence_role_bonus),
            "requested_roles": evidence_roles,
            "boosted_documents": role_coverage_documents,
            "max_matched_roles": max_role_coverage,
        }
        self.last_evidence_unit = {
            "enabled": bool(evidence_unit_weight),
            "units": self.evidence_unit_count,
            "matched_documents": matched_unit_documents,
            "rare_term_limit": evidence_unit_rare_n,
            "selected_terms": selected_unit_terms if evidence_unit_weight else [],
        }
        return scores

    def rank(self, raw_query: str, slots: dict, lexical_counts: list[int] | None = None,
             weights: dict | None = None, profile: str = "full", limit: int = 200):
        merged = {**profile_weights(profile), **(weights or {})}
        scores = self.score(raw_query, slots, lexical_counts, weights=merged)
        ranked = [i for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:limit]]

    @staticmethod
    def _identity_candidates(raw_query: str, slots: dict, limit: int = 2) -> list[str]:
        """라우터가 여러 판본/유사 특약을 열거했을 때 base identity별 대표만 고른다."""
        cq = compact(raw_query)
        grouped = {}
        for order, value in enumerate(list(slots.get("identity", [])) + list(slots.get("contract", []))):
            core = contract_core(value)
            if not core or core in grouped:
                continue
            # 질문에 명시된 더 긴 identity를 먼저, 나머지는 라우터 순서를 보존한다.
            grouped[core] = (core in cq, len(core), -order, str(value))
        return [item[3] for item in sorted(grouped.values(), reverse=True)[:limit]]

    @staticmethod
    def _focus_text(slots: dict, tokens: list[str]) -> str:
        """질문의 군더더기를 빼고 topic/function/constraint 중심의 재질의를 만든다."""
        parts = list(tokens)
        for field in ("subject", "topic", "role", "function", "qualifier", "constraint",
                      "schema", "structure", "locator", "relation"):
            parts.extend(map(str, slots.get(field, [])))
        return " ".join(dict.fromkeys(x for x in parts if x))

    @staticmethod
    def infer_evidence_roles(raw_query: str, slots: dict, limit: int = 3,
                             expanded: bool = False) -> list[str]:
        """원 질문이 동시에 요구하는 근거 역할을 규칙으로 찾는다.

        단일 역할이면 포트폴리오를 만들지 않는다. 이 불변식이 단일근거 질문의 기존
        C5 순위를 보존한다. 라우터가 잡은 역할을 먼저 두고, 범용 표면형 규칙을 보완한다.
        """
        return requested_evidence_roles(raw_query, slots, limit, expanded=expanded)

    @staticmethod
    def _evidence_query(role: str, slots: dict, tokens: list[str]) -> str:
        """identity는 슬롯으로 유지하고 topic/constraint + 역할 표면형만 질의에 넣는다."""
        parts = []
        for field in ("subject", "topic", "qualifier", "constraint", "schema", "structure"):
            parts.extend(map(str, slots.get(field, [])))
        if not parts:
            parts.extend(tokens[:8])
        parts.append(ROLE_KO.get(role, role))
        return " ".join(dict.fromkeys(x for x in parts if x))

    def build_portfolio_specs(self, raw_query: str, slots: dict, tokens: list[str],
                              mode: str, seed_ranked: list[tuple[int, float]] | None = None,
                              max_specs: int = 6) -> list[dict]:
        """LLM 없이 후속 검색 단계를 만든다. 각 단계 내부의 BM25F 순위는 그대로 둔다."""
        allowed = {"focus", "relax", "relax_gated", "identity", "role", "evidence",
                   "rrf_evidence", "rrf_axes", "rrf_safe_axes", "rrf_safe_axes_expanded",
                   "sequence", "facet", "all"}
        if mode not in allowed:
            raise ValueError(f"unknown portfolio mode: {mode}")
        use = allowed - {"all"} if mode == "all" else {mode}
        focus = self._focus_text(slots, tokens) or raw_query
        identities = self._identity_candidates(raw_query, slots, limit=2)
        roles = list(dict.fromkeys(list(slots.get("role", [])) + list(slots.get("function", []))))[:2]
        specs = []

        def add(name, query, query_slots, reason):
            specs.append({"phase": name, "query": query, "slots": query_slots, "reason": reason})

        if "focus" in use:
            add("focus", focus, dict(slots), "질문 군더더기를 제거한 슬롯 중심 질의")
        routed_identities = list(slots.get("contract", [])) + list(slots.get("identity", []))
        relax_requested = "relax" in use or "relax_gated" in use
        # 완화 후보는 유용하지만 무조건 첫 페이지에 섞으면 명확한 identity 질의에 노이즈가 된다.
        # 라우터가 3개 이상을 열거했을 때만 "초기 identity 과부하"로 간주한다. 이 규칙은
        # 상품명이나 암 도메인 어휘에 의존하지 않아 다른 약관/업무 문서에도 그대로 적용된다.
        relax_allowed = "relax" in use or len(routed_identities) >= 3
        if relax_requested and relax_allowed and routed_identities:
            relaxed = {k: list(v) for k, v in slots.items() if k not in {"contract", "identity"}}
            reason = ("라우터 identity 후보 3개 이상 과부하 완화"
                      if "relax_gated" in use else "초기 identity 오지정 회복용 완화 질의")
            add("identity_relax", focus, relaxed, reason)
        if "role" in use and len(roles) >= 2:
            for role in roles:
                role_slots = {k: list(v) for k, v in slots.items()}
                role_slots.pop("function", None)
                role_slots["role"] = [role]
                # identity 후보가 과다하면 역할 탐색에서는 identity를 풀어 잘못된 초기 지정을 회복한다.
                if len(identities) >= 2:
                    role_slots.pop("contract", None); role_slots.pop("identity", None)
                add(f"role:{role}", focus + " " + role, role_slots, "복수 근거의 역할별 하위질의")
        if ("evidence" in use or "rrf_evidence" in use or "rrf_axes" in use
                or "rrf_safe_axes" in use or "rrf_safe_axes_expanded" in use):
            evidence_roles = self.infer_evidence_roles(
                raw_query, slots, expanded="rrf_safe_axes_expanded" in use)
            for role in evidence_roles:
                role_slots = {k: list(v) if isinstance(v, list) else v for k, v in slots.items()}
                role_slots.pop("function", None)
                role_slots["role"] = [role]
                add(f"evidence:{role}", self._evidence_query(role, role_slots, tokens), role_slots,
                    "원질문 다중근거 역할별 규칙 질의")
                specs[-1]["coverage_axis"] = "function"
                specs[-1]["coverage_value"] = role
        if "rrf_axes" in use or "rrf_safe_axes" in use or "rrf_safe_axes_expanded" in use:
            identity_values = list(slots.get("contract", [])) + list(slots.get("identity", []))
            compact_query = compact(raw_query)

            # 질문에 직접 쓰인 판본 표지별로 관련 identity를 묶는다. 보험 상품명이
            # 아니라 임의 문서의 bracket/variant 축을 사용하므로 다른 문서군에도 같다.
            variant_groups = collections.defaultdict(list)
            for value in identity_values:
                for variant in variants_of(value):
                    if compact(variant) and compact(variant) in compact_query:
                        variant_groups[variant].append(value)
            if len(variant_groups) >= 2:
                for variant, values in list(variant_groups.items())[:3]:
                    variant_slots = {
                        k: list(v) if isinstance(v, list) else v
                        for k, v in slots.items() if k not in {"contract", "identity"}
                    }
                    variant_slots["identity"] = list(dict.fromkeys(values))
                    query = " ".join(dict.fromkeys([variant] +
                        list(variant_slots.get("subject", [])) +
                        list(variant_slots.get("topic", [])) +
                        list(variant_slots.get("role", [])) +
                        list(variant_slots.get("function", [])) +
                        list(variant_slots.get("qualifier", [])) +
                        list(variant_slots.get("constraint", []))))
                    add(f"variant:{variant}", query, variant_slots,
                        "질문에 명시된 판본별 독립 규칙 질의")
                    specs[-1]["coverage_axis"] = "variant"
                    specs[-1]["coverage_value"] = variant
                    specs[-1]["coverage_roles"] = (evidence_roles or
                        list(dict.fromkeys(list(slots.get("role", [])) +
                                           list(slots.get("function", [])))))

            # 서로 다른 identity가 질문에 직접 명시된 비교/목록 질문도 각 scope를
            # 독립 탐색한다. 같은 base의 판본은 위 variant 경로가 담당한다.
            base_groups = collections.OrderedDict()
            for value in identity_values:
                core = compact(contract_core(value))
                surface = explicit_identity_name(value)
                if core and len(surface) >= 6 and surface in compact_query:
                    base_groups.setdefault(core, []).append(value)
            if len(base_groups) >= 2:
                for core, values in list(base_groups.items())[:3]:
                    identity_slots = {
                        k: list(v) if isinstance(v, list) else v
                        for k, v in slots.items() if k not in {"contract", "identity"}
                    }
                    identity_slots["identity"] = list(dict.fromkeys(values))
                    query = " ".join(dict.fromkeys(
                        list(values[:1]) + list(identity_slots.get("subject", [])) +
                        list(identity_slots.get("topic", [])) +
                        list(identity_slots.get("role", [])) +
                        list(identity_slots.get("function", []))))
                    add(f"identity:{core[:24]}", query, identity_slots,
                        "질문에 명시된 서로 다른 identity별 독립 규칙 질의")
                    specs[-1]["coverage_axis"] = "identity"
                    specs[-1]["coverage_value"] = core
                    specs[-1]["coverage_roles"] = (evidence_roles or
                        list(dict.fromkeys(list(slots.get("role", [])) +
                                           list(slots.get("function", [])))))
        if "sequence" in use:
            # C9: 최초 identity를 버리거나 임의의 역할을 추론하지 않는다. 규칙 라우터가
            # 실제로 산출한 범용 축만 사용해 계층형 후속 질의를 만든다. 상품/질병 어휘와
            # 독립적이며, 각 하위 질의의 내부 BM25F 순위도 그대로 보존한다.
            identity_values = list(slots.get("contract", [])) + list(slots.get("identity", []))
            topic_values = list(slots.get("subject", [])) + list(slots.get("topic", []))
            function_values = list(slots.get("role", [])) + list(slots.get("function", []))
            constraint_values = list(slots.get("qualifier", [])) + list(slots.get("constraint", []))
            structure_values = list(slots.get("schema", [])) + list(slots.get("structure", []))

            common = {}
            if identity_values:
                common["identity"] = list(dict.fromkeys(identity_values[:2]))
            if topic_values:
                common["topic"] = list(dict.fromkeys(topic_values[:4]))

            if common and len(common) >= 2:
                query = " ".join(dict.fromkeys(common["identity"] + common["topic"]))
                add("sequence:identity_topic", query, common,
                    "정체성→주제 계층을 보존한 규칙 후속질의")

            # 역할/조건은 질문에서 명시적으로 라우팅된 경우에만 분리한다. C7의 암묵적
            # 역할 추론 오발화를 반복하지 않기 위해 표면 규칙으로 없는 역할은 만들지 않는다.
            for role in list(dict.fromkeys(function_values))[:2]:
                role_slots = {k: list(v) for k, v in common.items()}
                role_slots["function"] = [role]
                if constraint_values:
                    role_slots["constraint"] = list(dict.fromkeys(constraint_values[:3]))
                query_parts = list(role_slots.get("identity", [])) + list(role_slots.get("topic", []))
                query_parts += [ROLE_KO.get(role, role)] + list(role_slots.get("constraint", []))
                add(f"sequence:function:{role}", " ".join(dict.fromkeys(query_parts)), role_slots,
                    "정체성·주제 안에서 명시 역할/조건을 탐색")

            if structure_values and (common or constraint_values):
                structure_slots = {k: list(v) for k, v in common.items()}
                structure_slots["structure"] = list(dict.fromkeys(structure_values[:2]))
                if constraint_values:
                    structure_slots["constraint"] = list(dict.fromkeys(constraint_values[:3]))
                query_parts = list(structure_slots.get("identity", []))
                query_parts += list(structure_slots.get("topic", []))
                query_parts += list(structure_slots.get("constraint", []))
                query_parts += list(structure_slots["structure"])
                add("sequence:structure", " ".join(dict.fromkeys(query_parts)), structure_slots,
                    "명시된 표·구조 근거를 별도 탐색")
        if "identity" in use and len(identities) >= 2:
            for identity in identities:
                identity_slots = {k: list(v) for k, v in slots.items()
                                  if k not in {"contract", "identity"}}
                identity_slots["identity"] = [identity]
                add(f"identity:{contract_core(identity)[:24]}", focus, identity_slots,
                    "동명·파생 identity별 독립 탐색")
        if "facet" in use and seed_ranked and (not identities or len(identities) >= 2):
            facet_score = collections.Counter()
            for rank, (idx, _) in enumerate(seed_ranked[:100]):
                identity = self.fields[idx]["identity_base"]
                if identity:
                    facet_score[identity] += 1.0 / (10 + rank)
            for identity, _ in facet_score.most_common(2):
                facet_slots = {k: list(v) for k, v in slots.items()
                               if k not in {"contract", "identity"}}
                facet_slots["identity"] = [identity]
                add(f"facet:{identity[:24]}", focus, facet_slots,
                    "1차 결과 상위 facet의 identity 하강 탐색")
        return specs[:max_specs]

    def rank_portfolio(self, raw_query: str, slots: dict, tokens: list[str],
                       lexical_counts: list[int] | None = None, weights: dict | None = None,
                       profile: str = "core", mode: str = "all", limit: int = 400,
                       window: int = 40, seed_quota: int = 20,
                       process_query: str | None = None, rrf_k: int = 60,
                       coverage_seed: int = 3, coverage_per_role: int = 1):
        """BM25F 결과를 재점수화하지 않고 결정론적 단계별 quota로 교차 노출한다.

        첫 ``seed_quota``는 기존 core 순위를 완전히 보존한다. 나머지 첫 페이지 슬롯만
        후속 질의의 내부 순위를 round-robin으로 노출하고, 이후에는 seed 순위를 이어 붙인다.
        """
        seed = self.rank(raw_query, slots, lexical_counts, weights, profile, limit=max(limit, 100))
        specs = self.build_portfolio_specs(process_query or raw_query, slots, tokens, mode, seed)
        sublists = []
        for spec in specs:
            ranked = self.rank(spec["query"], spec["slots"], lexical_counts, weights,
                               profile, limit=max(window, 60))
            sublists.append((spec, ranked))

        if mode in {"rrf_evidence", "rrf_axes", "rrf_safe_axes", "rrf_safe_axes_expanded"}:
            # C21: 원질의와 규칙 기반 근거역할 하위질의를 학습 없이 융합한다.
            # RRF는 서로 다른 BM25F 질의의 점수 스케일을 직접 섞지 않는다. 상위
            # seed 일부를 안전판으로 보존한 뒤, 각 역할의 exact-function + identity
            # 일치 후보를 한 개씩 노출해 복수 근거가 같은 종류의 조항에 묻히지 않게 한다.
            if not sublists:
                self.last_portfolio = {
                    "mode": mode, "window": window, "seed_quota": 0,
                    "rrf_k": rrf_k, "coverage_seed": coverage_seed,
                    "coverage_per_role": coverage_per_role, "specs": [],
                    "trace": [{"index": idx, "phase": "seed",
                               "reason": "다중 근거 역할 없음; 기존 순위 완전 보존"}
                              for idx, _ in seed],
                }
                return seed[:limit]

            rankings = [({"phase": "seed", "reason": "원질의 BM25F"}, seed)] + sublists
            fused = collections.defaultdict(float)
            contributions = collections.defaultdict(list)
            original_score = {idx: score for idx, score in seed}
            for spec, ranked in rankings:
                for rank, (idx, _) in enumerate(ranked[:max(window, 100)], start=1):
                    value = 1.0 / (max(1, rrf_k) + rank)
                    fused[idx] += value
                    contributions[idx].append({"phase": spec["phase"], "rank": rank})

            # 동점은 원질의 순위, 원점수, element 순서로 결정해 완전히 재현 가능하게 한다.
            seed_rank = {idx: rank for rank, (idx, _) in enumerate(seed, start=1)}
            fused_ranked = sorted(
                fused.items(),
                key=lambda pair: (-pair[1], seed_rank.get(pair[0], 10 ** 9),
                                  -original_score.get(pair[0], 0.0), pair[0]),
            )

            out = []
            seen = set()
            trace = []

            def push(idx, score, phase, reason):
                if idx in seen or len(out) >= limit:
                    return False
                seen.add(idx)
                out.append((idx, score))
                trace.append({"index": idx, "phase": phase, "reason": reason,
                              "rrf": round(score, 10),
                              "contributions": contributions.get(idx, [])})
                return True

            for idx, _ in seed[:min(coverage_seed, window, limit)]:
                push(idx, fused.get(idx, 0.0), "seed_guard", "원질의 상위 안전판")

            query_cores = {
                compact(contract_core(value))
                for value in (list(slots.get("contract", [])) + list(slots.get("identity", [])))
                if contract_core(value)
            }
            rare_query_terms = {
                term for term in set(char_bigrams(raw_query))
                if self.df.get(term, 0) / max(1, self.N) <= 0.20
            }
            for spec, ranked in sublists:
                axis = spec.get("coverage_axis", "function")
                value = spec.get("coverage_value", spec["phase"].split(":", 1)[1])
                spec_cores = {
                    compact(contract_core(item))
                    for item in (list(spec.get("slots", {}).get("contract", [])) +
                                 list(spec.get("slots", {}).get("identity", [])))
                    if contract_core(item)
                }
                added = 0
                for idx, _ in ranked:
                    identity = compact(self.fields[idx]["identity_base"])
                    identity_scope = spec_cores or query_cores
                    identity_ok = (not identity_scope or
                                   any(core in identity for core in identity_scope))
                    if axis == "function":
                        exact_match = value in set(self.fields[idx]["function"].split())
                    elif axis == "variant":
                        exact_match = compact(value) in {
                            compact(x) for x in variants_of(self.fields[idx]["variant"])
                        }
                    else:
                        exact_match = value in identity
                    strict_ok = True
                    if mode in {"rrf_safe_axes", "rrf_safe_axes_expanded"}:
                        document_roles = set(self.fields[idx]["function"].split())
                        required_roles = set(spec.get("coverage_roles", []))
                        if axis in {"variant", "identity"} and required_roles:
                            strict_ok = bool(document_roles.intersection(required_roles))
                        answer_fields = " ".join(
                            self.fields[idx][field]
                            for field in ("topic", "table", "constraint", "locator", "extra")
                        )
                        support = rare_query_terms.intersection(char_bigrams(answer_fields))
                        strict_ok = strict_ok and len(support) >= 2
                    if exact_match and identity_ok and strict_ok:
                        if push(idx, fused.get(idx, 0.0), f"coverage:{axis}:{value}",
                                ("요청 축·개별 scope·answer-field 교차 커버리지"
                                 if mode in {"rrf_safe_axes", "rrf_safe_axes_expanded"}
                                 else "요청 축 exact-match·identity 커버리지")):
                            added += 1
                        if added >= max(0, coverage_per_role):
                            break

            for idx, score in fused_ranked:
                if len(out) >= limit:
                    break
                push(idx, score, "rrf", "원질의+역할별 BM25F reciprocal-rank 융합")

            # 어느 하위질의에도 없던 seed tail도 후보 집합에서 유실하지 않는다.
            for idx, _ in seed:
                if len(out) >= limit:
                    break
                push(idx, fused.get(idx, 0.0), "seed_tail", "기존 후보 보존")

            self.last_portfolio = {
                "mode": mode, "window": window, "seed_quota": 0,
                "rrf_k": rrf_k, "coverage_seed": coverage_seed,
                "coverage_per_role": coverage_per_role,
                "specs": [{k: v for k, v in spec.items() if k != "slots"}
                          for spec, _ in sublists],
                "trace": trace,
            }
            return out

        out = []
        seen = set()
        trace = []

        def push(item, phase, reason):
            idx, score = item
            if idx in seen:
                return False
            seen.add(idx); out.append(item)
            trace.append({"index": idx, "phase": phase, "reason": reason})
            return True

        for item in seed[:min(seed_quota, window, limit)]:
            push(item, "seed", "기존 core 순위 보존")

        cursors = [0] * len(sublists)
        while len(out) < min(window, limit) and sublists:
            progressed = False
            for j, (spec, ranked) in enumerate(sublists):
                while cursors[j] < len(ranked):
                    item = ranked[cursors[j]]; cursors[j] += 1
                    if push(item, spec["phase"], spec["reason"]):
                        progressed = True
                        break
                if len(out) >= min(window, limit):
                    break
            if not progressed:
                break

        # 탐색 단계가 새 후보를 충분히 못 만들면 seed로 채우고, 첫 페이지 이후도 seed 순서를 보존한다.
        for item in seed:
            if len(out) >= limit:
                break
            push(item, "seed_tail", "기존 core 후속 순위")

        self.last_portfolio = {"mode": mode, "window": window, "seed_quota": seed_quota,
                               "specs": [{k: v for k, v in spec.items() if k != "slots"}
                                         for spec, _ in sublists],
                               "trace": trace}
        return out
