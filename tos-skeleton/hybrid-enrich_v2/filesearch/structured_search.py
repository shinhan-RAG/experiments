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
from textmatch import compact, contract_core


FIELD_NAMES = (
    "identity_full", "identity_base", "variant", "topic", "function",
    "locator", "table", "constraint", "relation", "structure", "extra",
)

DEFAULT_WEIGHTS = {
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
    "extra": 0.15,
}

DEFAULT_B = {f: (0.2 if f in {"identity_full", "identity_base", "variant", "structure"} else 0.7)
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
BRACKET_RE = re.compile(r"\[([^\]]{2,40})\]")
KNOWN_VARIANTS = (
    "해약환급금 미지급형", "해지환급금 미지급형", "갱신형", "일반형",
    "삭감없음용", "기본", "간편심사형",
)


def char_bigrams(value: str) -> list[str]:
    """띄어쓰기/기호 차이에 강한 결정론적 문자 bigram."""
    value = compact(value)
    if not value:
        return []
    return [value[i:i + 2] for i in range(len(value) - 1)] if len(value) > 1 else [value]


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


class StructuredTagIndex:
    """Sparse posting 기반 BM25F 인덱스. 문서 원문 대신 Semantic Tag 필드만 색인한다."""

    def __init__(self, elements: list[dict], tags: list[dict], alias_path: str | Path | None = None):
        self.N = len(elements)
        self.fields = []
        self.length = {f: [0] * self.N for f in FIELD_NAMES}
        self.avglen = {}
        self.postings = {f: collections.defaultdict(list) for f in FIELD_NAMES}
        self.df = collections.Counter()
        self.alias_groups = self._load_aliases(alias_path)
        self.adapter_stats = collections.Counter()

        totals = collections.Counter()
        for i, (e, tag) in enumerate(zip(elements, tags)):
            row = adapt_tag(tag, e)
            identities = [str(x) for x in row["identity"] if str(x).strip()]
            full = " ".join(identities)
            vals = {
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
                "extra": " ".join(row["extra"]),
            }
            self.adapter_stats["documents"] += 1
            self.adapter_stats["recognized_paths"] += len(row["_recognized_paths"])
            self.adapter_stats["unknown_paths"] += len(row["_unknown_paths"])
            self.adapter_stats["documents_with_unknown"] += bool(row["_unknown_paths"])
            self.fields.append(vals)
            seen = set()
            for field, text in vals.items():
                tf = collections.Counter(char_bigrams(text))
                length = sum(tf.values())
                self.length[field][i] = length
                totals[field] += length
                for term, n in tf.items():
                    self.postings[field][term].append((i, n))
                    seen.add(term)
            for term in seen:
                self.df[term] += 1
        self.avglen = {f: totals[f] / max(1, self.N) for f in FIELD_NAMES}

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
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        b = {**DEFAULT_B, **(b or {})}
        qtext = self.expand_aliases(self.query_text(raw_query, slots))
        qterms = list(dict.fromkeys(char_bigrams(qtext)))
        scores = [0.0] * self.N

        for term in qterms:
            accum = collections.defaultdict(float)
            for field in FIELD_NAMES:
                avg = self.avglen[field]
                for i, tf in self.postings[field].get(term, ()):
                    norm = (1.0 - b[field]) + b[field] * (self.length[field][i] / avg if avg else 0.0)
                    accum[i] += weights[field] * tf / max(norm, 1e-9)
            if not accum:
                continue
            df = self.df[term]
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for i, x in accum.items():
                scores[i] += idf * x / (k1 + x)

        if lexical_counts:
            for i, n in enumerate(lexical_counts):
                if n:
                    scores[i] += lexical_weight * n

        # 계층/판본은 불완전 신호이므로 제거 조건이 아니라 유한 가산점만 준다.
        identity_values = list(slots.get("identity", [])) + list(slots.get("contract", []))
        contract_cores = {compact(contract_core(v)) for v in identity_values if contract_core(v)}
        query_variants = set(variants_of(raw_query))
        query_jo = set(jo_keys(raw_query))
        query_codes = {compact(x) for x in CODE_RE.findall(raw_query)}
        for i, vals in enumerate(self.fields):
            if contract_cores and any(core in compact(vals["identity_base"]) for core in contract_cores):
                scores[i] += 2.0
            if query_variants and query_variants.intersection(variants_of(vals["variant"])):
                scores[i] += 2.5
            if query_jo and query_jo.intersection(jo_keys(vals["locator"])):
                scores[i] += 2.0
            if query_codes:
                searchable = compact(" ".join(vals[f] for f in ("topic", "table", "constraint", "locator", "extra")))
                scores[i] += 2.5 * sum(1 for code in query_codes if code and code in searchable)
        return scores

    def rank(self, raw_query: str, slots: dict, lexical_counts: list[int] | None = None,
             weights: dict | None = None, profile: str = "full", limit: int = 200):
        merged = {**profile_weights(profile), **(weights or {})}
        scores = self.score(raw_query, slots, lexical_counts, weights=merged)
        ranked = [i for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:limit]]
