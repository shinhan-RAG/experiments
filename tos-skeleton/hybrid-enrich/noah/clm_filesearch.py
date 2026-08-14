#!/usr/bin/env python3
"""CLM (coordination-level matching) element FileSearch.

Winning element-tag retriever config: slot UNION + Q1 lexical channel, ranked
purely by the number of distinct matched conditions (coordination level), with
document order as the tie-break.

Explicitly NOT used (banned by the registered config): BM25, TF-IDF / IDF-style
weighting, rerankers, embeddings, or any learned scorer. A match contributes
exactly 1, no matter how rare or long the term is.

Contrast with slot_filesearch.py (the losing AND variant): there every slot had
to match, which left 214/337 questions with zero candidates. Here every slot is
an independent OR contributor.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

TAGS_PATH = OUT / "element_tags_v2.jsonl"
ELEMENTS_PATH = OUT / "elements_psection.jsonl"

# All eight slots are parsed so malformed rows never crash; only the five
# active ones participate in matching. article/reference/table are dead slots:
# measured query-side generation rates are .0000/.0000/.0200.
ALL_SLOTS = ("contract", "subject", "role", "article", "table", "qualifier", "reference", "schema")
ACTIVE_SLOTS = ("contract", "subject", "role", "qualifier", "schema")

# Closed vocabulary of 11 roles, with the Korean alias expansion so that a
# Korean query-side role hint can reach the canonical English code.
ROLE_ALIASES = {
    "exclusion_exception": "면책 제외 예외 부지급 지급하지 않는 사유",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}

TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")
# Common trailing Korean particles (josa). Longest-first so "으로"/"에서" etc.
# are stripped whole rather than leaving a dangling "로"/"서".
JOSA_SUFFIXES = sorted(
    ["은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
     "와", "과", "도", "만", "까지", "부터", "에게", "한테"],
    key=len, reverse=True,
)


def strip_josa(token: str) -> str | None:
    """Return `token` with one trailing particle removed, or None if none
    applies / the token is too short to bother. Query-side only -- this does
    not touch the indexed corpus text, so it is plain tokenization, not a
    learned or statistical scoring change."""
    if len(token) <= 2:
        return None
    for suf in JOSA_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 2:
            return token[:-len(suf)]
    return None
# Generic domain words that appear in nearly every element; keeping them would
# make the lexical channel match the whole corpus and flatten the ranking.
LEXICAL_STOPWORDS = {"보험", "약관", "보험금", "특약", "보험계약"}

CONTRACT_RE = re.compile(r"[(（]?\s*무\s*[)）]?[가-힣A-Za-z0-9()（）·\-]{0,40}?특약"
                         r"|[(（]\s*간편\s*[)）][가-힣A-Za-z0-9()（）·\-]{0,40}?특약"
                         r"|[가-힣A-Za-z0-9()（）·\-]{2,40}?(?:특약|주계약)")
SCHEMA_TABLE_RE = re.compile(r"금액표|구분표|보험금표|표|구분")
SCHEMA_FORMULA_RE = re.compile(r"계산|산출|공식")

# --- fuzzy matching -------------------------------------------------------
# Prefer the canonical helpers from slot_filesearch (same corpus, same
# canonicalization rules); fall back to an equivalent local implementation so
# this module never depends on that file being loadable. Only the plain
# functions are pulled in -- never the AND search class.
try:  # pragma: no cover - trivial import shim
    from slot_filesearch import compact, contract_core, fuzzy_contains
except Exception:  # pragma: no cover
    def compact(value):
        return re.sub(r"[^가-힣a-z0-9]", "", str(value).lower())

    def contract_core(value):
        value = re.sub(r"\(무배당[^)]*\)|\(간편\)|해약환급금\s*미지급형|갱신형|일반형", "", str(value))
        value = compact(value).replace("특약", "")
        return value.replace("허혈성심장질환", "허혈심장질환").replace("대상포진진단", "대상포진통풍진단")

    def _trigrams(value):
        value = compact(value)
        return {value[i:i + 3] for i in range(max(0, len(value) - 2))}

    def fuzzy_contains(query, target, field):
        q = contract_core(query) if field == "contract" else compact(query)
        t = contract_core(target) if field == "contract" else compact(target)
        if not q or not t:
            return False
        if q in t or t in q:
            return True
        if field != "contract" or min(len(q), len(t)) < 5:
            return False
        qg, tg = _trigrams(q), _trigrams(t)
        anchored = q[:3] in t and q[-4:] in t
        return anchored and len(qg & tg) / max(1, len(qg)) >= 0.72


def _read_jsonl(path):
    # Tolerant of a partial/growing file (e.g. a tagging job still writing
    # out/element_tags_v2.jsonl): a truncated trailing line is skipped rather
    # than raising, so this never crashes on an in-progress write.
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _as_list(value):
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if str(item).strip()]


def route_query(query: str) -> dict:
    """Rule-based query slot router. No LLM, no learned component."""
    slots = {slot: [] for slot in ACTIVE_SLOTS}
    seen = set()
    for match in CONTRACT_RE.finditer(query or ""):
        name = match.group(0).strip()
        key = compact(name)
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            slots["contract"].append(name)
    schema = []
    if SCHEMA_TABLE_RE.search(query or ""):
        schema.append("table")
    if SCHEMA_FORMULA_RE.search(query or ""):
        schema.append("formula")
    slots["schema"] = schema
    # subject / role / qualifier are intentionally left to the caller.
    return slots


def lexical_tokens(query: str) -> list:
    tokens, seen = [], set()
    for token in TOKEN_RE.findall(query or ""):
        if len(token) < 2 or token in LEXICAL_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


class CLMFileSearch:
    def __init__(self, tags_path=None, elements_path=None):
        self.tags_path = Path(tags_path) if tags_path else TAGS_PATH
        self.elements_path = Path(elements_path) if elements_path else ELEMENTS_PATH

        elements = _read_jsonl(self.elements_path)
        self.ids = []
        self.line_start = []
        self.line_end = []
        previews = []
        position = {}
        for index, row in enumerate(elements):
            eid = row.get("element_id", "")
            position[eid] = index
            self.ids.append(eid)
            self.line_start.append(row.get("line_start") or 0)
            self.line_end.append(row.get("line_end") or 0)
            previews.append(re.sub(r"\s+", " ", row.get("text") or "").strip())

        # Inverted index: (slot, tag value) -> set of element indices.
        self.slot_index = {slot: {} for slot in ALL_SLOTS}
        for row in _read_jsonl(self.tags_path):
            if not row.get("ok", True):
                continue
            index = position.get(row.get("element_id"))
            if index is None:
                continue
            for slot in ALL_SLOTS:
                for value in _as_list(row.get(slot)):
                    self.slot_index[slot].setdefault(value, set()).add(index)
                    if slot == "role" and value in ROLE_ALIASES:
                        self.slot_index[slot].setdefault(ROLE_ALIASES[value], set()).add(index)

        # Lexical postings: token -> set of element indices.
        self.lexical_index = {}
        for index, row in enumerate(elements):
            for token in set(TOKEN_RE.findall(row.get("text") or "")):
                if len(token) >= 2:
                    self.lexical_index.setdefault(token, set()).add(index)
        self.lexical_vocab = list(self.lexical_index)
        self.previews = previews

    # -- matching helpers --------------------------------------------------
    def _slot_hits(self, slot, query_value):
        hits = set()
        for value, postings in self.slot_index[slot].items():
            if fuzzy_contains(query_value, value, slot):
                hits |= postings
        return hits

    def _lexical_hits(self, token):
        # Korean text is not cleanly space-segmented, so a query token must be
        # allowed to match inside a longer eojeol ("납입면제" in "납입면제는").
        # Scanning the 23k-entry vocabulary is cheaper than a char-ngram index.
        #
        # Josa (particle) attachment also crippled this channel: a query token
        # like "해약환급금은" only matched elements containing that exact
        # inflected form. Try the raw token and, if a trailing particle is
        # present, the stripped form too -- union the hits either way.
        forms = [token]
        stripped = strip_josa(token)
        if stripped and stripped != token:
            forms.append(stripped)

        hits = set()
        for form in forms:
            exact = self.lexical_index.get(form)
            if exact:
                hits |= exact
            for word in self.lexical_vocab:
                if len(word) > len(form) and form in word:
                    hits |= self.lexical_index[word]
        return hits

    def search(self, query: str, slots: dict | None = None, limit: int = 20) -> dict:
        routed = route_query(query)
        query_slots = {slot: list(routed.get(slot) or []) for slot in ACTIVE_SLOTS}
        for slot, value in (slots or {}).items():
            if slot not in ACTIVE_SLOTS:
                continue  # article/reference/table hints are parsed then dropped
            for item in _as_list(value):
                if item not in query_slots[slot]:
                    query_slots[slot].append(item)

        # Coordination level: every satisfied (slot, value) or lexical token is
        # worth exactly 1. Candidates are the UNION of all conditions.
        matched = {}
        for slot in ACTIVE_SLOTS:
            for query_value in query_slots[slot]:
                label = "%s:%s" % (slot, query_value)
                for index in self._slot_hits(slot, query_value):
                    matched.setdefault(index, []).append(label)
        for token in lexical_tokens(query):
            label = "lex:%s" % token
            for index in self._lexical_hits(token):
                matched.setdefault(index, []).append(label)

        order = sorted(
            matched,
            key=lambda index: (-len(matched[index]), self.line_start[index],
                               self.line_end[index], self.ids[index]),
        )
        results = [{
            "id": self.ids[index],
            "score": len(matched[index]),
            "matched": matched[index],
            "line_start": self.line_start[index],
            "preview": self.previews[index][:180],
        } for index in order[:max(0, limit)]]
        return {
            "n_candidates": len(matched),
            "query_slots": query_slots,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="CLM element FileSearch (slot UNION + lexical)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--slots", default="{}")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    engine = CLMFileSearch()
    print(json.dumps(engine.search(args.query, json.loads(args.slots), args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
