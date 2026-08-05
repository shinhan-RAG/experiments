"""Part 0.5 improvement conditions — grounded implementations.

C2 BM25F : Robertson, Zaragoza & Taylor, CIKM 2004 (simple BM25F: field-weighted
           TF combined BEFORE saturation); Robertson & Zaragoza, FnTIR 2009.
           Note: with a single field, w=1, this is rank-equivalent to the PR#14
           gate2 BM25 (its (k1+1) numerator is a rank-preserving constant).
C3 cascade: Wang, Lin & Metzler, SIGIR 2011 — first stage narrows candidates,
           second stage re-ranks within them.
C4 router : Broder SIGIR Forum 2002 / Kang & Kim SIGIR 2003 — query-type
           dispatch; rule-based on surface features only (no suite labels).
C5 RRF    : Cormack, Clarke & Buettcher, SIGIR 2009 — score(d)=Σ 1/(k+rank).
Tokenization stays char-bigram (gate2) for comparability; C6 swaps it.
"""
import math
import re
from collections import Counter


# ---------- C2: simple BM25F ----------

class BM25F:
    """fields: list of dicts {field_name: text} per document (fixed order)."""

    def __init__(self, docs_fields: list, bigrams, field_names: tuple):
        self.bigrams = bigrams
        self.field_names = field_names
        self.N = len(docs_fields)
        self.tf = []      # per doc: {field: Counter}
        self.length = []  # per doc: {field: int}
        self.df = Counter()
        total_len = {f: 0 for f in field_names}
        for d in docs_fields:
            tfd, lend = {}, {}
            seen = set()
            for f in field_names:
                c = Counter(bigrams(d.get(f, "")))
                tfd[f] = c
                lend[f] = sum(c.values())
                total_len[f] += lend[f]
                seen.update(c)
            for t in seen:
                self.df[t] += 1
            self.tf.append(tfd)
            self.length.append(lend)
        self.avglen = {f: total_len[f] / max(1, self.N) for f in field_names}

    def score(self, q_terms, idx, weights, b, k1=1.2):
        s = 0.0
        tfd, lend = self.tf[idx], self.length[idx]
        for t in q_terms:
            x = 0.0
            for f in self.field_names:
                tf = tfd[f].get(t, 0)
                if not tf:
                    continue
                avg = self.avglen[f]
                B = (1 - b[f]) + b[f] * (lend[f] / avg if avg else 0.0)
                x += weights[f] * tf / B
            if x <= 0:
                continue
            df = self.df[t]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            s += idf * x / (k1 + x)
        return s

    def rank(self, query, candidates, weights, b, k1=1.2):
        q_terms = list(Counter(self.bigrams(query)))
        return sorted(candidates,
                      key=lambda i: -self.score(q_terms, i, weights, b, k1))


# ---------- C5: Reciprocal Rank Fusion ----------

def rrf_fuse(rankings: list, k: int = 60) -> list:
    """rankings: list of full orderings (lists of doc indices). SIGIR 2009 k=60."""
    score = Counter()
    for ranking in rankings:
        for pos, doc in enumerate(ranking, 1):
            score[doc] += 1.0 / (k + pos)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------- C3: lexical cascade ----------

def cascade_rank(index_ranking: list, fm_scorer, query: str, top_k: int) -> list:
    """Stage 1: index ranking. Stage 2: re-rank its top_k with the fm scorer.
    fm_scorer.rank(query, candidates) must honour the candidate restriction."""
    head = index_ranking[:top_k]
    reranked = fm_scorer.rank(query, head)
    return reranked + index_ranking[top_k:]


# ---------- C4: rule router (surface features only) ----------

_TYPE_KW = ("판매약관", "사업방법서", "공시약관", "상품요약서")
_YEAR = re.compile(r"(?:19|20)\d{2}\s*년")


def route(query: str) -> str:
    """Returns 'identity' | 'content' | 'mixed' from query surface only."""
    has_type = any(kw in query for kw in _TYPE_KW)
    has_content_marker = "내용" in query or "설명한" in query or "포함된" in query
    if has_content_marker and has_type:
        return "mixed"
    if has_content_marker:
        return "content"
    if has_type:
        return "identity"
    return "identity"  # conservative fallback: identity arm is the champion


# ---------- token-level BM25 (C6 morphological variant) ----------

class TokBM25:
    """Same Okapi form as gate2.BM25 but over a caller-supplied token list."""

    def __init__(self, docs_tokens: list, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.tf, self.dl, self.df = [], [], Counter()
        for toks in docs_tokens:
            c = Counter(toks)
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            for t in c:
                self.df[t] += 1
        self.avgdl = sum(self.dl) / max(1, self.N)

    def rank(self, q_tokens, candidates):
        q = list(Counter(q_tokens))
        def score(i):
            s = 0.0
            for t in q:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                B = 1 - self.b + self.b * (self.dl[i] / self.avgdl if self.avgdl else 0)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * B)
            return s
        return sorted(candidates, key=lambda i: -score(i))


# ---------- Part 0.6 R1: verify-style combination (T2) ----------
# Grounding: the agentic cascade (verify) succeeded where lexical rerank (C3)
# failed; ES query/filter context separates constraint from scoring — filters
# never alter scores. Here: stage-1 order is preserved; fm acts as a pure
# membership filter that stably partitions the head of the ranking.

_TEMPLATE_STOP = {"내용이", "내용을", "내용이포함된", "있는", "포함된", "담고",
                  "문서를", "문서는", "문서", "찾아줘", "알려줘", "관한",
                  "설명한", "최신", "보여줘", "무엇인가요",
                  "판매약관", "사업방법서", "공시약관", "상품요약서"}
_YEARTOK = re.compile(r"^(?:19|20)\d{2}년?$")


def content_tokens(query: str) -> list:
    """Content-bearing tokens of a query: Hangul 4..14 chars, minus template
    vocabulary, doc-type keywords and year tokens. Surface features only."""
    out = []
    for t in query.split():
        t = t.strip("()[]{}.,;:!?\"'?")
        if not (4 <= len(t) <= 14):
            continue
        if not re.search(r"[가-힣]", t):
            continue
        if t in _TEMPLATE_STOP or _YEARTOK.match(t):
            continue
        if any(k in t for k in _TYPE_KW):
            continue
        out.append(t)
    return out


def verify_partition(ranking: list, fm_texts: list, tokens: list,
                     depth: int) -> list:
    """Stable partition of ranking[:depth]: candidates whose fm text contains
    ANY content token come first (original relative order preserved), the rest
    follow, tail unchanged. Empty pass-set or no tokens -> original ranking."""
    if not tokens:
        return ranking
    head = ranking[:depth]
    passed = [i for i in head if any(t in fm_texts[i] for t in tokens)]
    if not passed:
        return ranking
    failed = [i for i in head if not any(t in fm_texts[i] for t in tokens)]
    return passed + failed + ranking[depth:]
