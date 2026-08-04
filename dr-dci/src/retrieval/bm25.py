"""Small deterministic BM25 implementation for controlled experiments."""

import math
import re
from collections import defaultdict


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lens = {}
        self.avg_dl = 0.0
        self.corpus_size = 0
        self.postings = defaultdict(dict)

    def fit(self, documents: list[dict]):
        self.doc_freqs.clear()
        self.doc_lens.clear()
        self.postings.clear()
        self.corpus_size = len(documents)
        total_len = 0

        for doc in documents:
            doc_id = doc["_id"]
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            terms = self._tokenize(text)
            self.doc_lens[doc_id] = len(terms)
            total_len += len(terms)

            term_freqs = defaultdict(int)
            for term in terms:
                term_freqs[term] += 1
            for term, frequency in term_freqs.items():
                self.doc_freqs[term] += 1
                self.postings[term][doc_id] = frequency

        self.avg_dl = total_len / self.corpus_size if self.corpus_size else 1.0

    def search(self, query: str, top_k: int = 20,
               allowed_ids: set[str] | None = None,
               subset_statistics: bool = False) -> list[dict]:
        """Search the index, optionally within an explicit document subset.

        ``subset_statistics=True`` recomputes BM25's collection size, average
        document length, and query-term document frequencies over
        ``allowed_ids``. This is equivalent to fitting a small BM25 index for
        the selected document's chunks, without rebuilding postings for every
        query.
        """
        allowed = set(allowed_ids) if allowed_ids is not None else None
        if allowed is not None and not allowed:
            return []

        corpus_size = self.corpus_size
        avg_dl = self.avg_dl
        if allowed is not None and subset_statistics:
            known = allowed & set(self.doc_lens)
            if not known:
                return []
            corpus_size = len(known)
            avg_dl = sum(self.doc_lens[doc_id] for doc_id in known) / corpus_size
            allowed = known

        scores = defaultdict(float)
        for term in self._tokenize(query):
            postings = self.postings.get(term, {})
            if allowed is None:
                matching = postings.items()
                df = self.doc_freqs.get(term, 0)
            else:
                matching = [
                    (doc_id, tf) for doc_id, tf in postings.items()
                    if doc_id in allowed
                ]
                df = len(matching) if subset_statistics else self.doc_freqs.get(term, 0)
            if not df:
                continue
            idf = math.log((corpus_size - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in matching:
                dl = self.doc_lens[doc_id]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / avg_dl
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
        return [
            {"doc_id": doc_id, "score": float(score)}
            for doc_id, score in ranked[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
