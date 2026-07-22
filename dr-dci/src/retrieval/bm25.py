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

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        scores = defaultdict(float)
        for term in self._tokenize(query):
            df = self.doc_freqs.get(term, 0)
            if not df:
                continue
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in self.postings[term].items():
                dl = self.doc_lens[doc_id]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / self.avg_dl
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
