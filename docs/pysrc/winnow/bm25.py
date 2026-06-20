"""Okapi BM25 ranking over a fixed set of chunks.

BM25 gives us a strong, well-understood relevance signal between the query and
each chunk without any embedding model or network call.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .text import terms


class BM25:
    def __init__(self, docs: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_terms = [terms(d) for d in docs]
        self._doc_len = [len(t) for t in self._doc_terms]
        self.n = len(docs)
        self.avgdl = (sum(self._doc_len) / self.n) if self.n else 0.0

        df: Counter[str] = Counter()
        for doc in self._doc_terms:
            df.update(set(doc))
        # BM25 idf with the +1 smoothing that keeps weights non-negative.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        self._tf = [Counter(doc) for doc in self._doc_terms]

    def score(self, query: str, index: int) -> float:
        if self.avgdl == 0:
            return 0.0
        tf = self._tf[index]
        dl = self._doc_len[index]
        score = 0.0
        for term in terms(query):
            freq = tf.get(term)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def scores(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.n)]
