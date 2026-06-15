"""Sparse TF-IDF vectors and cosine similarity between chunks.

Used by the Maximal Marginal Relevance step to measure how *redundant* a
candidate chunk is with the chunks already selected.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .text import terms


class TfidfSpace:
    def __init__(self, docs: Sequence[str]) -> None:
        self._doc_terms = [terms(d) for d in docs]
        self.n = len(docs)

        df: Counter[str] = Counter()
        for doc in self._doc_terms:
            df.update(set(doc))
        # Smoothed idf (ln) so a term present in every chunk still contributes.
        self.idf = {term: math.log((self.n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

        self._vectors = [self._vectorize(doc) for doc in self._doc_terms]
        self._norms = [math.sqrt(sum(w * w for w in vec.values())) for vec in self._vectors]

    def _vectorize(self, doc_terms: list[str]) -> dict[str, float]:
        tf = Counter(doc_terms)
        return {term: freq * self.idf.get(term, 0.0) for term, freq in tf.items()}

    def cosine(self, i: int, j: int) -> float:
        if i == j:
            return 1.0
        a, b = self._vectors[i], self._vectors[j]
        na, nb = self._norms[i], self._norms[j]
        if na == 0.0 or nb == 0.0:
            return 0.0
        # Iterate the smaller vector for the dot product.
        if len(a) > len(b):
            a, b = b, a
        dot = sum(weight * b.get(term, 0.0) for term, weight in a.items())
        return dot / (na * nb)
