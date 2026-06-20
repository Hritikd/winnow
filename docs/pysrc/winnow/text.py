"""Lightweight text tokenization for information-retrieval scoring.

This is intentionally separate from *token counting* (see ``tokenizer.py``).
Here we split text into normalized lexical terms used by BM25 and TF-IDF.
"""

from __future__ import annotations

import re

_TERM_RE = re.compile(r"[a-z0-9]+")

# A small, high-frequency stopword set. Removing these sharpens BM25/TF-IDF
# on short chunks without pulling in an NLP dependency.
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have he in is it its of on or
    that the their this to was were will with you your we our i they them
    """.split()
)


def terms(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokenize ``text`` into lowercase alphanumeric terms.

    >>> terms("The Quick, brown FOX-9!")
    ['quick', 'brown', 'fox', '9']
    """
    tokens = _TERM_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens
