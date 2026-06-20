"""Token counting abstractions.

The compressor needs to know how many tokens a chunk costs against a budget.
We default to a dependency-free heuristic that tracks real BPE counts closely
on English prose, and transparently upgrade to ``tiktoken`` when it is
installed (``pip install winnow-context[tiktoken]``).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_UNIT_RE = re.compile(r"\w+|[^\w\s]")


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


class HeuristicTokenizer:
    """Dependency-free approximation of BPE token counts.

    Strategy: split into word/punctuation units, then inflate long words to
    multiple sub-word tokens at roughly four characters per token (the
    empirical average for ``cl100k_base`` on English text). In practice this
    lands within ~10-15% of ``tiktoken`` — close enough for budgeting, and
    fully deterministic with no model download.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        total = 0
        for unit in _UNIT_RE.findall(text):
            if unit.isalnum():
                total += max(1, (len(unit) + 3) // 4)
            else:
                total += 1
        return total


class TiktokenTokenizer:
    """Exact token counts backed by OpenAI's ``tiktoken`` (optional)."""

    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # imported lazily so the package has zero hard deps

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def get_tokenizer(*, prefer_tiktoken: bool = True) -> Tokenizer:
    """Return the best available tokenizer.

    Falls back to the heuristic counter when ``tiktoken`` is not installed, so
    the library always works out of the box.
    """
    if prefer_tiktoken:
        try:
            return TiktokenTokenizer()
        except Exception:
            pass
    return HeuristicTokenizer()
