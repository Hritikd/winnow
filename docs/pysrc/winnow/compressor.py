"""The winnow compression algorithm.

Given a query, a list of context chunks, and a token budget, select the
subset of chunks that maximizes relevance to the query *while penalizing
redundancy*, packed to stay within budget.

The objective is Maximal Marginal Relevance (Carbonell & Goldstein, 1998)
over BM25 relevance and TF-IDF cosine redundancy:

    MMR(c) = lambda * relevance(c) - (1 - lambda) * max_{s in selected} sim(c, s)

with a knapsack-style budget constraint. ``lambda_ = 1.0`` reduces winnow to
pure BM25 packing; lower values trade a little relevance for broader coverage.
Everything is deterministic: same inputs always yield the same output.
"""

from __future__ import annotations

from collections.abc import Sequence

from .bm25 import BM25
from .tokenizer import Tokenizer, get_tokenizer
from .types import Chunk, CompressionResult, ScoredChunk
from .vectorize import TfidfSpace


def _normalize(scores: list[float]) -> list[float]:
    top = max(scores, default=0.0)
    if top <= 0.0:
        return [0.0 for _ in scores]
    return [s / top for s in scores]


def _coerce(chunks: Sequence[Chunk | str]) -> list[Chunk]:
    out: list[Chunk] = []
    for i, c in enumerate(chunks):
        out.append(c if isinstance(c, Chunk) else Chunk(id=str(i), text=c))
    return out


def compress(
    query: str,
    chunks: Sequence[Chunk | str],
    budget_tokens: int,
    *,
    lambda_: float = 0.7,
    tokenizer: Tokenizer | None = None,
    preserve_order: bool = True,
) -> CompressionResult:
    """Winnow ``chunks`` down to the best subset that fits ``budget_tokens``.

    Args:
        query: The user/agent query the context should support.
        chunks: ``Chunk`` objects or raw strings (auto-wrapped with index ids).
        budget_tokens: Maximum total tokens the kept chunks may consume.
        lambda_: Relevance/diversity trade-off in ``[0, 1]``. ``1.0`` is pure
            relevance; ``0.7`` (default) keeps strong relevance while pruning
            near-duplicate chunks.
        tokenizer: Token counter; defaults to ``get_tokenizer()``.
        preserve_order: If ``True`` (default) kept chunks are returned in their
            original input order; otherwise in selection (relevance) order.

    Returns:
        A :class:`CompressionResult` with kept/dropped chunks and token stats.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    if budget_tokens < 0:
        raise ValueError(f"budget_tokens must be >= 0, got {budget_tokens}")

    tok = tokenizer or get_tokenizer()
    items = _coerce(chunks)
    texts = [c.text for c in items]
    token_counts = [tok.count(t) for t in texts]
    original_tokens = sum(token_counts)

    n = len(items)
    if n == 0:
        return CompressionResult(query, [], [], budget_tokens, 0, 0)

    relevance = _normalize(BM25(texts).scores(query))
    space = TfidfSpace(texts)

    remaining = set(range(n))
    selected: list[int] = []
    selection_score: dict[int, float] = {}
    kept_tokens = 0

    while remaining:
        # Only consider chunks that still fit in the remaining budget.
        candidates = sorted(i for i in remaining if kept_tokens + token_counts[i] <= budget_tokens)
        if not candidates:
            break

        best_i = candidates[0]
        best_val = float("-inf")
        for i in candidates:
            redundancy = max((space.cosine(i, j) for j in selected), default=0.0)
            value = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if value > best_val:  # strict '>' keeps the lowest index on ties
                best_val = value
                best_i = i

        selected.append(best_i)
        selection_score[best_i] = best_val
        kept_tokens += token_counts[best_i]
        remaining.discard(best_i)

    selected_set = set(selected)
    order = sorted(selected) if preserve_order else selected
    kept = [
        ScoredChunk(items[i], relevance[i], selection_score[i], token_counts[i]) for i in order
    ]
    dropped = [
        ScoredChunk(items[i], relevance[i], 0.0, token_counts[i])
        for i in range(n)
        if i not in selected_set
    ]
    return CompressionResult(query, kept, dropped, budget_tokens, original_tokens, kept_tokens)
