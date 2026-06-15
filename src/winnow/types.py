"""Core data types for winnow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A unit of retrievable context (a passage, document, tool result, ...)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk annotated with the scores winnow assigned to it."""

    chunk: Chunk
    relevance: float
    """Normalized BM25 relevance in ``[0, 1]`` (1 == most relevant chunk)."""
    selection_score: float
    """The MMR objective value at the moment the chunk was selected."""
    tokens: int


@dataclass
class CompressionResult:
    """The outcome of a :func:`winnow.compress` call."""

    query: str
    kept: list[ScoredChunk]
    dropped: list[ScoredChunk]
    budget_tokens: int
    original_tokens: int
    kept_tokens: int

    @property
    def reduction(self) -> float:
        """Fraction of tokens removed, in ``[0, 1]``."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - self.kept_tokens / self.original_tokens

    @property
    def kept_ids(self) -> list[str]:
        return [s.chunk.id for s in self.kept]

    @property
    def dropped_ids(self) -> list[str]:
        return [s.chunk.id for s in self.dropped]

    def render(self, separator: str = "\n\n") -> str:
        """Join the kept chunks back into a single context string."""
        return separator.join(s.chunk.text for s in self.kept)
