"""winnow — budget-aware context compression for RAG and LLM agents.

Winnow a large context down to the highest-signal subset that fits a token
budget. Deterministic, dependency-free, no API keys.

    >>> from winnow import compress
    >>> result = compress("how do I reset my password?", chunks, budget_tokens=300)
    >>> result.render()          # the kept context, ready to prompt with
    >>> result.reduction         # e.g. 0.62  -> 62% fewer tokens
"""

from __future__ import annotations

from .compressor import compress
from .tokenizer import HeuristicTokenizer, TiktokenTokenizer, get_tokenizer
from .types import Chunk, CompressionResult, ScoredChunk

__all__ = [
    "compress",
    "Chunk",
    "CompressionResult",
    "ScoredChunk",
    "get_tokenizer",
    "HeuristicTokenizer",
    "TiktokenTokenizer",
]

__version__ = "0.1.0"
