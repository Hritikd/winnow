"""Command-line interface for winnow.

Examples
--------
Compress a JSON array of ``{"id", "text"}`` chunks to 200 tokens::

    winnow compress --query "refund policy" --budget 200 --input chunks.json

Pipe paragraphs in on stdin (blank-line separated) and print a report::

    cat notes.txt | winnow compress -q "deadline" -b 150 --format report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .compressor import compress
from .tokenizer import get_tokenizer
from .types import Chunk, CompressionResult


def _load_chunks(raw: str, *, as_json: bool) -> list[Chunk]:
    if as_json:
        data = json.loads(raw)
        chunks: list[Chunk] = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                chunks.append(Chunk(id=str(i), text=item))
            else:
                chunks.append(
                    Chunk(
                        id=str(item.get("id", i)),
                        text=item["text"],
                        metadata=item.get("metadata", {}),
                    )
                )
        return chunks
    # Plain text: split on blank lines into paragraph chunks.
    paragraphs = [p.strip() for p in raw.split("\n\n")]
    return [Chunk(id=str(i), text=p) for i, p in enumerate(paragraphs) if p]


def _render(result: CompressionResult, fmt: str) -> str:
    if fmt == "text":
        return result.render()
    if fmt == "json":
        return json.dumps(
            {
                "query": result.query,
                "budget_tokens": result.budget_tokens,
                "original_tokens": result.original_tokens,
                "kept_tokens": result.kept_tokens,
                "reduction": round(result.reduction, 4),
                "kept": [
                    {"id": s.chunk.id, "tokens": s.tokens, "relevance": round(s.relevance, 4)}
                    for s in result.kept
                ],
                "dropped_ids": result.dropped_ids,
            },
            indent=2,
        )
    # report
    lines = [
        f"query           : {result.query}",
        f"budget          : {result.budget_tokens} tokens",
        f"original        : {result.original_tokens} tokens "
        f"({len(result.kept) + len(result.dropped)} chunks)",
        f"kept            : {result.kept_tokens} tokens ({len(result.kept)} chunks)",
        f"reduction       : {result.reduction * 100:.1f}%",
        "",
        "kept chunks (id  rel   tokens):",
    ]
    for s in result.kept:
        lines.append(f"  {s.chunk.id:<8} {s.relevance:0.3f} {s.tokens:>6}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="winnow", description=__doc__)
    parser.add_argument("--version", action="version", version=f"winnow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="compress context to a token budget")
    c.add_argument("-q", "--query", required=True)
    c.add_argument("-b", "--budget", type=int, required=True, help="token budget")
    c.add_argument("-i", "--input", help="input file (defaults to stdin)")
    c.add_argument("--json", action="store_true", help="parse input as a JSON array")
    c.add_argument("--lambda", dest="lambda_", type=float, default=0.7)
    c.add_argument("--format", choices=["text", "json", "report"], default="text")
    c.add_argument(
        "--no-tiktoken",
        action="store_true",
        help="force the dependency-free heuristic tokenizer",
    )

    args = parser.parse_args(argv)

    raw = sys.stdin.read() if not args.input else open(args.input, encoding="utf-8").read()
    chunks = _load_chunks(raw, as_json=args.json)
    tokenizer = get_tokenizer(prefer_tiktoken=not args.no_tiktoken)
    result = compress(
        args.query, chunks, args.budget, lambda_=args.lambda_, tokenizer=tokenizer
    )
    print(_render(result, args.format))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
