<h1 align="center">winnow</h1>

<p align="center">
  <b>Budget-aware context compression for RAG and LLM agents.</b><br/>
  Winnow a large context down to the highest-signal subset that fits a token budget —
  <i>deterministic, dependency-free, no API keys.</i>
</p>

<p align="center">
  <img alt="CI" src="https://github.com/Hritikd/winnow/actions/workflows/ci.yml/badge.svg" />
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg" />
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img alt="Dependencies: 0" src="https://img.shields.io/badge/runtime%20deps-0-success.svg" />
</p>

<p align="center">
  <b><a href="https://hritikd.github.io/winnow/">▶ Try the live demo</a></b> — runs the real library in your browser (Pyodide), no install, no API keys.
</p>

---

## The problem

Every RAG pipeline and every agent eventually overflows its context window. The
retriever returns 20 passages, the agent accumulates a long scratchpad, the
budget says 2,000 tokens — so something has to go. The default answer is to
**truncate**: keep the top-k by similarity, or just cut the context to length.

Truncation throws away relevant passages that happen to sit lower in the list,
and it cheerfully keeps three near-duplicate chunks that say the same thing.
You pay for tokens that carry no new information, and you drop tokens that did.

**winnow** treats context-packing as what it actually is: a constrained
optimization. Given a query, a set of chunks, and a token budget, it selects
the subset that **maximizes relevance while penalizing redundancy**, packed to
stay under budget — and it does it deterministically, with no model and no
network call.

## Install

```bash
pip install winnow-context           # zero runtime dependencies
pip install "winnow-context[tiktoken]"   # optional: exact OpenAI token counts
```

## Quickstart

```python
from winnow import Chunk, compress

chunks = [
    Chunk("c0", "Welcome! This guide covers account creation and billing tiers."),
    Chunk("c1", "Our dashboard shows usage charts and monthly invoices."),
    Chunk("c3", "To rotate an expired API key without downtime: create a new key, "
                "deploy it everywhere, confirm traffic, then revoke the old one."),
    Chunk("c4", "Rate limits default to 600 requests per minute per key."),
]

result = compress("how do I rotate an expired API key?", chunks, budget_tokens=60)

result.render()      # -> the kept context, ready to drop into your prompt
result.reduction     # -> 0.61   (61% fewer tokens)
result.kept_ids      # -> ['c3', 'c4']
```

Or from the shell:

```bash
echo '[{"id":"a","text":"..."},{"id":"b","text":"..."}]' \
  | winnow compress --query "refund policy" --budget 200 --json --format report
```

## How it works

winnow scores every chunk against the query with **BM25** (a strong, classic
lexical relevance signal — no embeddings required), then selects chunks with
**Maximal Marginal Relevance** (Carbonell & Goldstein, 1998) under a
knapsack-style budget constraint:

```
MMR(c) = λ · relevance(c) − (1 − λ) · max  similarity(c, s)
                                      s ∈ selected
```

* **relevance** is normalized BM25 between the query and the chunk.
* **similarity** is TF-IDF cosine between a candidate chunk and the chunks
  already chosen — this is the redundancy penalty that stops winnow from
  spending budget on three paraphrases of the same fact.
* **λ** (default `0.7`) trades relevance against diversity. `λ = 1.0` reduces
  winnow to pure BM25 packing; lower values buy broader coverage.

Every step is deterministic: the same inputs always produce the same output, so
winnow is safe to put in a test and reason about in a review.

## Does it actually help? (reproducible benchmark)

`benchmarks/data.json` is a suite of 8 retrieval cases where the relevant
("gold") passages are **scattered**, not conveniently at the front — the
realistic situation, since you rarely know where in a retrieved context the
answer lives. We compress each context to a fixed fraction of its token budget
and measure **gold recall**: the fraction of human-labeled relevant chunks that
survive.

| method | gold recall @ 25% budget | gold recall @ 40% budget | gold recall @ 60% budget |
|---|---|---|---|
| **winnow** (relevance + diversity) | **0.38** | **0.75** | **0.88** |
| bm25 (relevance only) | 0.38 | 0.75 | 0.88 |
| head-truncation (naive default) | 0.00 | 0.19 | 0.44 |
| random | 0.19 | 0.38 | 0.44 |

```bash
python benchmarks/bench.py        # reproduce this table (deterministic)
```

**Reading this honestly:**

* The headline result is **winnow vs. truncation** — the thing people actually
  do. At a 40% budget winnow keeps **0.75** of the relevant context where
  head-truncation keeps **0.19**. As the budget grows the gap narrows (at 60%,
  0.88 vs 0.44), exactly as you'd expect: when you can afford to keep most of
  the context, selection matters less.
* On this recall metric, winnow **ties** pure BM25 — the diversity term doesn't
  change *which gold chunks* survive here. Its job is different: it stops winnow
  from filling the budget with near-duplicate chunks. That behavior is pinned by
  a unit test (`test_mmr_prefers_diverse_chunks`) rather than this metric, which
  can't see redundancy.
* The suite is small (8 cases) and uses the dependency-free heuristic tokenizer
  so anyone reproduces the same numbers. It's an illustration, not a leaderboard.

## API

```python
compress(
    query: str,
    chunks: Sequence[Chunk | str],   # Chunk objects, or raw strings
    budget_tokens: int,
    *,
    lambda_: float = 0.7,            # relevance vs. diversity, in [0, 1]
    tokenizer: Tokenizer | None = None,   # defaults to tiktoken if installed
    preserve_order: bool = True,     # keep original order, or selection order
) -> CompressionResult
```

`CompressionResult` exposes `.kept`, `.dropped`, `.kept_ids`, `.kept_tokens`,
`.original_tokens`, `.reduction`, and `.render(separator)` to rebuild the
context string. Each kept chunk carries its `relevance` and `selection_score`.

## Design principles

```text
Deterministic by default   same inputs -> same output, always testable
Zero runtime dependencies  pure-Python core; tiktoken is opt-in
No API keys to review       clone, run the benchmark, understand it in a minute
Honest measurement first    a benchmark you can reproduce, framed for what it is
```

## Limitations & scope

* winnow is **lexical** (BM25 / TF-IDF). It does not capture semantic matches
  that share no words. A natural extension is a pluggable embedding-backed
  relevance/similarity backend behind the same interface — the algorithm is
  agnostic to where the scores come from.
* The heuristic tokenizer is an *approximation* of BPE counts (within ~10–15%
  of `tiktoken` on English prose). Install the `tiktoken` extra for exact counts.
* The bundled benchmark is a small, hand-labeled illustration, not a claim of
  state-of-the-art retrieval.

## Development

```bash
make install   # pip install -e ".[dev]"
make test      # pytest  (32 tests)
make lint      # ruff
make bench     # regenerate benchmarks/results.md
```

## License

MIT © Hritik Datta
