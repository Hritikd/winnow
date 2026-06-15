"""Minimal end-to-end example. Run: python examples/quickstart.py"""

from winnow import Chunk, compress

# Imagine these came from a vector store / agent scratchpad.
chunks = [
    Chunk("c0", "Welcome! This guide covers account creation and billing tiers."),
    Chunk("c1", "Our dashboard shows usage charts and monthly invoices."),
    Chunk("c2", "Support is available 24/7 via chat for all customers."),
    Chunk(
        "c3",
        "To rotate an expired API key without downtime: create a new key, deploy "
        "it everywhere, confirm traffic on the new key, then revoke the old one.",
    ),
    Chunk("c4", "Rate limits default to 600 requests per minute per key."),
    Chunk(
        "c5",
        "The CLI automates rotation: `platform keys rotate --grace 24h` keeps the "
        "old key valid for a 24h overlap before revoking it.",
    ),
]

query = "how do I rotate an expired API key without downtime?"

result = compress(query, chunks, budget_tokens=60)

print(f"query        : {query}")
print(f"original     : {result.original_tokens} tokens across {len(chunks)} chunks")
print(f"kept         : {result.kept_tokens} tokens across {len(result.kept)} chunks")
print(f"reduction    : {result.reduction * 100:.0f}%")
print(f"kept chunks  : {result.kept_ids}")
print("\n--- context to send to the LLM ---")
print(result.render())
