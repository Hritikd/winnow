import pytest

from winnow import Chunk, compress
from winnow.tokenizer import HeuristicTokenizer

TOK = HeuristicTokenizer()


def test_budget_is_never_exceeded():
    chunks = [f"chunk number {i} with some filler words here" for i in range(20)]
    result = compress("filler words", chunks, budget_tokens=40, tokenizer=TOK)
    assert result.kept_tokens <= 40


def test_generous_budget_keeps_everything():
    chunks = ["alpha one", "beta two", "gamma three"]
    result = compress("alpha", chunks, budget_tokens=10_000, tokenizer=TOK)
    assert len(result.kept) == 3
    assert result.dropped == []


def test_relevant_chunk_is_selected_over_filler():
    chunks = [
        Chunk("filler1", "unrelated text about gardening and the weather today"),
        Chunk("filler2", "more unrelated text about cooking pasta for dinner"),
        Chunk("answer", "to reset your password open settings and click reset password"),
    ]
    # Budget only fits one chunk; it should be the relevant one, not filler.
    result = compress("how do I reset my password", chunks, budget_tokens=18, tokenizer=TOK)
    assert result.kept_ids == ["answer"]


def test_mmr_prefers_diverse_chunks_over_duplicates():
    chunks = [
        Chunk("dupA", "the deployment failed because the database migration timed out"),
        Chunk("dupB", "the deployment failed because the database migration timed out"),
        Chunk("other", "rollback is performed automatically when a health check fails"),
    ]
    # With lambda<1, after taking one duplicate the near-identical twin should be
    # penalized, letting the distinct 'other' chunk in.
    diverse = compress("deployment failure rollback", chunks, budget_tokens=22,
                       lambda_=0.5, tokenizer=TOK)
    assert "other" in diverse.kept_ids


def test_preserve_order_returns_original_order():
    chunks = [Chunk(str(i), f"sentence {i} alpha beta") for i in range(5)]
    result = compress("alpha", chunks, budget_tokens=10_000, tokenizer=TOK)
    ids = result.kept_ids
    assert ids == sorted(ids, key=int)


def test_reduction_property():
    chunks = ["a b c d e f g h", "i j k l m n o p", "q r s t u v w x"]
    result = compress("a", chunks, budget_tokens=6, tokenizer=TOK)
    assert 0.0 <= result.reduction <= 1.0
    assert result.kept_tokens <= result.original_tokens


def test_empty_input():
    result = compress("anything", [], budget_tokens=100, tokenizer=TOK)
    assert result.kept == [] and result.dropped == []
    assert result.reduction == 0.0


def test_render_roundtrips_kept_text():
    chunks = [Chunk("x", "hello world"), Chunk("y", "goodbye moon")]
    result = compress("hello", chunks, budget_tokens=10_000, tokenizer=TOK)
    assert "hello world" in result.render()


def test_determinism():
    chunks = [f"chunk {i} about retries and backoff and jitter" for i in range(12)]
    a = compress("retries backoff", chunks, budget_tokens=30, tokenizer=TOK)
    b = compress("retries backoff", chunks, budget_tokens=30, tokenizer=TOK)
    assert a.kept_ids == b.kept_ids


def test_invalid_lambda_raises():
    with pytest.raises(ValueError):
        compress("q", ["a"], 10, lambda_=1.5, tokenizer=TOK)


def test_negative_budget_raises():
    with pytest.raises(ValueError):
        compress("q", ["a"], -1, tokenizer=TOK)


def test_lambda_one_is_pure_relevance():
    # Pure-relevance mode must still respect the budget and return scored chunks.
    chunks = [f"doc {i} retries backoff jitter" for i in range(8)]
    result = compress("retries", chunks, budget_tokens=20, lambda_=1.0, tokenizer=TOK)
    assert result.kept_tokens <= 20
    assert all(s.relevance >= 0.0 for s in result.kept)
