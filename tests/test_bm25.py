from winnow.bm25 import BM25


def test_relevant_doc_scores_higher():
    docs = [
        "the cat sat on the mat",
        "exponential backoff retries with jitter avoid thundering herd",
        "a recipe for banana bread with walnuts",
    ]
    bm25 = BM25(docs)
    scores = bm25.scores("how to configure backoff and jitter for retries")
    assert scores[1] == max(scores)
    assert scores[1] > 0


def test_query_with_no_overlap_scores_zero():
    bm25 = BM25(["alpha beta gamma", "delta epsilon zeta"])
    assert bm25.scores("xylophone quokka") == [0.0, 0.0]


def test_empty_corpus_is_safe():
    bm25 = BM25([])
    assert bm25.scores("anything") == []


def test_idf_downweights_common_terms():
    # 'system' appears everywhere; 'kafka' is rare and should drive the ranking.
    docs = [
        "system overview and system goals",
        "system design for a kafka streaming pipeline",
        "system maintenance schedule",
    ]
    bm25 = BM25(docs)
    scores = bm25.scores("kafka system")
    assert scores[1] == max(scores)
