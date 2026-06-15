from winnow.tokenizer import HeuristicTokenizer, get_tokenizer


def test_empty_text_is_zero_tokens():
    assert HeuristicTokenizer().count("") == 0


def test_counts_are_positive_and_monotone():
    tok = HeuristicTokenizer()
    short = tok.count("hello world")
    longer = tok.count("hello world this is a longer sentence with more tokens")
    assert short > 0
    assert longer > short


def test_long_words_cost_more_than_one_token():
    tok = HeuristicTokenizer()
    # A 16-char word should map to roughly 4 sub-word tokens, not 1.
    assert tok.count("internationalization") >= 4


def test_punctuation_counts():
    tok = HeuristicTokenizer()
    assert tok.count("hi, there.") == tok.count("hi there") + 2  # comma + period


def test_get_tokenizer_falls_back_without_tiktoken():
    tok = get_tokenizer(prefer_tiktoken=False)
    assert tok.name == "heuristic"
    assert tok.count("anything") > 0


def test_heuristic_within_reason_of_word_count():
    tok = HeuristicTokenizer()
    text = "the quick brown fox jumps over the lazy dog"
    # 9 short words -> token count should be in a sane neighborhood of word count
    assert 8 <= tok.count(text) <= 14
