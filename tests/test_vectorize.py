from winnow.vectorize import TfidfSpace


def test_self_similarity_is_one():
    space = TfidfSpace(["alpha beta gamma", "delta epsilon"])
    assert abs(space.cosine(0, 0) - 1.0) < 1e-9


def test_identical_docs_are_maximally_similar():
    space = TfidfSpace(["machine learning pipeline", "machine learning pipeline"])
    assert abs(space.cosine(0, 1) - 1.0) < 1e-9


def test_disjoint_docs_are_orthogonal():
    space = TfidfSpace(["cat dog bird", "stock bond equity"])
    assert space.cosine(0, 1) == 0.0


def test_partial_overlap_is_between_zero_and_one():
    space = TfidfSpace(["machine learning model", "machine learning pipeline"])
    sim = space.cosine(0, 1)
    assert 0.0 < sim < 1.0
