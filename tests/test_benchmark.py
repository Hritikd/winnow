"""Guard-rail test: winnow must beat naive head-truncation on the benchmark.

This keeps the headline claim in the README honest and regression-proof — if a
change makes winnow no better than truncation, CI fails.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench  # noqa: E402


def test_winnow_beats_head_truncation():
    summary = bench.run()
    for col in summary["winnow"]:
        assert summary["winnow"][col] > summary["head"][col], (
            f"winnow ({summary['winnow'][col]:.2f}) did not beat "
            f"head-truncation ({summary['head'][col]:.2f}) on {col}"
        )


def test_winnow_beats_random_floor():
    summary = bench.run()
    for col in summary["winnow"]:
        assert summary["winnow"][col] >= summary["random"][col]


def test_winnow_at_least_matches_pure_bm25():
    # Diversity should never hurt overall gold recall on this suite.
    summary = bench.run()
    for col in summary["winnow"]:
        assert summary["winnow"][col] >= summary["bm25"][col] - 1e-9
