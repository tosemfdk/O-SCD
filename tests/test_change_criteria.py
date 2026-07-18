# Gate S1: criterion formulas, duplicate-view decay, float64 sums (spec §6.1).
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.criteria import CRITERIA, score_candidate  # noqa: E402


def test_formulas_match_closed_form():
    h = torch.tensor([1.0, 2.0, 4.0])
    b = torch.tensor([1.0, 0.0, 4.0])
    assert score_candidate("dopt", h, b) == pytest.approx(
        math.log(2.0) + 0.0 + math.log(2.0))
    assert score_candidate("trace_reduction", h, b) == pytest.approx(
        (1 - 1 / 2) + 0.0 + (1 / 4 - 1 / 8))
    assert score_candidate("fisher_ratio", h, b) == pytest.approx(1.0 + 0.0 + 1.0)
    assert score_candidate("candidate_only", h, b) == pytest.approx(5.0)


def test_duplicate_view_decays_strictly():
    lam = 1e-3
    b = torch.tensor([0.5, 0.0, 2.0])
    h = torch.full_like(b, lam)
    for crit in ("dopt", "trace_reduction", "fisher_ratio"):
        first = score_candidate(crit, h, b)
        second = score_candidate(crit, h + b, b)
        assert second < first, crit
    # candidate_only ignores the prior by design
    assert score_candidate("candidate_only", h, b) == \
        score_candidate("candidate_only", h + b, b)


def test_float64_sum_precision():
    n = 200_000
    h = torch.full((n,), 1.0)
    b = torch.full((n,), 1e-6)
    # float32 accumulation of n*log1p(1e-6) loses digits; float64 keeps them
    assert score_candidate("dopt", h, b) == pytest.approx(n * math.log1p(1e-6),
                                                          rel=1e-6)


def test_nan_raises_and_unknown_raises():
    h = torch.tensor([1.0]); b = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        score_candidate("dopt", h, b)
    with pytest.raises(ValueError):
        score_candidate("nope", h, torch.tensor([1.0]))
    assert set(CRITERIA) == {"dopt", "trace_reduction", "fisher_ratio",
                             "candidate_only"}
