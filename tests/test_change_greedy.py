# Gate S1: deterministic batch-static greedy properties (spec §11).
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.greedy import greedy_static_select  # noqa: E402


def _toy_infos(seed=0, n_frames=8, n_gauss=32):
    g = torch.Generator().manual_seed(seed)
    return {i: torch.rand(n_gauss, generator=g) * (i % 3 + 0.2)
            for i in range(n_frames)}


def test_budget_and_no_duplicates():
    res = greedy_static_select(_toy_infos(), budget=5, criterion="dopt",
                               lambda_value=1e-3)
    assert len(res.greedy_order) == 5
    assert len(set(res.greedy_order)) == 5
    assert res.replay_order == sorted(res.greedy_order)
    assert [s.step for s in res.steps] == list(range(5))


def test_shuffle_invariance():
    infos = _toy_infos(seed=1)
    a = greedy_static_select(infos, 4, "dopt", 1e-3).greedy_order
    for seed in range(3):
        items = list(infos.items())
        random.Random(seed).shuffle(items)
        b = greedy_static_select(dict(items), 4, "dopt", 1e-3).greedy_order
        assert a == b


def test_tie_break_smaller_frame_id():
    b = torch.tensor([1.0, 2.0, 0.5])
    infos = {7: b.clone(), 3: b.clone(), 5: b.clone()}
    res = greedy_static_select(infos, 2, "dopt", 1e-3)
    assert res.greedy_order == [3, 5]


def test_fixed_state_marginal_nonincreasing():
    infos = _toy_infos(seed=2)
    res = greedy_static_select(infos, len(infos), "dopt", 1e-3)
    gains = [s.selected_score for s in res.steps]
    for prev, nxt in zip(gains, gains[1:]):
        assert nxt <= prev + 1e-9


def test_error_paths():
    infos = _toy_infos()
    assert greedy_static_select(infos, 0, "dopt", 1e-3).greedy_order == []
    with pytest.raises(ValueError):
        greedy_static_select(infos, len(infos) + 1, "dopt", 1e-3)
    bad = dict(infos); bad[0] = bad[0].clone(); bad[0][0] = float("nan")
    with pytest.raises(FloatingPointError):
        greedy_static_select(bad, 2, "dopt", 1e-3)
    neg = dict(infos); neg[1] = neg[1].clone(); neg[1][0] = -1.0
    with pytest.raises(ValueError):
        greedy_static_select(neg, 2, "dopt", 1e-3)
