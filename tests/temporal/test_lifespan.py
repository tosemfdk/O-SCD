import pytest
import torch

from temporal import get_active_state_indices, temporal_gate


@pytest.mark.parametrize(
    "boundaries",
    ([95.0, 199.0], [105.0, 209.0]),
    ids=("scene_change1_2_3", "scene_change3_2_1"),
)
def test_temporal_gate_uses_sequence_boundaries(boundaries):
    starts = torch.tensor([[0.0, boundaries[0], boundaries[1]]]).repeat(2, 1)
    ends = torch.tensor([[boundaries[0], boundaries[1], float("inf")]]).repeat(2, 1)
    valid = torch.ones_like(starts, dtype=torch.bool)

    queries = (
        (0.0, 0),
        (boundaries[0] - 0.001, 0),
        (boundaries[0], 1),
        (boundaries[1] - 0.001, 1),
        (boundaries[1], 2),
        (boundaries[1] + 100.0, 2),
    )

    for timestamp, active_state in queries:
        expected = torch.zeros_like(valid)
        expected[:, active_state] = True
        assert torch.equal(temporal_gate(timestamp, starts, ends, valid), expected)


def test_temporal_gate_ignores_invalid_states():
    starts = torch.tensor([[0.0, 0.0]])
    ends = torch.tensor([[float("inf"), float("inf")]])
    valid = torch.tensor([[True, False]])

    active = temporal_gate(10.0, starts, ends, valid)

    assert active.shape == valid.shape
    assert active.dtype == torch.bool
    assert active.device == valid.device
    assert torch.equal(active, torch.tensor([[True, False]]))


def test_get_active_state_indices_selects_single_state_per_gaussian():
    starts = torch.tensor([[0.0, 95.0, 199.0], [0.0, 95.0, 199.0]])
    ends = torch.tensor([[95.0, 199.0, float("inf")]]).repeat(2, 1)
    valid = torch.ones_like(starts, dtype=torch.bool)

    assert torch.equal(get_active_state_indices(94.0, starts, ends, valid), torch.tensor([0, 0]))
    assert torch.equal(get_active_state_indices(95.0, starts, ends, valid), torch.tensor([1, 1]))
    assert torch.equal(get_active_state_indices(199.0, starts, ends, valid), torch.tensor([2, 2]))


@pytest.mark.parametrize(
    "mutate,exc,match",
    [
        (lambda s, e, v: (s[:1], e, v), ValueError, "exactly the same shape"),
        (lambda s, e, v: (s[:, :0], e[:, :0], v[:, :0]), ValueError, "nonempty"),
        (lambda s, e, v: (s[0], e[0], v[0]), ValueError, "rank-2"),
        (lambda s, e, v: (s.to(torch.int64), e, v), TypeError, "floating"),
        (lambda s, e, v: (s, e.to(torch.float64), v), TypeError, "same dtype"),
        (lambda s, e, v: (s, e, v.to(torch.float32)), TypeError, "bool"),
        (lambda s, e, v: (s, s.clone(), v), ValueError, "start < state_end"),
    ],
)
def test_get_active_state_indices_rejects_invalid_state_tensors(mutate, exc, match):
    starts = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    ends = torch.tensor([[10.0, float("inf")], [10.0, float("inf")]])
    valid = torch.ones_like(starts, dtype=torch.bool)

    bad_starts, bad_ends, bad_valid = mutate(starts, ends, valid)

    with pytest.raises(exc, match=match):
        get_active_state_indices(1.0, bad_starts, bad_ends, bad_valid)


@pytest.mark.parametrize(
    "timestamp",
    [float("nan"), float("inf"), True, torch.tensor([1.0, 2.0]), torch.tensor(False)],
)
def test_get_active_state_indices_rejects_invalid_timestamp(timestamp):
    starts = torch.tensor([[0.0]])
    ends = torch.tensor([[float("inf")]])
    valid = torch.tensor([[True]])

    with pytest.raises((TypeError, ValueError), match="timestamp"):
        get_active_state_indices(timestamp, starts, ends, valid)


def test_get_active_state_indices_uses_minus_one_for_outdated_gaussians():
    starts = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    ends = torch.tensor([[5.0, 10.0], [5.0, 10.0]])
    valid = torch.ones_like(starts, dtype=torch.bool)

    assert torch.equal(
        get_active_state_indices(11.0, starts, ends, valid),
        torch.tensor([-1, -1]),
    )


def test_get_active_state_indices_supports_mixed_gaussian_lifespans():
    starts = torch.tensor([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    ends = torch.tensor([[5.0, 10.0], [5.0, 10.0], [5.0, 10.0]])
    valid = torch.tensor([[True, False], [True, True], [False, True]])

    assert torch.equal(
        get_active_state_indices(6.0, starts, ends, valid),
        torch.tensor([-1, 1, 1]),
    )


def test_get_active_state_indices_supports_optional_manual_segments():
    starts = torch.tensor([[0.0, 95.0, 199.0]]).repeat(5, 1)
    ends = torch.tensor([[95.0, 199.0, float("inf")]]).repeat(5, 1)
    valid = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [False, True, True],
            [False, False, True],
        ]
    )

    assert torch.equal(
        get_active_state_indices(94.0, starts, ends, valid),
        torch.tensor([0, 0, 0, -1, -1]),
    )
    assert torch.equal(
        get_active_state_indices(95.0, starts, ends, valid),
        torch.tensor([-1, 1, 1, 1, -1]),
    )
    assert torch.equal(
        get_active_state_indices(199.0, starts, ends, valid),
        torch.tensor([-1, -1, 2, 2, 2]),
    )


def test_get_active_state_indices_rejects_overlapping_states():
    starts = torch.tensor([[0.0, 0.0]])
    ends = torch.tensor([[10.0, 10.0]])
    valid = torch.ones_like(starts, dtype=torch.bool)

    with pytest.raises(ValueError, match="at most one active state"):
        get_active_state_indices(1.0, starts, ends, valid)
