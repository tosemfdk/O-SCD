import math

import pytest
import torch

from temporal.binary_state_filter import BinaryStateFilter, BinaryStateFilterConfig


def manual_joint(b, q, w, eta=0.9, p01=0.01, p10=0.01):
    l1 = w * (q * math.log(eta) + (1.0 - q) * math.log(1.0 - eta))
    l0 = w * (q * math.log(1.0 - eta) + (1.0 - q) * math.log(eta))
    vals = torch.tensor([
        math.log(1.0 - b) + math.log(1.0 - p01) + l0,
        math.log(1.0 - b) + math.log(p01) + l1,
        math.log(b) + math.log(p10) + l0,
        math.log(b) + math.log(1.0 - p10) + l1,
    ], dtype=torch.float64)
    return torch.softmax(vals, dim=0)


def test_manual_math_matches_log_domain_joint():
    filt = BinaryStateFilter(1, dtype=torch.float64)
    update = filt.update(
        torch.tensor([9.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        timestamp=7,
    )

    expected = manual_joint(0.5, 0.9, 10.0)
    got = torch.stack([update.p_00[0], update.p_01[0], update.p_10[0], update.p_11[0]])
    assert torch.allclose(got, expected)
    assert update.p_active.item() == pytest.approx((expected[1] + expected[3]).item())
    assert update.p_flip.item() == pytest.approx((expected[1] + expected[2]).item())
    assert update.visible_observations.tolist() == [1]
    assert update.last_timestamp.tolist() == [7]


def test_vector_chunk_indices_update_only_selected_rows_and_reject_duplicates():
    filt = BinaryStateFilter(4, dtype=torch.float64)
    result = filt.update(
        torch.tensor([0.0, 5.0], dtype=torch.float64),
        torch.tensor([5.0, 0.0], dtype=torch.float64),
        indices=torch.tensor([2, 0]),
        timestamp=3,
    )

    assert result.indices.tolist() == [2, 0]
    assert filt.visible_observations.tolist() == [1, 0, 1, 0]
    assert filt.last_timestamp.tolist() == [3, -1, 3, -1]
    assert filt.p_active[0] > 0.5
    assert filt.p_active[2] < 0.5
    assert filt.p_active[1].item() == pytest.approx(0.5)
    assert filt.p_active[3].item() == pytest.approx(0.5)

    with pytest.raises(ValueError, match="unique"):
        filt.update(torch.ones(2), torch.ones(2), indices=torch.tensor([1, 1]), timestamp=4)


def test_soft_fractional_evidence_uses_same_emission_formula():
    filt = BinaryStateFilter(1, dtype=torch.float64)
    result = filt.update(torch.tensor([0.25]), torch.tensor([0.75]), timestamp=1)
    expected = manual_joint(0.5, 0.25, 1.0)
    got = torch.stack([result.p_00[0], result.p_01[0], result.p_10[0], result.p_11[0]])
    assert torch.allclose(got, expected, atol=1e-12)
    assert result.q.item() == pytest.approx(0.25)
    assert result.evidence_strength.item() == pytest.approx(1.0)


def test_unobserved_rows_keep_filter_state_bitwise_unchanged():
    filt = BinaryStateFilter(3, dtype=torch.float64)
    filt.update(torch.tensor([4.0]), torch.tensor([1.0]), indices=[1], timestamp=1)
    before = filt.state_dict()

    result = filt.update(
        torch.tensor([9.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        total_mass=torch.tensor([0.0, 0.0], dtype=torch.float64),
        indices=[1, 2],
        timestamp=2,
    )

    assert result.observed.tolist() == [False, False]
    for name, value in before.items():
        assert torch.equal(filt.state_dict()[name], value), name
    assert result.p_00.tolist() == [0.0, 0.0]
    assert result.p_flip.tolist() == [0.0, 0.0]


def test_min_mass_marks_low_mass_rows_unobserved():
    filt = BinaryStateFilter(1, BinaryStateFilterConfig(min_observation_mass=2.0), dtype=torch.float64)
    before = filt.state_dict()
    result = filt.update(torch.tensor([1.0]), torch.tensor([0.5]), total_mass=torch.tensor([1.5]), timestamp=5)
    assert result.observed.tolist() == [False]
    for name, value in before.items():
        assert torch.equal(filt.state_dict()[name], value), name


@pytest.mark.parametrize(
    "config",
    [
        BinaryStateFilterConfig(emission_reliability=0.5001),
        BinaryStateFilterConfig(initial_p_active=0.0),
        BinaryStateFilterConfig(initial_p_active=1.0),
    ],
)
def test_valid_boundary_configurations(config):
    BinaryStateFilter(1, config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"emission_reliability": 0.5},
        {"emission_reliability": 1.0},
        {"inactive_to_active_prior": 0.0},
        {"active_to_inactive_prior": 1.0},
        {"initial_p_active": -0.1},
        {"initial_p_active": 1.1},
    ],
)
def test_invalid_probabilistic_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BinaryStateFilterConfig(**kwargs)


def test_nonfinite_evidence_and_fractional_indices_are_rejected():
    filt = BinaryStateFilter(1)
    with pytest.raises(ValueError, match="finite"):
        filt.update(torch.tensor([float("nan")]), torch.tensor([0.0]), timestamp=0)
    with pytest.raises(TypeError, match="integers"):
        filt.update(torch.tensor([1.0]), torch.tensor([0.0]), indices=[0.0], timestamp=0)


def test_mass_equal_to_minimum_is_observed_but_zero_mass_is_not():
    filt = BinaryStateFilter(
        2,
        BinaryStateFilterConfig(min_observation_mass=1.0),
        dtype=torch.float64,
    )
    result = filt.update(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        total_mass=torch.tensor([1.0, 0.0]),
        timestamp=0,
    )
    assert result.observed.tolist() == [True, False]
