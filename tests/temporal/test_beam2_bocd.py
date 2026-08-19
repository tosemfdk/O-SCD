import torch

from temporal.beam2_bocd import BeamTwoBernoulliFilter
from temporal.bernoulli_bocd import (
    BernoulliBOCDConfig,
    MAPResetBernoulliFilter,
)


def _clone(filter_):
    return {name: value.clone() for name, value in filter_.state_dict().items()}


def test_beam2_preserves_unit_capped_reset_branch_that_map_discards():
    config = BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        expected_run_length=100.0,
        max_run_length=128,
        changepoint_probability=0.5,
    )
    map_reset = MAPResetBernoulliFilter(1, config, dtype=torch.float64)
    beam2 = BeamTwoBernoulliFilter(1, config, dtype=torch.float64)
    sequence = [(1.0, 0.0)] * 40 + [(0.0, 1.0)] * 40 + [(1.0, 0.0)] * 40

    map_changepoints = []
    beam_changepoints = []
    for timestamp, (positive, negative) in enumerate(sequence):
        positive_tensor = torch.tensor([positive], dtype=torch.float64)
        negative_tensor = torch.tensor([negative], dtype=torch.float64)
        map_update = map_reset.update(
            positive_tensor, negative_tensor, timestamp=timestamp
        )
        beam_update = beam2.update(
            positive_tensor, negative_tensor, timestamp=timestamp
        )
        if map_update.changepoint_probability.item() >= 0.5:
            map_changepoints.append(timestamp)
        if beam_update.changepoint_probability.item() >= 0.5:
            beam_changepoints.append(
                (timestamp, int(beam_update.estimated_run_start.item()))
            )

    assert map_changepoints == []
    assert beam_changepoints == [(41, 40), (81, 80)]


def test_beam2_branch_posterior_is_normalized_and_diagnostics_are_finite():
    config = BernoulliBOCDConfig(expected_run_length=100.0, max_run_length=16)
    filter_ = BeamTwoBernoulliFilter(1, config, dtype=torch.float64)
    for timestamp in range(8):
        update = filter_.update(
            torch.tensor([1.0 if timestamp < 4 else 0.0], dtype=torch.float64),
            torch.tensor([0.0 if timestamp < 4 else 1.0], dtype=torch.float64),
            timestamp=timestamp,
        )

    assert torch.allclose(
        update.branch_probability.sum(dim=1),
        torch.ones(1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        update.run_length_posterior.sum(dim=1),
        torch.ones(1, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.isfinite(update.log_predictive_continue).all()
    assert torch.isfinite(update.log_predictive_reset).all()
    assert torch.isfinite(update.log_bayes_factor).all()


def test_beam2_chunked_updates_match_full_batch_updates():
    config = BernoulliBOCDConfig(hazard=0.05, max_run_length=16)
    full = BeamTwoBernoulliFilter(5, config, dtype=torch.float64)
    chunked = BeamTwoBernoulliFilter(5, config, dtype=torch.float64)
    frames = [
        torch.tensor([0.2, 1.0, 0.0, 0.7, 0.3], dtype=torch.float64),
        torch.tensor([0.1, 0.9, 0.4, 0.6, 0.8], dtype=torch.float64),
        torch.tensor([0.8, 0.2, 0.3, 0.4, 0.9], dtype=torch.float64),
    ]
    for timestamp, positive in enumerate(frames):
        negative = 1.0 - positive
        full.update(positive, negative, timestamp=timestamp)
        chunked.update(
            positive[:2],
            negative[:2],
            row_indices=torch.tensor([0, 1]),
            timestamp=timestamp,
        )
        chunked.update(
            positive[2:],
            negative[2:],
            row_indices=torch.tensor([2, 3, 4]),
            timestamp=timestamp,
        )

    for name, expected in full.state_dict().items():
        actual = chunked.state_dict()[name]
        if expected.is_floating_point():
            assert torch.allclose(expected, actual, atol=1e-12, rtol=1e-12), name
        else:
            assert torch.equal(expected, actual), name


def test_beam2_unobserved_rows_preserve_every_persistent_state_tensor():
    config = BernoulliBOCDConfig(min_evidence_mass=0.5, max_run_length=8)
    filter_ = BeamTwoBernoulliFilter(2, config, dtype=torch.float64)
    filter_.update(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        timestamp=2,
    )
    before = _clone(filter_)

    result = filter_.update(
        torch.tensor([9.0], dtype=torch.float64),
        torch.tensor([9.0], dtype=torch.float64),
        total_mass=torch.tensor([0.0], dtype=torch.float64),
        row_indices=torch.tensor([1]),
        timestamp=20,
    )

    for name, expected in before.items():
        assert torch.equal(expected, filter_.state_dict()[name]), name
    assert result.observed.tolist() == [False]


def test_beam2_persistent_storage_is_linear():
    filter_ = BeamTwoBernoulliFilter(7, BernoulliBOCDConfig(max_run_length=64))
    for name, value in filter_.state_dict().items():
        assert value.ndim == 1, name
        assert value.shape == (7,), name
