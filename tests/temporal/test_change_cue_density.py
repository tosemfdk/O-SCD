import pytest
import torch

from temporal.change_cue_density import (
    aggregate_soft_change_masses,
    causal_random_view_indices,
    fastgs_change_candidate_masks,
)


def test_causal_sampling_includes_current_and_never_uses_future():
    for timestamp in range(15):
        for k in (1, 3, 5, 10):
            selected = causal_random_view_indices(timestamp, k, seed=7)
            assert selected[-1] == timestamp
            assert len(selected) == min(k, timestamp + 1)
            assert len(set(selected)) == len(selected)
            assert all(0 <= index <= timestamp for index in selected)


def test_k1_is_exact_current_view_mass_and_soft_values_are_preserved():
    positive = torch.tensor([0.25, 2.5, 7.75])
    negative = torch.tensor([0.75, 0.5, 0.25])
    plus, minus, total, ratio, importance = aggregate_soft_change_masses(
        [positive], [negative]
    )
    assert torch.equal(plus, positive)
    assert torch.equal(minus, negative)
    assert torch.equal(total, positive + negative)
    assert torch.equal(importance, positive)
    assert torch.allclose(ratio, positive / (positive + negative + 1e-8))


def test_persistent_support_passes_average_threshold_but_one_view_spike_does_not():
    persistent = [torch.tensor([6.0])] * 10
    spike = [torch.tensor([60.0])] + [torch.tensor([0.0])] * 9
    negative = [torch.tensor([1.0])] * 10
    *_, persistent_score = aggregate_soft_change_masses(persistent, negative)
    *_, spike_score = aggregate_soft_change_masses(spike, negative)
    assert persistent_score.item() > 5.0
    # FastGS uses a strict > threshold, so an average of exactly six would
    # pass. Use a 50-pixel spike to exercise a one-view average of exactly 5.
    spike[0] = torch.tensor([50.0])
    *_, spike_score = aggregate_soft_change_masses(spike, negative)
    assert not bool((spike_score > 5.0).item())


def test_empty_change_cue_produces_no_importance_candidate():
    positive = [torch.zeros(4) for _ in range(3)]
    negative = [torch.ones(4) for _ in range(3)]
    *_, importance = aggregate_soft_change_masses(positive, negative)
    assert torch.count_nonzero(importance > 0.0).item() == 0


class _FakeBank:
    def __init__(self):
        self._xyz = torch.zeros((4, 3))
        self._scaling_value = torch.tensor(
            [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]
        )
        self.xyz_gradient_accum = torch.tensor([[2.0], [2.0], [0.5], [2.0]])
        self.xyz_gradient_accum_abs = torch.tensor([[2.0], [2.0], [2.0], [0.5]])
        self.denom = torch.ones((4, 1))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling_value


def test_fastgs_masks_require_gradient_and_change_importance():
    bank = _FakeBank()
    masks = fastgs_change_candidate_masks(
        bank,
        torch.tensor([6.0, 4.0, 6.0, 6.0]),
        scene_extent=10.0,
        importance_threshold=5.0,
        grad_threshold=1.0,
        grad_abs_threshold=1.0,
        dense_fraction=0.01,
    )
    assert masks.importance.tolist() == [True, False, True, True]
    assert masks.clone.tolist() == [True, False, False, False]
    assert masks.split.tolist() == [False, False, True, False]


@pytest.mark.parametrize(
    "timestamp,k",
    [(-1, 1), (0, 0)],
)
def test_invalid_causal_sampling_is_rejected(timestamp, k):
    with pytest.raises(ValueError):
        causal_random_view_indices(timestamp, k, seed=0)
