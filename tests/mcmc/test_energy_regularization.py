import pytest
import torch

from temporal.mcmc_energy import MCMCEnergy
from temporal.mcmc_state import FixedCapacityChangeState
from .conftest import make_base


def _state_with_known_attrs():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=2), capacity=2)
    with torch.no_grad():
        state.current_raw_change_opacity.copy_(torch.logit(torch.tensor([[0.2], [0.4]])))
        state.current_scaling.copy_(torch.log(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
    return state


@pytest.mark.parametrize("reduction,expected_opacity,expected_scale", [
    ("mean", 0.3, 3.5),
    ("sum", 0.6, 21.0),
])
def test_regularizers_honor_mean_and_sum_reductions(reduction, expected_opacity, expected_scale):
    state = _state_with_known_attrs()
    energy = MCMCEnergy(opacity_weight=2.0, scale_weight=0.5, reduction=reduction)

    terms = energy.regularization_terms(state)

    assert terms["opacity"].item() == pytest.approx(expected_opacity * 2.0, rel=1e-6)
    assert terms["scale"].item() == pytest.approx(expected_scale * 0.5, rel=1e-6)


def test_regularizer_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        MCMCEnergy(reduction="median")
