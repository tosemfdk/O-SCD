import pytest
import torch

from temporal.mcmc_state import FixedCapacityChangeState
from .conftest import make_base


def test_fixed_capacity_preserves_parameter_identity_when_resetting_and_activating_slots():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=2), capacity=5, base_zero=False)
    before = {name: id(parameter) for name, parameter in state.current_parameter_items()}

    snapshot = state.snapshot(detach=True, cpu=False)
    snapshot["raw_change_opacity"].fill_(2.0)
    state.reset_current(warm_start=snapshot)
    assert torch.equal(state.current_raw_change_opacity, torch.full_like(state.current_raw_change_opacity, 2.0))
    state.activate_slots([0, 4])
    state.deactivate_slots([4])

    after = {name: id(parameter) for name, parameter in state.current_parameter_items()}
    assert after == before
    assert state.capacity == 5
    assert state.current_xyz.shape == (5, 3)
    assert torch.equal(state.support_mask, torch.tensor([True, False, False, False, False]))


def test_base_tensors_are_frozen_and_topology_methods_are_not_required():
    base = make_base(n=3)
    for method_name in ("densify_and_prune", "prune_points", "cat_tensors_to_optimizer"):
        setattr(base, method_name, lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(method_name)))

    state = FixedCapacityChangeState.from_gaussians(base, capacity=3)
    attrs = state.get_active_render_attributes(timestamp=10.0)

    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        assert getattr(base, name).requires_grad is False
    assert attrs["xyz"].shape[0] == 3


def test_change_opacity_is_raw_trainable_and_separate_from_cue_support_and_base_opacity():
    base = make_base(n=3)
    state = FixedCapacityChangeState.from_gaussians(base, capacity=3, base_zero=False)
    with torch.no_grad():
        base._opacity.fill_(10.0)
        state.current_raw_change_opacity.copy_(torch.tensor([[-2.0], [0.0], [2.0]]))
    state.activate_slots([1])

    attrs = state.get_active_render_attributes()

    assert attrs["raw_change_opacity"] is state.current_raw_change_opacity
    assert torch.equal(attrs["support_mask"], torch.tensor([False, True, False]))
    assert attrs["opacity"][0].item() == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    assert attrs["opacity"][1].item() == pytest.approx(0.1)
    assert attrs["opacity"][2].item() == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())


def test_cue_support_is_metadata_not_a_hard_gradient_gate():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=2), capacity=2)
    state.deactivate_slots()

    attrs = state.get_active_render_attributes()
    loss = attrs["dc"].sum() + attrs["opacity"].sum() + attrs["xyz"].sum()
    loss.backward()

    assert torch.equal(state.current_features_dc.grad, torch.ones_like(state.current_features_dc))
    assert torch.isfinite(state.current_raw_change_opacity.grad).all()
    assert torch.count_nonzero(state.current_raw_change_opacity.grad) == state.current_raw_change_opacity.numel()
    assert torch.equal(state.current_xyz.grad, torch.ones_like(state.current_xyz))


def test_causal_slot_evidence_and_carryover_protection_are_separate_from_opacity():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=4), capacity=4, base_zero=True)
    visible = torch.tensor([True, True, False, True])
    supported = torch.tensor([False, True, False, False])

    state.record_slot_evidence(visible, supported)
    warm = state.snapshot(detach=True, cpu=False)
    state.reset_current(warm_start=warm, preserve_cue_support=False)

    assert not state.cue_support_mask.any()
    assert torch.equal(state.carryover_protected_mask, supported)
    assert torch.equal(state.slot_observation_count, torch.zeros(4, dtype=torch.int32))
    assert torch.equal(state.slot_cue_support_count, torch.zeros(4, dtype=torch.int32))
    assert not state.removal_protected_mask.any()
    assert torch.allclose(
        torch.sigmoid(state.current_raw_change_opacity),
        torch.sigmoid(warm["raw_change_opacity"]),
    )
