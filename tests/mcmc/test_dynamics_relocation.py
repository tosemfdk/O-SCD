import pytest
import torch

from temporal.mcmc_dynamics import (
    eq9_relocation_terms,
    group_assignments,
    relocate_dead_gaussians_,
    reset_adam_rows_,
    sample_opacity_weighted_assignments,
    select_dead_live_indices,
)
from .conftest import make_mcmc_gaussians


def test_eq9_grouping_preserves_render_opacity_proxy_better_than_naive_split():
    old_opacity = torch.tensor(0.8)
    group_total = 4

    new_opacity, covariance_scale = eq9_relocation_terms(old_opacity, group_total)
    composed = 1.0 - (1.0 - new_opacity).pow(group_total)
    naive = 1.0 - (1.0 - old_opacity / group_total).pow(group_total)

    assert composed.item() == pytest.approx(old_opacity.item(), abs=1e-6)
    assert abs(composed.item() - old_opacity.item()) < abs(naive.item() - old_opacity.item())
    assert covariance_scale.item() > 0.0


def test_relocation_groups_assignments_before_mutating_and_preserves_fixed_capacity():
    gaussians = make_mcmc_gaussians(raw_opacity=[-10.0, -9.0, 2.0])
    before_count = gaussians.xyz.shape[0]
    before_ids = {name: id(getattr(gaussians, name)) for name in ("xyz", "change_dc", "change_opacity_raw", "scaling_raw", "rotation_raw")}

    dead, live = select_dead_live_indices(gaussians.change_opacity_raw, opacity_threshold=0.005)
    assignments = sample_opacity_weighted_assignments(gaussians.change_opacity_raw, dead, live, seed=0)
    groups = group_assignments(assignments)
    result = relocate_dead_gaussians_(gaussians, opacity_threshold=0.005, seed=0)

    assert set(groups) == {2}
    assert result.applied_count == 2
    assert gaussians.xyz.shape[0] == before_count
    assert {name: id(getattr(gaussians, name)) for name in before_ids} == before_ids
    assert torch.allclose(gaussians.xyz[0], gaussians.xyz[2])
    assert torch.allclose(gaussians.xyz[1], gaussians.xyz[2])


def test_relocation_uses_change_opacity_not_unrelated_base_opacity_attribute():
    gaussians = make_mcmc_gaussians(raw_opacity=[-10.0, 2.0])
    gaussians.opacity = torch.tensor([[0.99], [0.0]])

    result = relocate_dead_gaussians_(gaussians, opacity_threshold=0.005, seed=0)

    assert result.selected_dead_indices == (0,)
    assert result.selected_live_indices == (1,)


def test_adam_reset_zeros_target_rows_but_retains_source_rows():
    gaussians = make_mcmc_gaussians(raw_opacity=[-10.0, -9.0, 2.0])
    params = {name: getattr(gaussians, name) for name in ("xyz", "change_dc", "change_opacity_raw", "scaling_raw", "rotation_raw")}
    optimizer = torch.optim.Adam(params.values(), lr=0.01)
    loss = sum(param.sum() for param in params.values())
    loss.backward()
    optimizer.step()

    for param in params.values():
        optimizer.state[param]["exp_avg"].fill_(1.0)
        optimizer.state[param]["exp_avg_sq"].fill_(2.0)

    reset_adam_rows_(optimizer, params, rows=[2])

    for param in params.values():
        state = optimizer.state[param]
        assert torch.equal(state["exp_avg"][2], torch.zeros_like(state["exp_avg"][2]))
        assert torch.equal(state["exp_avg_sq"][2], torch.zeros_like(state["exp_avg_sq"][2]))
        assert torch.equal(state["exp_avg"][0], torch.ones_like(state["exp_avg"][0]))
        assert torch.equal(state["exp_avg_sq"][0], torch.full_like(state["exp_avg_sq"][0], 2.0))
