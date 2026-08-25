from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from temporal.active_density_topology import (
    TemporalDensityScore,
    TemporalTopologyManager,
    aggregate_temporal_view_evidence,
)
from temporal.geometry_change_model import TemporalGeometryChangeModel
from temporal.masked_optimizer import MaskedRowSlotAdam
from utils.fastgs_topology import fastgs_split_children
from utils.general_utils import build_rotation


class _Base:
    def __init__(self, n: int = 2) -> None:
        self._xyz = nn.Parameter(torch.zeros(n, 3))
        self._features_dc = nn.Parameter(torch.zeros(n, 1, 3))
        self._features_rest = nn.Parameter(torch.zeros(n, 15, 3))
        self._opacity = nn.Parameter(torch.full((n, 1), 2.0))
        self._scaling = nn.Parameter(torch.full((n, 3), math.log(1e-4)))
        rotation = torch.zeros(n, 4)
        rotation[:, 0] = 1.0
        self._rotation = nn.Parameter(rotation)
        self.max_radii2D = torch.zeros(n)
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = lambda value: torch.nn.functional.normalize(value)

    @property
    def get_xyz(self):
        return self._xyz


def _model_optimizer_manager():
    model = TemporalGeometryChangeModel(_Base(), max_states=3)
    model.reset_all_lifespans_closed()
    optimizer = MaskedRowSlotAdam(
        dict(model.state_parameter_items()),
        thaw_names=("dc", "xyz", "opacity", "scaling", "rotation"),
        lrs={name: 1e-2 for name, _ in model.state_parameter_items()},
    )
    model.open_rows(torch.tensor([0, 1]), 0, optimizer=optimizer)
    manager = TemporalTopologyManager(model, optimizer, immutable_count=2)
    return model, optimizer, manager


def _score(n: int, *, important_row: int | None = None, prune_row: int | None = None):
    importance = torch.zeros(n)
    visible = torch.zeros(n, dtype=torch.long)
    support = torch.zeros(n, dtype=torch.long)
    if important_row is not None:
        importance[important_row] = 6.0
        visible[important_row] = 10
        support[important_row] = 10
    if prune_row is not None:
        visible[prune_row] = 10
        support[prune_row] = 1
    return TemporalDensityScore(
        view_indices=tuple(range(10)),
        positive_mass=importance * 10,
        negative_mass=torch.zeros(n),
        total_mass=importance * 10,
        importance_score=importance,
        visible_view_count=visible,
        support_view_count=support,
        change_ratio=torch.where(importance > 0, torch.ones(n), torch.zeros(n)),
    )


def _build_momentum(model, optimizer, row: int, slot: int = 0):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter[row, slot].sum() for _, parameter in model.state_parameter_items())
    loss.backward()
    mask = torch.zeros_like(model.state_valid)
    mask[row, slot] = True
    optimizer.step(mask)


def test_aggregate_temporal_view_evidence_respects_episode_eligibility():
    positive = [torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0])]
    negative = [torch.tensor([2.0, 1.0]), torch.tensor([1.0, 5.0])]
    eligible = [torch.tensor([True, False]), torch.tensor([True, True])]
    result = aggregate_temporal_view_evidence(
        positive,
        negative,
        eligible,
        min_mass=0.1,
        support_threshold=0.5,
    )
    assert torch.equal(result[0], torch.tensor([6.0, 5.0]))
    assert torch.equal(result[1], torch.tensor([3.0, 5.0]))
    assert torch.equal(result[3], torch.tensor([3.0, 2.5]))
    assert torch.equal(result[4], torch.tensor([2, 1]))
    assert torch.equal(result[5], torch.tensor([2, 1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FastGS split primitive is CUDA-only")
def test_shared_fastgs_split_primitive_matches_original_port_equation():
    device = torch.device("cuda")
    xyz = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], device=device)
    scaling = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]], device=device)
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device
    )
    torch.manual_seed(17)
    expected_stds = scaling.repeat(2, 1)
    samples = torch.normal(torch.zeros_like(expected_stds), expected_stds)
    expected_xyz = (
        torch.bmm(build_rotation(rotation).repeat(2, 1, 1), samples.unsqueeze(-1))
        .squeeze(-1)
        .add(xyz.repeat(2, 1))
    )
    expected_scaling = torch.log(expected_stds / 1.6)
    torch.manual_seed(17)
    actual_xyz, actual_scaling = fastgs_split_children(
        xyz, scaling, rotation, torch.log, children_per_source=2
    )
    assert torch.equal(actual_xyz, expected_xyz)
    assert torch.equal(actual_scaling, expected_scaling)


def test_density_clone_preserves_prefix_and_optimizer_state():
    model, optimizer, manager = _model_optimizer_manager()
    _build_momentum(model, optimizer, row=0)
    prefix_before = {
        name: getattr(model.base, name).detach().clone()
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    }
    state_before = {
        name: {
            key: value.detach()[0, 0].clone()
            for key, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2
        }
        for name, parameter in model.state_parameter_items()
    }
    manager.xyz_gradient_accum[0] = 1.0
    manager.xyz_gradient_accum_abs[0] = 1.0
    manager.denom[0] = 1.0
    result = manager.apply_density_control(
        _score(manager.count, important_row=0),
        timestamp=1,
        scene_extent=1.0,
    )
    assert result.clone_count == 1
    assert result.split_child_count == 0
    assert manager.count == 3
    assert manager.root_index.tolist() == [0, 1, 0]
    assert manager.episode_slot.tolist() == [-1, -1, 0]
    assert model.current_state_index.tolist() == [0, 0, 0]
    assert model.state_start[2, 0].item() == 1
    for name, expected in prefix_before.items():
        assert torch.equal(getattr(model.base, name)[:2], expected)
    for name, parameter in model.state_parameter_items():
        state = optimizer.state[parameter]
        for key, expected in state_before[name].items():
            assert torch.equal(state[key][0, 0], expected)
            assert torch.count_nonzero(state[key][2]).item() == 0


def test_closed_residual_is_frozen_and_reopen_uses_new_episode_slot():
    model, optimizer, manager = _model_optimizer_manager()
    manager.xyz_gradient_accum[0] = 1.0
    manager.xyz_gradient_accum_abs[0] = 1.0
    manager.denom[0] = 1.0
    manager.apply_density_control(
        _score(manager.count, important_row=0),
        timestamp=1,
        scene_extent=1.0,
    )
    old_child_id = int(manager.stable_id[2].item())
    old_slot = model.current_state_index[0].clone()
    model.close_rows(torch.tensor([0]), 2)
    assert manager.close_descendants(
        torch.tensor([0]), old_slot.reshape(1), timestamp=2
    ) == 1
    for _ in range(3):
        _build_momentum(model, optimizer, row=1)
    audit = manager.verify_closed_residuals()
    assert audit["passed"]
    assert audit["max_abs"] == 0.0

    new_slot = model.open_rows(torch.tensor([0]), 3, optimizer=optimizer)
    assert new_slot.tolist() == [1]
    manager.xyz_gradient_accum[0] = 1.0
    manager.xyz_gradient_accum_abs[0] = 1.0
    manager.denom[0] = 1.0
    manager.apply_density_control(
        _score(manager.count, important_row=0),
        timestamp=4,
        scene_extent=1.0,
    )
    assert manager.count == 4
    assert int(manager.stable_id[2].item()) == old_child_id
    assert model.current_state_index[2].item() == -1
    assert manager.episode_slot[3].item() == 1
    assert model.current_state_index[3].item() == 1


def test_vcp_removes_only_active_residual_and_never_prefix():
    model, _optimizer, manager = _model_optimizer_manager()
    manager.xyz_gradient_accum[0] = 1.0
    manager.xyz_gradient_accum_abs[0] = 1.0
    manager.denom[0] = 1.0
    manager.apply_density_control(
        _score(manager.count, important_row=0),
        timestamp=1,
        scene_extent=1.0,
    )
    residual_id = int(manager.stable_id[2].item())
    result = manager.apply_density_control(
        _score(manager.count, prune_row=2),
        timestamp=20,
        scene_extent=1.0,
        prune_grace_frames=10,
    )
    assert result.vcp_pruned_count == 1
    assert result.split_residual_removed_count == 0
    assert result.removed_count == 1
    assert manager.count == 2
    assert manager.stable_id.tolist() == [0, 1]
    deletion = [
        event for event in manager.lineage_events
        if event["action"] == "DELETE" and event["stable_id"] == residual_id
    ]
    assert len(deletion) == 1
    assert manager.validate()
