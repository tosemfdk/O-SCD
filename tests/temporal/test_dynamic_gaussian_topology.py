from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from temporal.binary_state_filter import BinaryStateFilter
from temporal.dynamic_gaussian_topology import DynamicGaussianTopologyManager
from temporal.masked_optimizer import MaskedRowAdam
from temporal.persistent_gaussian_lifespan_model import PersistentGaussianLifespanModel
from temporal.bayesian_lifespan_controller import BayesianLifespanController
from temporal.bernoulli_bocd import BernoulliBOCDConfig
from temporal.single_candidate_beta import SingleCandidateBetaFilter
from temporal.view_consistent_binary_lifespan_controller import (
    ViewConsistentBinaryLifespanController,
)


class _Base:
    def __init__(self, n: int, device: torch.device | str = "cpu") -> None:
        self._xyz = nn.Parameter(torch.zeros(n, 3, device=device))
        self._features_dc = nn.Parameter(torch.zeros(n, 1, 3, device=device))
        self._features_rest = nn.Parameter(torch.zeros(n, 15, 3, device=device))
        self._opacity = nn.Parameter(torch.full((n, 1), 2.0, device=device))
        self._scaling = nn.Parameter(
            torch.full((n, 3), math.log(1e-4), device=device)
        )
        rotation = torch.zeros(n, 4, device=device)
        rotation[:, 0] = 1.0
        self._rotation = nn.Parameter(rotation)
        self.max_radii2D = torch.zeros(n, device=device)
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = lambda value: torch.nn.functional.normalize(
            value, dim=-1
        )

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


def _objects(n: int = 3, device: torch.device | str = "cpu"):
    model = PersistentGaussianLifespanModel(_Base(n, device), max_states=4)
    model.reset_all_lifespans_closed()
    optimizer = MaskedRowAdam(
        dict(model.persistent_parameter_items()),
        lrs={name: 1e-2 for name, _ in model.persistent_parameter_items()},
    )
    tracker = BinaryStateFilter(n, device=device)
    controller = ViewConsistentBinaryLifespanController(model)
    manager = DynamicGaussianTopologyManager(
        model, optimizer, tracker, controller, percent_dense=0.01
    )
    return model, optimizer, tracker, controller, manager


def _build_momentum(model, optimizer, row: int) -> None:
    optimizer.zero_grad(set_to_none=True)
    for _name, parameter in model.persistent_parameter_items():
        parameter.grad = torch.ones_like(parameter)
    selected = torch.zeros(model.current_state_index.shape[0], dtype=torch.bool, device=model.xyz.device)
    selected[row] = True
    optimizer.step(selected)


def test_clone_copies_detector_lifecycle_and_then_becomes_independent():
    model, optimizer, tracker, controller, manager = _objects()
    model.open_rows(torch.tensor([0]), timestamp=0)
    tracker.p_active[0] = 0.93
    tracker.visible_observations[0] = 7
    tracker.last_timestamp[0] = 4
    controller.open_support_count[0] = 2
    controller.close_support_count[0] = 1
    model.change_dc.data[0].fill_(0.7)
    _build_momentum(model, optimizer, 0)
    parent_parameter = model.change_dc.detach()[0].clone()
    parent_state = {
        name: value[0].detach().clone()
        for name, value in optimizer.state[model.change_dc].items()
        if isinstance(value, torch.Tensor)
    }

    manager.xyz_gradient_accum[0] = 1.0
    manager.denom[0] = 1.0
    # An equally strong inactive row is not eligible for density control.
    manager.xyz_gradient_accum[1] = 1.0
    manager.denom[1] = 1.0
    result = manager.apply_active_oscd_density_control(
        timestamp=5,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
    )

    assert result.clone_source_count == 1
    assert result.clone_child_count == 1
    assert result.split_source_count == 0
    assert manager.count == 4
    child = 3
    assert manager.parent_stable_id[child].item() == manager.stable_id[0].item()
    assert model.current_state_index[child].item() == model.current_state_index[0].item()
    assert torch.equal(model.state_valid[child], model.state_valid[0])
    assert tracker.p_active[child].item() == pytest.approx(tracker.p_active[0].item())
    assert tracker.visible_observations[child].item() == 7
    assert tracker.last_timestamp[child].item() == 4
    assert controller.open_support_count[child].item() == 2
    assert controller.close_support_count[child].item() == 1
    assert torch.equal(model.change_dc[child], parent_parameter)
    for state_name, expected in parent_state.items():
        assert torch.equal(optimizer.state[model.change_dc][state_name][0], expected)
        assert torch.count_nonzero(
            optimizer.state[model.change_dc][state_name][child]
        ).item() == 0

    # The copied row is independent after birth: detector and parameter updates
    # on the child do not broadcast back to its source.
    parent_after_clone = model.change_dc.detach()[0].clone()
    tracker.p_active[child] = 0.11
    optimizer.zero_grad(set_to_none=True)
    for _name, parameter in model.persistent_parameter_items():
        parameter.grad = torch.ones_like(parameter)
    selected = torch.zeros(manager.count, dtype=torch.bool)
    selected[child] = True
    optimizer.step(selected)
    assert tracker.p_active[0].item() != tracker.p_active[child].item()
    assert torch.equal(model.change_dc[0], parent_after_clone)
    assert not torch.equal(model.change_dc[child], parent_parameter)
    assert manager.validate()


def test_clone_copies_declared_single_candidate_and_bayesian_controller_buffers():
    model = PersistentGaussianLifespanModel(_Base(3), max_states=4)
    model.reset_all_lifespans_closed()
    optimizer = MaskedRowAdam(
        dict(model.persistent_parameter_items()),
        lrs={name: 1e-2 for name, _ in model.persistent_parameter_items()},
    )
    tracker = SingleCandidateBetaFilter(3)
    controller = BayesianLifespanController(
        model,
        BernoulliBOCDConfig(
            prior_a=1.0,
            prior_b=1.0,
            hazard=0.01,
            open_probability=0.6,
            close_probability=0.4,
            changepoint_probability=0.5,
            min_run_evidence=1.0,
            min_visible_observations=1,
        ),
    )
    manager = DynamicGaussianTopologyManager(
        model, optimizer, tracker, controller, percent_dense=0.01
    )
    model.open_rows(torch.tensor([0]), timestamp=0)
    tracker.initialized[0] = True
    tracker.stable_a[0] = 8.0
    tracker.stable_b[0] = 2.0
    tracker.candidate_active[0] = True
    tracker.candidate_delta_b[0] = 1.0
    tracker.candidate_start[0] = 4
    tracker.candidate_visible_observations[0] = 1
    controller.committed_run_label[0] = 1
    controller.committed_run_start[0] = 0
    controller.pending_run_start[0] = 4
    controller.pending_changepoint_probability[0] = 1.0
    manager.xyz_gradient_accum[0] = 1.0
    manager.denom[0] = 1.0

    result = manager.apply_active_oscd_density_control(
        timestamp=5,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
    )

    assert result.clone_child_count == 1
    child = 3
    for name in tracker.topology_buffer_names:
        assert torch.equal(getattr(tracker, name)[child], getattr(tracker, name)[0]), name
    for name in controller.topology_buffer_names:
        assert torch.equal(
            getattr(controller, name)[child], getattr(controller, name)[0]
        ), name
    tracker.stable_a[child] = 3.0
    assert tracker.stable_a[0].item() == pytest.approx(8.0)
    assert manager.validate()


def test_active_only_prune_can_remove_an_initial_row_but_not_inactive_row():
    model, _optimizer, tracker, _controller, manager = _objects()
    model.open_rows(torch.tensor([0, 2]), timestamp=0)
    # Rows 1 and 2 both have low opacity.  Row 1 is INACTIVE and must survive;
    # row 2 is ACTIVE and is removed even though it was in the initial bank.
    model.opacity.data[1] = -10.0
    model.opacity.data[2] = -10.0
    tracker.p_active.copy_(torch.tensor([0.8, 0.2, 0.9]))
    removed_stable_id = int(manager.stable_id[2].item())
    inactive_stable_id = int(manager.stable_id[1].item())

    result = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.4,
    )

    assert result.opacity_pruned_count == 1
    assert result.total_removed_count == 1
    assert result.final_count == 2
    assert removed_stable_id not in manager.stable_id.tolist()
    assert inactive_stable_id in manager.stable_id.tolist()
    inactive_row = manager.stable_id.tolist().index(inactive_stable_id)
    assert model.current_state_index[inactive_row].item() == -1
    assert tracker.p_active[inactive_row].item() == pytest.approx(0.2)
    assert manager.validate()


def test_pruning_preserves_unaffected_parameter_and_optimizer_rows_exactly():
    model, optimizer, _tracker, _controller, manager = _objects()
    model.open_rows(torch.tensor([0, 2]), timestamp=0)
    _build_momentum(model, optimizer, 0)
    expected_parameter = {
        name: parameter.detach()[0].clone()
        for name, parameter in model.persistent_parameter_items()
    }
    expected_state = {
        name: {
            state_name: state_value[0].detach().clone()
            for state_name, state_value in optimizer.state[parameter].items()
            if isinstance(state_value, torch.Tensor)
        }
        for name, parameter in model.persistent_parameter_items()
    }
    model.opacity.data[2] = -10.0
    manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.4,
    )
    parameters = dict(model.persistent_parameter_items())
    for name, expected in expected_parameter.items():
        assert torch.equal(parameters[name][0], expected)
        for state_name, expected_value in expected_state[name].items():
            assert torch.equal(
                optimizer.state[parameters[name]][state_name][0], expected_value
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="split primitive is CUDA-only")
def test_split_replaces_any_active_source_with_two_equal_status_children():
    model, optimizer, tracker, controller, manager = _objects(device="cuda")
    model.open_rows(torch.tensor([0], device="cuda"), timestamp=0)
    model.scaling.data[0].fill_(math.log(0.1))
    tracker.p_active[0] = 0.88
    tracker.visible_observations[0] = 9
    controller.close_support_count[0] = 2
    source_id = int(manager.stable_id[0].item())
    manager.xyz_gradient_accum[0] = 1.0
    manager.denom[0] = 1.0

    result = manager.apply_active_oscd_density_control(
        timestamp=3,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
    )

    assert result.split_source_count == 1
    assert result.split_child_count == 2
    assert result.split_source_removed_count == 1
    assert manager.count == 4
    assert source_id not in manager.stable_id.tolist()
    child_rows = torch.nonzero(
        manager.parent_stable_id == source_id, as_tuple=False
    ).flatten()
    assert child_rows.numel() == 2
    assert bool((model.current_state_index[child_rows] >= 0).all())
    assert bool(torch.allclose(tracker.p_active[child_rows], torch.full((2,), 0.88, device="cuda")))
    assert bool((tracker.visible_observations[child_rows] == 9).all())
    assert bool((controller.close_support_count[child_rows] == 2).all())
    for _name, parameter in model.persistent_parameter_items():
        state = optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][child_rows]).item() == 0
        assert torch.count_nonzero(state["exp_avg_sq"][child_rows]).item() == 0
        assert torch.count_nonzero(state["step"][child_rows]).item() == 0
    assert manager.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="split primitive is CUDA-only")
def test_cue_mixture_can_split_large_active_row_without_xyz_gradient():
    model, _optimizer, _tracker, _controller, manager = _objects(device="cuda")
    model.open_rows(torch.tensor([0], device="cuda"), timestamp=0)
    model.scaling.data[0].fill_(math.log(0.1))
    source_id = int(manager.stable_id[0].item())

    result = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
        cue_mixture_score=torch.tensor([0.75, 0.0, 0.0], device="cuda"),
        cue_mixture_threshold=0.5,
    )

    assert result.gradient_split_source_count == 0
    assert result.cue_mixture_split_source_count == 1
    assert result.cue_mixture_only_split_source_count == 1
    assert result.split_source_count == 1
    assert result.split_child_count == 2
    assert result.masks.cue_mixture_split.tolist() == [True, False, False]
    assert source_id not in manager.stable_id.tolist()
    assert manager.count == 4
    assert manager.validate()


def test_cue_mixture_does_not_clone_small_or_inactive_rows():
    model, _optimizer, _tracker, _controller, manager = _objects()
    model.open_rows(torch.tensor([0]), timestamp=0)

    result = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
        cue_mixture_score=torch.tensor([1.0, 1.0, 0.0]),
        cue_mixture_threshold=0.5,
    )

    assert result.clone_source_count == 0
    assert result.split_source_count == 0
    assert result.cue_mixture_split_source_count == 0
    assert manager.count == 3


def test_black_child_pruning_never_removes_initial_rows_or_same_event_children():
    model, _optimizer, _tracker, _controller, manager = _objects()
    model.open_rows(torch.tensor([0]), timestamp=0)
    model.change_dc.data[0].fill_(-1.0)
    initial_stable_id = int(manager.stable_id[0].item())
    manager.xyz_gradient_accum[0] = 1.0
    manager.denom[0] = 1.0

    born = manager.apply_active_oscd_density_control(
        timestamp=0,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
        black_child_prune_threshold=0.5,
        black_child_prune_min_age_frames=1,
    )

    assert born.clone_child_count == 1
    assert born.black_child_prune_candidate_count == 0
    assert born.black_child_pruned_count == 0
    assert manager.count == 4
    assert initial_stable_id in manager.stable_id.tolist()

    pruned = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.0,
        black_child_prune_threshold=0.5,
        black_child_prune_min_age_frames=1,
    )

    assert pruned.black_child_prune_candidate_count == 1
    assert pruned.black_child_pruned_count == 1
    assert pruned.black_child_retained_for_support_count == 0
    assert manager.count == 3
    assert initial_stable_id in manager.stable_id.tolist()
    assert bool((manager.generation == 0).all())
    assert manager.validate()


def test_black_child_pruning_keeps_one_child_when_removed_parent_has_no_support():
    model, _optimizer, _tracker, _controller, manager = _objects()
    model.open_rows(torch.tensor([0]), timestamp=0)
    model.change_dc.data[0].fill_(-1.0)
    parent_stable_id = int(manager.stable_id[0].item())

    for timestamp in (0, 1):
        parent_row = manager.stable_id.tolist().index(parent_stable_id)
        manager.xyz_gradient_accum[parent_row] = 1.0
        manager.denom[parent_row] = 1.0
        result = manager.apply_active_oscd_density_control(
            timestamp=timestamp,
            scene_extent=1.0,
            grad_threshold=1e-3,
            min_opacity=0.0,
        )
        assert result.clone_child_count == 1

    parent_row = manager.stable_id.tolist().index(parent_stable_id)
    child_rows = torch.nonzero(
        manager.parent_stable_id == parent_stable_id, as_tuple=False
    ).flatten()
    assert child_rows.numel() == 2
    model.opacity.data[parent_row] = -10.0
    model.opacity.data[child_rows] = 2.0

    result = manager.apply_active_oscd_density_control(
        timestamp=2,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.4,
        black_child_prune_threshold=0.5,
        black_child_prune_min_age_frames=1,
    )

    assert result.opacity_pruned_count == 1
    assert result.black_child_prune_candidate_count == 2
    assert result.black_child_pruned_count == 1
    assert result.black_child_retained_for_support_count == 1
    assert parent_stable_id not in manager.stable_id.tolist()
    surviving_children = torch.nonzero(
        manager.parent_stable_id == parent_stable_id, as_tuple=False
    ).flatten()
    assert surviving_children.numel() == 1
    assert manager.generation[surviving_children].item() == 1
    assert manager.validate()


def test_signed_score_replaces_gradient_gate_and_uses_stable_topk_for_clones():
    model, _optimizer, _tracker, _controller, manager = _objects(n=4)
    model.open_rows(torch.tensor([0, 1, 2]), timestamp=0)
    # Row 0 has enough gradient for legacy cloning, but signed-score mode must
    # ignore the gradient path entirely.
    manager.xyz_gradient_accum[0] = 10.0
    manager.denom[0] = 1.0

    result = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1e-3,
        min_opacity=0.0,
        signed_density_score=torch.tensor([0.1, 0.9, 0.9, 1.0]),
        signed_density_threshold=0.5,
        signed_density_max_sources=1,
    )

    assert result.gradient_split_source_count == 0
    assert result.cue_mixture_split_source_count == 0
    assert result.signed_score_candidate_count == 1
    assert result.signed_score_clone_source_count == 1
    assert result.clone_source_count == 1
    assert result.masks.clone.tolist() == [False, True, False, False]
    assert result.masks.signed_score_candidate.tolist() == [False, True, False, False]
    assert manager.parent_stable_id[-1].item() == 1
    assert manager.validate()


def test_signed_score_topk_ties_use_stable_id_not_row_order_and_zero_selects_none():
    model, _optimizer, _tracker, _controller, manager = _objects(n=4)
    model.open_rows(torch.tensor([0, 1, 2]), timestamp=0)
    manager.stable_id.copy_(torch.tensor([10, 2, 5, 7]))
    manager.next_stable_id = 11

    result = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.0,
        signed_density_score=torch.tensor([0.9, 0.9, 0.9, 1.0]),
        signed_density_threshold=0.5,
        signed_density_max_sources=2,
    )

    assert result.masks.signed_score_candidate.tolist() == [False, True, True, False]
    assert result.clone_source_count == 2
    assert sorted(manager.parent_stable_id[-2:].tolist()) == [2, 5]
    assert manager.validate()

    model, _optimizer, _tracker, _controller, manager = _objects(n=3)
    model.open_rows(torch.tensor([0, 1]), timestamp=0)
    zero = manager.apply_active_oscd_density_control(
        timestamp=1,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.0,
        signed_density_score=torch.tensor([0.9, 0.8, 0.7]),
        signed_density_threshold=0.5,
        signed_density_max_sources=0,
    )
    assert zero.signed_score_candidate_count == 0
    assert zero.clone_source_count == 0
    assert manager.count == 3
    assert manager.validate()


def test_negative_signed_score_prunes_only_active_children_and_protects_generation_zero():
    model, _optimizer, _tracker, _controller, manager = _objects(n=3)
    model.open_rows(torch.tensor([0]), timestamp=0)

    born = manager.apply_active_oscd_density_control(
        timestamp=10,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.0,
        signed_density_score=torch.tensor([1.0, 0.0, 0.0]),
        signed_density_threshold=0.5,
    )
    assert born.clone_child_count == 1
    child_stable_id = int(manager.stable_id[-1].item())
    child_row = manager.stable_id.tolist().index(child_stable_id)
    assert manager.creation_timestamp[child_row].item() == 10
    child_slot = model.current_state_index[child_row]
    assert model.state_start[child_row, child_slot].item() == pytest.approx(0.0)

    # Same timestamp as child creation: pruning succeeds only because age is
    # computed from inherited lifecycle state_start, not creation_timestamp.
    score = torch.zeros(manager.count)
    score[0] = -1.0  # generation-zero source must be protected.
    score[child_row] = -1.0
    pruned = manager.apply_active_oscd_density_control(
        timestamp=10,
        scene_extent=1.0,
        grad_threshold=1.0,
        min_opacity=0.0,
        signed_density_score=score,
        signed_density_threshold=0.5,
        signed_density_prune_threshold=0.5,
        signed_density_prune_min_age_frames=5,
    )

    assert pruned.signed_score_prune_candidate_count == 1
    assert pruned.signed_score_pruned_count == 1
    assert pruned.total_removed_count == 1
    assert child_stable_id not in manager.stable_id.tolist()
    assert int(manager.stable_id[0].item()) == 0
    assert bool((manager.generation == 0).all())
    assert manager.validate()
