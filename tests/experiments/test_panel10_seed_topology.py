from argparse import Namespace

import pytest
import torch

from experiments.panel10_seed_topology import (
    append_typed_seeds,
    apply_typed_density,
    prune_typed_seeds,
)
from experiments.view_bayesian_detector_steps import DetectorReplayLifespan
from temporal.active_new_gaussians import ActiveNewGaussianModel
from temporal.lifespan_gate_beta import LifespanGateBetaFilter
from temporal.masked_optimizer import MaskedRowAdam
from temporal.new_seed_gaussians import NewSeedGaussianModel


def _optimizer(model: ActiveNewGaussianModel) -> MaskedRowAdam:
    return MaskedRowAdam(
        {
            "xyz": model._xyz,
            "dc": model.new_dc,
            "opacity": model._opacity,
            "scaling": model._scaling,
            "rotation": model._rotation,
        },
        thaw_names=("xyz", "dc", "opacity", "scaling", "rotation"),
        lrs={name: 1e-3 for name in ("xyz", "dc", "opacity", "scaling", "rotation")},
        eps=1e-15,
    )


def _all_rows_mask(model: ActiveNewGaussianModel) -> dict[str, torch.Tensor]:
    mask = torch.ones(model.num_gaussians, dtype=torch.bool)
    return {name: mask.clone() for name in ("xyz", "dc", "opacity", "scaling", "rotation")}


def _initialize_masked_adam(replay) -> None:
    loss = sum(parameter.square().sum() for parameter in replay.seed_model.parameters())
    loss.backward()
    replay.seed_optimizer.step(_all_rows_mask(replay.seed_model))
    replay.seed_optimizer.zero_grad(set_to_none=True)


def _snapshot_prefix(replay):
    params = {
        name: parameter.detach().clone()
        for name, parameter in replay.seed_model._parameter_map().items()
    }
    state = {}
    for group in replay.seed_optimizer.param_groups:
        parameter = group["params"][0]
        state[str(group["name"])] = {
            key: value.detach().clone()
            for key, value in replay.seed_optimizer.state[parameter].items()
            if key in {"exp_avg", "exp_avg_sq", "step"}
        }
    return params, state


def _assert_prefix_preserved(replay, params, state, rows: int) -> None:
    for name, before in params.items():
        after = replay.seed_model._parameter_map()[name].detach()[:rows]
        assert torch.equal(after, before[:rows])
    for group in replay.seed_optimizer.param_groups:
        parameter = group["params"][0]
        name = str(group["name"])
        after_state = replay.seed_optimizer.state[parameter]
        for key, before in state[name].items():
            assert torch.equal(after_state[key][:rows], before[:rows])


def _snapshot_tracker_prefix(replay):
    names = replay.tracker.topology_buffer_names
    return {name: getattr(replay.tracker, name)[: replay.count].detach().clone() for name in names}


def _assert_tracker_prefix_preserved(replay, before) -> None:
    for name, value in before.items():
        assert torch.equal(getattr(replay.tracker, name)[: replay.count], value)


def _tracker_row(replay, seed_row: int) -> int:
    return int(replay.count) + int(seed_row)


def _replay(*, max_rows=20, max_states=4, base_count=5):
    model = ActiveNewGaussianModel()
    tracker = LifespanGateBetaFilter(base_count + max_rows, device=torch.device("cpu"))
    # Non-default base-prefix sentinels make accidental seed-topology writes into
    # the immutable base detector rows visible.
    tracker.stable_a[:base_count] = torch.arange(base_count, dtype=torch.float32) + 101.0
    tracker.candidate_active[:base_count] = True
    tracker.last_timestamp[:base_count] = torch.arange(base_count, dtype=torch.long) + 17
    return Namespace(
        args=Namespace(
            da3_max_rows=max_rows,
            da3_initial_opacity=0.2,
            max_states=max_states,
            da3_density_max_children=128,
            da3_max_generation=2,
            da3_max_root_children=32,
            da3_split_scale_ratio=1.0,
            da3_density_grad_threshold=2e-4,
            da3_density_abs_grad_threshold=1.2e-3,
            da3_prune_opacity=0.02,
            da3_prune_grace_frames=3,
            da3_prune_min_updates=4,
        ),
        device=torch.device("cpu"),
        count=base_count,
        seed_model=model,
        seed_detector_probe=NewSeedGaussianModel(),
        seed_lifecycle=DetectorReplayLifespan(0, max_states=max_states, device=torch.device("cpu")),
        seed_optimizer=_optimizer(model),
        tracker=tracker,
        seed_tracker=None,
        accepted_da3_source_rows=[],
        accepted_da3_birth_global=[],
        seed_retired=torch.empty(0, dtype=torch.bool),
        seed_geometry_update_counts=torch.empty(0, dtype=torch.long),
        seed_last_visible_rows=torch.empty(0, dtype=torch.bool),
    )


def _append_two(replay, timestamp=3):
    return append_typed_seeds(
        replay,
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        torch.full((2, 3), -3.0),
        timestamp,
        [
            {"pixel_xy": (10, 20), "panel10_depth": 0.4},
            {"pixel_xy": (11, 20), "panel10_depth": 0.5},
        ],
    )


def test_append_typed_seeds_aligns_append_only_identity_and_never_open_rows():
    replay = _replay(max_rows=3)
    before_tracker_prefix = _snapshot_tracker_prefix(replay)

    accepted = _append_two(replay, timestamp=5)

    assert accepted == 2
    assert replay.seed_model.num_gaussians == 2
    assert replay.seed_detector_probe.num_seeds == 2
    assert replay.seed_lifecycle.count == 2
    assert replay.accepted_da3_source_rows == [0, 1]
    assert replay.accepted_da3_birth_global == [5, 5]
    assert replay.seed_model.never_open_mask().tolist() == [True, True]
    assert replay.seed_lifecycle.never_open_mask(5).tolist() == [True, True]
    assert replay.seed_retired.tolist() == [False, False]
    assert replay.seed_geometry_update_counts.tolist() == [0, 0]
    assert replay.seed_last_visible_rows.tolist() == [False, False]
    assert replay.seed_model.metadata[0]["pixel_xy"] == (10, 20)
    assert replay.seed_model.metadata[0]["root_seed_row"] == 0
    _assert_tracker_prefix_preserved(replay, before_tracker_prefix)
    suffix = slice(replay.count, replay.count + replay.seed_model.num_gaussians)
    assert replay.tracker.last_timestamp[suffix].tolist() == [-1, -1]
    assert replay.tracker.visible_observations[suffix].tolist() == [0, 0]

    _initialize_masked_adam(replay)
    before_params, before_state = _snapshot_prefix(replay)
    before_tracker_prefix = _snapshot_tracker_prefix(replay)

    # Capacity counts archived rows, even though the tracker has a larger preallocated state.
    accepted = append_typed_seeds(
        replay,
        torch.tensor([[2.0, 0.0, 2.0], [3.0, 0.0, 2.0]]),
        torch.full((2, 3), -3.0),
        6,
    )
    assert accepted == 1
    assert replay.accepted_da3_source_rows == [0, 1, 2]
    _assert_prefix_preserved(replay, before_params, before_state, rows=2)
    _assert_tracker_prefix_preserved(replay, before_tracker_prefix)
    assert replay.tracker.last_timestamp[_tracker_row(replay, 2)].item() == -1
    for group in replay.seed_optimizer.param_groups:
        state = replay.seed_optimizer.state[group["params"][0]]
        assert torch.equal(state["step"][2:], torch.zeros_like(state["step"][2:]))
        assert torch.equal(state["exp_avg"][2:], torch.zeros_like(state["exp_avg"][2:]))
        assert torch.equal(state["exp_avg_sq"][2:], torch.zeros_like(state["exp_avg_sq"][2:]))


@pytest.mark.parametrize("max_rows", [0, 10])
def test_density_clone_copies_tracker_once_and_child_has_fresh_lifecycle(max_rows):
    replay = _replay(max_rows=max_rows)
    _append_two(replay, timestamp=1)
    replay.seed_lifecycle.open_rows([0], 2)
    replay.seed_model.open_rows(torch.tensor([0]), 2)
    replay.seed_last_visible_rows[0] = True
    replay.seed_model.xyz_gradient_accum[0] = 1e-3
    replay.seed_model.xyz_gradient_accum_abs[0] = 1e-3
    replay.seed_model.gradient_denom[0] = 1.0
    parent_tracker_row = _tracker_row(replay, 0)
    replay.tracker.stable_a[parent_tracker_row] = 7.0
    replay.tracker.candidate_active[parent_tracker_row] = True
    replay.tracker.last_timestamp[parent_tracker_row] = 2
    if max_rows == 0:
        # Force this density event to cross a storage boundary, independently
        # of the implementation's allocation chunk size.
        for name in replay.tracker.topology_buffer_names:
            setattr(replay.tracker, name, getattr(replay.tracker, name)[:replay.count + 2].clone())
    before_tracker_prefix = _snapshot_tracker_prefix(replay)

    result = apply_typed_density(replay, timestamp=2, random_seed=4)

    assert result["clone_count"] == 1
    assert result["children"] == 1
    assert replay.seed_model.num_gaussians == 3
    child = 2
    assert replay.seed_lifecycle.active_mask(2).tolist() == [True, False, True]
    assert replay.seed_lifecycle.num_states.tolist() == [1, 0, 1]
    assert replay.seed_lifecycle.state_start[child, 0].item() == pytest.approx(2.0)
    child_tracker_row = _tracker_row(replay, child)
    assert replay.tracker.stable_a[child_tracker_row].item() == pytest.approx(7.0)
    assert bool(replay.tracker.candidate_active[child_tracker_row]) is True
    assert replay.tracker.last_timestamp[child_tracker_row].item() == 2
    _assert_tracker_prefix_preserved(replay, before_tracker_prefix)
    assert replay.seed_model.metadata[child]["root_seed_row"] == 0
    assert replay.seed_detector_probe.num_seeds == 3
    assert replay.accepted_da3_source_rows == [0, 1, 2]
    assert replay.seed_model.densification_count[0].item() == 1
    assert replay.seed_model.gradient_denom.sum().item() == 0.0


def test_density_split_retires_parent_without_compaction_and_preserves_optimizer_state():
    replay = _replay(max_rows=10)
    append_typed_seeds(replay, torch.tensor([[0.0, 0.0, 2.0]]), torch.full((1, 3), 0.0), 1)
    replay.seed_lifecycle.open_rows([0], 2)
    replay.seed_model.open_rows(torch.tensor([0]), 2)
    replay.seed_model._scaling.data[0].fill_(0.5)
    replay.seed_last_visible_rows[0] = True
    replay.seed_geometry_update_counts[0] = 10
    replay.seed_model.xyz_gradient_accum_abs[0] = 2e-3
    replay.seed_model.gradient_denom[0] = 1.0
    _initialize_masked_adam(replay)
    before_params, before_state = _snapshot_prefix(replay)

    result = apply_typed_density(replay, timestamp=4, random_seed=0)

    assert result["split_source_count"] == 1
    assert result["children"] == 2
    assert replay.seed_model.num_gaussians == 3
    assert replay.seed_retired.tolist() == [True, False, False]
    assert replay.seed_lifecycle.closed_mask(4).tolist() == [True, False, False]
    assert replay.seed_model.closed_mask(4).tolist() == [True, False, False]
    assert replay.seed_lifecycle.active_mask(4).tolist() == [False, True, True]
    # Parent archive row and optimizer-addressed parameters were not compacted/deleted.
    _assert_prefix_preserved(replay, before_params, before_state, rows=1)
    for group in replay.seed_optimizer.param_groups:
        parameter = group["params"][0]
        state = replay.seed_optimizer.state[parameter]
        assert parameter.shape[0] == 3
        assert state["exp_avg"].shape[0] == 3
        assert state["exp_avg_sq"].shape[0] == 3
        assert state["step"].shape[0] == 3


def test_density_protects_retired_not_visible_and_fresh_without_capacity_exit():
    replay = _replay(max_rows=8)
    append_typed_seeds(replay, torch.zeros((4, 3)), torch.full((4, 3), -3.0), 1)
    replay.seed_lifecycle.open_rows([0, 1, 2, 3], 2)
    replay.seed_model.open_rows(torch.tensor([0, 1, 2, 3]), 2)
    replay.seed_last_visible_rows[:] = torch.tensor([True, False, True, True])
    replay.seed_retired[2] = True
    replay.seed_model.xyz_gradient_accum[:] = 0.0
    replay.seed_model.xyz_gradient_accum_abs[:] = 1e-2
    replay.seed_model.gradient_denom[:] = 1.0
    replay.seed_model.densification_count[0] = 1
    replay.seed_model._scaling.data[3].fill_(0.5)

    result = apply_typed_density(replay, timestamp=2, random_seed=0)

    assert result["children"] == 0
    assert replay.seed_model.num_gaussians == 4
    assert replay.seed_model.gradient_denom.sum().item() == 0.0


def test_density_enforces_root_child_capacity_for_split_and_siblings():
    replay = _replay(max_rows=20)
    replay.args.da3_max_root_children = 1
    append_typed_seeds(replay, torch.tensor([[0.0, 0.0, 2.0]]), torch.full((1, 3), 0.5), 1)
    replay.seed_lifecycle.open_rows([0], 2)
    replay.seed_model.open_rows(torch.tensor([0]), 2)
    replay.seed_last_visible_rows[0] = True
    replay.seed_model.xyz_gradient_accum_abs[0] = 1e-2
    replay.seed_model.gradient_denom[0] = 1.0

    result = apply_typed_density(replay, timestamp=4, random_seed=0)

    assert result["split_source_count"] == 0
    assert result["children"] == 0
    assert replay.seed_model.num_gaussians == 1

    replay = _replay(max_rows=20)
    replay.args.da3_max_root_children = 3
    append_typed_seeds(replay, torch.zeros((3, 3)), torch.full((3, 3), -3.0), 1)
    replay.seed_lifecycle.open_rows([0, 1, 2], 2)
    replay.seed_model.open_rows(torch.tensor([0, 1, 2]), 2)
    replay.seed_model.generation[1:] = 1
    replay.seed_model.parent_stable_id[1:] = replay.seed_model.stable_id[0]
    replay.seed_model.metadata[1]["root_seed_row"] = 0
    replay.seed_model.metadata[2]["root_seed_row"] = 0
    replay.seed_last_visible_rows[:] = True
    replay.seed_model.xyz_gradient_accum[:, 0] = torch.tensor([3e-3, 2e-3, 1e-3])
    replay.seed_model.xyz_gradient_accum_abs[:, 0] = replay.seed_model.xyz_gradient_accum[:, 0]
    replay.seed_model.gradient_denom[:] = 1.0

    result = apply_typed_density(replay, timestamp=4, random_seed=0)

    assert result["clone_count"] == 1
    assert result["children"] == 1
    assert replay.seed_model.num_gaussians == 4
    assert replay.seed_model.metadata[3]["root_seed_row"] == 0


def test_prune_typed_seeds_only_retires_old_visible_low_opacity_open_rows():
    replay = _replay(max_rows=6)
    append_typed_seeds(replay, torch.zeros((3, 3)), torch.full((3, 3), -3.0), 1)
    replay.seed_lifecycle.open_rows([0, 1, 2], 2)
    replay.seed_model.open_rows(torch.tensor([0, 1, 2]), 2)
    replay.seed_last_visible_rows[:] = torch.tensor([True, True, False])
    replay.seed_geometry_update_counts[:] = torch.tensor([4, 3, 4])
    replay.seed_model._opacity.data.fill_(-10.0)

    result = prune_typed_seeds(replay, timestamp=5)

    assert result["pruned"] == 1
    assert replay.seed_retired.tolist() == [True, False, False]
    assert replay.seed_lifecycle.closed_mask(5).tolist() == [True, False, False]
    assert replay.seed_model.num_gaussians == 3

    # Just-opened or fresh rows are protected even if their learned opacity is tiny.
    replay.seed_last_visible_rows[1] = True
    replay.seed_lifecycle.close_rows([1], 6)
    replay.seed_model.close_rows(torch.tensor([1]), 6)
    replay.seed_lifecycle.open_rows([1], 7)
    replay.seed_model.open_rows(torch.tensor([1]), 7)
    result = prune_typed_seeds(replay, timestamp=7)
    assert result["pruned"] == 0
    assert replay.seed_retired.tolist() == [True, False, False]


def test_unlimited_archive_grows_past_twenty_thousand_without_reusing_closed_rows():
    replay = _replay(max_rows=0)
    prefix = _snapshot_tracker_prefix(replay)
    count = 20000
    assert append_typed_seeds(replay, torch.zeros(count, 3), torch.full((count, 3), -3.), 0) == count
    replay.seed_lifecycle.open_rows([0], 0)
    replay.seed_model.open_rows(torch.tensor([0]), 0)
    replay.seed_lifecycle.close_rows([0], 1)
    replay.seed_model.close_rows(torch.tensor([0]), 1)
    replay.seed_retired[0] = True
    replay.tracker.stable_a[replay.count] = 9.
    assert _append_two(replay, timestamp=2) == 2
    assert replay.seed_model.num_gaussians == 20002
    assert replay.seed_detector_probe.num_seeds == replay.seed_lifecycle.count == 20002
    assert replay.accepted_da3_source_rows[-2:] == [20000, 20001]
    assert replay.seed_retired[0] and replay.tracker.stable_a[replay.count] == 9.
    assert replay.tracker.last_timestamp[replay.count + 20000:replay.count + 20002].eq(-1).all()
    _assert_tracker_prefix_preserved(replay, prefix)


def test_tracker_allocation_failure_does_not_append_root_archive(monkeypatch):
    replay = _replay(max_rows=0)
    def fail(*args):
        raise RuntimeError("allocation failed")
    monkeypatch.setattr(replay.tracker, "ensure_capacity", fail, raising=False)
    with pytest.raises(RuntimeError, match="allocation failed"):
        _append_two(replay)
    assert replay.seed_model.num_gaussians == replay.seed_detector_probe.num_seeds == replay.seed_lifecycle.count == 0
    assert replay.accepted_da3_source_rows == []
