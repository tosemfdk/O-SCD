from types import SimpleNamespace

import pytest
import torch

from temporal.active_new_density import (
    ActiveNewDensityConfig,
    ActiveNewPruneConfig,
    densify_active_new,
    prune_active_new,
    root_anchor_hinge_loss,
    update_causal_new_support,
)
from temporal.active_new_gaussians import ActiveNewGaussianModel, ActiveNewGeometryView


def _model_with_optimizer(count=1):
    model = ActiveNewGaussianModel()
    optimizer = torch.optim.Adam(
        model.optimizer_parameter_groups(
            xyz_lr=1e-3,
            dc_lr=1e-3,
            opacity_lr=1e-3,
            scaling_lr=1e-3,
            rotation_lr=1e-3,
        ),
        eps=1e-15,
    )
    if count:
        model.append_xfeat_anchors(
            xyz=torch.tensor([[0.0, 0.0, 2.0]]).repeat(count, 1),
            start=1.0,
            scaling=torch.full((count, 3), -3.0),
            optimizer=optimizer,
        )
    return model, optimizer


def _initialize_adam(model, optimizer):
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_active_new_append_preserves_all_adam_rows():
    model, optimizer = _model_with_optimizer()
    _initialize_adam(model, optimizer)
    before = {}
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        before[group["name"]] = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state[parameter].items()
        }

    model.append_xfeat_anchors(
        xyz=torch.tensor([[1.0, 0.0, 2.0]]),
        start=2.0,
        scaling=torch.full((1, 3), -2.5),
        optimizer=optimizer,
    )

    assert model.num_gaussians == 2
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        state = optimizer.state[parameter]
        assert torch.equal(state["exp_avg"][:1], before[group["name"]]["exp_avg"])
        assert torch.equal(state["exp_avg_sq"][:1], before[group["name"]]["exp_avg_sq"])
        assert torch.equal(state["exp_avg"][1:], torch.zeros_like(state["exp_avg"][1:]))
        assert torch.equal(state["exp_avg_sq"][1:], torch.zeros_like(state["exp_avg_sq"][1:]))
        assert torch.equal(state["step"], before[group["name"]]["step"])


def test_active_new_prune_keeps_parameter_state_and_metadata_aligned():
    model, optimizer = _model_with_optimizer(count=3)
    model.metadata[0]["tag"] = "zero"
    model.metadata[1]["tag"] = "one"
    model.metadata[2]["tag"] = "two"
    _initialize_adam(model, optimizer)
    old_state = {}
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        old_state[group["name"]] = optimizer.state[parameter]["exp_avg"].clone()

    model.prune_rows(
        torch.tensor([False, True, False]), optimizer=optimizer, reason="test"
    )

    assert model.num_gaussians == 2
    assert [row["tag"] for row in model.metadata] == ["zero", "two"]
    assert model.stable_id.tolist() == [0, 2]
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        assert torch.equal(
            optimizer.state[parameter]["exp_avg"],
            old_state[group["name"]][torch.tensor([True, False, True])],
        )


def test_split_children_are_new_only_and_replace_the_parent():
    model, optimizer = _model_with_optimizer()
    _initialize_adam(model, optimizer)
    parent_id = int(model.stable_id[0])
    rows = model.append_gradient_children(
        parent_rows=torch.tensor([0]),
        split=True,
        timestamp=3,
        optimizer=optimizer,
        trigger_scores=torch.tensor([0.2]),
        random_seed=7,
    )

    assert model.num_gaussians == 2
    assert rows.tolist() == [0, 1]
    assert parent_id not in model.stable_id.tolist()
    assert model.parent_stable_id.tolist() == [parent_id, parent_id]
    assert model.generation.tolist() == [1, 1]
    assert all(row["birth_kind"] == "gradient_densified" for row in model.metadata)
    assert all(row["parent_row"] == 0 for row in model.metadata)
    assert not torch.equal(model._xyz[0], model._xyz[1])
    with pytest.raises(IndexError, match="NEW sidecar"):
        model.append_gradient_children(
            parent_rows=torch.tensor([1_283_501]),
            split=False,
            timestamp=4,
            optimizer=optimizer,
            trigger_scores=torch.tensor([1.0]),
        )


def test_density_cap_and_half_open_lifespan():
    model, optimizer = _model_with_optimizer(count=2)
    model._scaling.data.fill_(-5.0)
    model.xyz_gradient_accum.fill_(1.0)
    model.gradient_denom.fill_(1.0)
    result = densify_active_new(
        model,
        optimizer,
        timestamp=2,
        scene_extent=1.0,
        config=ActiveNewDensityConfig(
            grad_threshold=0.1,
            abs_grad_threshold=0.1,
            percent_dense=0.1,
            max_gaussians=3,
        ),
        random_seed=0,
    )
    assert result["count_after"] == 3
    assert result["clone_count"] == 1
    assert result["events"][0]["trigger_score"] == pytest.approx(1.0)
    model.close_active(4.0)
    assert not bool(model.active_mask(4.0).any())
    assert bool((model.end == 4.0).all())


def test_active_new_detector_lifecycle_hides_closed_and_keeps_never_open_black():
    model, optimizer = _model_with_optimizer(count=0)
    rows = model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        start=3.0,
        scaling=torch.full((2, 3), -3.0),
        optimizer=optimizer,
        start_active=False,
    )

    assert rows.tolist() == [0, 1]
    assert model.birth_frame.tolist() == [3, 3]
    assert model.never_open_mask().tolist() == [True, True]
    lifecycle = model.get_lifecycle_render_attributes(3.0)
    assert lifecycle["rows"].tolist() == [0, 1]
    assert torch.allclose(
        lifecycle["dc"],
        torch.full_like(lifecycle["dc"], -0.5 / 0.28209479177387814),
    )

    model.open_rows(torch.tensor([0]), 4.0)
    assert model.active_mask(4.0).tolist() == [True, False]
    model.close_rows(torch.tensor([0]), 5.0)
    assert model.closed_mask(5.0).tolist() == [True, False]
    lifecycle = model.get_lifecycle_render_attributes(5.0)
    assert lifecycle["rows"].tolist() == [1]


def test_never_open_geometry_view_trains_xyz_scale_rotation_but_not_dc_opacity():
    model, optimizer = _model_with_optimizer(count=0)
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        start=1.0,
        scaling=torch.full((2, 3), -3.0),
        optimizer=optimizer,
        start_active=False,
    )
    model.open_rows(torch.tensor([0]), 2.0)
    pending = ActiveNewGeometryView(
        model,
        2.0,
        row_mask=model.never_open_mask(),
        detach_dc=True,
        detach_opacity=True,
    )

    loss = (
        pending.get_xyz.sum()
        + pending.get_scaling.sum()
        + pending.get_rotation.sum()
        + pending.get_opacity.sum()
        + pending.get_features.sum()
    )
    loss.backward()

    assert pending.global_rows.tolist() == [1]
    assert torch.equal(model._xyz.grad[0], torch.zeros(3))
    assert bool((model._xyz.grad[1].abs() > 0).any())
    assert torch.equal(model._scaling.grad[0], torch.zeros(3))
    assert bool((model._scaling.grad[1].abs() > 0).any())
    assert torch.equal(model._rotation.grad[0], torch.zeros(4))
    assert bool((model._rotation.grad[1].abs() > 0).any())
    assert model._opacity.grad is None
    assert model.new_dc.grad is None


def test_probability_opacity_stays_in_probability_semantics_at_saturation():
    model, optimizer = _model_with_optimizer(count=0)
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        start=1.0,
        scaling=torch.full((1, 3), -3.0),
        opacity=1.0,
        optimizer=optimizer,
    )
    assert float(model.get_opacity[0].detach()) > 0.999
    model.append_gradient_children(
        parent_rows=torch.tensor([0]),
        split=False,
        timestamp=2,
        optimizer=optimizer,
        trigger_scores=torch.tensor([1.0]),
    )
    assert float(model.get_opacity[1].detach()) > 0.999


def test_screen_radius_sanity_prune_consumes_accumulated_radius():
    model, optimizer = _model_with_optimizer()
    model.max_radii2d[0] = 101.0
    result = prune_active_new(
        model,
        optimizer,
        timestamp=20,
        scene_extent=10.0,
        config=ActiveNewPruneConfig(
            grace_frames=1,
            max_screen_radius=100.0,
            max_world_scale_ratio=10.0,
        ),
    )
    assert result["count_after"] == 0
    assert result["events"][0]["reasons"] == ["excessive_screen_radius"]


def test_pruning_preserves_closed_lifespan_history():
    model, optimizer = _model_with_optimizer()
    model.max_radii2d[0] = 101.0
    model.close_active(5.0)
    result = prune_active_new(
        model,
        optimizer,
        timestamp=6,
        scene_extent=10.0,
        config=ActiveNewPruneConfig(
            grace_frames=1,
            max_screen_radius=100.0,
            max_world_scale_ratio=10.0,
        ),
    )
    assert result["count_after"] == 1
    assert result["events"] == []
    assert not bool(model.active_mask(6.0)[0])


def test_active_new_checkpoint_roundtrip_preserves_complete_state():
    model, _ = _model_with_optimizer(count=2)
    model.metadata[0]["track_id"] = 17
    model.positive_support[:] = torch.tensor([2, 3])
    model.contradiction_support[:] = torch.tensor([1, 0])
    model.xyz_gradient_accum[:] = torch.tensor([[0.2], [0.4]])
    model.xyz_gradient_accum_abs[:] = torch.tensor([[0.3], [0.5]])
    model.gradient_denom[:] = 2.0
    model.max_radii2d[:] = torch.tensor([4.0, 5.0])
    model.observed_view_ids[0] = {2, 3}

    restored = ActiveNewGaussianModel.from_checkpoint(model.to_checkpoint())

    for name in (
        "_xyz",
        "new_dc",
        "_opacity",
        "_scaling",
        "_rotation",
        "_features_rest",
        "start",
        "end",
        "stable_id",
        "parent_stable_id",
        "generation",
        "birth_frame",
        "initial_xyz",
        "root_anchor_xyz",
        "root_anchor_scale",
        "densification_count",
        "positive_support",
        "contradiction_support",
        "xyz_gradient_accum",
        "xyz_gradient_accum_abs",
        "gradient_denom",
        "max_radii2d",
    ):
        assert torch.equal(getattr(restored, name), getattr(model, name)), name
    assert restored.metadata == model.metadata
    assert restored.observed_view_ids == model.observed_view_ids
    assert restored.next_stable_id == model.next_stable_id


def test_gradient_children_inherit_root_anchor_and_radius_hinge_pullback():
    model, optimizer = _model_with_optimizer()
    root_xyz = model.root_anchor_xyz[0].clone()
    root_scale = model.root_anchor_scale[0].clone()
    child_rows = model.append_gradient_children(
        parent_rows=torch.tensor([0]),
        split=False,
        timestamp=2,
        optimizer=optimizer,
        trigger_scores=torch.tensor([1.0]),
    )
    child = int(child_rows[0])
    assert torch.equal(model.root_anchor_xyz[child], root_xyz)
    assert torch.equal(model.root_anchor_scale[child], root_scale)

    model._xyz.data[child] = root_xyz + torch.tensor(
        [float(root_scale) * 3.0, 0.0, 0.0]
    )
    loss, diagnostics = root_anchor_hinge_loss(
        model, timestamp=2, radius_multiplier=2.0
    )
    loss.backward()

    assert diagnostics["rows"] == 1
    assert diagnostics["outside_rows"] == 1
    assert diagnostics["max_distance_ratio"] == pytest.approx(1.5)
    assert model._xyz.grad is not None
    assert float(model._xyz.grad[child, 0]) > 0.0
    assert torch.equal(model._xyz.grad[0], torch.zeros(3))


def test_xfeat_xyz_restore_is_exact_and_clears_adam_momentum():
    model, optimizer = _model_with_optimizer()
    child = int(
        model.append_gradient_children(
            parent_rows=torch.tensor([0]),
            split=False,
            timestamp=2,
            optimizer=optimizer,
            trigger_scores=torch.tensor([1.0]),
        )[0]
    )
    anchor_before = model._xyz[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = model._xyz[:, 0].sum()
    loss.backward()
    assert float(model._xyz.grad[0, 0]) == 1.0
    model.zero_xfeat_anchor_xyz_gradient()
    optimizer.step()
    restored = model.restore_xfeat_anchor_xyz(optimizer)

    assert restored == 1
    assert torch.equal(model._xyz[0].detach(), anchor_before)
    assert not torch.equal(model._xyz[child].detach(), anchor_before)
    xyz_state = optimizer.state[model._xyz]
    assert torch.equal(xyz_state["exp_avg"][0], torch.zeros(3))
    assert torch.equal(xyz_state["exp_avg_sq"][0], torch.zeros(3))


def test_preserved_xfeat_anchor_densifies_once_and_is_not_pruned():
    model, optimizer = _model_with_optimizer()
    model.xyz_gradient_accum_abs[0] = 1.0
    model.gradient_denom[0] = 1.0
    config = ActiveNewDensityConfig(
        grad_threshold=0.1,
        abs_grad_threshold=0.1,
        percent_dense=0.01,
        max_gaussians=10,
        preserve_xfeat_anchors=True,
        one_shot_anchor_densification=True,
    )
    first = densify_active_new(
        model,
        optimizer,
        timestamp=2,
        scene_extent=1.0,
        config=config,
        random_seed=3,
    )

    assert first["count_after"] == 3
    assert first["events"][0]["action"] == "split_preserve_xfeat_anchor"
    assert model.generation.tolist() == [0, 1, 1]
    assert model.densification_count.tolist() == [1, 0, 0]
    anchor_id = int(model.stable_id[0])

    model.xyz_gradient_accum_abs[0] = 1.0
    model.gradient_denom[0] = 1.0
    second = densify_active_new(
        model,
        optimizer,
        timestamp=3,
        scene_extent=1.0,
        config=config,
        random_seed=4,
    )
    assert second["events"] == []
    assert model.num_gaussians == 3

    model.max_radii2d[:] = 101.0
    pruned = prune_active_new(
        model,
        optimizer,
        timestamp=20,
        scene_extent=10.0,
        config=ActiveNewPruneConfig(
            grace_frames=1,
            max_screen_radius=100.0,
            max_world_scale_ratio=10.0,
            preserve_xfeat_anchors=True,
        ),
    )
    assert pruned["count_after"] == 1
    assert model.stable_id.tolist() == [anchor_id]
    assert int(model.generation[0]) == 0


def test_causal_support_rejects_future_and_deduplicates_views():
    model, _ = _model_with_optimizer()
    view = SimpleNamespace(
        world_view_transform=torch.eye(4),
        image_width=8,
        image_height=8,
        FoVx=1.0,
        FoVy=1.0,
    )
    # Put the point at the image center with positive camera depth.
    new_mask = torch.zeros(8, 8, dtype=torch.bool)
    new_mask[3:6, 3:6] = True
    stable = ~new_mask
    config = ActiveNewPruneConfig(erosion_pixels=0, min_visible_mass=0.0)
    with pytest.raises(ValueError, match="future"):
        update_causal_new_support(
            model,
            decision_timestamp=3,
            observation_timestamp=4,
            view=view,
            new_mask=new_mask,
            stable_mask=stable,
            active_rows=torch.tensor([0]),
            visibility_mass=torch.ones(1),
            config=config,
        )
    first = update_causal_new_support(
        model,
        decision_timestamp=3,
        observation_timestamp=2,
        view=view,
        new_mask=new_mask,
        stable_mask=stable,
        active_rows=torch.tensor([0]),
        visibility_mass=torch.ones(1),
        config=config,
    )
    second = update_causal_new_support(
        model,
        decision_timestamp=3,
        observation_timestamp=2,
        view=view,
        new_mask=new_mask,
        stable_mask=stable,
        active_rows=torch.tensor([0]),
        visibility_mass=torch.ones(1),
        config=config,
    )
    assert first["positive"] == 1
    assert second["positive"] == 0
    assert model.positive_support.tolist() == [1]
