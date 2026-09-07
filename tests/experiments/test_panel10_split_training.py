from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments import panel10_split_training as split


class ToyLifecycle:
    def __init__(self, active_by_t, never_by_t=None, materialized_by_t=None):
        self.active_by_t = {int(k): torch.tensor(v, dtype=torch.bool) for k, v in active_by_t.items()}
        first = next(iter(self.active_by_t.values()), torch.empty(0, dtype=torch.bool))
        self.never_by_t = (
            {int(k): torch.tensor(v, dtype=torch.bool) for k, v in never_by_t.items()}
            if never_by_t is not None
            else {k: torch.zeros_like(first) for k in self.active_by_t}
        )
        self.materialized_by_t = (
            {int(k): torch.tensor(v, dtype=torch.bool) for k, v in materialized_by_t.items()}
            if materialized_by_t is not None
            else {k: torch.ones_like(first) for k in self.active_by_t}
        )

    def active_mask(self, timestamp):
        return self.active_by_t[int(timestamp)].clone()

    def never_open_mask(self, timestamp):
        return self.never_by_t[int(timestamp)].clone()

    def materialized_mask(self, timestamp):
        return self.materialized_by_t[int(timestamp)].clone()

    def closed_mask(self, timestamp):
        t = int(timestamp)
        return self.materialized_mask(t) & ~(self.active_mask(t) | self.never_open_mask(t))


class CaptureOptimizer:
    def __init__(self, params):
        self.params = params
        self.calls = []
        self.step_count = torch.zeros(next(iter(params.values())).shape[0], dtype=torch.long)

    def zero_grad(self, set_to_none=True):
        for p in self.params.values():
            p.grad = None if set_to_none else torch.zeros_like(p)

    @torch.no_grad()
    def step(self, mask):
        self.calls.append(mask.clone() if isinstance(mask, torch.Tensor) else {k: v.clone() for k, v in mask.items()})
        if isinstance(mask, dict):
            for name, p in self.params.items():
                rows = mask[name]
                if p.grad is not None and bool(rows.any()):
                    p[rows] -= 0.01 * p.grad[rows]
                    self.step_count[rows] += 1
        else:
            p = self.params["dc"]
            if p.grad is not None and bool(mask.any()):
                p[mask] -= 0.01 * p.grad[mask]
                self.step_count[mask] += 1


class ToyBase:
    max_sh_degree = 0
    active_sh_degree = 0

    def __init__(self, n):
        self._xyz = torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10 + 1)
        self._features_dc = torch.nn.Parameter(torch.zeros(n, 1, 3))
        self._features_rest = torch.empty(n, 0, 3)
        self._opacity = torch.nn.Parameter(torch.zeros(n, 1))
        self._scaling = torch.nn.Parameter(torch.zeros(n, 3))
        rot = torch.zeros(n, 4)
        rot[:, 0] = 1
        self._rotation = torch.nn.Parameter(rot)
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = lambda v: torch.nn.functional.normalize(v, dim=1)
        self.covariance_activation = lambda scaling, modifier, rotation: scaling * modifier

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)


class ToySeeds(ToyBase):
    def __init__(self, n):
        super().__init__(n)
        self.new_dc = torch.nn.Parameter(torch.zeros(n, 1, 3))
        self._features_dc = self.new_dc
        self.num_gaussians = n
        self.num_seeds = n
        self.xyz_gradient_accum = torch.zeros(n, 1)
        self.xyz_gradient_accum_abs = torch.zeros(n, 1)
        self.gradient_denom = torch.zeros(n, 1)
        self.max_radii2d = torch.zeros(n)


def make_replay(*, current=1, with_seeds=True):
    base = ToyBase(3)
    seeds = ToySeeds(3) if with_seeds else ToySeeds(0)
    probe = ToySeeds(seeds.num_gaussians)
    if seeds.num_gaussians:
        with torch.no_grad():
            probe._xyz.copy_(seeds._xyz + 10.0)
            probe._opacity.copy_(seeds._opacity + 1.0)
            probe._scaling.copy_(seeds._scaling + 0.5)
    replay = SimpleNamespace(
        current_index=current,
        base=base,
        seed_model=seeds,
        seed_detector_probe=probe,
        lifecycle=ToyLifecycle(
            active_by_t={0: [1, 1, 0], 1: [1, 0, 0], 2: [1, 0, 0]},
            never_by_t={0: [0, 0, 1], 1: [0, 0, 1], 2: [0, 0, 1]},
        ),
        seed_lifecycle=ToyLifecycle(
            active_by_t={0: [1, 1, 0], 1: [1, 0, 0], 2: [0, 0, 0]},
            never_by_t={0: [0, 0, 1], 1: [0, 0, 1], 2: [0, 0, 1]},
        ),
        change_dc=torch.nn.Parameter(torch.zeros(3, 1, 3)),
        args=SimpleNamespace(representation_cue_amplitude=2.0),
        pipe=SimpleNamespace(),
        evidence_background=torch.zeros(3),
        seed_geometry_update_counts=torch.zeros(seeds.num_gaussians, dtype=torch.long),
    )
    replay.base_optimizer = CaptureOptimizer({"dc": replay.change_dc})
    replay.seed_optimizer = CaptureOptimizer(
        {
            "xyz": seeds._xyz,
            "dc": seeds.new_dc,
            "opacity": seeds._opacity,
            "scaling": seeds._scaling,
            "rotation": seeds._rotation,
        }
    )
    replay.constrained = []
    replay._constrain_typed_seeds = lambda mask: replay.constrained.append(mask.clone())
    return replay


def fake_render(_camera, model, _pipe, _background, **_kwargs):
    n = model.get_xyz.shape[0]
    points = model.get_xyz
    if points.requires_grad:
        points.retain_grad()
    if n == 0:
        scalar = torch.zeros((), requires_grad=True)
        radii = torch.empty(0)
    else:
        # Every attribute contributes so seed DC, geometry, opacity, scaling and
        # rotation all receive gradients when their rows are trainable.
        row_signal = (
            model.get_features.mean(dim=(1, 2))
            + 0.05 * points.sum(dim=1)
            + 0.03 * model.get_opacity.flatten()
            + 0.02 * model.get_scaling.sum(dim=1)
            + 0.01 * model.get_rotation.sum(dim=1)
        )
        scalar = row_signal.sum()
        radii = torch.ones(n)
    return {"render": scalar.expand(3, 2, 2), "radii": radii, "viewspace_points": points}


@pytest.fixture(autouse=True)
def patch_renderer(monkeypatch):
    monkeypatch.setattr(split, "render_change", fake_render)


def item(timestamp=1, new=0.4, cue=0.9):
    return SimpleNamespace(
        timestamp=timestamp,
        view=SimpleNamespace(),
        new_target=torch.full((1, 2, 2), float(new)),
        cue_target=torch.full((1, 2, 2), float(cue)),
    )


def test_build_partition_view_branch_membership_and_global_mappings():
    replay = make_replay(current=1)

    base = split.build_partition_view(replay, 1, "base")
    assert base.base_rows.tolist() == [0, 2, -1]
    assert base.seed_rows.tolist() == [-1, -1, 2]
    assert base.trainable_render_rows.tolist() == [True, False, False]

    new = split.build_partition_view(replay, 1, "new")
    assert new.base_rows.tolist() == [2, -1, -1]
    assert new.seed_rows.tolist() == [-1, 0, 2]
    assert new.trainable_render_rows.tolist() == [False, True, False]

    joint = split.build_partition_view(replay, 1, "joint", train=False)
    assert joint.base_rows.tolist() == [0, 2, -1, -1]
    assert joint.seed_rows.tolist() == [-1, -1, 0, 2]
    assert joint.trainable_render_rows.tolist() == [False, False, False, False]


def test_historical_open_current_closed_rows_render_detached_and_do_not_step():
    replay = make_replay(current=1)
    # At t=0 base row1 and seed row1 were OPEN; at current t=1 they are CLOSED.
    view = split.build_partition_view(replay, 0, "joint")
    assert view.base_rows.tolist() == [0, 1, 2, -1, -1, -1]
    assert view.seed_rows.tolist() == [-1, -1, -1, 0, 1, 2]
    assert view.trainable_render_rows.tolist() == [True, False, False, True, False, False]

    out = split.train_partition_update(replay, item(timestamp=0), current_timestamp=1)

    assert replay.base_optimizer.calls[-1].tolist() == [True, False, False]
    assert replay.seed_optimizer.calls[-1]["dc"].tolist() == [True, False, False]
    assert out["trainable_open_rows"] == 1
    assert out["visible_seed_rows"] == 1
    assert replay.seed_geometry_update_counts.tolist() == [1, 0, 0]


def test_train_partition_update_isolates_base_and_seed_gradients_and_moments():
    replay = make_replay(current=1)
    old_base = replay.change_dc.detach().clone()
    old_seed_xyz = replay.seed_model._xyz.detach().clone()
    old_seed_dc = replay.seed_model.new_dc.detach().clone()

    out = split.train_partition_update(replay, item(timestamp=1), current_timestamp=1)

    assert out["active_seed_rows"] == 1
    assert replay.base_optimizer.calls[-1].tolist() == [True, False, False]
    assert replay.seed_optimizer.calls[-1]["xyz"].tolist() == [True, False, False]
    assert not torch.equal(replay.change_dc[0], old_base[0])
    assert torch.equal(replay.change_dc[1:], old_base[1:])
    assert not torch.equal(replay.seed_model._xyz[0], old_seed_xyz[0])
    assert torch.equal(replay.seed_model._xyz[1:], old_seed_xyz[1:])
    assert not torch.equal(replay.seed_model.new_dc[0], old_seed_dc[0])
    assert torch.equal(replay.seed_model.new_dc[1:], old_seed_dc[1:])
    assert replay.constrained[-1].tolist() == [True, False, False]


def test_seed_screen_gradient_stats_are_seed_only_and_current_visible():
    replay = make_replay(current=1)

    split.train_partition_update(replay, item(timestamp=1), current_timestamp=1)

    assert replay.seed_last_visible_rows.tolist() == [True, False, False]
    assert replay.seed_model.gradient_denom.flatten().tolist() == [1.0, 0.0, 0.0]
    assert float(replay.seed_model.xyz_gradient_accum[0]) > 0.0
    assert float(replay.seed_model.xyz_gradient_accum_abs[0]) > 0.0
    assert replay.seed_model.max_radii2d.tolist() == [1.0, 0.0, 0.0]


def test_dc_item_replaces_item_for_both_new_and_base_targets(monkeypatch):
    replay = make_replay(current=1)
    seen = []
    real_ssf = split.compute_ssf_loss

    def spy(target, rendered):
        seen.append(target.detach().clone())
        return real_ssf(target, rendered)

    monkeypatch.setattr(split, "compute_ssf_loss", spy)
    split.train_partition_update(
        replay,
        item(timestamp=1, new=0.1, cue=0.3),
        current_timestamp=1,
        dc_item=item(timestamp=1, new=0.4, cue=0.9),
    )

    assert len(seen) == 2
    assert torch.allclose(seen[0], torch.full((1, 2, 2), 0.8))
    assert torch.allclose(seen[1], torch.full((1, 2, 2), 1.0))


def test_no_seeds_and_no_new_target_do_not_crash_or_mutate_seed_state():
    replay = make_replay(current=1, with_seeds=False)

    out = split.train_partition_update(replay, item(timestamp=1, new=0.0, cue=0.5), current_timestamp=1)

    assert out["active_seed_rows"] == 0
    assert out["visible_seed_rows"] == 0
    assert replay.seed_last_visible_rows.numel() == 0
    assert replay.seed_geometry_update_counts.numel() == 0
    assert replay.seed_optimizer.calls == []


def test_rejects_future_replay_timestamp():
    replay = make_replay(current=1)
    with pytest.raises(RuntimeError, match="causal range"):
        split.build_partition_view(replay, 2, "base")
    with pytest.raises(RuntimeError, match="causal range"):
        split.train_partition_update(replay, item(timestamp=2), current_timestamp=1)


def test_partition_targets_validate_shape_range_and_amplitude():
    good = item(timestamp=1, new=0.2, cue=0.7)
    split._validate_partition_targets(good.cue_target, good.new_target, amplitude=2.0)

    with pytest.raises(ValueError, match="0 <= new_target <= cue_target <= 1"):
        split._validate_partition_targets(
            torch.full((1, 2, 2), 0.4),
            torch.full((1, 2, 2), 0.5),
            amplitude=2.0,
        )
    with pytest.raises(ValueError, match="shape"):
        split._validate_partition_targets(
            torch.full((2, 2), 0.4),
            torch.full((2, 2), 0.2),
            amplitude=2.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        split._validate_partition_targets(
            torch.full((1, 2, 2), 0.4),
            torch.full((1, 2, 2), 0.2),
            amplitude=0.0,
        )


def test_partition_update_rejects_invalid_new_complement_before_render():
    replay = make_replay(current=1)
    bad = item(timestamp=1, new=0.8, cue=0.4)

    with pytest.raises(ValueError, match="new_target <= cue_target"):
        split.train_partition_update(replay, bad, current_timestamp=1)

    assert replay.base_optimizer.calls == []
    assert replay.seed_optimizer.calls == []


def test_production_masked_row_adam_preserves_frozen_rows_and_moments():
    from temporal.masked_optimizer import MaskedRowAdam

    params = {
        "xyz": torch.nn.Parameter(torch.zeros(3, 3)),
        "dc": torch.nn.Parameter(torch.zeros(3, 1, 3)),
        "opacity": torch.nn.Parameter(torch.zeros(3, 1)),
        "scaling": torch.nn.Parameter(torch.zeros(3, 3)),
        "rotation": torch.nn.Parameter(torch.zeros(3, 4)),
    }
    optimizer = MaskedRowAdam(params, thaw_names=params.keys(), lrs={name: 0.1 for name in params})
    for param in params.values():
        param.grad = torch.ones_like(param)
    row_masks = {
        "xyz": torch.tensor([True, False, False]),
        "dc": torch.tensor([True, False, False]),
        "opacity": torch.tensor([True, False, False]),
        "scaling": torch.tensor([True, False, False]),
        "rotation": torch.tensor([True, False, False]),
    }

    before_values = {name: param.detach().clone() for name, param in params.items()}
    optimizer.step(row_masks)

    for name, param in params.items():
        assert not torch.equal(param[0], before_values[name][0])
        assert torch.equal(param[1:], before_values[name][1:])
        state = optimizer.state[param]
        assert state["step"].tolist() == [1, 0, 0]
        assert float(state["exp_avg"][0].abs().sum()) > 0.0
        assert float(state["exp_avg"][1:].abs().sum()) == 0.0
        assert float(state["exp_avg_sq"][0].abs().sum()) > 0.0
        assert float(state["exp_avg_sq"][1:].abs().sum()) == 0.0


def test_sampled_never_open_seed_uses_frozen_detector_probe_geometry():
    replay = make_replay(current=1)
    with torch.no_grad():
        replay.seed_model._xyz[2].fill_(123.0)
        replay.seed_model._opacity[2].fill_(5.0)
        replay.seed_detector_probe._xyz[2] = torch.tensor([7.0, 8.0, 9.0])
        replay.seed_detector_probe._opacity[2].fill_(-2.0)

    view = split.build_partition_view(replay, 1, "new")
    never_render_row = torch.nonzero(view.seed_rows == 2, as_tuple=False).flatten().item()

    assert torch.allclose(view.get_xyz[never_render_row], torch.tensor([7.0, 8.0, 9.0]))
    assert torch.allclose(
        view.get_opacity[never_render_row], replay.seed_detector_probe.get_opacity[2]
    )

    view.get_xyz[never_render_row].sum().backward(retain_graph=True)
    assert replay.seed_model._xyz.grad is None or float(replay.seed_model._xyz.grad[2].abs().sum()) == 0.0
    assert replay.seed_detector_probe._xyz.grad is None or float(replay.seed_detector_probe._xyz.grad[2].abs().sum()) == 0.0


def test_real_active_seed_with_fixed_probe_and_detector_lifespan_contract():
    from experiments.view_bayesian_detector_steps import DetectorReplayLifespan
    from temporal.active_new_gaussians import ActiveNewGaussianModel
    from temporal.new_seed_gaussians import NewSeedGaussianModel

    replay = make_replay(current=1, with_seeds=False)
    seeds = ActiveNewGaussianModel(device="cpu")
    seeds.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        start=0.0,
        scaling=torch.zeros(2, 3),
        opacity=0.5,
    )
    with torch.no_grad():
        seeds.start[1] = float("inf")
        seeds.end[1] = float("inf")
        seeds._xyz[1] = torch.tensor([99.0, 99.0, 99.0])
    probe = NewSeedGaussianModel(device="cpu")
    probe.append(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [1.0, 2.0, 3.0]]),
        start=float("inf"),
        end=float("inf"),
        scaling=torch.zeros(2, 3),
        opacity=0.25,
    )
    lifecycle = DetectorReplayLifespan(
        2, max_states=2, device=torch.device("cpu"), materialized_timestamp=0
    )
    lifecycle.open_rows([0], timestamp=0)

    replay.seed_model = seeds
    replay.seed_detector_probe = probe
    replay.seed_lifecycle = lifecycle
    replay.seed_geometry_update_counts = torch.zeros(2, dtype=torch.long)
    replay.seed_optimizer = CaptureOptimizer(
        {
            "xyz": seeds._xyz,
            "dc": seeds.new_dc,
            "opacity": seeds._opacity,
            "scaling": seeds._scaling,
            "rotation": seeds._rotation,
        }
    )

    view = split.build_partition_view(replay, 1, "new")

    assert view.max_sh_degree == 0
    assert view.get_features.shape[1] == 1
    assert view.seed_rows.tolist() == [-1, 0, 1]
    assert view.trainable_render_rows.tolist() == [False, True, False]
    assert torch.allclose(view.get_xyz[2], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(view.get_opacity[2], probe.get_opacity[1])


@pytest.mark.cuda
@pytest.mark.parametrize("render_mode", ["split", "joint_channels"])
def test_real_cuda_partition_update_trains_base_and_seed_dc(monkeypatch, render_mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real FastGS DC gradient regression")
    pytest.importorskip("diff_gaussian_rasterization_fastgs")
    pytest.importorskip("simple_knn._C")

    from gaussian_renderer import render_change as cuda_render_change
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from temporal.active_new_gaussians import ActiveNewGaussianModel
    from temporal.masked_optimizer import MaskedRowAdam
    from temporal.new_seed_gaussians import NewSeedGaussianModel
    from utils.general_utils import inverse_sigmoid
    from utils.graphics_utils import getProjectionMatrix

    device = torch.device("cuda")
    monkeypatch.setattr(split, "render_change", cuda_render_change)

    fov = math.radians(60.0)
    world = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).t().to(device)
    camera = MiniCam(
        32,
        32,
        fov,
        fov,
        0.01,
        100.0,
        world,
        world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
    )
    base = GaussianModel(sh_degree=0, active_sh_degree=0)
    base._xyz = nn.Parameter(torch.tensor([[-0.18, 0.0, 2.0]], device=device))
    base._features_dc = nn.Parameter(torch.zeros((1, 1, 3), device=device))
    base._features_rest = nn.Parameter(torch.zeros((1, 0, 3), device=device))
    base._opacity = nn.Parameter(inverse_sigmoid(torch.full((1, 1), 0.8, device=device)))
    base._scaling = nn.Parameter(torch.full((1, 3), math.log(0.22), device=device))
    base._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))

    seeds = ActiveNewGaussianModel(device="cuda")
    seeds.append_xfeat_anchors(
        xyz=torch.tensor([[0.18, 0.0, 2.0]], device=device),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.22), device=device),
        opacity=0.8,
    )
    probe = NewSeedGaussianModel(device="cuda")
    probe.append(
        xyz=torch.tensor([[0.18, 0.0, 2.0]], device=device),
        start=float("inf"),
        end=float("inf"),
        scaling=torch.full((1, 3), math.log(0.22), device=device),
        opacity=0.8,
    )
    replay = SimpleNamespace(
        current_index=0,
        base=base,
        seed_model=seeds,
        seed_detector_probe=probe,
        lifecycle=ToyLifecycle(active_by_t={0: [1]}, never_by_t={0: [0]}),
        seed_lifecycle=ToyLifecycle(active_by_t={0: [1]}, never_by_t={0: [0]}),
        change_dc=nn.Parameter(torch.zeros((1, 1, 3), device=device)),
        args=SimpleNamespace(representation_cue_amplitude=2.0),
        pipe=SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False),
        evidence_background=torch.zeros(3, device=device),
        seed_geometry_update_counts=torch.zeros(1, device=device, dtype=torch.long),
    )
    for lifecycle in (replay.lifecycle, replay.seed_lifecycle):
        lifecycle.active_by_t = {
            key: value.to(device=device) for key, value in lifecycle.active_by_t.items()
        }
        lifecycle.never_by_t = {
            key: value.to(device=device) for key, value in lifecycle.never_by_t.items()
        }
        lifecycle.materialized_by_t = {
            key: value.to(device=device)
            for key, value in lifecycle.materialized_by_t.items()
        }
    replay.base_optimizer = MaskedRowAdam(
        {"dc": replay.change_dc}, thaw_names=("dc",), lrs={"dc": 5e-2}
    )
    replay.seed_optimizer = MaskedRowAdam(
        {
            "xyz": seeds._xyz,
            "dc": seeds.new_dc,
            "opacity": seeds._opacity,
            "scaling": seeds._scaling,
            "rotation": seeds._rotation,
        },
        thaw_names=("xyz", "dc", "opacity", "scaling", "rotation"),
        lrs={"xyz": 1e-4, "dc": 5e-2, "opacity": 1e-4, "scaling": 1e-4, "rotation": 1e-4},
    )
    replay._constrain_typed_seeds = lambda _mask: None
    replay.args.panel10_render_mode = render_mode

    sample = SimpleNamespace(
        timestamp=0,
        view=camera,
        new_target=torch.full((1, 32, 32), 0.45, device=device),
        cue_target=torch.full((1, 32, 32), 0.90, device=device),
    )
    old_base_dc = replay.change_dc.detach().clone()
    old_seed_dc = seeds.new_dc.detach().clone()

    # The dummy buffer repairs backward only; degree-zero forward is identical.
    for branch in ("base", "new", "joint"):
        model = split.build_partition_view(replay, 0, branch)
        with torch.no_grad():
            fixed = cuda_render_change(camera, model, replay.pipe, replay.evidence_background)["render"]
            model._features_rest = model._features_rest[:, :0, :].contiguous()
            old = cuda_render_change(camera, model, replay.pipe, replay.evidence_background)["render"]
        assert torch.equal(fixed, old)

    if render_mode == "joint_channels":
        model = split.build_partition_view(replay, 0, "joint")
        colors = split.joint_partition_colors(model)
        with torch.no_grad():
            typed = cuda_render_change(camera, model, replay.pipe, replay.evidence_background,
                                       override_color=colors, clamp_output=False)["render"]
            original = cuda_render_change(camera, model, replay.pipe, replay.evidence_background,
                                          clamp_output=False)["render"]
            assert torch.allclose(typed[:2].sum(0), original.mean(0), atol=2e-6, rtol=2e-6)
            assert torch.count_nonzero(typed[2]) == 0
            # Independent per-source renders retain ALL geometry/opacity, so T
            # is shared. Channel packing must produce the same two signals.
            for channel in range(2):
                monochrome = colors[:, channel:channel + 1].expand(-1, 3).contiguous()
                expected = cuda_render_change(camera, model, replay.pipe, replay.evidence_background,
                                             override_color=monochrome, clamp_output=False)["render"]
                assert torch.allclose(typed[channel], expected.mean(0), atol=2e-6, rtol=2e-6)
        # One packed backward must equal two shared-geometry reference passes.
        joint = cuda_render_change(camera, model, replay.pipe, replay.evidence_background,
                                   override_color=colors, clamp_output=False)["render"]
        targets = (2 * sample.new_target, 2 * (sample.cue_target - sample.new_target))
        joint_loss = sum(split.compute_ssf_loss(targets[c], joint[c:c + 1].expand(3, -1, -1))[0]
                         for c in range(2))
        reference_losses = []
        for c in range(2):
            monochrome = colors[:, c:c + 1].expand(-1, 3).contiguous()
            rgb = cuda_render_change(camera, model, replay.pipe, replay.evidence_background,
                                     override_color=monochrome, clamp_output=False)["render"]
            reference_losses.append(split.compute_ssf_loss(targets[c], rgb)[0])
        params = (replay.change_dc, seeds.new_dc, seeds._xyz, seeds._opacity,
                  seeds._scaling, seeds._rotation)
        packed_grads = torch.autograd.grad(joint_loss, params, retain_graph=True)
        reference_grads = torch.autograd.grad(sum(reference_losses), params)
        for actual, expected in zip(packed_grads, reference_grads):
            assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-4)

    out = split.train_partition_update(replay, sample, current_timestamp=0)

    assert out["visible_seed_rows"] == 1
    assert replay.change_dc.grad is not None
    assert seeds.new_dc.grad is not None
    assert float(replay.change_dc.grad.abs().sum()) > 0.0
    assert float(seeds.new_dc.grad.abs().sum()) > 0.0
    assert not torch.equal(replay.change_dc.detach(), old_base_dc)
    assert not torch.equal(seeds.new_dc.detach(), old_seed_dc)
    if render_mode == "joint_channels":
        assert seeds._xyz.grad is not None and float(seeds._xyz.grad.abs().sum()) > 0
        assert seeds._opacity.grad is not None and float(seeds._opacity.grad.abs().sum()) > 0


def test_joint_colors_route_mean_post_sh_clamp_and_keep_never_open_black():
    from utils.sh_utils import SH2RGB
    replay = make_replay()
    with torch.no_grad():
        replay.change_dc[0, 0] = torch.tensor([-4., 0., 2.])
        replay.seed_model.new_dc[0, 0] = torch.tensor([1., 2., 3.])
    view = split.build_partition_view(replay, 1, "joint")
    colors = split.joint_partition_colors(view)
    expected_base = SH2RGB(replay.change_dc[0, 0]).clamp_min(0).mean()
    expected_new = SH2RGB(replay.seed_model.new_dc[0, 0]).clamp_min(0).mean()
    assert torch.allclose(colors[0], torch.stack((expected_base * 0, expected_base, expected_base * 0)))
    assert torch.allclose(colors[2], torch.stack((expected_new, expected_new * 0, expected_new * 0)))
    assert torch.count_nonzero(colors[[1, 3]]) == 0
    colors[:, 0].sum().backward(retain_graph=True)
    assert replay.change_dc.grad is None or not bool(replay.change_dc.grad.any())
    assert bool(replay.seed_model.new_dc.grad[0].any())


def typed_fake_render(_camera, model, _pipe, _background, *, override_color=None, **_kwargs):
    assert override_color is not None
    points = model.get_xyz
    if points.requires_grad:
        points.retain_grad()
    weights = (torch.sigmoid(points.sum(1) * .03) * model.get_opacity.flatten()
               * model.get_scaling.mean(1) * (1 + .01 * model.get_rotation.sum(1)))
    rgb = (override_color * weights[:, None]).sum(0)
    return {"render": rgb[:, None, None].expand(3, 2, 2),
            "radii": torch.ones(len(points)), "viewspace_points": points}


@pytest.mark.parametrize("timestamp", [0, 1])
def test_joint_update_one_scene_one_render_preserves_targets_and_frozen_rows(monkeypatch, timestamp):
    replay = make_replay()
    replay.args.panel10_render_mode = "joint_channels"
    calls, targets = [], []
    def render(*a, **kw):
        calls.append(a[1])
        return typed_fake_render(*a, **kw)
    real_ssf = split.compute_ssf_loss
    def ssf(target, rgb):
        targets.append(target.clone())
        return real_ssf(target, rgb)
    monkeypatch.setattr(split, "render_change", render)
    monkeypatch.setattr(split, "compute_ssf_loss", ssf)
    before_base = replay.change_dc.detach().clone()
    before_seed = replay.seed_model.new_dc.detach().clone()
    out = split.train_partition_update(replay, item(timestamp=timestamp), current_timestamp=1)
    assert len(calls) == 1
    assert (calls[0].base_rows >= 0).any() and (calls[0].seed_rows >= 0).any()
    assert torch.equal(targets[0], torch.full((1, 2, 2), .8))
    assert torch.allclose(targets[1], torch.ones(1, 2, 2))
    assert not torch.equal(replay.change_dc[0], before_base[0])
    assert not torch.equal(replay.seed_model.new_dc[0], before_seed[0])
    assert torch.equal(replay.change_dc[1:], before_base[1:])
    assert torch.equal(replay.seed_model.new_dc[1:], before_seed[1:])
    assert replay.seed_geometry_update_counts.tolist() == [1, 0, 0]
    assert out['loss'] == pytest.approx(out['new_loss'] + out['base_loss'], abs=1e-6)


def test_joint_update_without_seeds(monkeypatch):
    replay = make_replay(with_seeds=False)
    replay.args.panel10_render_mode = "joint_channels"
    monkeypatch.setattr(split, "render_change", typed_fake_render)
    out = split.train_partition_update(replay, item(new=0), current_timestamp=1)
    assert out['visible_seed_rows'] == 0
    assert replay.seed_optimizer.calls == []


def test_joint_cli_is_opt_in_and_rejects_non_panel10():
    from experiments.view_bayesian_detector_steps import parse_args
    assert parse_args([]).panel10_render_mode == 'split'
    with pytest.raises(SystemExit):
        parse_args(['--panel10-render-mode', 'joint_channels'])
