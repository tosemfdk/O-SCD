# Gate B: persistent Gaussian identity + episode freeze (GPU).
import numpy as np
import pytest
import torch

from tests.conftest import make_toy_model, make_opt_args
from target_nbv import target_registry as reg
from target_nbv.target_registry import UnknownTargetError

pytestmark = pytest.mark.gpu


def _ids(model):
    return model._persistent_id


def _assert_id_invariants(model):
    ids = _ids(model)
    assert ids.shape[0] == model.get_xyz.shape[0]
    assert ids.unique().numel() == ids.numel(), "duplicate persistent IDs"
    assert int(ids.max()) < model._next_persistent_id


def test_ids_initialized_and_resolvable(toy_model):
    _assert_id_invariants(toy_model)
    row = reg.resolve_target_by_persistent_id(toy_model, 42)
    assert row == 42  # fresh model: identity mapping
    assert reg.resolve_target_by_current_index(toy_model, 7) == 7
    with pytest.raises(UnknownTargetError):
        reg.resolve_target_by_persistent_id(toy_model, 10**9)
    with pytest.raises(UnknownTargetError):
        reg.resolve_target_by_current_index(toy_model, 10**9)


def test_surgery_stress_50_cycles():
    torch.manual_seed(0)
    model = make_toy_model(n=100)
    # frozen targets must survive every cycle content-exact; unprotected
    # gaussians may legitimately die via split (originals pruned) or prune
    frozen = [3, 55]
    tracked = frozen + [17, 42, 90]
    snapshots = {pid: model._xyz[reg.resolve_target_by_persistent_id(model, pid)].detach().clone()
                 for pid in tracked}
    for pid in frozen:
        reg.begin_target_episode(model, pid)
    extent = 1.0
    model.percent_dense = 0.05  # split for scales > 0.05, clone for <= 0.05

    for cycle in range(50):
        P = model.get_xyz.shape[0]
        grads = torch.rand((P, 1), device="cuda") * 2e-4
        model.densify_and_clone(grads, 1e-4, extent)
        P = model.get_xyz.shape[0]
        grads = torch.rand((P, 1), device="cuda") * 2e-4
        model.densify_and_split(grads, 1e-4, extent)
        # random prune of ~5%
        P = model.get_xyz.shape[0]
        prune_mask = torch.rand(P, device="cuda") < 0.05
        model.prune_points(prune_mask)
        _assert_id_invariants(model)
        if model.get_xyz.shape[0] > 5000:
            break

    for pid in frozen:
        row = reg.resolve_target_by_persistent_id(model, pid)  # must not raise
        assert torch.equal(model._xyz[row].detach(), snapshots[pid]), f"frozen pid {pid} drifted"
        reg.end_target_episode(model, pid)

    for pid in tracked:
        try:
            row = reg.resolve_target_by_persistent_id(model, pid)
        except UnknownTargetError:
            continue  # legitimately pruned/split away
        assert torch.equal(model._xyz[row].detach(), snapshots[pid]), f"pid {pid} row content drifted"


def test_freeze_blocks_prune_and_split():
    model = make_toy_model(n=50)
    pid = 10
    reg.begin_target_episode(model, pid)
    assert reg.is_target_frozen(model, pid)

    # prune everything: the frozen target must survive
    model.prune_points(torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda"))
    assert model.get_xyz.shape[0] == 1
    assert reg.resolve_target_by_persistent_id(model, pid) == 0

    # split with a threshold that selects everyone: target must not be split away
    model.percent_dense = 0.0  # everything qualifies as "large" -> split candidate
    grads = torch.ones((1, 1), device="cuda")
    model.densify_and_split(grads, 0.0, 1.0)
    assert reg.resolve_target_by_persistent_id(model, pid) is not None

    reg.end_target_episode(model, pid)
    assert not reg.is_target_frozen(model, pid)
    # unfrozen: prune-all now removes it
    model.prune_points(torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda"))
    with pytest.raises(UnknownTargetError):
        reg.resolve_target_by_persistent_id(model, pid)


def test_episode_context_manager_unfreezes_on_error():
    model = make_toy_model(n=20)
    with pytest.raises(RuntimeError):
        with reg.target_episode(model, 5) as handle:
            assert handle.persistent_id == 5
            assert reg.is_target_frozen(model, 5)
            raise RuntimeError("boom")
    assert not reg.is_target_frozen(model, 5)


def test_capture_restore_round_trip():
    model = make_toy_model(n=30)
    model.prune_points((torch.arange(30, device="cuda") % 3 == 0))  # drop 10
    state = model.capture()
    ids_before = _ids(model).clone()

    model2 = make_toy_model(n=30)
    model2.restore(state, make_opt_args())
    assert torch.equal(_ids(model2), ids_before)
    assert model2._next_persistent_id == model._next_persistent_id
    _assert_id_invariants(model2)


def test_ply_sidecar_round_trip(tmp_path):
    model = make_toy_model(n=25)
    model.prune_points((torch.arange(25, device="cuda") < 5))  # ids 5..24 remain
    path = str(tmp_path / "toy.ply")
    model.save_ply(path)
    assert (tmp_path / "toy.ply.pid.pt").exists()

    model2 = make_toy_model(n=1, setup_optimizer=False)
    model2.max_sh_degree = 0
    model2.load_ply(path)
    assert torch.equal(_ids(model2), _ids(model))
    assert model2._next_persistent_id == model._next_persistent_id


def test_baseline_render_unaffected(toy_model):
    """The ID buffer must be inert for rendering paths."""
    from argparse import ArgumentParser
    from arguments import PipelineParams
    from gaussian_renderer import render
    from tests.test_candidates_util import simple_cam  # shared helper (Gate C)

    parser = ArgumentParser()
    pp = PipelineParams(parser)
    pipe = pp.extract(parser.parse_args([]))
    bg = torch.zeros(3, device="cuda")
    cam = simple_cam()
    out = render(cam, toy_model, pipe, bg)
    assert out["render"].shape[0] == 3 and torch.isfinite(out["render"]).all()
