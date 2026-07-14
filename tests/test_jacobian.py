# Gate D: FD Jacobian backend (GPU).
import pytest
import torch

from tests.conftest import make_scene, make_pipe
from tests.test_candidates_util import simple_cam
from target_nbv.config import TargetNBVConfig
from target_nbv.jacobian import compute_target_jacobian
from target_nbv.types import TargetParameterSpec

pytestmark = pytest.mark.gpu

SPEC = TargetParameterSpec()
CFG = TargetNBVConfig().validate()


def _jac(model, cam, row=0, cfg=CFG):
    return compute_target_jacobian(model, cam, row, SPEC, make_pipe(),
                                   torch.zeros(3, device="cuda"), cfg)


def test_shape_and_validity():
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    res = _jac(model, simple_cam(distance=2.0))
    assert res.valid
    n, d = res.J.shape
    assert d == 6 and n % 3 == 0 and n > 0
    assert torch.isfinite(res.J).all()


def test_ray_direction_insensitive_face_on():
    """Camera on -z looking +z: moving the mean along z (the viewing ray)
    changes the image far less than moving it in-plane (x/y)."""
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    res = _jac(model, simple_cam(distance=2.0))
    norms = res.J.norm(dim=0)  # [mu_x, mu_y, mu_z, ls_x, ls_y, ls_z]
    assert norms[0] > 5 * norms[2], f"x vs z (ray): {norms.tolist()}"
    assert norms[1] > 5 * norms[2], f"y vs z (ray): {norms.tolist()}"


def test_model_restored_bitwise():
    model = make_scene(points=[[0.1, -0.2, 0.3], [0.5, 0.5, 0.5]],
                       scales=[[0.05, 0.05, 0.05], [0.04, 0.04, 0.04]])
    snap = {n: getattr(model, n).detach().clone()
            for n in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc")}
    res = _jac(model, simple_cam(distance=2.0), row=0)
    assert res.valid
    for name, before in snap.items():
        assert torch.equal(getattr(model, name).detach(), before), f"{name} mutated"


def test_non_target_unaffected_by_perturbation():
    from target_nbv.adapter import perturbed, get_theta
    model = make_scene(points=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                       scales=[[0.05, 0.05, 0.05], [0.04, 0.04, 0.04]])
    other_before = model._xyz[1].detach().clone()
    theta = get_theta(model, 0, SPEC)
    with perturbed(model, 0, SPEC, theta + 0.01):
        assert torch.equal(model._xyz[1].detach(), other_before)


def test_invisible_target_invalid():
    model = make_scene(points=[[0.0, 0.0, -5.0]], scales=[[0.05, 0.05, 0.05]])
    res = _jac(model, simple_cam(distance=2.0))  # target behind camera
    assert not res.valid and res.invalid_reason == "behind_camera"


def test_eps_halving_stability():
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    cam = simple_cam(distance=2.0)
    cfg2 = TargetNBVConfig()
    cfg2.jacobian.mean_epsilon_rel = CFG.jacobian.mean_epsilon_rel / 2
    cfg2.jacobian.log_scale_epsilon = CFG.jacobian.log_scale_epsilon / 2
    J1 = _jac(model, cam).J
    J2 = _jac(model, cam, cfg=cfg2).J
    # crops are identical by construction (frozen from unperturbed view)
    n1, n2 = J1.norm(dim=0), J2.norm(dim=0)
    rel = ((n1 - n2).abs() / n1.clamp_min(1e-12)).max()
    assert rel < 0.10, f"FD not converged: rel change {rel:.3f}"
