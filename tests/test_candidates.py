# Gate C: candidate camera generation.
import math

import numpy as np
import pytest

from target_nbv.candidates import (
    fibonacci_sphere, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    look_at_wxyz,
)
from target_nbv.config import TargetNBVConfig


# ---------------- CPU ----------------

def test_fibonacci_sphere_unit_and_deterministic():
    d1, d2 = fibonacci_sphere(64), fibonacci_sphere(64)
    assert np.allclose(d1, d2)
    assert np.allclose(np.linalg.norm(d1, axis=1), 1.0, atol=1e-12)
    assert np.abs(d1.mean(axis=0)).max() < 0.05  # roughly isotropic


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.standard_normal(4); q /= np.linalg.norm(q)
        R = quaternion_to_rotation_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        q2 = rotation_matrix_to_quaternion(R)
        # q and -q encode the same rotation
        assert np.allclose(q, q2, atol=1e-8) or np.allclose(q, -q2, atol=1e-8)


def test_look_at_points_camera_forward_at_target():
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos, tgt = rng.standard_normal(3) * 3, rng.standard_normal(3)
        if np.linalg.norm(tgt - pos) < 1e-3:
            continue
        R = quaternion_to_rotation_matrix(look_at_wxyz(pos, tgt))
        forward_world = R @ np.array([0.0, 0.0, -1.0])  # OpenGL forward is -z
        expected = (tgt - pos) / np.linalg.norm(tgt - pos)
        assert np.allclose(forward_world, expected, atol=1e-8)


def test_look_at_pole_singularity():
    # looking straight along the default up axis must not blow up
    q = look_at_wxyz(np.array([0.0, 5.0, 0.0]), np.zeros(3))
    R = quaternion_to_rotation_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)


# ---------------- GPU ----------------

@pytest.mark.gpu
def test_candidates_center_target_in_ndc():
    from tests.conftest import make_scene
    from target_nbv.candidates import generate_candidates, project_point

    model = make_scene(points=[[0.3, -0.2, 0.5]], scales=[[0.05, 0.05, 0.05]])
    cfg = TargetNBVConfig(candidates=__import__("target_nbv.config", fromlist=["CandidateConfig"]).CandidateConfig(count=32))
    cfg.validate()
    cands = generate_candidates(model, 0, cfg, width=128, height=128, fovy=math.radians(60))
    assert len(cands) > 0

    mu = model._xyz[0].detach().cpu().numpy()
    for cam in cands[:20]:
        ndc, depth = project_point(cam.minicam, mu)
        assert abs(ndc[0]) < 1e-3 and abs(ndc[1]) < 1e-3, "target not centered"
        assert depth == pytest.approx(cam.meta["distance"], rel=1e-4)
        # generated position must equal the MiniCam's own camera center
        cc = cam.minicam.camera_center.detach().cpu().numpy()
        assert np.allclose(cc, cam.position, atol=1e-4)


@pytest.mark.gpu
def test_projected_radius_decreases_with_distance():
    from tests.conftest import make_scene
    from target_nbv.candidates import generate_candidates

    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    cfg = TargetNBVConfig()
    cands = generate_candidates(model, 0, cfg, width=128, height=128, fovy=math.radians(60))
    by_shell = {}
    for c in cands:
        by_shell.setdefault(c.shell_index, c.meta["projected_radius_px"])
    shells = sorted(by_shell)
    radii = [by_shell[s] for s in shells]
    assert radii == sorted(radii, reverse=True), "radius must shrink with distance"


@pytest.mark.gpu
def test_generation_deterministic():
    from tests.conftest import make_scene
    from target_nbv.candidates import generate_candidates

    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    cfg = TargetNBVConfig()
    a = generate_candidates(model, 0, cfg, 128, 128, math.radians(60))
    b = generate_candidates(model, 0, cfg, 128, 128, math.radians(60))
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert ca.cand_id == cb.cand_id
        assert np.allclose(ca.position, cb.position)
