import math

import pytest
import torch

from poses.new_seed_triangulation import (
    GeometryGateConfig,
    camera_center_from_w2c,
    fundamental_from_w2c,
    match_and_triangulate_masked_pair,
    points_inside_mask,
    project_points,
    ray_angles_deg,
    sampson_error_px,
    triangulate_multiview_dlt,
    triangulate_pair_dlt,
    triangulate_pair_with_diagnostics,
    triangulate_track_with_diagnostics,
    validate_K,
    validate_w2c,
    world_rays,
)


def K(width=640.0, height=480.0, fx=500.0, fy=530.0):
    return torch.tensor([[fx, 0.0, (width - 1.0) / 2.0], [0.0, fy, (height - 1.0) / 2.0], [0.0, 0.0, 1.0]])


def w2c_from_center(center, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)
    C = torch.tensor(center, dtype=torch.float32)
    Rt = torch.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = -R @ C
    return Rt


def synthetic_setup():
    k = K()
    w0 = w2c_from_center([0.0, 0.0, 0.0])
    w1 = w2c_from_center([0.5, 0.0, 0.0])
    w2 = w2c_from_center([0.0, 0.4, 0.0])
    pts = torch.tensor([[0.10, 0.05, 4.0], [-0.25, -0.1, 3.5], [0.4, 0.15, 5.0]], dtype=torch.float32)
    uv0, _ = project_points(pts, k, w0)
    uv1, _ = project_points(pts, k, w1)
    uv2, _ = project_points(pts, k, w2)
    return k, w0, w1, w2, pts, uv0, uv1, uv2


def test_validate_w2c_k_and_camera_center_catches_bad_pose():
    k = K(fx=500.0, fy=550.0)
    assert validate_K(k).shape == (3, 3)
    w = w2c_from_center([1.0, 2.0, 3.0], yaw_deg=10.0)
    assert torch.allclose(camera_center_from_w2c(w), torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)

    bad = w.clone()
    bad[:3, :3] = bad[:3, :3] * 2.0
    with pytest.raises(ValueError, match="orthonormal|transposed"):
        validate_w2c(bad)

    bad_k = k.clone()
    bad_k[0, 0] = -1.0
    with pytest.raises(ValueError, match="focal"):
        validate_K(bad_k)


def test_world_rays_and_known_pose_epipolar_error_are_consistent():
    k, w0, w1, _, pts, uv0, uv1, _ = synthetic_setup()
    rays = world_rays(uv0[:1], k, w0)
    expected = pts[:1] / torch.linalg.norm(pts[:1], dim=-1, keepdim=True)
    assert torch.allclose(rays, expected, atol=1e-5)

    F = fundamental_from_w2c(k, w0, k, w1)
    err = sampson_error_px(uv0, uv1, F)
    assert torch.max(err) < 1e-4

    shifted = uv1.clone()
    shifted[:, 1] += 25.0
    bad_err = sampson_error_px(uv0, shifted, F)
    assert torch.min(bad_err) > 1.0


def test_pair_and_multiview_dlt_recover_world_points():
    k, w0, w1, w2, pts, uv0, uv1, uv2 = synthetic_setup()
    pair_pts = triangulate_pair_dlt(uv0, uv1, k, w0, k, w1)
    assert torch.max(torch.linalg.norm(pair_pts - pts, dim=-1)) < 1e-4

    Ks = torch.stack([k, k, k])
    w2cs = torch.stack([w0, w1, w2])
    one = triangulate_multiview_dlt(torch.stack([uv0[0], uv1[0], uv2[0]]), Ks, w2cs)
    assert torch.linalg.norm(one - pts[0]) < 1e-4

    diag = triangulate_track_with_diagnostics(torch.stack([uv0[0], uv1[0], uv2[0]]), Ks, w2cs)
    assert bool(diag["valid"].item())
    assert torch.linalg.norm(diag["point_world"] - pts[0]) < 1e-4
    assert float(diag["reprojection_rmse_px"]) < 1e-4


def test_pair_diagnostics_reject_named_failure_modes():
    k, w0, w1, _, pts, uv0, uv1, _ = synthetic_setup()
    config = GeometryGateConfig(min_translation=0.1, min_ray_angle_deg=1.5, max_epipolar_error_px=2.0, max_reprojection_rmse_px=3.0)

    ok = triangulate_pair_with_diagnostics(uv0[:1], uv1[:1], k, w0, k, w1, config)
    assert bool(ok.valid[0].item())
    assert ok.rejection_reasons[0] == ("ok",)

    # Negative depth: mirror the true point behind both cameras and use its exact projections.
    behind = torch.tensor([[0.0, 0.0, -4.0]])
    b0, _ = project_points(behind, k, w0)
    b1, _ = project_points(behind, k, w1)
    neg = triangulate_pair_with_diagnostics(b0, b1, k, w0, k, w1, config)
    assert not bool(neg.valid[0].item())
    assert "positive_depth" in neg.rejection_reasons[0]

    small_baseline_w = w2c_from_center([0.001, 0.0, 0.0])
    small_uv, _ = project_points(pts[:1], k, small_baseline_w)
    small = triangulate_pair_with_diagnostics(uv0[:1], small_uv, k, w0, k, small_baseline_w, config)
    assert not bool(small.valid[0].item())
    assert "min_translation" in small.rejection_reasons[0]
    assert "min_ray_angle" in small.rejection_reasons[0]

    shifted = uv1[:1].clone()
    shifted[:, 1] += 30.0
    bad_reproj = triangulate_pair_with_diagnostics(uv0[:1], shifted, k, w0, k, w1, config)
    assert not bool(bad_reproj.valid[0].item())
    assert "epipolar" in bad_reproj.rejection_reasons[0]
    assert "reprojection" in bad_reproj.rejection_reasons[0]

    mask = torch.zeros(64, 64, dtype=torch.bool)
    masked = triangulate_pair_with_diagnostics(
        uv0[:1], uv1[:1], k, w0, k, w1, config, mask1=mask, mask2=torch.ones(64, 64, dtype=torch.bool), image_size1=(640, 480), image_size2=(640, 480)
    )
    assert not bool(masked.valid[0].item())
    assert "mask1" in masked.rejection_reasons[0]


def test_ray_angle_and_mask_lookup_use_xy_order():
    k, w0, w1, _, _, uv0, uv1, _ = synthetic_setup()
    angle = ray_angles_deg(uv0[:1], k, w0, uv1[:1], k, w1)
    assert float(angle[0]) > 1.5

    mask = torch.zeros(64, 64, dtype=torch.bool)
    # x near image center maps to mask column ~32, y maps to row ~32.
    mask[31:33, 31:33] = True
    assert bool(points_inside_mask(torch.tensor([[319.5, 239.5]]), mask, image_size=(640, 480))[0].item())
    assert not bool(points_inside_mask(torch.tensor([[10.0, 239.5]]), mask, image_size=(640, 480))[0].item())


def test_masked_descriptor_matching_returns_original_indices_and_geometry_gates():
    k, w0, w1, _, pts, uv0, uv1, _ = synthetic_setup()
    # Add an invalid/out-of-mask distractor before and after the true feature.
    kpts0 = torch.cat([torch.tensor([[5.0, 5.0]]), uv0[:1], torch.tensor([[630.0, 470.0]])], dim=0)
    kpts1 = torch.cat([torch.tensor([[6.0, 6.0]]), uv1[:1], torch.tensor([[620.0, 470.0]])], dim=0)
    desc0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    desc1 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    mask0 = torch.zeros(64, 64, dtype=torch.bool)
    mask1 = torch.zeros(64, 64, dtype=torch.bool)
    # Fill a generous central window containing the projected point in both frames.
    mask0[25:40, 25:40] = True
    mask1[25:40, 20:40] = True

    out = match_and_triangulate_masked_pair(kpts0, desc0, mask0, k, w0, kpts1, desc1, mask1, k, w1, image_size1=(640, 480), image_size2=(640, 480))
    assert out.idx1.tolist() == [1]
    assert out.idx2.tolist() == [1]
    assert bool(out.valid[0].item())
    assert torch.linalg.norm(out.points_world[0] - pts[0]) < 1e-4


def test_masked_matching_does_not_silently_hide_invalid_camera_geometry():
    """Configuration/pose bugs must abort the experiment, not become zero seeds."""

    k, w0, w1, _, _, uv0, uv1, _ = synthetic_setup()
    desc = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones(64, 64, dtype=torch.bool)
    malformed_w2c = w1.clone()
    malformed_w2c[:3, :3] *= 2.0

    with pytest.raises(ValueError, match="orthonormal|transposed"):
        match_and_triangulate_masked_pair(
            uv0[:1],
            desc,
            mask,
            k,
            w0,
            uv1[:1],
            desc,
            mask,
            k,
            malformed_w2c,
            image_size1=(640, 480),
            image_size2=(640, 480),
        )
