import math

import pytest
import torch

from temporal.seed_projected_occupancy import (
    projected_covariance_ellipse_occupancy,
    scale_depth_intrinsics_to_image,
)


def _K(f=40.0, cx=32.0, cy=32.0):
    return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def _quat_z(degrees: float):
    radians = math.radians(degrees)
    return torch.tensor([[math.cos(radians / 2.0), 0.0, 0.0, math.sin(radians / 2.0)]])


def test_depth_jitter_far_3d_same_2d_center_is_occupied_without_depth_filter():
    K = _K()
    w2c = torch.eye(4)
    existing = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[0.25, 0.25, 0.25]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=K,
        w2c=w2c,
        height=64,
        width=64,
        sigma=2.0,
    )
    assert bool(existing.mask[32, 32]) is True
    # A candidate at z=8 projects to the same pixel center; no strict depth or
    # occlusion agreement is applied by this occupancy helper.
    candidate = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 8.0]]),
        torch.tensor([[0.25, 0.25, 0.25]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=K,
        w2c=w2c,
        height=64,
        width=64,
        sigma=2.0,
    )
    assert bool(candidate.mask[32, 32]) is True


def test_pixels_outside_actual_ellipse_remain_free_including_bbox_corners():
    result = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[0.10, 0.10, 0.10]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=_K(),
        w2c=torch.eye(4),
        height=64,
        width=64,
        sigma=2.0,
    )
    assert bool(result.mask[32, 32]) is True
    assert bool(result.mask[27, 27]) is False
    assert bool(result.mask[37, 37]) is False


def test_tiny_spd_covariance_does_not_inflate_to_bounding_box():
    # Projected covariance is 1e-4 I at z=1 when f=1 and scale=0.01.  The old
    # determinant floor inflated inv_cov downward enough to mark bbox-neighbor
    # pixels.  Exact SPD inverse should keep only the center pixel at 2σ=0.02px.
    result = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.01, 0.01, 0.01]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=torch.tensor([[1.0, 0.0, 8.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]]),
        w2c=torch.eye(4),
        height=17,
        width=17,
        sigma=2.0,
    )
    assert result.pixels == 1
    assert bool(result.mask[8, 8]) is True
    assert bool(result.mask[8, 9]) is False
    assert bool(result.mask[9, 8]) is False


def test_rows_used_counts_all_intersecting_rows_not_only_new_union_pixels():
    result = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        torch.tensor([[0.10, 0.10, 0.10], [0.10, 0.10, 0.10]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        K=_K(),
        w2c=torch.eye(4),
        height=64,
        width=64,
        sigma=2.0,
        row_indices=torch.tensor([10, 11]),
    )
    assert result.rows_used == 2
    assert result.rows_projected.tolist() == [10, 11]
    assert result.pixels > 0


def test_anisotropy_and_rotation_change_projected_ellipse_support():
    xyz = torch.tensor([[0.0, 0.0, 2.0]])
    scaling = torch.tensor([[0.40, 0.05, 0.05]])
    identity = projected_covariance_ellipse_occupancy(
        xyz,
        scaling,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=_K(),
        w2c=torch.eye(4),
        height=64,
        width=64,
        sigma=2.0,
    )
    rotated = projected_covariance_ellipse_occupancy(
        xyz,
        scaling,
        _quat_z(90.0),
        K=_K(),
        w2c=torch.eye(4),
        height=64,
        width=64,
        sigma=2.0,
    )
    assert bool(identity.mask[32, 45]) is True
    assert bool(identity.mask[45, 32]) is False
    assert bool(rotated.mask[32, 45]) is False
    assert bool(rotated.mask[45, 32]) is True


def test_translated_rotated_camera_offscreen_and_behind_cases():
    K = _K(f=20.0, cx=20.0, cy=20.0)
    w2c = torch.eye(4)
    w2c[:3, 3] = torch.tensor([1.0, 0.0, 0.0])
    shifted = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[0.10, 0.10, 0.10]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=K,
        w2c=w2c,
        height=48,
        width=48,
        sigma=2.0,
    )
    assert bool(shifted.mask[20, 30]) is True

    theta = math.radians(180.0)
    w2c_rot = torch.eye(4)
    w2c_rot[:3, :3] = torch.tensor(
        [[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]]
    )
    behind = projected_covariance_ellipse_occupancy(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[0.20, 0.20, 0.20]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=K,
        w2c=w2c_rot,
        height=48,
        width=48,
        sigma=2.0,
    )
    assert behind.pixels == 0
    offscreen = projected_covariance_ellipse_occupancy(
        torch.tensor([[100.0, 0.0, 2.0]]),
        torch.tensor([[0.01, 0.01, 0.01]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        K=K,
        w2c=torch.eye(4),
        height=48,
        width=48,
        sigma=2.0,
    )
    assert offscreen.pixels == 0


def test_scale_depth_intrinsics_matches_panel10_row_scaling():
    K = torch.tensor([[10.0, 0.0, 3.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]])
    scaled = scale_depth_intrinsics_to_image(K, native_height=10, native_width=20, height=40, width=100)
    assert scaled[0].tolist() == pytest.approx([50.0, 0.0, 15.0])
    assert scaled[1].tolist() == pytest.approx([0.0, 80.0, 16.0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_inputs_match_cpu_inputs():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0]])
    scaling = torch.tensor([[0.12, 0.08, 0.05], [0.10, 0.10, 0.10]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    kwargs = dict(K=_K(), w2c=torch.eye(4), height=64, width=64, sigma=2.0)
    cpu = projected_covariance_ellipse_occupancy(xyz, scaling, rotation, **kwargs)
    cuda = projected_covariance_ellipse_occupancy(
        xyz.cuda(), scaling.cuda(), rotation.cuda(), K=kwargs["K"].cuda(),
        w2c=kwargs["w2c"].cuda(), height=64, width=64, sigma=2.0,
    )
    assert torch.equal(cpu.mask, cuda.mask)
    assert cpu.rows_used == cuda.rows_used
    assert cpu.pixels == cuda.pixels
