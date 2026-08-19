import math

import pytest
import torch

from temporal import pick_gaussian_along_ray


def _identity_rotations(count: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)


def test_picker_returns_frontmost_intersected_gaussian():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    scaling = torch.full((2, 3), 0.1)

    pick = pick_gaussian_along_ray(
        xyz,
        scaling,
        _identity_rotations(2),
        torch.tensor([True, True]),
        ray_origin=[0.0, 0.0, 0.0],
        ray_direction=[0.0, 0.0, 1.0],
    )

    assert pick is not None
    assert pick.index == 0
    assert pick.entry_depth == pytest.approx(1.7)
    assert pick.perpendicular_distance == pytest.approx(0.0)


def test_picker_respects_candidate_mask_and_rejects_a_miss():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.5, 0.0, 3.0]])
    scaling = torch.full((2, 3), 0.1)
    rotations = _identity_rotations(2)

    masked_pick = pick_gaussian_along_ray(
        xyz,
        scaling,
        rotations,
        torch.tensor([False, True]),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    assert masked_pick is None

    second_pick = pick_gaussian_along_ray(
        xyz,
        scaling,
        rotations,
        torch.tensor([False, True]),
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 3.0],
    )
    assert second_pick is not None
    assert second_pick.index == 1


def test_picker_uses_rotated_anisotropic_ellipsoid():
    angle = math.pi / 2.0
    rotation_z_90 = torch.tensor(
        [[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]]
    )
    xyz = torch.tensor([[0.20, 0.0, 2.0]])
    scaling = torch.tensor([[0.05, 0.10, 0.05]])

    pick = pick_gaussian_along_ray(
        xyz,
        scaling,
        rotation_z_90,
        torch.tensor([True]),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    )

    assert pick is not None
    assert pick.index == 0


def test_picker_rejects_numerically_flat_ellipsoid_behind_ray():
    xyz = torch.tensor([[0.714851, 3.56779, 2.50841]])
    scaling = torch.tensor([[0.297884, 0.138014, 1.42504e-7]])
    rotation = torch.tensor([[0.106431, 0.852685, 0.511467, -0.00164194]])
    origin = torch.tensor([0.0, 0.0, 0.0])
    direction = -xyz[0] / torch.linalg.vector_norm(xyz[0])

    pick = pick_gaussian_along_ray(
        xyz,
        scaling,
        rotation,
        torch.tensor([True]),
        origin,
        direction,
    )

    assert pick is None
