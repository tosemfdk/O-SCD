import math

import torch

from temporal.new_seed_densification import (
    NewSeedDensificationConfig,
    densify_confirmed_new_region,
    project_points,
    unproject_at_camera_depth,
)
from temporal.new_seed_observation import SignedXFeatObservation


def _observation(frame: int, camera_x: float, mask: torch.Tensor) -> SignedXFeatObservation:
    # Camera center C=(camera_x,0,0), hence t=-C for identity rotation.
    w2c = torch.eye(4)
    w2c[0, 3] = -camera_x
    return SignedXFeatObservation(
        frame_index=frame,
        timestamp=float(frame),
        frame_name=f"frame_{frame:04d}.png",
        w2c=w2c,
        K=torch.tensor([[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]),
        image_size=(64, 64),
        keypoints=torch.empty((0, 2)),
        descriptors=torch.empty((0, 8)),
        valid=torch.empty((0,), dtype=torch.bool),
        plus_mask64=mask,
        minus_mask64=torch.zeros_like(mask),
        cue_strength64=mask.float(),
        pca_margin64=mask.float(),
    )


def test_projection_and_unprojection_round_trip_at_camera_depth():
    mask = torch.ones((64, 64), dtype=torch.bool)
    observation = _observation(0, 0.2, mask)
    xyz = torch.tensor([[0.1, -0.2, 2.0], [0.4, 0.3, 3.0]])

    uv, depth = project_points(xyz, observation)
    restored = unproject_at_camera_depth(uv, depth, observation)

    torch.testing.assert_close(restored, xyz, atol=1e-5, rtol=0.0)


def test_densification_adds_only_undercovered_cells_with_three_view_support():
    mask = torch.zeros((64, 64), dtype=torch.bool)
    mask[24:41, 24:41] = True
    history = tuple(
        _observation(frame, camera_x, mask)
        for frame, camera_x in ((0, -0.2), (1, 0.0), (2, 0.2))
    )
    current = history[-1]
    anchors = torch.tensor([[0.0, 0.0, 2.0]])
    config = NewSeedDensificationConfig(
        min_support_views=3,
        min_support_ratio=1.0,
        min_baseline=0.1,
        min_view_angle_deg=1.0,
        erosion_cells=1,
        coverage_radius_cells=1.0,
        min_child_separation_cells=2.5,
        max_parent_distance_cells=9.0,
        min_world_separation=0.01,
        max_new_per_frame=12,
        footprint_px=3.0,
    )

    batch = densify_confirmed_new_region(
        current=current,
        history=history,
        source_sign="+",
        active_xyz=anchors,
        active_rows=torch.tensor([7]),
        config=config,
    )

    assert 0 < batch.count <= 12
    assert batch.uncovered_cells > batch.count
    assert batch.parent_rows.tolist() == [7] * batch.count
    assert all(len(frames) == 3 for frames in batch.support_frames)
    assert all(math.isclose(ratio, 1.0) for ratio in batch.support_ratios)
    assert all(baseline >= 0.1 for baseline in batch.max_baselines)
    for observation in history:
        uv, depth = project_points(batch.xyz, observation)
        assert bool((depth > 0).all())
        cells = torch.floor(uv).long()
        assert bool(mask[cells[:, 1], cells[:, 0]].all())


def test_densification_rejects_region_without_multiview_signed_support():
    current_mask = torch.zeros((64, 64), dtype=torch.bool)
    current_mask[24:41, 24:41] = True
    empty = torch.zeros_like(current_mask)
    history = (
        _observation(0, -0.2, empty),
        _observation(1, 0.0, empty),
        _observation(2, 0.2, current_mask),
    )
    config = NewSeedDensificationConfig(
        min_support_views=3,
        min_support_ratio=0.7,
        min_baseline=0.1,
        min_view_angle_deg=1.0,
        max_new_per_frame=8,
    )

    batch = densify_confirmed_new_region(
        current=history[-1],
        history=history,
        source_sign="+",
        active_xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        active_rows=torch.tensor([0]),
        config=config,
    )

    assert batch.count == 0
    assert batch.rejection_counts.get("support_views", 0) > 0
