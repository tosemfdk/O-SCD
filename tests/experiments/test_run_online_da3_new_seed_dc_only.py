import json
from types import SimpleNamespace

import pytest
import torch

from experiments.run_online_da3_new_seed_dc_only import (
    LearnedCueBoundary,
    _accept_new_voxels,
    _apply_seed_detector_update,
    _audit_seed_geometry,
    _calibrate_stage2_cue,
    _load_learned_cue_artifact,
    _occupied_voxels,
    _save_temporal_confusion_frame,
    _seed_geometry_snapshot,
)
from experiments.run_online_xfeat_new_seed import (
    train_seed_dc_from_projected_coverage,
)
from temporal.lifespan_gate_beta import (
    LifespanGateBetaConfig,
    LifespanGateBetaFilter,
)
from temporal.new_seed_gaussians import NewSeedGaussianModel


def test_voxel_filter_keeps_first_causal_point_and_respects_capacity():
    occupied: set[tuple[int, int, int]] = set()
    xyz = torch.tensor(
        [
            [0.001, 0.0, 0.0],
            [0.019, 0.0, 0.0],
            [0.021, 0.0, 0.0],
            [0.041, 0.0, 0.0],
        ]
    )

    selected = _accept_new_voxels(xyz, occupied, voxel_size=0.02, maximum=2)

    assert selected.tolist() == [0, 2]
    assert occupied == {(0, 0, 0), (1, 0, 0)}


def test_load_learned_cue_artifact_validates_and_indexes_frames(tmp_path):
    path = tmp_path / "boundaries.json"
    path.write_text(
        json.dumps(
            {
                "remap": "sigmoid",
                "edge_probability": 0.05,
                "cue_scale": 2.0,
                "cue_formula": "q = pixel * SAM",
                "frames": {"frame.png": {"tau": 0.3, "width": 0.1}},
            }
        )
    )

    artifact = _load_learned_cue_artifact(path)

    assert artifact.edge_probability == pytest.approx(0.05)
    assert artifact.cue_scale == pytest.approx(2.0)
    assert artifact.boundaries["frame.png"] == LearnedCueBoundary(0.3, 0.1)


def test_calibrated_stage2_cue_is_soft_and_centered_at_tau():
    height = width = 16
    reference = torch.zeros((3, height, width))
    online = torch.zeros_like(reference)
    cached_sum = torch.zeros((1, height, width))
    cached_sum[:, :, width // 2 :] = 2.0
    online[:, :, width // 2 :] = 1.0

    normalized, calibrated = _calibrate_stage2_cue(
        cached_sum=cached_sum,
        reference_rgb=reference,
        online_rgb=online,
        boundary=LearnedCueBoundary(tau=0.3, width=0.1),
        l1_exponent=0.3,
        edge_probability=0.05,
    )

    assert normalized.shape == cached_sum.shape
    assert calibrated.shape == cached_sum.shape
    assert float(calibrated.min()) > 0.0
    assert float(calibrated.max()) <= 1.0
    assert bool(((calibrated > 0.0) & (calibrated < 1.0)).any())
    assert float(calibrated[:, :, width // 2 :].mean()) > float(
        calibrated[:, :, : width // 2].mean()
    )


def test_rchange_occupancy_rejects_base_surface_and_accepts_free_space():
    occupied = _occupied_voxels(
        torch.tensor([[0.001, 0.0, 1.001], [0.041, 0.0, 1.001]]),
        voxel_size=0.02,
    )
    candidates = torch.tensor(
        [[0.019, 0.0, 1.019], [0.021, 0.0, 0.951], [0.059, 0.0, 1.019]]
    )

    selected = _accept_new_voxels(
        candidates, occupied, voxel_size=0.02, maximum=10
    )

    assert selected.tolist() == [1]


def test_bf30_detector_opens_dormant_seed_after_causal_positive_evidence():
    model = NewSeedGaussianModel(device="cpu")
    model.append(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        start=float("inf"),
        end=float("inf"),
    )
    tracker = LifespanGateBetaFilter(
        1, LifespanGateBetaConfig(bayes_factor_threshold=30.0)
    )

    for timestamp in (1, 2):
        result = _apply_seed_detector_update(
            tracker=tracker,
            seeds=model,
            delta_a=torch.ones(1),
            delta_b=torch.zeros(1),
            total_mass=torch.ones(1),
            timestamp=timestamp,
        )
        assert result["opened"] == 0
        assert result["never_open"] == 1

    result = _apply_seed_detector_update(
        tracker=tracker,
        seeds=model,
        delta_a=torch.ones(1),
        delta_b=torch.zeros(1),
        total_mass=torch.ones(1),
        timestamp=3,
    )

    assert result["opened"] == 1
    assert result["active"] == 1
    assert result["never_open"] == 0
    assert model.active_mask(3.0).tolist() == [True]


def test_seed_dc_optimization_leaves_all_geometry_bitwise_fixed():
    model = NewSeedGaussianModel(sh_degree=3, device="cpu")
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(0.01), lr=0.01)
    model.append(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        start=1.0,
        scaling=torch.full((2, 3), -4.0),
        opacity=0.1,
        optimizer=optimizer,
    )
    geometry = _seed_geometry_snapshot(model)

    optimizer.zero_grad(set_to_none=True)
    model.seed_dc.sum().backward()
    optimizer.step()

    audit = _audit_seed_geometry(model, geometry)
    assert audit["all_fields_bitwise_equal"]
    assert not torch.equal(model.seed_dc.detach(), torch.zeros_like(model.seed_dc))


def test_seed_geometry_audit_detects_xyz_drift():
    model = NewSeedGaussianModel(sh_degree=3, device="cpu")
    model.append(xyz=torch.tensor([[0.0, 0.0, 2.0]]), start=1.0)
    geometry = _seed_geometry_snapshot(model)

    model._xyz[0, 0] += 0.1

    audit = _audit_seed_geometry(model, geometry)
    assert not audit["all_fields_bitwise_equal"]
    assert not audit["fields"]["xyz"]["bitwise_equal"]
    assert audit["fields"]["xyz"]["max_abs_difference"] > 0.0


def test_temporal_confusion_frame_matches_saved_metrics(tmp_path):
    gt = torch.tensor([[False, True], [True, False]])
    score = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
    row = {
        "frame_local": 3,
        "frame_global": 201,
        "frame_name": "scene_change3_frame_000003.png",
        "seed_count": 17,
        "seed_active_count": 11,
        "seed_never_open_count": 4,
        "seed_closed_count": 2,
        "da3_full_tp": 1,
        "da3_full_tn": 1,
        "da3_full_fp": 1,
        "da3_full_fn": 1,
        "da3_full_pred_positive": 2,
        "da3_full_gt_positive": 2,
        "da3_full_pixels": 4,
    }

    panel = _save_temporal_confusion_frame(
        tmp_path,
        rgb=torch.zeros((3, 2, 2)),
        gt=gt.numpy(),
        score=score.numpy(),
        row=row,
        panel_width=32,
    )

    assert panel.exists()
    assert len(list((tmp_path / "raw_confusion").glob("*.png"))) == 1
    assert len(list((tmp_path / "pred_binary").glob("*.png"))) == 1


def test_seed_replay_uses_current_lifecycle_in_historical_camera():
    model = NewSeedGaussianModel(device="cpu")
    model.append(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        start=5.0,
        scaling=torch.full((1, 3), -4.0),
        opacity=0.1,
    )
    optimizer = torch.optim.Adam(model.optimizer_parameter_groups(0.01), lr=0.01)
    historical_view = SimpleNamespace(
        timestamp=1.0,
        image_width=8,
        image_height=8,
        FoVx=1.0,
        FoVy=1.0,
        world_view_transform=torch.eye(4),
    )
    target = torch.ones((1, 8, 8))

    historical_lifecycle = train_seed_dc_from_projected_coverage(
        view=historical_view,
        seeds=model,
        optimizer=optimizer,
        target=target,
        updates=1,
        loss_weight=1.0,
    )
    current_lifecycle = train_seed_dc_from_projected_coverage(
        view=historical_view,
        seeds=model,
        optimizer=optimizer,
        target=target,
        updates=1,
        loss_weight=1.0,
        active_timestamp=5.0,
    )

    assert historical_lifecycle["active_seed_rows"] == 0
    assert current_lifecycle["active_seed_rows"] == 1
    assert current_lifecycle["last"] is not None
