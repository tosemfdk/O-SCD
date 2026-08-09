from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from experiments.train_real_temporal_rchange import (
    PoseResult,
    boundaries_from_manifest,
    build_frame_records,
    classify_slot_gradients,
    completed_state_drift_max,
    exact_dataset_contract_audit,
    make_state_optimizer,
    make_training_schedule,
    oracle_training_indices,
    require_valid_training_poses,
    segment_id,
    segment_ranges,
    training_schedule_audit,
    validate_boundaries,
    validate_exact_dataset_contract,
)


@pytest.mark.parametrize(
    ("gradients", "active_state", "isolated", "has_active_signal"),
    [
        ([0.0, 2.0, 0.0], 1, True, True),
        ([0.0, 0.0, 0.0], 1, True, False),
        ([0.1, 2.0, 0.0], 1, False, True),
    ],
)
def test_gradient_classification_separates_isolation_from_zero_signal(
    gradients, active_state, isolated, has_active_signal
):
    result = classify_slot_gradients(gradients, active_state)

    assert result[0] is isolated
    assert result[1] is has_active_signal


def test_manifest_counts_define_half_open_state_boundaries():
    manifest = {
        "sequence_order": ["scene_change1", "scene_change2", "scene_change3"],
        "counts": {
            "per_source_scene": {
                "scene_change1": 95,
                "scene_change2": 104,
                "scene_change3": 105,
            }
        },
    }

    boundaries = boundaries_from_manifest(manifest, fallback=(1, 2))

    assert boundaries == (95, 199)
    assert segment_ranges(304, boundaries) == [(0, 95), (95, 199), (199, 304)]
    assert [segment_id(t, boundaries) for t in (94, 95, 198, 199)] == [0, 1, 1, 2]


def test_oracle_training_selection_uses_nonempty_interior_frames(tmp_path: Path):
    names = [f"frame_{index:03d}.png" for index in range(10)]
    nonempty = {1, 2, 3, 4, 5, 6, 7, 8}
    for index, name in enumerate(names):
        mask = np.zeros((4, 4), dtype=np.uint8)
        if index in nonempty:
            mask[1:3, 1:3] = 255
        assert cv2.imwrite(str(tmp_path / name), mask)

    selected = oracle_training_indices(tmp_path, names, start=0, end=10, count=3)

    assert len(selected) == 3
    assert set(selected) <= nonempty
    assert min(selected) > min(nonempty)
    assert max(selected) < max(nonempty)


def test_oracle_training_selection_falls_back_when_state_mask_is_empty(tmp_path: Path):
    names = [f"frame_{index:03d}.png" for index in range(5)]
    for name in names:
        assert cv2.imwrite(str(tmp_path / name), np.zeros((2, 2), dtype=np.uint8))

    assert oracle_training_indices(tmp_path, names, 0, 5, count=3) == [0, 2, 4]


@pytest.mark.parametrize("boundaries", [(), (0, 5), (5, 5), (6, 5), (5, 10)])
def test_boundary_validation_rejects_invalid_manual_segments(boundaries):
    with pytest.raises(ValueError):
        validate_boundaries(total_frames=10, boundaries=boundaries)


def _write_fake_real_source(root: Path, count: int, nonempty: set[int]) -> None:
    image_dir = root / "inference_scene" / "images"
    mask_dir = root / "gt_mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(count):
        name = f"frame_{index:03d}.png"
        assert cv2.imwrite(str(image_dir / name), np.zeros((2, 2, 3), dtype=np.uint8))
        mask = np.zeros((2, 2), dtype=np.uint8)
        if index in nonempty:
            mask[0, 0] = 255
        assert cv2.imwrite(str(mask_dir / name), mask)


def test_all_training_frame_mode_keeps_empty_gt_frames(tmp_path: Path):
    _write_fake_real_source(tmp_path, count=6, nonempty={1, 4})

    legacy_records, _, _ = build_frame_records(
        tmp_path, frames_per_state=1, probes=[], boundaries=(3,), all_training_frames=False
    )
    exact_records, _, names = build_frame_records(
        tmp_path, frames_per_state=1, probes=[], boundaries=(3,), all_training_frames=True
    )

    assert names == [f"frame_{index:03d}.png" for index in range(6)]
    assert [record.global_index for record in legacy_records] == [1, 4]
    assert [record.global_index for record in exact_records] == list(range(6))
    assert [record.segment_id for record in exact_records] == [0, 0, 0, 1, 1, 1]


def test_exact_schedule_is_epoch_major_and_counts_every_frame_k_times():
    views = [
        SimpleNamespace(image_name="s0_a", segment_id=0),
        SimpleNamespace(image_name="s0_b", segment_id=0),
        SimpleNamespace(image_name="s1_a", segment_id=1),
    ]
    views_by_state = {0: views[:2], 1: views[2:]}
    args = SimpleNamespace(updates_per_frame=3, steps_state=99)

    schedule = make_training_schedule(views_by_state, max_states=2, args=args)
    dataset_audit = {
        "expected_dataset_frames": 3,
        "actual_dataset_frames": 3,
        "actual_train_records": 3,
        "actual_train_views": 3,
        "expected_exact_total_updates": 9,
        "dataset_contract_errors": [],
        "dataset_contract_passed": True,
    }
    audit = training_schedule_audit(views, schedule, args, dataset_audit)

    assert [(step.state, step.view_index, step.epoch) for step in schedule] == [
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
        (0, 1, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ]
    assert audit["mode"] == "updates_per_frame_exact"
    assert audit["expected_total_updates"] == 9
    assert audit["actual_total_updates"] == 9
    assert audit["min_updates_per_frame"] == 3
    assert audit["max_updates_per_frame"] == 3
    assert audit["per_frame_update_counts"] == {"s0_a": 3, "s0_b": 3, "s1_a": 3}
    assert audit["exact_all_images_guarantee"] is True


def test_exact_schedule_matches_instance1_audit_total():
    views = [
        SimpleNamespace(image_name=f"frame_{index:03d}", segment_id=0 if index < 95 else 1 if index < 199 else 2)
        for index in range(304)
    ]
    views_by_state = {0: views[:95], 1: views[95:199], 2: views[199:]}
    args = SimpleNamespace(updates_per_frame=120, steps_state=30)

    schedule = make_training_schedule(views_by_state, max_states=3, args=args)
    dataset_audit = {
        "expected_dataset_frames": 304,
        "actual_dataset_frames": 304,
        "actual_train_records": 304,
        "actual_train_views": 304,
        "expected_exact_total_updates": 36_480,
        "dataset_contract_errors": [],
        "dataset_contract_passed": True,
    }
    audit = training_schedule_audit(views, schedule, args, dataset_audit)

    assert audit["expected_total_updates"] == 36_480
    assert audit["actual_total_updates"] == 36_480
    assert audit["min_updates_per_frame"] == 120
    assert audit["max_updates_per_frame"] == 120
    assert len(audit["per_frame_update_counts"]) == 304


def test_exact_pose_requirement_reports_missing_or_invalid_records():
    records = [
        SimpleNamespace(name="ok.png", global_index=0),
        SimpleNamespace(name="bad.png", global_index=1),
    ]
    poses = {
        "ok.png": PoseResult(True, "ok.png", "ref.png", [[1.0]], 10, 10, 0.1),
        "bad.png": PoseResult(False, "bad.png", "ref.png", None, 3, 0, None, "too_few_matches"),
    }

    with pytest.raises(RuntimeError, match="valid pose for every training record") as exc:
        require_valid_training_poses(records, poses)

    assert "bad.png" in str(exc.value)
    assert "too_few_matches" in str(exc.value)


def test_exact_dataset_contract_rejects_manifest_mismatch_and_blocks_guarantee():
    manifest = {
        "sequence_order": ["scene_a", "scene_b"],
        "counts": {"per_source_scene": {"scene_a": 2, "scene_b": 2}},
    }
    all_names = ["frame_000.png", "frame_001.png", "frame_002.png"]
    train_records = [SimpleNamespace(name=name, global_index=index) for index, name in enumerate(all_names)]
    train_views = [SimpleNamespace(image_name=Path(name).stem, segment_id=0) for name in all_names]
    args = SimpleNamespace(updates_per_frame=5, steps_state=30)
    schedule = make_training_schedule({0: train_views}, max_states=1, args=args)

    dataset_audit = exact_dataset_contract_audit(manifest, all_names, train_records, train_views, args.updates_per_frame)
    schedule_audit = training_schedule_audit(train_views, schedule, args, dataset_audit)

    assert dataset_audit["dataset_contract_passed"] is False
    assert dataset_audit["expected_dataset_frames"] == 4
    assert dataset_audit["actual_dataset_frames"] == 3
    assert schedule_audit["expected_exact_total_updates"] == 20
    assert schedule_audit["actual_total_updates"] == 15
    assert schedule_audit["exact_all_images_guarantee"] is False
    with pytest.raises(RuntimeError, match="dataset contract failed"):
        validate_exact_dataset_contract(manifest, all_names, train_records, train_views, args.updates_per_frame)


def test_reset_optimizer_prevents_completed_slot_adam_momentum_drift():
    args = SimpleNamespace(lr=0.1)
    slots = torch.nn.Parameter(torch.zeros(1, 2, 1, 1))

    optimizer = make_state_optimizer(slots, args)
    optimizer.zero_grad(set_to_none=True)
    slots[:, 0].sum().backward()
    optimizer.step()

    slot0_after_seed = slots[:, :1].detach().clone()
    optimizer = make_state_optimizer(slots, args)
    optimizer.zero_grad(set_to_none=True)
    slots[:, 1].sum().backward()
    before = slots[:, :1].detach().clone()
    optimizer.step()

    assert torch.equal(before, slot0_after_seed)
    assert completed_state_drift_max(before, slots, state=1) == 0.0
    assert torch.equal(slots[:, :1], slot0_after_seed)
