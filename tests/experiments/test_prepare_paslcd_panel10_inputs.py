from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from experiments.prepare_paslcd_fixed_pose_cues import SceneSpec
from experiments.prepare_paslcd_panel10_inputs import (
    DEFAULT_METRIC_MODEL,
    DEFAULT_PROCESS_RES,
    _complete_cache_hit,
    _expected_contract,
    _train_and_write_boundaries,
    prepare_scene,
    reconstruct_panel10_raw_cue,
    stable_mask_from_candidate_sum,
)
from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL
from temporal.change_cue_fusion import (
    fuse_power_product,
    normalized_oscd_pixel_cue_from_terms,
    oscd_pixel_terms,
    semantic_from_cached_sum,
)


def test_reconstruct_panel10_raw_cue_uses_powered_pixel_times_cached_sam() -> None:
    reference = torch.zeros((3, 4, 4), dtype=torch.float32)
    online = torch.linspace(0.0, 1.0, 48, dtype=torch.float32).reshape(3, 4, 4)
    terms = oscd_pixel_terms(reference, online)
    original = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=1.0)
    semantic_true = torch.full_like(original, 0.7)
    cached_sum = original + semantic_true

    cue = reconstruct_panel10_raw_cue(
        reference_rgb=reference,
        online_rgb=online,
        candidate_map=cached_sum,
    )

    powered = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=0.3)
    semantic = semantic_from_cached_sum(cached_sum, original)
    expected = fuse_power_product(powered, semantic, exponent=1.0) / 2.0
    torch.testing.assert_close(cue, expected)
    assert float(cue.min()) >= 0.0
    assert float(cue.max()) <= 1.0


def test_stable_mask_is_area_downsampled_raw_candidate_sum() -> None:
    candidate = torch.zeros((1, 128, 128), dtype=torch.float32)
    candidate[:, :2, :2] = 1.0  # one 2x2 block maps to a single 64x64 cell average of 1

    stable = stable_mask_from_candidate_sum(candidate, threshold=0.2)

    assert stable.shape == (64, 64)
    assert stable[0, 0].item() is False
    assert stable[0, 1].item() is True
    assert int((~stable).sum().item()) == 1


def test_train_and_write_boundaries_is_prequential_and_writes_schema(tmp_path: Path) -> None:
    arrays = {
        "input_counts": np.ones((3, 64), dtype=np.int64),
        "loss_counts": np.ones((3, 512), dtype=np.int64),
        "teacher_sums": np.zeros((3, 512), dtype=np.float32),
        "frame_names": np.asarray(["f000.png", "f001.png", "f002.png"]),
        "segment_names": np.asarray(["scene"] * 3),
    }
    arrays["teacher_sums"][:, 300:] = 0.75

    audit, history = _train_and_write_boundaries(
        output_dir=tmp_path,
        arrays=arrays,
        updates_per_arrival=1,
        seed=0,
    )

    artifact = json.loads((tmp_path / "learned_boundaries_causal.json").read_text())
    assert artifact["training_mode"] == "causal_prequential"
    assert artifact["edge_probability"] == 0.05
    assert artifact["frames"]["f000.png"]["tau"] == pytest.approx(0.25, abs=1e-7)
    assert audit["causal_audit"]["future_teacher_accesses"] == 0
    assert audit["causal_audit"]["maximum_teacher_index_before_prediction"] == [-1, 0, 1]
    assert len(history) == 3


def _write_tiny_scene(root: Path) -> tuple[SceneSpec, Path, Path]:
    source = root / "data" / "Instance_1" / "Cantina"
    image_dir = source / "inference_scene" / "images"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "frame_000.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    base_ply = source / BASE_PLY_REL
    base_ply.parent.mkdir(parents=True)
    base_ply.write_text("ply\n", encoding="utf-8")
    cameras_json = root / "cameras.json"
    cameras_json.write_text(json.dumps({"frame_000": {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0}}), encoding="utf-8")
    cue_cache = root / "cue_cache"
    cue_cache.mkdir()
    teacher_dir = root / "teacher"
    teacher_dir.mkdir()
    cv2.imwrite(str(teacher_dir / "frame_000.png"), np.zeros((2, 2), dtype=np.uint8))
    spec = SceneSpec("Instance_1", "Cantina", source, cameras_json, cue_cache)
    return spec, teacher_dir, root / "prepared"


def test_prepare_scene_cache_hit_requires_complete_schema_and_avoids_cuda(tmp_path: Path) -> None:
    spec, teacher_dir, output_dir = _write_tiny_scene(tmp_path)
    expected = _expected_contract(
        spec=spec,
        output_dir=output_dir,
        teacher_dir=teacher_dir,
        resolution=4.0,
        input_bins=64,
        loss_bins=512,
        frames=1,
        process_res=DEFAULT_PROCESS_RES,
        metric_model=DEFAULT_METRIC_MODEL,
    )
    output_dir.mkdir()
    (output_dir / "input_preparation_metadata.json").write_text(json.dumps(expected), encoding="utf-8")
    for name in [
        "boundary_statistics.npz",
        "causal_pca_posterior_arrays.npz",
    ]:
        np.savez_compressed(output_dir / name, frame_names=np.asarray(["frame_000.png"]))
    for name in [
        "boundary_statistics.json",
        "learned_boundaries_causal.json",
        "training_history.json",
    ]:
        (output_dir / name).write_text("{}", encoding="utf-8")
    torch.save({"configuration": expected["da3_metric"]}, output_dir / "da3_seed_replay.pt")
    (output_dir / "summary.json").write_text(json.dumps({"frames": 1, "cache_hit": False}), encoding="utf-8")
    cache_dir = output_dir / "da3metric_cache"
    cache_dir.mkdir()
    np.savez_compressed(cache_dir / "frame_000000.npz", depth_meters=np.zeros((1, 1), dtype=np.float32), K=np.eye(3, dtype=np.float32))

    assert _complete_cache_hit(output_dir, expected, ["frame_000.png"]) is True
    summary = prepare_scene(spec, output_dir, teacher_dir)

    assert summary["cache_hit"] is True
    assert summary["frames"] == 1
