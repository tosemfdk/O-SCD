import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.run_online_bayesian_lifespan_thaw import (
    OUTPUT_FILES,
    RunConfig,
    build_causal_records,
    enforce_bocd_memory_limit,
    parse_thaw_parameters,
    parse_visualization_frame_indices,
    run_detector_only_synthetic_smoke,
    run_detector_sequence,
    main,
    choose_visualization_indices,
    SyntheticTemporalModel,
    write_visualization_artifacts,
)
from types import SimpleNamespace


def _base():
    return SimpleNamespace(
        _xyz=torch.zeros(1, 3),
        _features_dc=torch.zeros(1, 1, 3),
        _features_rest=torch.zeros(1, 0, 3),
        _opacity=torch.zeros(1, 1),
        _scaling=torch.zeros(1, 3),
        _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )


def test_parse_thaw_parameters_requires_ordered_known_unique_names():
    assert parse_thaw_parameters("dc,xyz,opacity") == ("dc", "xyz", "opacity")
    for value in ["xyz,dc", "dc,dc", "dc,birth"]:
        with pytest.raises(Exception):
            parse_thaw_parameters(value)


def test_parse_visualization_frame_indices_deduplicates_and_rejects_negative():
    assert parse_visualization_frame_indices("2,0,2") == (2, 0)
    assert parse_visualization_frame_indices("") is None
    with pytest.raises(Exception):
        parse_visualization_frame_indices("-1")


def test_causal_record_construction_does_not_require_gt_or_boundaries(tmp_path: Path):
    image_dir = tmp_path / "inference_scene" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_002.png").touch()
    (image_dir / "frame_001.png").touch()

    records, names = build_causal_records(tmp_path)

    assert names == ["frame_001.png", "frame_002.png"]
    assert [record.global_index for record in records] == [0, 1]
    assert all(record.mask_path == "" for record in records)


def test_exact_bocd_memory_guard_requires_explicit_capacity():
    config = RunConfig(
        bocd_mode="exact", exact_bocd_memory_limit_gb=1e-9
    )
    with pytest.raises(MemoryError, match="explicitly use --bocd-mode map_reset"):
        enforce_bocd_memory_limit(config, gaussian_count=1000, dtype=torch.float32)


def test_detector_smoke_opens_keeps_closes_and_reopens_without_false_active_split():
    result = run_detector_only_synthetic_smoke()
    actions = [event.action for event in result["events"]]

    assert actions == ["OPEN", "CLOSE", "OPEN"]
    assert result["frame_metrics"][2]["keep_count"] == 1
    assert result["frame_metrics"][2]["active_lifespan_count"] == 1
    assert result["final_num_states"] == [2]
    assert result["base_drift"]["bitwise_equal"] is True


def test_no_observation_preserves_lifecycle_and_reports_uncertain():
    model = SyntheticTemporalModel(_base(), n=1, max_states=2)
    cfg = RunConfig(
        evidence_count_mode="raw",
        min_evidence_mass=0.5,
        bocd_mode="map_reset",
        hazard=0.2,
        expected_run_length=None,
        open_probability=0.5,
        close_probability=0.5,
        min_run_evidence=1.0,
    )
    out = run_detector_sequence(
        [(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))],
        model,
        cfg,
    )

    assert out["frame_metrics"][0]["observed_gaussian_count"] == 0
    assert out["frame_metrics"][0]["active_lifespan_count"] == 0
    assert out["events"] == []
    assert model.num_states.tolist() == [0]


def test_detector_only_smoke_writes_only_machine_readable_outputs(tmp_path: Path):
    rc = main(["--detector-only-smoke", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUTPUT_FILES)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["detector_only_smoke"] is True
    assert summary["active_to_active_false_split_count"] == 0
    with (tmp_path / "frame_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and "observed_gaussian_count" in rows[0]
    events = [json.loads(line) for line in (tmp_path / "lifecycle_events.jsonl").read_text().splitlines()]
    assert [event["action"] for event in events] == ["OPEN", "CLOSE", "OPEN"]
    assert {
        "gaussian_index",
        "decision_timestamp",
        "bocd_estimated_changepoint_timestamp",
        "old_binary_label",
        "new_binary_label",
        "action",
        "old_slot",
        "new_current_slot",
        "posterior_probability",
        "changepoint_probability",
        "concentration",
        "visible_observation_count",
    } <= set(events[0])
    stats = np.load(tmp_path / "per_frame_bayesian_stats.npz")
    assert {"change_probability", "changepoint_probability", "observed"} <= set(stats.files)
    assert stats["observed"].shape == (5,)
    assert stats["change_probability"].shape == (5, 5)


def test_visualization_selection_uses_anchors_and_metric_extremes():
    rows = [
        {"timestamp": i, "iou": value}
        for i, value in enumerate([0.5, 0.1, 0.7, 0.2, 0.9])
    ]

    assert choose_visualization_indices(rows, explicit=[4, 2], sample_count=3) == [4, 2]
    selected = choose_visualization_indices(rows, explicit=None, sample_count=9)

    assert 0 in selected
    assert 4 in selected
    assert 1 in selected  # worst IoU


def test_optional_visualizations_write_separate_pngs_and_summary(tmp_path: Path):
    source = tmp_path / "scene_change_tiny"
    image_dir = source / "inference_scene" / "images"
    mask_dir = source / "gt_mask"
    cue_dir = tmp_path / "cue_cache" / "cues"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    cue_dir.mkdir(parents=True)
    records = []
    frame_rows = []
    predictions = []
    for i in range(2):
        name = f"frame_{i:06d}.png"
        image = np.full((8, 8, 3), 40 + i * 60, dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.imwrite(str(image_dir / name), image)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        cv2.imwrite(str(mask_dir / name), mask)
        torch.save(torch.full((1, 8, 8), 0.25 + i * 0.5), cue_dir / f"frame_{i:06d}.pt")
        records.append(SimpleNamespace(name=name, image_path=str(image_dir / name)))
        pred = np.zeros((8, 8), dtype=bool)
        pred[2:6, 2:6] = i == 1
        predictions.append(pred)
        frame_rows.append(
            {
                "timestamp": i,
                "iou": float(i),
                "f1": float(i),
                "precision": float(i),
                "recall": float(i),
                "active_lifespan_count": 10 + i,
                "open_count": i,
                "keep_count": 2 * i,
                "predicted_positive_fraction": float(pred.mean()),
            }
        )

    summary = write_visualization_artifacts(
        source_path=source,
        cue_cache_root=tmp_path / "cue_cache",
        records=records,
        predictions=predictions,
        score_maps=[pred.astype(np.uint8) * 255 for pred in predictions],
        frame_rows=frame_rows,
        output_dir=tmp_path / "visuals",
        panel_width=64,
        sample_count=2,
        frame_indices=(0, 1),
    )

    assert Path(summary["timeline_metrics_png"]).is_file()
    assert len(summary["panel_paths"]) == 2
    assert all(Path(path).is_file() for path in summary["panel_paths"])
    assert (tmp_path / "visuals" / "visual_summary.json").is_file()
