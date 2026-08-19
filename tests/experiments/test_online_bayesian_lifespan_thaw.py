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
    run_detector_only_synthetic_smoke,
    run_detector_sequence,
    main,
    SyntheticTemporalModel,
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
