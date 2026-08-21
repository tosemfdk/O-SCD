import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.run_online_binary_state_lifespan_thaw import (
    ExactClosedPairArchive,
    validate_run_config,
    OUTPUT_FILES,
    RunConfig,
    SyntheticTemporalModel,
    _synthetic_base,
    main,
    make_optimizer,
    q_sequence_to_evidence,
    run_detector_only_synthetic_smoke,
    run_detector_sequence,
    same_scene_repeated_transition_diagnostics,
    validate_cue_camera_checksum,
)


def _cfg(**kwargs):
    defaults = dict(
        evidence_count_mode="raw",
        min_evidence_mass=0.1,
        inactive_to_active_prior=0.2,
        active_to_inactive_prior=0.2,
        initial_active_probability=0.1,
        max_states=6,
        detector_only=True,
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def _actions(result):
    return [event.action for event in result["events"]]


def test_requested_repeated_flip_sequence_opens_closes_and_reopens_new_slot():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=6)
    result = run_detector_sequence(
        q_sequence_to_evidence([
            0.05, 0.08, 0.10,
            0.90, 0.92, 0.88,
            0.07, 0.05, 0.10,
            0.91, 0.94, 0.90,
        ]),
        model,
        _cfg(),
    )

    assert _actions(result) == ["OPEN", "CLOSE", "OPEN"]
    open_events = [event for event in result["events"] if event.action == "OPEN"]
    assert [event.new_current_slot for event in open_events] == [0, 1]
    assert model.num_states.tolist() == [2]
    assert model.current_state_index.tolist() == [1]
    assert model.state_end[0, 0].item() == pytest.approx(6.0)
    assert torch.isinf(model.state_end[0, 1])

    same_scene = same_scene_repeated_transition_diagnostics(
        result["events"],
        boundaries=(6,),
    )
    assert same_scene == {
        "same_scene_repeated_transition_event_count": 1,
        "same_scene_repeated_gaussian_segment_count": 1,
        "same_scene_repeated_gaussian_count": 1,
    }


def test_stable_active_opens_once_then_keeps_without_false_split():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    result = run_detector_sequence(
        q_sequence_to_evidence([0.9, 0.85, 0.95, 0.88]),
        model,
        _cfg(),
    )

    assert _actions(result) == ["OPEN"]
    assert [row["keep_count"] for row in result["frame_metrics"]] == [0, 1, 1, 1]
    assert model.num_states.tolist() == [1]
    assert model.current_state_index.tolist() == [0]


def test_stable_inactive_never_opens():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    result = run_detector_sequence(
        q_sequence_to_evidence([0.1, 0.05, 0.12]),
        model,
        _cfg(),
    )

    assert _actions(result) == []
    assert [row["none_count"] for row in result["frame_metrics"]] == [1, 1, 1]
    assert model.num_states.tolist() == [0]


def test_ambiguous_sequence_does_not_flip():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    result = run_detector_sequence(
        q_sequence_to_evidence([0.48, 0.52, 0.49], strength=1.0),
        model,
        _cfg(inactive_to_active_prior=0.01, active_to_inactive_prior=0.01, initial_active_probability=0.5),
    )

    assert _actions(result) == []
    assert model.num_states.tolist() == [0]
    assert all(row["open_count"] == 0 and row["close_count"] == 0 for row in result["frame_metrics"])


def test_unobserved_frame_preserves_filter_and_lifecycle_bitwise():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    result = run_detector_sequence(
        q_sequence_to_evidence([0.9, None, 0.9]),
        model,
        _cfg(),
    )

    assert _actions(result) == ["OPEN"]
    assert result["frame_metrics"][1]["observed_gaussian_count"] == 0
    assert result["frame_metrics"][1]["open_count"] == 0
    assert result["frame_metrics"][1]["keep_count"] == 0
    assert result["frame_metrics"][1]["active_lifespan_count"] == 1
    # Fallback and real filters expose p_active in state_dict; unobserved frame must not change it.
    stats = result["binary_stats"]["p_active"]
    assert stats[1, 0] == 0
    assert np.isnan(stats[1, 1:]).all()  # no observed rows are included in per-frame quantiles


def test_detector_only_smoke_writes_binary_state_outputs(tmp_path: Path):
    rc = main(["--detector-only-smoke", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUTPUT_FILES)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["detector_only_smoke"] is True
    assert summary["reused_slot_violations"] == 0
    events = [json.loads(line) for line in (tmp_path / "lifecycle_events.jsonl").read_text().splitlines()]
    assert [event["action"] for event in events] == ["OPEN", "CLOSE", "OPEN"]
    assert {"p_active", "p_01", "p_10", "p_flip", "new_current_slot"} <= set(events[0])
    with (tmp_path / "frame_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and "p_active_stats" in rows[0]
    stats = np.load(tmp_path / "per_frame_binary_state_stats.npz")
    assert {"p_active", "p_flip", "p_01", "p_10", "open_count", "reopen_count"} <= set(stats.files)
    assert stats["p_active"].shape[1] == 5


def test_detector_only_smoke_can_omit_large_checkpoint(tmp_path: Path):
    (tmp_path / "checkpoint.pt").write_bytes(b"stale checkpoint")
    rc = main([
        "--detector-only-smoke",
        "--skip-checkpoint",
        "--output-dir",
        str(tmp_path),
    ])

    assert rc == 0
    assert not (tmp_path / "checkpoint.pt").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["checkpoint_saved"] is False
    assert "checkpoint.pt" not in summary["output_files"]


def test_cue_camera_checksum_is_enforced_when_metadata_provides_it():
    validate_cue_camera_checksum({}, "actual")
    validate_cue_camera_checksum({"fixed_cameras_sha256": "actual"}, "actual")
    with pytest.raises(ValueError, match="camera checksum mismatch"):
        validate_cue_camera_checksum(
            {"fixed_cameras_sha256": "cached"},
            "actual",
        )


def test_smoke_helper_reports_expected_final_state():
    result = run_detector_only_synthetic_smoke()

    assert _actions(result) == ["OPEN", "CLOSE", "OPEN"]
    assert result["final_num_states"] == [2]
    assert result["final_current_state_index"] == [1]
    assert result["base_drift"]["bitwise_equal"] is True


def test_production_capped_strength_one_requested_sequence_flips_with_default_detector_priors():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=6)
    cfg = RunConfig(
        evidence_count_mode="capped",
        evidence_mass_saturation=1.0,
        min_evidence_mass=1e-6,
        inactive_to_active_prior=0.01,
        active_to_inactive_prior=0.01,
        initial_active_probability=0.5,
        open_probability=0.6,
        close_probability=0.4,
        max_states=6,
        detector_only=True,
    )

    result = run_detector_sequence(
        q_sequence_to_evidence([
            0.05, 0.08, 0.10,
            0.90, 0.92, 0.88,
            0.07, 0.05, 0.10,
            0.91, 0.94, 0.90,
        ], strength=1.0),
        model,
        cfg,
    )

    assert _actions(result) == ["OPEN", "CLOSE", "OPEN"]
    assert model.num_states.tolist() == [2]
    assert model.state_status[0, :2].tolist() == [2, 1]
    assert result["events"][0].new_current_slot == 0
    assert result["events"][-1].new_current_slot == 1


def test_run_config_rejects_non_discriminative_emission_and_zero_transition_priors():
    for kwargs in (
        {"state_emission_reliability": 0.5},
        {"state_emission_reliability": 1.0},
        {"inactive_to_active_prior": 0.0},
        {"active_to_inactive_prior": 0.0},
        {"open_probability": 0.4, "close_probability": 0.4},
    ):
        with pytest.raises(ValueError):
            validate_run_config(RunConfig(**kwargs))


def test_evaluation_after_inference_records_pre_and_post_open_delta(tmp_path: Path):
    import cv2
    from experiments.run_online_binary_state_lifespan_thaw import evaluate_after_inference

    source = tmp_path / "source"
    (source / "gt_mask").mkdir(parents=True)
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255
    cv2.imwrite(str(source / "gt_mask" / "frame_000.png"), mask)
    rows = [{"open_count": 1}]
    record = type("Record", (), {"name": "frame_000.png"})()

    metrics = evaluate_after_inference(
        source,
        [record],
        [np.zeros((2, 2), dtype=bool)],
        [mask >= 128],
        rows,
    )

    assert metrics["evaluated"] is True
    assert metrics["pre_opt"]["mean_frame_iou"] == 0.0
    assert metrics["post_opt"]["mean_frame_iou"] == 1.0
    assert metrics["open_frame_pre_post"]["count"] == 1
    assert metrics["open_frame_pre_post"]["mean_delta_iou"] == 1.0
    assert rows[0]["pre_iou"] == 0.0
    assert rows[0]["iou"] == 1.0
    assert rows[0]["open_frame_iou_delta"] == 1.0


def test_exact_closed_pair_archive_detects_any_future_parameter_drift():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=2)
    model.open_rows([0], timestamp=0)
    with torch.no_grad():
        model.state_change_dc[0, 0].fill_(2.0)
    slot = model.close_rows([0], timestamp=1)
    archive = ExactClosedPairArchive()
    archive.add(model, None, torch.tensor([0]), slot)

    assert archive.verify(model, None)["passed"] is True
    with torch.no_grad():
        model.state_change_dc[0, 0, 0, 0] += 1.0
    failed = archive.verify(model, None)
    assert failed["passed"] is False
    assert failed["max_abs"] == 1.0
    assert failed["exhaustive"] is True


def test_training_optimizer_factory_uses_real_masked_row_slot_adam():
    from torch import nn
    from torch.nn import functional as F
    from temporal.geometry_change_model import TemporalGeometryChangeModel
    from temporal.masked_optimizer import MaskedRowSlotAdam

    rotation = torch.zeros(1, 4)
    rotation[:, 0] = 1.0
    base = SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(1, 3)),
        _features_dc=nn.Parameter(torch.zeros(1, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(1, 0, 3)),
        _opacity=nn.Parameter(torch.zeros(1, 1)),
        _scaling=nn.Parameter(torch.zeros(1, 3)),
        _rotation=nn.Parameter(rotation),
        opacity_activation=torch.sigmoid,
        scaling_activation=torch.exp,
        rotation_activation=F.normalize,
    )
    model = TemporalGeometryChangeModel.from_gaussians(base, max_states=2)
    args = SimpleNamespace(
        dc_lr=0.0025,
        xyz_lr=0.00016,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        adam_eps=1e-15,
    )
    optimizer = make_optimizer(model, RunConfig(thaw_parameters=("dc",)), args)
    assert isinstance(optimizer, MaskedRowSlotAdam)


def test_view_consistent_lifecycle_controller_config_is_wired_into_runner():
    from experiments.run_online_binary_state_lifespan_thaw import make_controller
    from temporal.view_consistent_binary_lifespan_controller import ViewConsistentBinaryLifespanController

    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    cfg = RunConfig(
        lifecycle_controller="view_consistent",
        transition_confirmation_views=3,
        min_transition_bayes_factor=2.0,
        min_transition_evidence_strength=0.25,
        inactive_to_active_prior=0.03,
        active_to_inactive_prior=0.04,
        detector_only=True,
    )

    controller = make_controller(model, cfg)

    assert isinstance(controller, ViewConsistentBinaryLifespanController)
    assert controller.config.confirmation_views == 3
    assert controller.config.min_transition_bayes_factor == pytest.approx(2.0)
    assert controller.config.min_evidence_strength == pytest.approx(0.25)
    assert controller.config.inactive_to_active_prior == pytest.approx(0.03)
    assert controller.config.active_to_inactive_prior == pytest.approx(0.04)


def test_view_consistent_detector_sequence_waits_for_confirmed_transition():
    model = SyntheticTemporalModel(_synthetic_base(), n=1, max_states=4)
    cfg = _cfg(
        evidence_count_mode="capped",
        min_evidence_mass=1e-6,
        inactive_to_active_prior=0.01,
        active_to_inactive_prior=0.01,
        initial_active_probability=0.5,
        lifecycle_controller="view_consistent",
        transition_confirmation_views=2,
        min_transition_bayes_factor=3.0,
    )

    result = run_detector_sequence(
        q_sequence_to_evidence([0.9, 0.2, 0.9, 0.9], strength=1.0),
        model,
        cfg,
    )

    assert _actions(result) == ["OPEN"]
    assert result["events"][0].decision_timestamp == 3
    assert model.current_state_index.tolist() == [0]
