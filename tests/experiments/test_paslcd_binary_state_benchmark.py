from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.prepare_paslcd_fixed_pose_cues import (
    CUE_VALUE_STORAGE,
    SceneSpec,
    select_cameras_json,
    sha256_file,
)
from experiments.run_paslcd_binary_state_benchmark import (
    ConditionSpec,
    aggregate_rows,
    command_for_scene,
    detector_consistency,
    lifecycle_event_structure_sha256,
    load_baseline_rows,
    parse_update_budgets,
    sanitize_extra_args,
    scene_outputs_complete,
    summarize_scene,
    validate_scene_summary,
)


def test_load_baseline_and_scene_summary(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "metrics_per_scene.csv"
    baseline_csv.write_text(
        "instance,scene,miou_online,f1_online,miou_refined,f1_refined\n"
        "Instance_1,Garden,0.4,0.5,0.6,0.7\n",
        encoding="utf-8",
    )
    baseline = load_baseline_rows(baseline_csv)
    spec = SceneSpec(
        instance="Instance_1",
        scene="Garden",
        source_path=tmp_path / "data",
        cameras_json=tmp_path / "cameras.json",
        output_dir=tmp_path / "cues",
    )
    row = summarize_scene(
        {
            "frames": 2,
            "metrics": {
                "mean_frame_iou": 0.45,
                "mean_frame_f1": 0.55,
                "precision": 0.8,
                "recall": 0.6,
            "aggregate_iou": 0.5,
            "aggregate_f1": 0.66,
            "pre_opt": {"mean_frame_iou": 0.4, "mean_frame_f1": 0.5},
            "open_frame_pre_post": {
                "count": 2,
                "mean_delta_iou": 0.05,
                "mean_delta_f1": 0.05,
            },
            },
            "open_count": 3,
            "close_count": 1,
            "reopen_count": 0,
            "base_max_drift": 0.0,
            "closed_slot_max_drift": 0.0,
        },
        spec,
        ConditionSpec("B0_binary_dc", "binary", "dc"),
        baseline[("Instance_1", "Garden")],
    )

    assert row["miou"] == 0.45
    assert row["pre_opt_miou"] == 0.4
    assert row["open_frame_count"] == 2
    assert row["delta_vs_oscd_online_miou"] == 0.04999999999999999
    assert row["delta_vs_oscd_refined_f1"] == -0.1499999999999999


def test_aggregate_rows_keeps_oscd_comparison() -> None:
    rows = [
        {
            "condition": "B0_binary_dc",
            "miou": 0.4,
            "f1": 0.5,
            "precision": 0.6,
            "recall": 0.7,
            "pre_opt_miou": 0.3,
            "pre_opt_f1": 0.4,
            "open_frame_count": 2,
            "open_frame_miou_delta": 0.2,
            "open_frame_f1_delta": 0.2,
            "oscd_online_miou": 0.3,
            "oscd_online_f1": 0.4,
            "oscd_refined_miou": 0.5,
            "oscd_refined_f1": 0.6,
            "delta_vs_oscd_online_miou": 0.1,
            "delta_vs_oscd_online_f1": 0.1,
            "delta_vs_oscd_refined_miou": -0.1,
            "delta_vs_oscd_refined_f1": -0.1,
            "open": 2,
            "close": 1,
            "reopen": 0,
            "keep": 4,
            "uncertain": 3,
            "same_scene_repeated_transition_events": 2,
            "same_scene_repeated_gaussians": 1,
            "active_to_active_false_splits": 0,
            "reused_slot_violations": 0,
            "final_active_gs": 10,
            "used_fallback_camera": False,
            "gt_used_for_training": False,
            "manual_boundaries_used_for_inference": False,
            "checkpoint_saved": False,
            "runtime_seconds": 4.0,
            "cuda_peak_memory_bytes": 100,
            "base_max_drift": 0.0,
            "closed_slot_max_drift": 0.0,
            "inactive_gradient_violations": 0,
        },
        {
            "condition": "B0_binary_dc",
            "miou": 0.6,
            "f1": 0.7,
            "precision": 0.8,
            "recall": 0.9,
            "pre_opt_miou": 0.5,
            "pre_opt_f1": 0.6,
            "open_frame_count": 3,
            "open_frame_miou_delta": 0.1,
            "open_frame_f1_delta": 0.1,
            "oscd_online_miou": 0.4,
            "oscd_online_f1": 0.5,
            "oscd_refined_miou": 0.7,
            "oscd_refined_f1": 0.8,
            "delta_vs_oscd_online_miou": 0.2,
            "delta_vs_oscd_online_f1": 0.2,
            "delta_vs_oscd_refined_miou": -0.1,
            "delta_vs_oscd_refined_f1": -0.1,
            "open": 4,
            "close": 2,
            "reopen": 1,
            "keep": 6,
            "uncertain": 5,
            "same_scene_repeated_transition_events": 4,
            "same_scene_repeated_gaussians": 2,
            "active_to_active_false_splits": 0,
            "reused_slot_violations": 0,
            "final_active_gs": 20,
            "used_fallback_camera": True,
            "gt_used_for_training": False,
            "manual_boundaries_used_for_inference": False,
            "checkpoint_saved": False,
            "runtime_seconds": 6.0,
            "cuda_peak_memory_bytes": 200,
            "base_max_drift": 0.0,
            "closed_slot_max_drift": 0.0,
            "inactive_gradient_violations": 0,
        },
    ]

    agg = aggregate_rows(rows)["B0_binary_dc"]
    assert agg["scene_count"] == 2
    assert agg["miou"] == 0.5
    assert agg["open"] == 6
    assert agg["same_scene_repeated_transition_events"] == 6
    assert agg["scene_wins_vs_oscd_online_miou"] == 2
    assert agg["fallback_camera_scene_count"] == 1
    assert agg["open_frame_miou_delta"] == pytest.approx(0.14)
    assert agg["peak_cuda_memory_bytes"] == 200
    assert agg["delta_vs_oscd_online_miou"] == 0.15000000000000002


def test_command_owns_causal_paths_and_boundary_diagnostics(tmp_path: Path) -> None:
    spec = SceneSpec(
        instance="Instance_1",
        scene="Garden",
        source_path=tmp_path / "data",
        cameras_json=tmp_path / "cameras.json",
        output_dir=tmp_path / "cues",
    )
    cmd, output_dir = command_for_scene(
        spec,
        ConditionSpec("B1_binary_all_geometry", "binary", "dc,xyz,opacity,scaling,rotation"),
        output_root=tmp_path / "out",
        resolution=4.0,
        updates_per_frame=120,
        max_frames=3,
        skip_checkpoint=True,
        extra_args=["--state-emission-reliability", "0.9"],
    )
    assert output_dir == tmp_path / "out" / "u120" / "B1_binary_all_geometry" / "Instance_1" / "Garden"
    assert "--disable-boundary-diagnostics" in cmd
    assert "--skip-checkpoint" in cmd
    assert cmd[cmd.index("--cue-cache-root") + 1] == str(tmp_path / "cues")
    assert cmd[cmd.index("--thaw-parameters") + 1] == "dc,xyz,opacity,scaling,rotation"
    assert cmd[-2:] == ["--state-emission-reliability", "0.9"]


def test_resume_summary_must_match_owned_benchmark_contract(tmp_path: Path) -> None:
    source = tmp_path / "data"
    cameras = tmp_path / "cameras.json"
    cues = tmp_path / "cues"
    cameras.write_text("[]", encoding="utf-8")
    spec = SceneSpec("Instance_1", "Garden", source, cameras, cues)
    condition = ConditionSpec("B0_binary_dc", "binary", "dc")
    summary = {
        "algorithm": "direct_binary_state_filter",
        "frames": 25,
        "run_config": {
            "bayes_cue_mode": "binary",
            "thaw_parameters": ["dc"],
            "updates_per_frame": 120,
            "evidence_count_mode": "capped",
            "seed": 0,
        },
        "run_arguments": {
            "source_path": str(source),
            "fixed_cameras_json": str(cameras),
            "cue_cache_root": str(cues),
            "max_frames": None,
        },
        "cue_cache_metadata": {
            "cue_value_storage": CUE_VALUE_STORAGE,
            "fixed_cameras_sha256": sha256_file(cameras),
        },
        "metrics": {"evaluated": True},
        "gt_used_for_training": False,
        "manual_boundaries_used_for_inference": False,
        "checkpoint_saved": False,
    }
    summary["run_arguments"]["output_dir"] = str(tmp_path / "relocated")

    validate_scene_summary(
        summary,
        spec,
        condition,
        updates_per_frame=120,
        max_frames=None,
        checkpoint_expected=False,
    )
    summary["run_config"]["updates_per_frame"] = 16
    with pytest.raises(ValueError, match="updates_per_frame"):
        validate_scene_summary(
            summary,
            spec,
            condition,
            updates_per_frame=120,
            max_frames=None,
            checkpoint_expected=False,
        )


def test_resume_summary_rejects_changed_extra_runner_arguments(tmp_path: Path) -> None:
    source = tmp_path / "data"
    cameras = tmp_path / "cameras.json"
    cues = tmp_path / "cues"
    cameras.write_text("[]", encoding="utf-8")
    spec = SceneSpec("Instance_1", "Garden", source, cameras, cues)
    condition = ConditionSpec("B0_binary_dc", "binary", "dc")
    command, _ = command_for_scene(
        spec,
        condition,
        output_root=tmp_path / "out",
        resolution=4.0,
        updates_per_frame=120,
        max_frames=None,
        skip_checkpoint=True,
        extra_args=["--dc-lr", "0.003"],
    )
    from experiments.run_online_binary_state_lifespan_thaw import (
        parse_args as parse_direct_args,
        run_config_from_args,
        serializable_arguments,
    )
    from dataclasses import asdict

    parsed = parse_direct_args(command[3:])
    summary = {
        "algorithm": "direct_binary_state_filter",
        "frames": 25,
        "run_config": json.loads(json.dumps(asdict(run_config_from_args(parsed)))),
        "run_arguments": json.loads(json.dumps(serializable_arguments(parsed))),
        "cue_cache_metadata": {
            "cue_value_storage": CUE_VALUE_STORAGE,
            "fixed_cameras_sha256": sha256_file(cameras),
        },
        "metrics": {"evaluated": True},
        "gt_used_for_training": False,
        "manual_boundaries_used_for_inference": False,
        "checkpoint_saved": False,
    }
    summary["run_arguments"]["output_dir"] = str(tmp_path / "relocated")

    validate_scene_summary(
        summary,
        spec,
        condition,
        updates_per_frame=120,
        max_frames=None,
        checkpoint_expected=False,
        expected_command=command,
    )
    changed_command = [
        "0.004" if value == "0.003" else value for value in command
    ]
    with pytest.raises(ValueError, match="run_arguments"):
        validate_scene_summary(
            summary,
            spec,
            condition,
            updates_per_frame=120,
            max_frames=None,
            checkpoint_expected=False,
            expected_command=changed_command,
        )


def test_sanitize_extra_args_rejects_owned_flags() -> None:
    for flag, values in (
        ("--source-path", ["--source-path", "bad"]),
        ("--max-frames", ["--max-frames", "2"]),
        ("--skip-post-inference-evaluation", ["--skip-post-inference-evaluation"]),
        ("--detector-only", ["--detector-only"]),
    ):
        with pytest.raises(ValueError, match=flag):
            sanitize_extra_args(values)


def test_parse_update_budgets_labels_primary_and_budget_matched() -> None:
    assert parse_update_budgets("120,16") == (120, 16)


def test_lifecycle_event_structure_ignores_posterior_but_detects_slot_change(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    event = {
        "gaussian_index": 1,
        "decision_timestamp": 2,
        "old_binary_label": 0,
        "new_binary_label": 1,
        "action": "OPEN",
        "old_slot": -1,
        "new_current_slot": 0,
        "p_active": 0.9,
    }
    first.write_text(json.dumps(event) + "\n", encoding="utf-8")
    event["p_active"] = 0.900001
    second.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert lifecycle_event_structure_sha256(first) == lifecycle_event_structure_sha256(
        second
    )

    event["new_current_slot"] = 1
    second.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert lifecycle_event_structure_sha256(first) != lifecycle_event_structure_sha256(
        second
    )


def test_detector_consistency_compares_dc_and_geometry_event_structure() -> None:
    common = {
        "updates_per_frame": 16,
        "instance": "Instance_1",
        "scene": "Garden",
    }
    rows = [
        {
            **common,
            "condition_base": "B0_binary_dc",
            "lifecycle_event_structure_sha256": "same",
        },
        {
            **common,
            "condition_base": "B1_binary_all_geometry",
            "lifecycle_event_structure_sha256": "same",
        },
    ]
    assert detector_consistency(rows)["u16"] == {
        "comparable_scene_count": 1,
        "event_structure_mismatch_scene_count": 0,
        "passed": True,
    }


def test_detector_consistency_compares_update_budgets() -> None:
    rows = []
    for budget in (16, 120):
        for condition in ("B0_binary_dc", "B1_binary_all_geometry"):
            rows.append(
                {
                    "updates_per_frame": budget,
                    "instance": "Instance_1",
                    "scene": "Garden",
                    "condition_base": condition,
                    "lifecycle_event_structure_sha256": "same",
                }
            )
    assert detector_consistency(rows)["cross_budget"] == {
        "comparable_condition_scene_count": 2,
        "event_structure_mismatch_count": 0,
        "passed": True,
    }


def test_scene_outputs_complete_honors_checkpoint_policy(tmp_path: Path) -> None:
    for name in (
        "summary.json",
        "frame_metrics.csv",
        "lifecycle_events.jsonl",
        "per_frame_binary_state_stats.npz",
    ):
        (tmp_path / name).write_bytes(b"")
    assert scene_outputs_complete(tmp_path, checkpoint_expected=False)
    assert not scene_outputs_complete(tmp_path, checkpoint_expected=True)
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint")
    assert scene_outputs_complete(tmp_path, checkpoint_expected=True)
    assert not scene_outputs_complete(tmp_path, checkpoint_expected=False)


def test_cue_metadata_invalidates_old_clamped_cache(tmp_path: Path, monkeypatch):
    import json
    import experiments.prepare_paslcd_fixed_pose_cues as prep

    base = tmp_path / "point_cloud.ply"
    cameras = tmp_path / "cameras.json"
    base.write_text("ply", encoding="utf-8")
    cameras.write_text("[]", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "candidate_map_definition": prep.CANDIDATE_MAP_DEFINITION,
                "resolution": 4.0,
                "reference_ply_sha256": "hash",
                "fixed_cameras_sha256": "hash",
                "frame_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prep, "sha256_file", lambda path: "hash")

    assert not prep._metadata_matches(
        metadata,
        base_ply=base,
        cameras_json=cameras,
        resolution=4.0,
        frame_count=1,
    )

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["cue_value_storage"] = "raw_generate_candidate_map_float32_not_clamped"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    assert prep._metadata_matches(
        metadata,
        base_ply=base,
        cameras_json=cameras,
        resolution=4.0,
        frame_count=1,
    )


def test_cue_preparation_saves_raw_unclamped_generate_candidate_map_output():
    source = Path("experiments/prepare_paslcd_fixed_pose_cues.py").read_text(encoding="utf-8")
    assert "raw_generate_candidate_map_float32_not_clamped" in source
    assert ").detach().float().cpu().clamp(0, 1)" not in source


def test_camera_selection_uses_compatible_fallback_or_fails_explicitly(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.prepare_paslcd_fixed_pose_cues as prep

    primary = tmp_path / "primary" / "Instance_1" / "Lounge" / "cameras.json"
    fallback_root = tmp_path / "fallback"
    fallback = (
        fallback_root
        / "Instance_1"
        / "Lounge"
        / "seed_0"
        / "reference"
        / "cameras.json"
    )
    primary.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    primary.write_text("[]", encoding="utf-8")
    fallback.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        prep,
        "_camera_json_matches_scene",
        lambda _source, path, *, resolution: path == fallback,
    )

    assert select_cameras_json(
        tmp_path / "scene",
        primary,
        fallback_root=fallback_root,
        instance="Instance_1",
        scene="Lounge",
        resolution=4.0,
    ) == fallback

    monkeypatch.setattr(
        prep,
        "_camera_json_matches_scene",
        lambda _source, _path, *, resolution: False,
    )
    with pytest.raises(ValueError, match="camera/image mismatch"):
        select_cameras_json(
            tmp_path / "scene",
            primary,
            fallback_root=fallback_root,
            instance="Instance_1",
            scene="Lounge",
            resolution=4.0,
        )


def test_valid_cue_cache_hit_needs_neither_cuda_nor_sam(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.prepare_paslcd_fixed_pose_cues as prep

    source = tmp_path / "scene"
    image_dir = source / "inference_scene" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame.png").write_bytes(b"cached-image-placeholder")
    base = source / prep.BASE_PLY_REL
    base.parent.mkdir(parents=True)
    base.write_text("ply", encoding="utf-8")
    cameras = tmp_path / "cameras.json"
    cameras.write_text(json.dumps([{"img_name": "frame"}]), encoding="utf-8")
    cache = tmp_path / "cache"
    (cache / "cues").mkdir(parents=True)
    (cache / "cues" / "frame.pt").write_bytes(b"cached-cue-placeholder")
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "candidate_map_definition": prep.CANDIDATE_MAP_DEFINITION,
                "cue_value_storage": prep.CUE_VALUE_STORAGE,
                "resolution": 4.0,
                "reference_ply_sha256": prep.sha256_file(base),
                "fixed_cameras_sha256": prep.sha256_file(cameras),
                "frame_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prep.torch.cuda, "is_available", lambda: False)

    result = prep.prepare_all(
        [SceneSpec("Instance_1", "Garden", source, cameras, cache)],
        resolution=4.0,
    )

    assert result == [
        {
            "instance": "Instance_1",
            "scene": "Garden",
            "frame_count": 1,
            "cache_hit": True,
            "output_dir": str(cache),
        }
    ]
