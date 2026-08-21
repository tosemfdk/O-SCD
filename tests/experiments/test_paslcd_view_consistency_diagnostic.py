from __future__ import annotations

import json
import math
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("plyfile") is None:
    class _BinaryLifespanAction(int):
        _names = {0: "NONE", 1: "OPEN", 2: "KEEP", 3: "CLOSE", 4: "UNCERTAIN"}

        @property
        def name(self) -> str:
            return self._names[int(self)]

        def __new__(cls, value: int) -> "_BinaryLifespanAction":
            if int(value) not in cls._names:
                raise ValueError(value)
            return int.__new__(cls, int(value))

        @classmethod
        def __iter__(cls):
            return iter(cls(i) for i in sorted(cls._names))

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub
    temporal_stub = types.ModuleType("temporal")
    binary_filter_stub = types.ModuleType("temporal.binary_state_filter")
    binary_filter_stub.BinaryStateFilter = object
    binary_controller_stub = types.ModuleType("temporal.binary_state_lifespan_controller")
    binary_controller_stub.BinaryLifespanAction = _BinaryLifespanAction
    sys.modules["temporal"] = temporal_stub
    sys.modules["temporal.binary_state_filter"] = binary_filter_stub
    sys.modules["temporal.binary_state_lifespan_controller"] = binary_controller_stub
    prepare_stub = types.ModuleType("experiments.prepare_paslcd_fixed_pose_cues")
    prepare_stub.DEFAULT_CUE_ROOT = Path("outputs/cues")
    prepare_stub.DEFAULT_DATASET_ROOT = Path("data/paslcd")
    prepare_stub.DEFAULT_INSTANCES = ()
    prepare_stub.DEFAULT_OSCD_FALLBACK_OUTPUT_ROOT = Path("outputs/oscd_fallback")
    prepare_stub.DEFAULT_OSCD_OUTPUT_ROOT = Path("outputs/oscd")
    prepare_stub.DEFAULT_SCENES = ()
    prepare_stub.SceneSpec = object
    prepare_stub.discover_scenes = lambda *args, **kwargs: []
    prepare_stub.parse_csv = lambda value, default: default if value is None else tuple(value.split(","))
    prepare_stub.sha256_file = lambda path: "sha256"
    sys.modules["experiments.prepare_paslcd_fixed_pose_cues"] = prepare_stub
    run_online_stub = types.ModuleType("experiments.run_online_binary_state_lifespan_thaw")
    run_online_stub.BASE_PLY_REL = "point_cloud/iteration_30000/point_cloud.ply"
    run_online_stub.LifecycleEvent = object
    run_online_stub.RunConfig = object
    run_online_stub.controller_events = lambda decision: []
    run_online_stub.event_diagnostics = lambda events: {}
    run_online_stub.filter_config = lambda config: config
    run_online_stub.make_controller = lambda model, config: None
    run_online_stub.posterior_run_diagnostics = lambda rows: {}
    run_online_stub.quantile_summary = lambda values: {}
    run_online_stub.run_config_from_args = lambda args: args
    run_online_stub.same_scene_repeated_transition_diagnostics = lambda events, boundaries: {}
    run_online_stub.serializable_arguments = lambda args: {}
    sys.modules["experiments.run_online_binary_state_lifespan_thaw"] = run_online_stub

from experiments.analyze_paslcd_view_consistency import (
    collect_cohort_stats,
    discover_scene_infos,
    emission_llr_active,
    load_frame_fields,
    transition_odds_arrays,
    update_scene,
)
from experiments.run_paslcd_view_consistency_diagnostic import (
    DiagnosticEvent,
    SparseObservedRows,
    angular_delta_degrees,
    camera_center_from_w2c,
    camera_forward_from_w2c,
    event_structure_key,
    event_structure_sha256,
    lifecycle_event_structure_sha256,
    load_resumable_scene_summary,
    max_transition_llr_without_prior,
    merge_sparse_observed_rows,
    read_sparse_observed_rows_jsonl,
    run_detector_only_view_consistency_diagnostic,
    summarize_transition_cohorts,
    transition_llr_without_prior,
    write_sparse_observed_rows_jsonl,
)


def test_transition_llr_removes_flip_prior_and_uses_max_evidence_view() -> None:
    """View-consistency scores must measure evidence, not Markov prior strength."""
    weak = transition_llr_without_prior(
        p_transition=0.60,
        transition_prior=0.20,
    )
    strong = transition_llr_without_prior(
        p_transition=0.90,
        transition_prior=0.20,
    )
    same_posterior_weaker_prior = transition_llr_without_prior(
        p_transition=0.90,
        transition_prior=0.05,
    )

    expected_strong = math.log(0.90 / 0.10) - math.log(0.20 / 0.80)
    assert strong == pytest.approx(expected_strong)
    assert same_posterior_weaker_prior > strong
    assert max_transition_llr_without_prior([weak, strong]) == pytest.approx(strong)


@pytest.mark.parametrize(
    ("p_transition", "transition_prior"),
    [(0.0, 0.2), (1.0, 0.2), (0.5, 0.0), (0.5, 1.0)],
)
def test_transition_llr_rejects_degenerate_probabilities(
    p_transition: float,
    transition_prior: float,
) -> None:
    with pytest.raises(ValueError, match="probab|prior|open interval"):
        transition_llr_without_prior(
            p_transition=p_transition,
            transition_prior=transition_prior,
        )


def test_sparse_observed_rows_schema_round_trips_jsonl_without_densifying(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observed_rows.jsonl"
    rows = [
        SparseObservedRows(
            frame_index=3,
            view_id="cam_0003",
            observed_gaussian_indices=[2, 7],
            transition_llr=[1.25, -0.5],
            p_transition=[0.8, 0.1],
            camera_center=[1.0, 2.0, 3.0],
            camera_forward=[0.0, 0.0, 1.0],
        ),
        SparseObservedRows(
            frame_index=4,
            view_id="cam_0004",
            observed_gaussian_indices=[],
            transition_llr=[],
            p_transition=[],
            camera_center=[1.0, 2.0, 4.0],
            camera_forward=[0.0, 0.0, 1.0],
        ),
    ]

    write_sparse_observed_rows_jsonl(path, rows)
    loaded = read_sparse_observed_rows_jsonl(path)

    assert loaded == rows
    payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert payload[0]["schema"] == "paslcd_view_consistency_sparse_observed_rows_v1"
    assert payload[0]["observed_gaussian_indices"] == [2, 7]
    assert "dense_gaussian_count" not in payload[0]
    assert payload[1]["observed_gaussian_indices"] == []


def test_camera_helpers_return_center_forward_and_angular_delta() -> None:
    identity_w2c = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    translated_w2c = [
        [1.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -2.0],
        [0.0, 0.0, 1.0, -3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    assert camera_center_from_w2c(identity_w2c) == pytest.approx([0.0, 0.0, 0.0])
    assert camera_center_from_w2c(translated_w2c) == pytest.approx([1.0, 2.0, 3.0])
    assert camera_forward_from_w2c(identity_w2c) == pytest.approx([0.0, 0.0, 1.0])
    assert angular_delta_degrees([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]) == pytest.approx(90.0)
    assert angular_delta_degrees([0.0, 0.0, 1.0], [0.0, 0.0, 2.0]) == pytest.approx(0.0)


def test_event_structure_hash_ignores_scores_but_detects_view_or_action_change() -> None:
    base = DiagnosticEvent(
        gaussian_index=5,
        frame_index=12,
        view_id="cam_0012",
        action="OPEN",
        old_binary_label=0,
        new_binary_label=1,
        old_slot=-1,
        new_current_slot=0,
        transition_llr=3.0,
        p_transition=0.9,
    )
    score_changed = DiagnosticEvent(
        gaussian_index=5,
        frame_index=12,
        view_id="cam_0012",
        action="OPEN",
        old_binary_label=0,
        new_binary_label=1,
        old_slot=-1,
        new_current_slot=0,
        transition_llr=4.0,
        p_transition=0.95,
    )
    view_changed = DiagnosticEvent(
        gaussian_index=5,
        frame_index=12,
        view_id="cam_0013",
        action="OPEN",
        old_binary_label=0,
        new_binary_label=1,
        old_slot=-1,
        new_current_slot=0,
        transition_llr=3.0,
        p_transition=0.9,
    )
    action_changed = DiagnosticEvent(
        gaussian_index=5,
        frame_index=12,
        view_id="cam_0012",
        action="CLOSE",
        old_binary_label=1,
        new_binary_label=0,
        old_slot=0,
        new_current_slot=-1,
        transition_llr=3.0,
        p_transition=0.9,
    )

    assert event_structure_key(base) == event_structure_key(score_changed)
    assert event_structure_sha256([base]) == event_structure_sha256([score_changed])
    assert event_structure_key(base) != event_structure_key(view_changed)
    assert event_structure_key(base) != event_structure_key(action_changed)


def test_unobserved_rows_remain_missing_not_zero_scored() -> None:
    dense = merge_sparse_observed_rows(
        num_gaussians=5,
        sparse=SparseObservedRows(
            frame_index=8,
            view_id="cam_0008",
            observed_gaussian_indices=[1, 4],
            transition_llr=[2.0, -1.0],
            p_transition=[0.88, 0.12],
            camera_center=[0.0, 0.0, 0.0],
            camera_forward=[0.0, 0.0, 1.0],
        ),
    )

    assert [row["observed"] for row in dense] == [False, True, False, False, True]
    assert dense[1]["transition_llr"] == pytest.approx(2.0)
    assert dense[4]["p_transition"] == pytest.approx(0.12)
    assert dense[0]["transition_llr"] is None
    assert dense[2]["p_transition"] is None


def test_no_gt_no_optimizer_path_uses_stubs_without_touching_forbidden_hooks(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def detector_stub() -> list[SparseObservedRows]:
        calls.append("detector")
        return [
            SparseObservedRows(
                frame_index=0,
                view_id="cam_0000",
                observed_gaussian_indices=[0],
                transition_llr=[2.5],
                p_transition=[0.92],
                camera_center=[0.0, 0.0, 0.0],
                camera_forward=[0.0, 0.0, 1.0],
            )
        ]

    def forbidden_gt_loader():  # pragma: no cover - should never be reached
        calls.append("gt")
        raise AssertionError("diagnostic must not load GT masks")

    def forbidden_optimizer_step():  # pragma: no cover - should never be reached
        calls.append("optimizer")
        raise AssertionError("diagnostic must not optimize state geometry")

    summary = run_detector_only_view_consistency_diagnostic(
        output_dir=tmp_path,
        detector=detector_stub,
        gt_loader=forbidden_gt_loader,
        optimizer_step=forbidden_optimizer_step,
    )

    assert calls == ["detector"]
    assert summary["gt_used_for_training"] is False
    assert summary["optimizer_steps"] == 0
    assert summary["observed_row_count"] == 1
    assert (tmp_path / "observed_rows.jsonl").is_file()


def test_cohort_statistics_are_deterministic_and_order_independent() -> None:
    events = [
        DiagnosticEvent(2, 10, "cam_b", "OPEN", 0, 1, -1, 0, 2.0, 0.9),
        DiagnosticEvent(1, 10, "cam_a", "OPEN", 0, 1, -1, 0, 4.0, 0.98),
        DiagnosticEvent(1, 11, "cam_c", "KEEP", 1, 1, 0, 0, 1.0, 0.7),
    ]

    first = summarize_transition_cohorts(events)
    second = summarize_transition_cohorts(list(reversed(events)))

    assert first == second
    assert first["event_count"] == 3
    assert first["transition_event_count"] == 2
    assert first["unique_gaussian_count"] == 2
    assert first["max_transition_llr_by_gaussian"] == {1: 4.0, 2: 2.0}
    assert first["event_structure_sha256"] == event_structure_sha256(events)


def _write_analysis_npz(
    path: Path,
    *,
    gaussian_index: list[int],
    action: list[int] | list[str],
    current_slot: list[int] | None = None,
    q: list[float] | None = None,
    evidence_strength: list[float] | None = None,
    raw_mass: list[float] | None = None,
    p_active_pre: list[float] | None = None,
    p_active_post: list[float] | None = None,
    p00: list[float] | None = None,
    p01: list[float] | None = None,
    p10: list[float] | None = None,
    p11: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(gaussian_index)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "gaussian_index": np.asarray(gaussian_index, dtype=np.int64),
        "e_plus": np.full(n, 0.75, dtype=np.float32),
        "e_minus": np.full(n, 0.25, dtype=np.float32),
        "raw_mass": np.asarray(raw_mass if raw_mass is not None else [1.0] * n, dtype=np.float32),
        "delta_a": np.full(n, 0.75, dtype=np.float32),
        "delta_b": np.full(n, 0.25, dtype=np.float32),
        "q": np.asarray(q if q is not None else [0.75] * n, dtype=np.float32),
        "evidence_strength": np.asarray(
            evidence_strength if evidence_strength is not None else [1.0] * n,
            dtype=np.float32,
        ),
        "p_active_pre": np.asarray(p_active_pre if p_active_pre is not None else [0.2] * n, dtype=np.float32),
        "p_active_post": np.asarray(p_active_post if p_active_post is not None else [0.8] * n, dtype=np.float32),
        "p00": np.asarray(p00 if p00 is not None else [0.90] * n, dtype=np.float32),
        "p01": np.asarray(p01 if p01 is not None else [0.10] * n, dtype=np.float32),
        "p10": np.asarray(p10 if p10 is not None else [0.20] * n, dtype=np.float32),
        "p11": np.asarray(p11 if p11 is not None else [0.80] * n, dtype=np.float32),
        "pflip": np.full(n, 0.30, dtype=np.float32),
        "old_slot": np.full(n, -1, dtype=np.int64),
        "current_slot": np.asarray(current_slot if current_slot is not None else [-1] * n, dtype=np.int64),
        "visible_observation_count": np.ones(n, dtype=np.int64),
    }
    if action and isinstance(action[0], str):
        payload["action_name"] = np.asarray(action, dtype=np.str_)
    else:
        payload["action"] = np.asarray(action, dtype=np.int16)
    np.savez_compressed(path, **payload)


def test_load_frame_fields_decodes_real_npz_action_schema_and_legacy_aliases(tmp_path: Path) -> None:
    npz = tmp_path / "frame.npz"
    np.savez_compressed(
        npz,
        gaussian_index=np.asarray([3, 4, 5, 6], dtype=np.int64),
        delta_a=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        delta_b=np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        total_mass=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        p_active=np.asarray([0.1, 0.6, 0.7, 0.2], dtype=np.float32),
        p_00=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        p_01=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        p_10=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        p_11=np.asarray([0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        p_flip=np.asarray([0.3, 0.5, 0.7, 0.9], dtype=np.float32),
        action=np.asarray([1, 2, 3, 4], dtype=np.int16),
        new_current_slot=np.asarray([0, 0, -1, -1], dtype=np.int64),
        visible_observations=np.asarray([1, 2, 3, 4], dtype=np.int64),
    )

    fields = load_frame_fields(npz)

    assert fields["gaussian_index"].tolist() == [3, 4, 5, 6]
    assert fields["action_name"].tolist() == ["OPEN", "KEEP", "CLOSE", "UNCERTAIN"]
    assert fields["raw_mass"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert fields["current_slot"].tolist() == [0, 0, -1, -1]
    assert fields["visible_observation_count"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert fields["q"].tolist() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert fields["evidence_strength"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_lifecycle_event_structure_hash_filters_events_at_or_after_max_timestamp(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.jsonl"
    prefix = tmp_path / "prefix.jsonl"
    rows = [
        {"gaussian_index": 1, "decision_timestamp": 0, "old_binary_label": 0, "new_binary_label": 1, "action": "OPEN", "old_slot": -1, "new_current_slot": 0, "score": 0.1},
        {"gaussian_index": 2, "decision_timestamp": 1, "old_binary_label": 1, "new_binary_label": 1, "action": "KEEP", "old_slot": 0, "new_current_slot": 0, "score": 0.2},
        {"gaussian_index": 3, "decision_timestamp": 2, "old_binary_label": 1, "new_binary_label": 0, "action": "CLOSE", "old_slot": 0, "new_current_slot": -1, "score": 0.3},
    ]
    full.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    prefix.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:2]), encoding="utf-8")

    assert lifecycle_event_structure_sha256(full, max_timestamp=2) == lifecycle_event_structure_sha256(prefix)
    assert lifecycle_event_structure_sha256(full, max_timestamp=2) != lifecycle_event_structure_sha256(full)


def test_nested_multi_scene_manifest_keeps_gaussian_namespaces_separate(tmp_path: Path) -> None:
    for instance, scene in [("u01", "scene_a"), ("u02", "scene_a")]:
        scene_dir = tmp_path / instance / scene
        frame_path = scene_dir / "frames" / "000000_same_index.npz"
        _write_analysis_npz(frame_path, gaussian_index=[0], action=["OPEN"], current_slot=[0])
        (scene_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "frames": [{"timestamp": 0, "frame_name": "same_index", "npz": "frames/000000_same_index.npz"}],
                    "scene_summary": {"instance": instance, "scene": scene, "gaussian_count": 1},
                }
            ),
            encoding="utf-8",
        )

    scenes = discover_scene_infos(tmp_path)
    states_and_events = [update_scene(scene, eta=0.9) for scene in scenes]
    states = [state for state, _events in states_and_events]

    assert sorted((scene.instance, scene.scene) for scene in scenes) == [("u01", "scene_a"), ("u02", "scene_a")]
    stable_open = next(row for row in collect_cohort_stats(states) if row["cohort"] == "stable-open")
    assert stable_open["gaussian_count"] == 2
    assert stable_open["open_count"] == 2


def test_eta_emission_and_transition_odds_math_are_reported_at_events(tmp_path: Path) -> None:
    scene_dir = tmp_path / "u01" / "scene_a"
    frame_path = scene_dir / "frames" / "000000_event.npz"
    _write_analysis_npz(
        frame_path,
        gaussian_index=[7],
        action=["OPEN"],
        current_slot=[0],
        q=[0.75],
        evidence_strength=[2.0],
        p_active_pre=[0.4],
        p_active_post=[0.7],
        p00=[0.8],
        p01=[0.2],
        p10=[0.3],
        p11=[0.7],
    )
    (scene_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frames": [{"timestamp": 0, "frame_name": "event", "npz": "frames/000000_event.npz"}],
                "scene_summary": {"instance": "u01", "scene": "scene_a", "gaussian_count": 8},
            }
        ),
        encoding="utf-8",
    )

    scene = discover_scene_infos(scene_dir)[0]
    _state, events = update_scene(scene, eta=0.9)
    odds01, odds10 = transition_odds_arrays(np.asarray([0.8]), np.asarray([0.2]), np.asarray([0.3]), np.asarray([0.7]))

    assert emission_llr_active(np.asarray([0.75]), np.asarray([2.0]), 0.9).tolist() == pytest.approx([math.log(9.0)])
    assert odds01.tolist() == pytest.approx([0.25])
    assert odds10.tolist() == pytest.approx([3.0 / 7.0])
    assert len(events) == 1
    assert events[0]["selected_transition_odds"] == pytest.approx(0.25)
    assert events[0]["emission_llr_active"] == pytest.approx(math.log(9.0))
    assert events[0]["threshold_crossing"] == "up"


def test_stable_inactive_cohort_requires_at_least_five_observations(tmp_path: Path) -> None:
    scene_dir = tmp_path / "u01" / "scene_a"
    frames: list[dict[str, str | int]] = []
    for t in range(5):
        frame_path = scene_dir / "frames" / f"{t:06d}.npz"
        _write_analysis_npz(frame_path, gaussian_index=[0, 1], action=["NONE", "NONE"], current_slot=[-1, -1])
        frames.append({"timestamp": t, "frame_name": f"frame_{t}", "npz": f"frames/{t:06d}.npz"})
    (scene_dir / "manifest.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "scene_summary": {"instance": "u01", "scene": "scene_a", "gaussian_count": 2},
            }
        ),
        encoding="utf-8",
    )

    state, events = update_scene(discover_scene_infos(scene_dir)[0], eta=0.9)
    cohorts = {row["cohort"]: row for row in collect_cohort_stats([state])}

    assert events == []
    assert state.cohort_codes().tolist() == [3, 3]
    assert cohorts["stable-inactive"]["gaussian_count"] == 2
    assert cohorts["stable-inactive"]["observed_row_count"] == 10
    assert cohorts["insufficient-observation"]["gaussian_count"] == 0


def test_full_resume_rejects_cached_partial_scene(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_dir = source / "inference_scene" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_000.png").write_bytes(b"first")
    (image_dir / "frame_001.png").write_bytes(b"second")
    output_root = tmp_path / "diagnostic"
    scene_dir = output_root / "Instance_1" / "Scene"
    frames_dir = scene_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "000000.npz").write_bytes(b"cached")
    for name in ("frame_metrics.csv", "lifecycle_events.jsonl"):
        (scene_dir / name).write_text("", encoding="utf-8")
    (scene_dir / "summary.json").write_text(
        json.dumps({"frames": 1, "processed_frame_count": 1, "run_config": {}}),
        encoding="utf-8",
    )
    (scene_dir / "manifest.json").write_text(
        json.dumps({"frames": [{"npz": "frames/000000.npz"}]}),
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        instance="Instance_1",
        scene="Scene",
        source_path=source,
        cameras_json=tmp_path / "cameras.json",
        output_dir=tmp_path / "cue",
    )
    args = Namespace(
        output_root=output_root,
        max_frames=None,
        baseline_lifecycle_events=None,
        baseline_lifecycle_root=None,
        allow_baseline_mismatch=True,
    )

    assert load_resumable_scene_summary(args, spec, object()) is None
