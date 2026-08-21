from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.replay_paslcd_transition_confirmation import (
    ReplayConfig,
    ReplayState,
    build_configs,
    normalized_bayes_factors,
    replay_scene,
    run_replay,
    update_state_for_frame,
)


def _write_scene(root: Path, instance: str, scene: str, frames: list[dict[str, np.ndarray]], gaussian_count: int = 4) -> Path:
    scene_dir = root / instance / scene
    frame_dir = scene_dir / "frames"
    frame_dir.mkdir(parents=True)
    manifest_frames = []
    for t, arrays in enumerate(frames):
        path = frame_dir / f"frame_{t:05d}.npz"
        np.savez_compressed(path, **arrays)
        manifest_frames.append(
            {
                "timestamp": t,
                "frame_index": t,
                "frame_name": f"frame_{t:05d}",
                "npz": f"frames/{path.name}",
                "translation_delta_from_previous": 0.0,
                "angular_delta_from_previous_degrees": 0.0,
            }
        )
    manifest = {
        "scene_summary": {"instance": instance, "scene": scene, "gaussian_count": gaussian_count},
        "frames": manifest_frames,
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scene_dir


def _frame(idx, p00, p01, p10, p11, strength=1.0):
    idx = np.asarray(idx, dtype=np.int64)
    n = idx.size
    return {
        "gaussian_index": idx,
        "evidence_strength": np.full(n, strength, dtype=np.float32),
        "p00": np.full(n, p00, dtype=np.float32),
        "p01": np.full(n, p01, dtype=np.float32),
        "p10": np.full(n, p10, dtype=np.float32),
        "p11": np.full(n, p11, dtype=np.float32),
    }


def _config(k=2, bf=3.0, min_strength=1e-6):
    return ReplayConfig(k, bf, min_strength, 0.01, 0.01)


def test_normalized_bayes_factor_removes_markov_prior_odds() -> None:
    p00 = np.asarray([0.9])
    p01 = np.asarray([0.09 / 0.99 * 0.9])
    p10 = np.asarray([0.09 / 0.99 * 0.8])
    p11 = np.asarray([0.8])
    open_bf, close_bf = normalized_bayes_factors(p00, p01, p10, p11, p01_prior=0.01, p10_prior=0.01)
    assert open_bf[0] == pytest.approx(9.0)
    assert close_bf[0] == pytest.approx(9.0)


def test_k_confirmation_requires_consecutive_support() -> None:
    cfg = _config(k=2, bf=3.0)
    state = ReplayState.make(2)
    idx = np.asarray([0], dtype=np.int64)
    strength = np.asarray([1.0])
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=0, config=cfg)
    assert int(state.open_count[0]) == 0
    assert int(state.support_count[0]) == 1
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=1, config=cfg)
    assert bool(state.active[0]) is True
    assert int(state.open_count[0]) == 1
    assert int(state.support_count[0]) == 0
    assert state.decision_count == 1
    assert state.observed_delay_sum == pytest.approx(1.0)


def test_observed_contradiction_resets_streak_but_unobserved_holds() -> None:
    cfg = _config(k=2, bf=3.0)
    state = ReplayState.make(1)
    idx = np.asarray([0], dtype=np.int64)
    strength = np.asarray([1.0])
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=0, config=cfg)
    update_state_for_frame(state, np.asarray([], dtype=np.int64), np.asarray([]), np.asarray([]), np.asarray([]), frame_index=1, config=cfg)
    assert int(state.support_count[0]) == 1
    update_state_for_frame(state, idx, strength, np.asarray([1.0]), np.asarray([1.0]), frame_index=2, config=cfg)
    assert int(state.support_count[0]) == 0
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=3, config=cfg)
    assert int(state.open_count[0]) == 0
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=4, config=cfg)
    assert int(state.open_count[0]) == 1


def test_close_then_reopen_allocates_reopen_metric() -> None:
    cfg = _config(k=1, bf=3.0)
    state = ReplayState.make(1)
    idx = np.asarray([0], dtype=np.int64)
    strength = np.asarray([1.0])
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=0, config=cfg)
    update_state_for_frame(state, idx, strength, np.asarray([1.0]), np.asarray([9.0]), frame_index=1, config=cfg)
    update_state_for_frame(state, idx, strength, np.asarray([9.0]), np.asarray([1.0]), frame_index=2, config=cfg)
    assert int(state.open_count[0]) == 2
    assert int(state.close_count[0]) == 1
    assert int(state.transition_count[0]) == 3


def test_low_strength_does_not_count_as_support() -> None:
    cfg = _config(k=1, bf=3.0, min_strength=0.5)
    state = ReplayState.make(1)
    update_state_for_frame(
        state,
        np.asarray([0], dtype=np.int64),
        np.asarray([0.1]),
        np.asarray([99.0]),
        np.asarray([1.0]),
        frame_index=0,
        config=cfg,
    )
    assert int(state.open_count[0]) == 0
    assert int(state.support_count[0]) == 0


def test_multiscene_namespace_is_not_shared(tmp_path: Path) -> None:
    root = tmp_path / "d1"
    # Same Gaussian id appears in two scenes. Each scene should get one OPEN;
    # it must not become a reopen by leaking state across scene namespaces.
    arrays = [_frame([0], 0.9, 0.08181818, 0.01, 0.9)]
    _write_scene(root, "Instance_A", "Scene_1", arrays, gaussian_count=1)
    _write_scene(root, "Instance_A", "Scene_2", arrays, gaussian_count=1)
    out = tmp_path / "out"
    summary = run_replay(root, out, [_config(k=1, bf=3.0)])
    aggregate = summary["aggregate"][0]
    assert aggregate["open"] == 2
    assert aggregate["reopen"] == 0
    assert aggregate["unique_opened_rows"] == 2
    assert (out / "summary.json").exists()
    assert (out / "condition_metrics.csv").exists()


def test_replay_scene_uses_d1_schema_and_strength(tmp_path: Path) -> None:
    scene_dir = _write_scene(
        tmp_path,
        "Instance_1",
        "Garden",
        [
            _frame([0], 0.9, 0.08181818, 0.01, 0.9, strength=1.0),
            _frame([0], 0.9, 0.08181818, 0.01, 0.9, strength=1.0),
        ],
        gaussian_count=1,
    )
    from experiments.analyze_paslcd_view_consistency import discover_scene_infos

    rows = replay_scene(discover_scene_infos(scene_dir)[0], [_config(k=2, bf=3.0)])
    assert rows[0]["open"] == 1
    assert rows[0]["mean_decision_delay_observed_supports"] == pytest.approx(1.0)


def test_build_configs_grid_order_and_validation() -> None:
    configs = build_configs(consecutive_k=[1, 2], bf_threshold=[1.0, 3.0], min_strength=1e-6, p01_prior=0.01, p10_prior=0.01)
    assert [c.key for c in configs] == [
        "K1_BF1_M1e-06_P010.01_P100.01",
        "K1_BF3_M1e-06_P010.01_P100.01",
        "K2_BF1_M1e-06_P010.01_P100.01",
        "K2_BF3_M1e-06_P010.01_P100.01",
    ]
    with pytest.raises(ValueError):
        build_configs(consecutive_k=[0], bf_threshold=[1.0], min_strength=1e-6, p01_prior=0.01, p10_prior=0.01)
