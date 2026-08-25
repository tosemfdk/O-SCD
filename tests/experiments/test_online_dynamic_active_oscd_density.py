from pathlib import Path

from experiments.run_online_dynamic_active_oscd_density import (
    _config,
    _sample_training_view,
    parse_args,
)
import numpy as np


def test_dynamic_runner_defaults_lock_active_oscd_16_step_schedule():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density")),
        ]
    )
    config = _config(args)
    assert args.density_policy == "active_oscd"
    assert config.updates_per_frame == 16
    assert args.densify_update_index == 4
    assert args.oscd_grad_threshold == 0.001
    assert args.percent_dense == 0.01
    assert args.min_opacity == 0.4
    assert args.max_screen_size == 0.0
    assert config.lifecycle_controller == "view_consistent"
    assert config.transition_confirmation_views == 3
    assert config.min_transition_bayes_factor == 3.0


def test_online_replay_samples_only_already_processed_views():
    views = [object() for _ in range(7)]
    rng = np.random.default_rng(3)
    for _ in range(100):
        selected, index = _sample_training_view(views, rng, 0.33)
        assert selected is views[index]
        assert 0 <= index < len(views)


def test_current_probability_one_always_selects_current_view():
    views = [object() for _ in range(4)]
    rng = np.random.default_rng(9)
    for _ in range(20):
        _selected, index = _sample_training_view(views, rng, 1.0)
        assert index == 3
