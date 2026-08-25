from pathlib import Path

from experiments.run_online_persistent_gaussian_lifespan import (
    DIRECT_PARAMETER_NAMES,
    _config,
    parse_args,
)


def test_persistent_runner_defaults_lock_matched_k3_bf3_protocol():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-persistent")),
        ]
    )
    config = _config(args)
    assert config.lifecycle_controller == "view_consistent"
    assert config.transition_confirmation_views == 3
    assert config.min_transition_bayes_factor == 3.0
    assert config.updates_per_frame == 120
    assert config.bayes_cue_mode == "binary"
    assert DIRECT_PARAMETER_NAMES == (
        "dc",
        "xyz",
        "features_rest",
        "opacity",
        "scaling",
        "rotation",
    )


def test_persistent_runner_rejects_nonpositive_updates():
    try:
        parse_args(
            [
                "--scope",
                "scene_change1",
                "--output-dir",
                "outputs/test-persistent",
                "--updates-per-frame",
                "0",
            ]
        )
    except SystemExit as error:
        assert error.code != 0
    else:  # pragma: no cover - argparse must reject before execution.
        raise AssertionError("zero updates should be rejected")
