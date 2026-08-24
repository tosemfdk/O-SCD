from pathlib import Path

import pytest

from experiments.run_ref_sc1_change_cue_density import (
    deterministic_training_view_index,
    parse_args,
    validate_args,
)


def test_training_schedule_never_accesses_a_future_view_and_is_repeatable():
    first = []
    second = []
    for timestamp in range(20):
        for update in range(16):
            first.append(
                deterministic_training_view_index(timestamp, update, seed=4)
            )
            second.append(
                deterministic_training_view_index(timestamp, update, seed=4)
            )
            assert first[-1] <= timestamp
    assert first == second


def test_cli_locks_ablation_to_ref_sc1_and_valid_density_update(tmp_path: Path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--condition",
            "cue_vcd",
            "--max-frames",
            "95",
            "--updates-per-frame",
            "16",
            "--densify-update-index",
            "4",
        ]
    )
    validate_args(args)
    args.max_frames = 96
    with pytest.raises(ValueError, match="ref -> SC1"):
        validate_args(args)
    args.max_frames = 95
    args.densify_update_index = 16
    with pytest.raises(ValueError, match="update schedule"):
        validate_args(args)


def test_baseline_ignores_k_as_a_density_condition(tmp_path: Path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--condition",
            "baseline",
            "--k-views",
            "10",
        ]
    )
    validate_args(args)
    assert args.condition == "baseline"
