from pathlib import Path

import pytest
import torch

from experiments import run_online_bayesian_lifespan_thaw as runner
from experiments.run_online_bocd_branch_ablation import (
    beam2_persistent_state_bytes,
    build_mode_args,
    experimental_beam2_runner_support,
    sanitize_runner_args,
)


def test_beam2_memory_estimate_is_linear_and_between_map_and_exact():
    count = 1000
    beam = beam2_persistent_state_bytes(count, torch.float32)
    map_bytes = runner.estimated_bocd_state_bytes(
        "map_reset", count, 128, torch.float32
    )
    exact_bytes = runner.estimated_bocd_state_bytes(
        "exact", count, 128, torch.float32
    )

    assert beam == count * (9 * 4 + 8 * 8)
    assert map_bytes < beam < exact_bytes


def test_experimental_context_accepts_beam2_and_restores_runner_functions():
    original_validate = runner.validate_run_config
    original_estimate = runner.estimated_bocd_state_bytes
    original_factory = runner.make_bocd_filter
    config = runner.RunConfig(bocd_mode="beam2")

    with pytest.raises(ValueError, match="bocd_mode"):
        runner.validate_run_config(config)

    with experimental_beam2_runner_support():
        runner.validate_run_config(config)
        assert runner.estimated_bocd_state_bytes(
            "beam2", 3, 128, torch.float32
        ) == beam2_persistent_state_bytes(3, torch.float32)
        assert runner.make_bocd_filter(
            "beam2", 2, runner.bocd_config(config)
        ).algorithm == "beam2"

    assert runner.validate_run_config is original_validate
    assert runner.estimated_bocd_state_bytes is original_estimate
    assert runner.make_bocd_filter is original_factory


def test_build_mode_args_forces_detector_only_and_mode_specific_output(
    tmp_path: Path,
):
    args = build_mode_args(
        ["--max-frames", "2", "--skip-post-inference-evaluation"],
        mode="beam2",
        output_dir=tmp_path / "beam2",
    )

    assert args.bocd_mode == "beam2"
    assert args.detector_only is True
    assert args.detector_only_smoke is False
    assert args.output_dir == tmp_path / "beam2"
    assert args.max_frames == 2
    assert args.skip_post_inference_evaluation is True


def test_sanitize_runner_args_rejects_ablation_owned_flags():
    assert sanitize_runner_args(["--", "--max-frames", "3"]) == [
        "--max-frames",
        "3",
    ]
    for flag in ("--bocd-mode", "--output-dir", "--detector-only-smoke"):
        with pytest.raises(ValueError, match=flag):
            sanitize_runner_args([flag, "x"])
