from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.run_online_binary_lifespan_active_density import (
    _density_event_row,
    _run_config,
    _validate_args,
    parse_args,
)
from temporal.active_density_topology import (
    TemporalDensityResult,
    TemporalDensityScore,
)


def _args(*extra: str):
    return parse_args(
        [
            "--condition",
            "active_cue_vcd_vcp",
            "--scope",
            "scene_change1",
            "--output-dir",
            "outputs/test",
            *extra,
        ]
    )


def test_primary_config_locks_binary_k3bf3_all_geometry():
    args = _args()
    config = _run_config(args)
    assert config.bayes_cue_mode == "binary"
    assert config.lifecycle_controller == "view_consistent"
    assert config.transition_confirmation_views == 3
    assert config.min_transition_bayes_factor == 3.0
    assert config.thaw_parameters == (
        "dc",
        "xyz",
        "opacity",
        "scaling",
        "rotation",
    )
    assert args.k_views == 10
    assert args.updates_per_frame == 120


def test_density_thresholds_cannot_exceed_causal_view_bank():
    args = _args("--k-views", "3", "--min-prune-views", "4")
    with pytest.raises(ValueError, match="cannot exceed K"):
        _validate_args(args)


def test_density_event_reports_zero_future_access_and_separate_vcp_count():
    score = TemporalDensityScore(
        view_indices=(0, 2, 4),
        positive_mass=torch.tensor([6.0]),
        negative_mass=torch.tensor([1.0]),
        total_mass=torch.tensor([7.0]),
        importance_score=torch.tensor([2.0]),
        visible_view_count=torch.tensor([3]),
        support_view_count=torch.tensor([2]),
        change_ratio=torch.tensor([6.0 / 7.0]),
    )
    result = TemporalDensityResult(
        initial_count=2,
        final_count=3,
        importance_count=1,
        clone_count=0,
        split_source_count=1,
        split_child_count=2,
        vcp_pruned_count=1,
        split_residual_removed_count=0,
        removed_count=1,
        clone_mask=torch.tensor([False, False]),
        split_mask=torch.tensor([True, False]),
        prune_mask=torch.tensor([False, True]),
    )
    records = [SimpleNamespace(name=f"frame_{index}") for index in range(5)]
    event = _density_event_row(result, score, records, timestamp=4)
    assert event["future_view_access_count"] == 0
    assert event["vcp_pruned_count"] == 1
    assert event["split_residual_removed_count"] == 0
    assert event["selected_view_names"] == ["frame_0", "frame_2", "frame_4"]
