from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.run_ref_sc1_change_cue_density import (
    deterministic_training_view_index,
    oracle_state_local_view_indices,
    parse_args,
    select_scope_records,
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


def test_cli_validates_scope_and_density_update(tmp_path: Path):
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
    with pytest.raises(ValueError, match="scene_change1"):
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


def test_independent_scope_selection_reindexes_scene_change2():
    records = [
        SimpleNamespace(global_index=index, segment_name=segment, segment_id=sid)
        for index, (segment, sid) in enumerate(
            [("scene_change1", 0)] * 2 + [("scene_change2", 1)] * 3
        )
    ]
    selected = select_scope_records(records, scope="scene_change2", max_frames=3)
    assert [record.global_index for record in selected] == [0, 1, 2]
    assert {record.segment_name for record in selected} == {"scene_change2"}


def test_oracle_k10_never_samples_a_previous_state():
    records = [
        SimpleNamespace(segment_name="scene_change1", segment_id=0)
        for _ in range(12)
    ] + [
        SimpleNamespace(segment_name="scene_change2", segment_id=1)
        for _ in range(12)
    ]
    for current in range(len(records)):
        selected = oracle_state_local_view_indices(
            records[: current + 1], current, 10, seed=0
        )
        assert selected[-1] == current
        assert all(
            records[index].segment_name == records[current].segment_name
            for index in selected
        )


def test_oracle_view_bank_is_only_allowed_for_continuous_scope(tmp_path: Path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--condition",
            "cue_vcd",
            "--oracle-state-local-density-views",
        ]
    )
    with pytest.raises(ValueError, match="continuous"):
        validate_args(args)
