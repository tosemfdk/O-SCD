from pathlib import Path

import pytest

from experiments.summarize_single_candidate_beta_threshold_sweep import (
    _row,
    parse_args,
)


def test_summary_row_flattens_metrics_candidate_counts_and_invariants(tmp_path: Path):
    summary = {
        "detector": {
            "algorithm": "single_candidate_beta_log_bayes_factor",
            "single_candidate_beta": {"bayes_factor_threshold": 30.0},
        },
        "metrics": {
            "mean_frame_iou": 0.5,
            "mean_frame_f1": 0.6,
            "aggregate_iou": 0.51,
            "aggregate_f1": 0.61,
            "precision": 0.7,
            "recall": 0.8,
            "segments": [
                {
                    "segment": f"scene_change{index}",
                    "mean_frame_iou": 0.1 * index,
                    "mean_frame_f1": 0.2 * index,
                }
                for index in (1, 2, 3)
            ],
        },
        "open_count": 10,
        "close_count": 4,
        "reopen_count": 2,
        "same_scene_repeated_transition_event_count": 3,
        "candidate_diagnostics": {
            "candidate_started_count": 20,
            "candidate_rejected_count": 12,
            "candidate_committed_count": 5,
            "max_live_candidate_count": 7,
            "final_live_candidate_count": 3,
        },
        "density_control": {
            "initial_gaussian_count": 100,
            "final_gaussian_count": 110,
        },
        "runtime_seconds": 9.0,
        "peak_cuda_memory_bytes": 123,
        "closed_row_persistence_audit": {"passed": True, "max_abs": 0.0},
        "inactive_gradient_violations": 0,
        "topology_integrity": True,
        "gt_used_in_causal_loop": False,
    }

    (tmp_path / "lifecycle_events.jsonl").write_text(
        '{"action":"OPEN","decision_timestamp":1,"gaussian_index":4}\n'
        '{"action":"KEEP","decision_timestamp":2,"gaussian_index":4}\n'
        '{"action":"CLOSE","decision_timestamp":3,"gaussian_index":4}\n',
        encoding="utf-8",
    )
    row = _row("bf30", tmp_path, summary)

    assert row["bayes_factor_threshold"] == 30.0
    assert row["scene_change3_mean_frame_iou"] == pytest.approx(0.3)
    assert row["candidate_committed_count"] == 5
    assert row["same_scene_representation_repeated_transition_event_count"] == 1
    assert row["closed_drift_passed"] is True


def test_sweep_summary_cli_requires_explicit_output_locations():
    args = parse_args(
        [
            "--sweep-root",
            "/tmp/sweep",
            "--output-dir",
            "/tmp/sweep/report",
        ]
    )
    assert args.sweep_root == Path("/tmp/sweep")
    assert args.output_dir == Path("/tmp/sweep/report")
