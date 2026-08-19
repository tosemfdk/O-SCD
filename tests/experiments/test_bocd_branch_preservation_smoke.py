import json
from pathlib import Path

from experiments.run_bocd_branch_preservation_smoke import (
    SmokeConfig,
    main,
    run_filter,
)


def test_default_capped_smoke_reproduces_close_zero_and_beam2_recovery():
    config = SmokeConfig()
    map_result = run_filter("map_reset", config)
    beam_result = run_filter("beam2", config)

    assert map_result["lifecycle_events"] == ["OPEN"]
    assert map_result["close_count"] == 0
    assert map_result["reopen_count"] == 0

    assert beam_result["lifecycle_events"] == ["OPEN", "CLOSE", "OPEN"]
    assert beam_result["close_count"] == 1
    assert beam_result["reopen_count"] == 1
    assert beam_result["close_decision_delay"] == 1
    assert beam_result["reopen_decision_delay"] == 1
    assert beam_result["close_estimated_start_error"] == 0
    assert beam_result["reopen_estimated_start_error"] == 0
    assert beam_result["state_intervals"] == [
        {"slot": 0, "start": 0.0, "end": 41.0},
        {"slot": 1, "start": 81.0, "end": None},
    ]


def test_smoke_cli_writes_reproducible_machine_readable_outputs(tmp_path: Path):
    assert main(["--output-dir", str(tmp_path)]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["per_frame_evidence_mass"] == 1.0
    assert summary["results"]["map_reset"]["close_count"] == 0
    assert summary["results"]["beam2"]["close_count"] == 1
    assert (tmp_path / "frame_metrics.csv").is_file()
    assert (tmp_path / "lifecycle_events.jsonl").is_file()
