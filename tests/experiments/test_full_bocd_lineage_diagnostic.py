import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.run_full_bocd_lineage_diagnostic import (
    DiagnosticConfig,
    bocd_config,
    build_cohort_mapping,
    detector_local_rows,
    parse_action_list,
    parse_cohort_events,
    posterior_mass_by_start,
    recent_start_mass,
    snapshot_filter_start_masses,
    start_mass_stats,
    summarize_boundary_lineages,
    summarize_lineage_recoveries,
    validate_config,
    validate_full_run_max_r,
    load_bocd_classes,
)


def test_parse_cohort_events_dedups_and_filters_actions(tmp_path: Path):
    path = tmp_path / "lifecycle_events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"gaussian_index": 3, "action": "OPEN"}),
                json.dumps({"gaussian_index": 2, "action": "CLOSE"}),
                json.dumps({"gaussian_index": 2, "action": "CLOSE"}),
                json.dumps({"gaussian_index": 5, "action": "REOPEN"}),
                json.dumps({"gaussian_index": 7, "action": "close"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parse_action_list("close, reopen") == {"CLOSE", "REOPEN"}
    assert parse_cohort_events(path, {"CLOSE"}) == [2, 7]
    assert parse_cohort_events(path, {"CLOSE", "REOPEN"}) == [2, 5, 7]


def test_event_cohort_uses_compact_detector_local_rows():
    global_rows, global_to_local, detector_count = build_cohort_mapping(
        10,
        [8, 2, 8, 5],
        device=torch.device("cpu"),
    )

    assert detector_count == 3
    assert global_rows.tolist() == [2, 5, 8]
    assert detector_local_rows(
        torch.tensor([8, 2, 5]), global_to_local
    ).tolist() == [2, 0, 1]
    with pytest.raises(RuntimeError, match="outside the detector cohort"):
        detector_local_rows(torch.tensor([1]), global_to_local)


def test_all_cohort_mapping_is_identity_without_dense_lookup():
    global_rows, global_to_local, detector_count = build_cohort_mapping(
        10,
        None,
        device=torch.device("cpu"),
    )

    assert global_rows is None
    assert global_to_local is None
    assert detector_count == 10
    rows = torch.tensor([9, 1, 4])
    assert torch.equal(detector_local_rows(rows, global_to_local), rows)


def test_filter_snapshot_includes_persistent_unobserved_rows_without_advancing():
    cfg = DiagnosticConfig(hazard=0.01, expected_run_length=None, max_run_length=4)
    bocd_cfg = bocd_config(cfg)
    _Config, BetaBernoulliBOCD, BeamTwoBernoulliFilter = load_bocd_classes()
    exact = BetaBernoulliBOCD(2, bocd_cfg, dtype=torch.float64)
    beam = BeamTwoBernoulliFilter(2, bocd_cfg, dtype=torch.float64)
    one = torch.tensor([1.0], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)
    for filt in (exact, beam):
        filt.update(one, zero, row_indices=torch.tensor([0]), timestamp=0)
        filt.update(zero, one, row_indices=torch.tensor([1]), timestamp=1)

    exact_before = exact.log_run_probs.clone()
    beam_before = beam.log_incumbent_weight.clone()
    exact_mass, beam_mass = snapshot_filter_start_masses(
        exact,
        beam,
        torch.tensor([0, 1]),
        frame_count=3,
        chunk_size=1,
    )

    assert exact_mass.shape == beam_mass.shape == (2, 3)
    assert torch.allclose(exact_mass.sum(dim=1), torch.ones(2))
    assert torch.allclose(beam_mass.sum(dim=1), torch.ones(2))
    assert torch.equal(exact.log_run_probs, exact_before)
    assert torch.equal(beam.log_incumbent_weight, beam_before)


def test_posterior_mass_by_start_ignores_missing_and_bins_duplicates():
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.9, 0.4]])
    starts = torch.tensor([[0, 1, 1], [-1, 2, 2]])

    mass = posterior_mass_by_start(probs, starts, frame_count=4)

    assert torch.allclose(mass[0], torch.tensor([0.2, 0.8, 0.0, 0.0]))
    assert torch.allclose(mass[1], torch.tensor([0.0, 0.0, 1.3, 0.0]))


def test_recent_start_mass_and_start_stats_are_causal():
    mass = torch.tensor(
        [
            [0.1, 0.2, 0.7, 0.0],
            [0.0, 0.4, 0.6, 0.0],
        ]
    )

    assert torch.allclose(recent_start_mass(mass, current_t=2, recent_window=2), torch.tensor([0.9, 1.0]))
    mean, q95 = start_mass_stats(mass, current_t=2, frame_count=4)
    assert np.allclose(mean, np.array([0.05, 0.3, 0.65, 0.0], dtype=np.float32))
    assert q95[3] == 0.0


def test_validate_full_run_max_r_rejects_truncation():
    validate_full_run_max_r(4, 4)
    with pytest.raises(ValueError, match="max-run-length"):
        validate_full_run_max_r(3, 4)


def test_boundary_summaries_are_posthoc_lineage_slices():
    mean = np.zeros((5, 5), dtype=np.float32)
    q95 = np.zeros((5, 5), dtype=np.float32)
    mean[2, 2] = 0.1
    mean[3, 2] = 0.6
    mean[4, 2] = 0.4
    q95[:, 2] = mean[:, 2] + 0.1

    summaries = summarize_boundary_lineages(mean, q95, boundaries=[2, 9])

    assert summaries[0]["boundary"] == 2
    assert summaries[0]["peak_mean_timestamp"] == 3
    assert summaries[0]["mean_crosses_0_5_at"] == 3
    assert [point["age"] for point in summaries[0]["lineage"]] == [0, 1, 2]
    assert summaries[1] == {"boundary": 9, "in_range": False}


def test_lineage_recovery_counts_low_initial_hypotheses_that_later_cross():
    initial = torch.tensor(
        [
            [0.1, float("nan"), 0.7],
            [0.2, 0.4, 0.0],
            [0.8, 0.3, 0.1],
        ]
    )
    first_cross = torch.tensor(
        [
            [3, -1, 2],
            [-1, 4, -1],
            [0, 1, 5],
        ]
    )

    summary = summarize_lineage_recoveries(initial, first_cross, threshold=0.5)

    assert summary["low_initial_lineage_count"] == 5
    assert summary["recovered_lineage_count"] == 3
    assert summary["recovery_rate"] == pytest.approx(0.6)
    assert summary["per_start"][0]["mean_recovery_delay"] == 3.0
    assert summary["per_start"][1]["mean_recovery_delay"] == 3.0
    assert summary["per_start"][2]["mean_recovery_delay"] == 3.0


def test_config_validation_rejects_bad_cue_and_accepts_defaults():
    validate_config(DiagnosticConfig())
    with pytest.raises(ValueError, match="cue_mode"):
        validate_config(DiagnosticConfig(cue_mode="other"))


def test_exact_lineage_can_become_dominant_after_initial_small_r0_and_beam_candidate_survives():
    cfg = DiagnosticConfig(
        hazard=0.01,
        expected_run_length=None,
        max_run_length=16,
        changepoint_probability=0.5,
        min_evidence_mass=1e-9,
    )
    bocd_cfg = bocd_config(cfg)
    _Config, BetaBernoulliBOCD, BeamTwoBernoulliFilter = load_bocd_classes()
    exact = BetaBernoulliBOCD(1, bocd_cfg, dtype=torch.float64)
    beam = BeamTwoBernoulliFilter(1, bocd_cfg, dtype=torch.float64)

    # Build a strong old no-change run, then switch to repeated change evidence.
    observations = [(0.0, 1.0)] * 8 + [(1.0, 0.0)] * 6
    exact_start_mass = []
    beam_candidate = []
    beam_candidate_start = []
    exact_p0_at_switch = None
    switch_t = 8
    for t, (s, f) in enumerate(observations):
        s_t = torch.tensor([s], dtype=torch.float64)
        f_t = torch.tensor([f], dtype=torch.float64)
        mass = s_t + f_t
        ex = exact.update(s_t, f_t, total_mass=mass, timestamp=t)
        be = beam.update(s_t, f_t, total_mass=mass, timestamp=t)
        binned = posterior_mass_by_start(ex.run_length_posterior, exact.run_start, frame_count=len(observations))
        exact_start_mass.append(float(binned[0, switch_t].item()))
        beam_candidate.append(float(be.candidate_probability[0].item()))
        beam_candidate_start.append(int(be.candidate_run_start[0].item()))
        if t == switch_t:
            exact_p0_at_switch = float(ex.changepoint_probability[0].item())

    assert exact_p0_at_switch is not None and exact_p0_at_switch < 0.5
    # Full BOCD kept the low-prior switch-t lineage and later makes it dominant.
    assert max(exact_start_mass[switch_t + 1 :]) > 0.5
    # Protected Beam-2 also does not immediately starve the recent candidate.
    assert any(start == switch_t and prob > 0.0 for start, prob in zip(beam_candidate_start[switch_t:], beam_candidate[switch_t:]))
