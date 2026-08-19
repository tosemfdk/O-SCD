"""Production-scale capped-evidence smoke for branch-preserving BOCD.

The experiment isolates the detector/controller from rendering and geometry. One
Gaussian receives unit-mass observations (the maximum contribution of the
current capped evidence mode) in an ACTIVE -> INACTIVE -> ACTIVE sequence. It
compares the current single-branch MAP-reset filter against the two-branch
filter that retains a reset candidate across frames.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

from temporal.bayesian_lifespan_controller import (
    BayesianLifespanController,
    LifespanAction,
)
from temporal.beam2_bocd import BeamTwoBernoulliFilter
from temporal.bernoulli_bocd import (
    BernoulliBOCDConfig,
    MAPResetBernoulliFilter,
)


@dataclass(frozen=True)
class SmokeConfig:
    segment_length: int = 40
    expected_run_length: float = 100.0
    changepoint_probability: float = 0.5
    open_probability: float = 0.6
    close_probability: float = 0.4
    min_run_evidence: float = 1.0
    min_visible_observations: int = 1
    max_run_length: int = 128
    dtype: str = "float64"


class SyntheticTemporalModel:
    """Minimal lifecycle model matching the production controller contract."""

    def __init__(self, max_states: int = 4, dtype: torch.dtype = torch.float64):
        self.max_states = int(max_states)
        self.state_valid = torch.zeros((1, max_states), dtype=torch.bool)
        self.state_start = torch.zeros((1, max_states), dtype=dtype)
        self.state_end = torch.full((1, max_states), float("inf"), dtype=dtype)
        self.state_status = torch.zeros((1, max_states), dtype=torch.int8)
        self.num_states = torch.zeros(1, dtype=torch.long)
        self.current_state_index = torch.full((1,), -1, dtype=torch.long)
        self.state_change_dc = torch.zeros((1, max_states, 1, 3), dtype=dtype)
        self.base = SimpleNamespace(
            _xyz=torch.zeros(1, 3, dtype=dtype),
            _features_dc=torch.zeros(1, 1, 3, dtype=dtype),
            _features_rest=torch.zeros(1, 0, 3, dtype=dtype),
            _opacity=torch.zeros(1, 1, dtype=dtype),
            _scaling=torch.zeros(1, 3, dtype=dtype),
            _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
        )

    def state_parameter_items(self):
        return (("dc", self.state_change_dc),)

    @torch.no_grad()
    def open_rows(self, rows, timestamp, initialization="zero", optimizer=None):
        del initialization, optimizer
        rows = torch.as_tensor(rows, dtype=torch.long).flatten()
        slots = self.num_states[rows].clone()
        if bool((self.current_state_index[rows] >= 0).any()):
            raise RuntimeError("row already open")
        if bool((slots >= self.max_states).any()):
            raise RuntimeError("temporal state capacity exceeded")
        self.state_valid[rows, slots] = True
        self.state_status[rows, slots] = 1
        self.state_start[rows, slots] = float(timestamp)
        self.state_end[rows, slots] = float("inf")
        self.current_state_index[rows] = slots
        self.num_states[rows] = slots + 1
        return slots

    @torch.no_grad()
    def close_rows(self, rows, timestamp):
        rows = torch.as_tensor(rows, dtype=torch.long).flatten()
        slots = self.current_state_index[rows].clone()
        active = slots >= 0
        rows, slots = rows[active], slots[active]
        self.state_end[rows, slots] = float(timestamp)
        self.state_status[rows, slots] = 2
        self.current_state_index[rows] = -1
        return slots


FILTERS = {
    "map_reset": MAPResetBernoulliFilter,
    "beam2": BeamTwoBernoulliFilter,
}


def detector_config(config: SmokeConfig) -> BernoulliBOCDConfig:
    return BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        expected_run_length=float(config.expected_run_length),
        hazard=None,
        max_run_length=int(config.max_run_length),
        min_evidence_mass=1e-6,
        open_probability=float(config.open_probability),
        close_probability=float(config.close_probability),
        changepoint_probability=float(config.changepoint_probability),
        min_run_evidence=float(config.min_run_evidence),
        min_visible_observations=int(config.min_visible_observations),
    )


def capped_sequence(segment_length: int) -> list[tuple[float, float, str]]:
    if segment_length < 2:
        raise ValueError("segment_length must be at least 2")
    return (
        [(1.0, 0.0, "active_1")] * segment_length
        + [(0.0, 1.0, "inactive")] * segment_length
        + [(1.0, 0.0, "active_2")] * segment_length
    )


def run_filter(mode: str, config: SmokeConfig) -> dict:
    if mode not in FILTERS:
        raise ValueError(f"unknown mode: {mode}")
    dtype = getattr(torch, config.dtype)
    bayes_config = detector_config(config)
    tracker = FILTERS[mode](1, bayes_config, dtype=dtype)
    model = SyntheticTemporalModel(max_states=4, dtype=dtype)
    controller = BayesianLifespanController(model, bayes_config)

    rows: list[dict] = []
    events: list[dict] = []
    for timestamp, (positive, negative, segment) in enumerate(
        capped_sequence(config.segment_length)
    ):
        update = tracker.update(
            torch.tensor([positive], dtype=dtype),
            torch.tensor([negative], dtype=dtype),
            total_mass=torch.tensor([1.0], dtype=dtype),
            timestamp=timestamp,
        )
        decision = controller.update(update, timestamp=timestamp)
        action = LifespanAction(int(decision.action.item())).name
        event = None
        if bool(decision.event_mask.item()):
            event = decision.event_records()[0]
            events.append(event)
        rows.append(
            {
                "mode": mode,
                "timestamp": timestamp,
                "segment": segment,
                "positive_count": positive,
                "negative_count": negative,
                "posterior_change_probability": float(
                    update.change_probability.item()
                ),
                "changepoint_probability": float(
                    update.changepoint_probability.item()
                ),
                "candidate_probability": float(
                    getattr(
                        update,
                        "candidate_probability",
                        update.changepoint_probability,
                    ).item()
                ),
                "log_predictive_continue": _finite_or_none(
                    getattr(update, "log_predictive_continue", None)
                ),
                "log_predictive_reset": _finite_or_none(
                    getattr(update, "log_predictive_reset", None)
                ),
                "log_bayes_factor": _finite_or_none(
                    getattr(update, "log_bayes_factor", None)
                ),
                "estimated_run_start": int(update.estimated_run_start.item()),
                "action": action,
                "active": bool(model.current_state_index.item() >= 0),
                "num_states": int(model.num_states.item()),
                "event": event,
            }
        )

    action_counts = {
        action.name: sum(row["action"] == action.name for row in rows)
        for action in LifespanAction
    }
    lifecycle_events = [event["action"] for event in events]
    close_events = [event for event in events if event["action"] == "CLOSE"]
    reopen_events = [
        event
        for event in events
        if event["action"] == "OPEN" and event["new_or_current_slot"] == 1
    ]
    boundaries = [config.segment_length, 2 * config.segment_length]
    return {
        "algorithm": tracker.algorithm,
        "rows": rows,
        "events": events,
        "action_counts": action_counts,
        "lifecycle_events": lifecycle_events,
        "close_count": len(close_events),
        "reopen_count": len(reopen_events),
        "close_decision_delay": _first_delay(close_events, boundaries[0]),
        "reopen_decision_delay": _first_delay(reopen_events, boundaries[1]),
        "close_estimated_start_error": _first_start_error(
            close_events, boundaries[0]
        ),
        "reopen_estimated_start_error": _first_start_error(
            reopen_events, boundaries[1]
        ),
        "final_num_states": int(model.num_states.item()),
        "final_active": bool(model.current_state_index.item() >= 0),
        "state_intervals": [
            {
                "slot": slot,
                "start": float(model.state_start[0, slot].item()),
                "end": (
                    None
                    if bool(torch.isinf(model.state_end[0, slot]))
                    else float(model.state_end[0, slot].item())
                ),
            }
            for slot in range(int(model.num_states.item()))
        ],
    }


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    scalar = float(torch.as_tensor(value).flatten()[0].item())
    return scalar if torch.isfinite(torch.tensor(scalar)) else None


def _first_delay(events: list[dict], boundary: int) -> int | None:
    return int(events[0]["decision_timestamp"]) - int(boundary) if events else None


def _first_start_error(events: list[dict], boundary: int) -> int | None:
    return (
        int(events[0]["bocd_estimated_changepoint_timestamp"]) - int(boundary)
        if events
        else None
    )


def write_outputs(
    output_dir: Path, config: SmokeConfig, results: dict[str, dict]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "contract": "capped_evidence_branch_preservation_smoke",
        "config": asdict(config),
        "sequence": "ACTIVE -> INACTIVE -> ACTIVE",
        "per_frame_evidence_mass": 1.0,
        "boundaries": [config.segment_length, 2 * config.segment_length],
        "results": {
            mode: {key: value for key, value in result.items() if key != "rows"}
            for mode, result in results.items()
        },
        "acceptance": {
            "map_reset_close_count_expected": 0,
            "beam2_close_count_expected": 1,
            "beam2_reopen_count_expected": 1,
            "beam2_estimated_start_error_expected": 0,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [row for result in results.values() for row in result["rows"]]
    with (output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [key for key in rows[0] if key != "event"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    with (output_dir / "lifecycle_events.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for mode, result in results.items():
            for event in result["events"]:
                file.write(
                    json.dumps({"mode": mode, **event}, sort_keys=True) + "\n"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bocd_branch_preservation_smoke"),
    )
    parser.add_argument("--segment-length", type=int, default=40)
    parser.add_argument("--expected-run-length", type=float, default=100.0)
    parser.add_argument("--changepoint-probability", type=float, default=0.5)
    parser.add_argument("--open-probability", type=float, default=0.6)
    parser.add_argument("--close-probability", type=float, default=0.4)
    parser.add_argument("--min-run-evidence", type=float, default=1.0)
    parser.add_argument("--min-visible-observations", type=int, default=1)
    parser.add_argument("--max-run-length", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = SmokeConfig(
        segment_length=int(args.segment_length),
        expected_run_length=float(args.expected_run_length),
        changepoint_probability=float(args.changepoint_probability),
        open_probability=float(args.open_probability),
        close_probability=float(args.close_probability),
        min_run_evidence=float(args.min_run_evidence),
        min_visible_observations=int(args.min_visible_observations),
        max_run_length=int(args.max_run_length),
    )
    results = {mode: run_filter(mode, config) for mode in FILTERS}
    write_outputs(args.output_dir, config, results)
    print(
        json.dumps(
            {
                mode: {
                    "lifecycle_events": result["lifecycle_events"],
                    "close_count": result["close_count"],
                    "reopen_count": result["reopen_count"],
                    "close_decision_delay": result["close_decision_delay"],
                    "reopen_decision_delay": result["reopen_decision_delay"],
                }
                for mode, result in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
