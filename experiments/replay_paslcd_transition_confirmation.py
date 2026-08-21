"""Replay view-consistent transition confirmation policies on D1 PASLCD diagnostics.

This is a detector-policy ablation only: it consumes sparse immutable-reference
D1 NPZ shards, never reads GT masks or temporal representation parameters, and
writes compact aggregate metrics for consecutive-view confirmation variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.analyze_paslcd_view_consistency import discover_scene_infos, load_frame_fields

EPS = 1.0e-12
DEFAULT_D1_ROOT = Path("outputs/paslcd_d1_view_consistency_diagnostic")
DEFAULT_OUTPUT_DIR = Path("outputs/paslcd_d2_transition_confirmation_replay")
SCHEMA_VERSION = "paslcd_transition_confirmation_replay_v1"


@dataclass(frozen=True)
class ReplayConfig:
    consecutive_k: int
    bf_threshold: float
    min_strength: float
    p01_prior: float
    p10_prior: float

    @property
    def key(self) -> str:
        return (
            f"K{self.consecutive_k}_BF{self.bf_threshold:g}"
            f"_M{self.min_strength:g}_P01{self.p01_prior:g}_P10{self.p10_prior:g}"
        )


@dataclass
class ReplayState:
    active: np.ndarray
    support_count: np.ndarray
    support_start_frame: np.ndarray
    open_count: np.ndarray
    close_count: np.ndarray
    ever_closed: np.ndarray
    ever_opened: np.ndarray
    transition_count: np.ndarray
    frame_delay_sum: float = 0.0
    observed_delay_sum: float = 0.0
    decision_count: int = 0
    supported_observation_count: int = 0
    observed_row_count: int = 0

    @classmethod
    def make(cls, size: int) -> "ReplayState":
        size = max(int(size), 1)
        return cls(
            active=np.zeros(size, dtype=bool),
            support_count=np.zeros(size, dtype=np.uint16),
            support_start_frame=np.full(size, -1, dtype=np.int32),
            open_count=np.zeros(size, dtype=np.uint16),
            close_count=np.zeros(size, dtype=np.uint16),
            ever_closed=np.zeros(size, dtype=bool),
            ever_opened=np.zeros(size, dtype=bool),
            transition_count=np.zeros(size, dtype=np.uint16),
        )

    def ensure(self, max_index: int) -> None:
        if max_index < self.active.size:
            return
        old = self.active.size
        new_size = max(max_index + 1, old * 2)

        def grow(arr: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full(new_size, fill, dtype=arr.dtype)
            out[:old] = arr
            return out

        self.active = grow(self.active, False)
        self.support_count = grow(self.support_count, 0)
        self.support_start_frame = grow(self.support_start_frame, -1)
        self.open_count = grow(self.open_count, 0)
        self.close_count = grow(self.close_count, 0)
        self.ever_closed = grow(self.ever_closed, False)
        self.ever_opened = grow(self.ever_opened, False)
        self.transition_count = grow(self.transition_count, 0)


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return out


def probability(value: str) -> float:
    out = float(value)
    if not (0.0 < out < 1.0):
        raise argparse.ArgumentTypeError("probability must be in (0, 1)")
    return out


def positive_float(value: str) -> float:
    out = float(value)
    if out <= 0.0 or not math.isfinite(out):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return out


def parse_csv_numbers(raw: str, cast: Any) -> tuple[Any, ...]:
    values: list[Any] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return tuple(values)


def build_configs(
    *,
    consecutive_k: Sequence[int],
    bf_threshold: Sequence[float],
    min_strength: float,
    p01_prior: float,
    p10_prior: float,
) -> list[ReplayConfig]:
    configs: list[ReplayConfig] = []
    for k in consecutive_k:
        if k <= 0:
            raise ValueError("consecutive K must be positive")
        for bf in bf_threshold:
            if bf <= 0.0 or not math.isfinite(bf):
                raise ValueError("BF threshold must be positive and finite")
            configs.append(
                ReplayConfig(
                    consecutive_k=int(k),
                    bf_threshold=float(bf),
                    min_strength=float(min_strength),
                    p01_prior=float(p01_prior),
                    p10_prior=float(p10_prior),
                )
            )
    return configs


def normalized_bayes_factors(
    p00: np.ndarray,
    p01: np.ndarray,
    p10: np.ndarray,
    p11: np.ndarray,
    *,
    p01_prior: float,
    p10_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return likelihood Bayes factors with the Markov prior odds removed."""
    prior_open_odds = p01_prior / (1.0 - p01_prior)
    prior_close_odds = p10_prior / (1.0 - p10_prior)
    bf_open = (p01 / np.maximum(p00, EPS)) / max(prior_open_odds, EPS)
    bf_close = (p10 / np.maximum(p11, EPS)) / max(prior_close_odds, EPS)
    return bf_open, bf_close


def update_state_for_frame(
    state: ReplayState,
    idx: np.ndarray,
    strength: np.ndarray,
    bf_open: np.ndarray,
    bf_close: np.ndarray,
    *,
    frame_index: int,
    config: ReplayConfig,
) -> None:
    if idx.size == 0:
        return
    state.ensure(int(np.max(idx)))
    active = state.active[idx]
    branch_bf = np.where(active, bf_close, bf_open)
    support = np.isfinite(branch_bf) & np.isfinite(strength) & (strength >= config.min_strength) & (branch_bf >= config.bf_threshold)

    state.observed_row_count += int(idx.size)
    state.supported_observation_count += int(np.count_nonzero(support))

    # Observed non-support is contradictory/insufficient evidence for the current
    # committed transition candidate, so it resets the consecutive confirmation.
    if np.any(~support):
        reset_idx = idx[~support]
        state.support_count[reset_idx] = 0
        state.support_start_frame[reset_idx] = -1

    if not np.any(support):
        return

    sup_idx = idx[support]
    previous_count = state.support_count[sup_idx].astype(np.int32, copy=False)
    starting = previous_count == 0
    if np.any(starting):
        state.support_start_frame[sup_idx[starting]] = int(frame_index)
    new_count = np.minimum(previous_count + 1, np.iinfo(np.uint16).max).astype(np.uint16)
    state.support_count[sup_idx] = new_count

    ready = new_count >= config.consecutive_k
    if not np.any(ready):
        return

    ready_idx = sup_idx[ready]
    ready_was_active = state.active[ready_idx].copy()
    starts = state.support_start_frame[ready_idx]
    frame_delays = np.maximum(0, int(frame_index) - starts)
    state.frame_delay_sum += float(np.sum(frame_delays))
    state.observed_delay_sum += float((config.consecutive_k - 1) * ready_idx.size)
    state.decision_count += int(ready_idx.size)

    opening = ~ready_was_active
    closing = ready_was_active
    if np.any(opening):
        rows = ready_idx[opening]
        state.open_count[rows] = np.minimum(state.open_count[rows] + 1, np.iinfo(np.uint16).max)
        state.ever_opened[rows] = True
        state.active[rows] = True
    if np.any(closing):
        rows = ready_idx[closing]
        state.close_count[rows] = np.minimum(state.close_count[rows] + 1, np.iinfo(np.uint16).max)
        state.ever_closed[rows] = True
        state.active[rows] = False
    state.transition_count[ready_idx] = np.minimum(state.transition_count[ready_idx] + 1, np.iinfo(np.uint16).max)
    state.support_count[ready_idx] = 0
    state.support_start_frame[ready_idx] = -1


def summarize_state(state: ReplayState, *, instance: str, scene: str, config: ReplayConfig) -> dict[str, Any]:
    open_total = int(np.sum(state.open_count, dtype=np.int64))
    close_total = int(np.sum(state.close_count, dtype=np.int64))
    reopen_total = int(np.sum(np.maximum(state.open_count.astype(np.int64) - 1, 0)))
    repeated_extra = int(np.sum(np.maximum(state.transition_count.astype(np.int64) - 1, 0)))
    decision_count = int(state.decision_count)
    return {
        "schema": SCHEMA_VERSION,
        "instance": instance,
        "scene": scene,
        "condition": config.key,
        "consecutive_k": int(config.consecutive_k),
        "bf_threshold": float(config.bf_threshold),
        "min_strength": float(config.min_strength),
        "p01_prior": float(config.p01_prior),
        "p10_prior": float(config.p10_prior),
        "open": open_total,
        "close": close_total,
        "reopen": reopen_total,
        "repeated_transition_extras": repeated_extra,
        "final_active": int(np.count_nonzero(state.active)),
        "unique_opened_rows": int(np.count_nonzero(state.open_count > 0)),
        "unique_closed_rows": int(np.count_nonzero(state.close_count > 0)),
        "rows_with_reopen": int(np.count_nonzero(state.open_count > 1)),
        "rows_with_repeated_transition": int(np.count_nonzero(state.transition_count > 1)),
        "observed_rows": int(state.observed_row_count),
        "supported_observations": int(state.supported_observation_count),
        "support_rate": safe_ratio(state.supported_observation_count, state.observed_row_count),
        "decision_count": decision_count,
        "mean_decision_delay_frames": none_if_nan(safe_ratio(state.frame_delay_sum, decision_count)),
        "mean_decision_delay_observed_supports": none_if_nan(safe_ratio(state.observed_delay_sum, decision_count)),
    }


def safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num) / float(den)


def none_if_nan(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def replay_scene(scene: Any, configs: Sequence[ReplayConfig]) -> list[dict[str, Any]]:
    max_idx = int(scene.gaussian_count or 1)
    states = [ReplayState.make(max_idx) for _ in configs]
    for frame in scene.frames:
        fields = load_frame_fields(frame.path)
        idx = np.asarray(fields["gaussian_index"], dtype=np.int64).reshape(-1)
        if idx.size == 0:
            continue
        strength = np.asarray(fields["evidence_strength"], dtype=np.float64).reshape(-1)
        p00 = np.asarray(fields["p00"], dtype=np.float64).reshape(-1)
        p01 = np.asarray(fields["p01"], dtype=np.float64).reshape(-1)
        p10 = np.asarray(fields["p10"], dtype=np.float64).reshape(-1)
        p11 = np.asarray(fields["p11"], dtype=np.float64).reshape(-1)
        for state, config in zip(states, configs, strict=True):
            bf_open, bf_close = normalized_bayes_factors(
                p00,
                p01,
                p10,
                p11,
                p01_prior=config.p01_prior,
                p10_prior=config.p10_prior,
            )
            update_state_for_frame(
                state,
                idx,
                strength,
                bf_open,
                bf_close,
                frame_index=int(frame.index),
                config=config,
            )
    return [summarize_state(state, instance=scene.instance, scene=scene.scene, config=config) for state, config in zip(states, configs, strict=True)]


def aggregate_rows(rows: Sequence[Mapping[str, Any]], configs: Sequence[ReplayConfig]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for config in configs:
        subset = [row for row in rows if row["condition"] == config.key]
        if not subset:
            continue
        decision_count = sum(int(row["decision_count"]) for row in subset)
        frame_delay_num = sum(float(row["mean_decision_delay_frames"] or 0.0) * int(row["decision_count"]) for row in subset)
        obs_delay_num = sum(float(row["mean_decision_delay_observed_supports"] or 0.0) * int(row["decision_count"]) for row in subset)
        observed_rows = sum(int(row["observed_rows"]) for row in subset)
        supported = sum(int(row["supported_observations"]) for row in subset)
        out.append(
            {
                "schema": SCHEMA_VERSION,
                "instance": "ALL",
                "scene": "ALL",
                "condition": config.key,
                "consecutive_k": int(config.consecutive_k),
                "bf_threshold": float(config.bf_threshold),
                "min_strength": float(config.min_strength),
                "p01_prior": float(config.p01_prior),
                "p10_prior": float(config.p10_prior),
                "open": sum(int(row["open"]) for row in subset),
                "close": sum(int(row["close"]) for row in subset),
                "reopen": sum(int(row["reopen"]) for row in subset),
                "repeated_transition_extras": sum(int(row["repeated_transition_extras"]) for row in subset),
                "final_active": sum(int(row["final_active"]) for row in subset),
                "unique_opened_rows": sum(int(row["unique_opened_rows"]) for row in subset),
                "unique_closed_rows": sum(int(row["unique_closed_rows"]) for row in subset),
                "rows_with_reopen": sum(int(row["rows_with_reopen"]) for row in subset),
                "rows_with_repeated_transition": sum(int(row["rows_with_repeated_transition"]) for row in subset),
                "observed_rows": observed_rows,
                "supported_observations": supported,
                "support_rate": none_if_nan(safe_ratio(supported, observed_rows)),
                "decision_count": decision_count,
                "mean_decision_delay_frames": none_if_nan(safe_ratio(frame_delay_num, decision_count)),
                "mean_decision_delay_observed_supports": none_if_nan(safe_ratio(obs_delay_num, decision_count)),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: Sequence[Mapping[str, Any]], aggregate: Sequence[Mapping[str, Any]], *, d1_root: Path, output_dir: Path, scene_count: int) -> dict[str, Any]:
    best_by_repeated = min(aggregate, key=lambda row: (int(row["repeated_transition_extras"]), -int(row["final_active"]))) if aggregate else None
    return {
        "schema": SCHEMA_VERSION,
        "d1_root": str(d1_root),
        "output_dir": str(output_dir),
        "scene_count": int(scene_count),
        "gt_used": False,
        "temporal_representation_read": False,
        "rows": list(rows),
        "aggregate": list(aggregate),
        "best_by_repeated_transition_extras": best_by_repeated,
    }


def run_replay(d1_root: Path, output_dir: Path, configs: Sequence[ReplayConfig]) -> dict[str, Any]:
    scenes = discover_scene_infos(d1_root)
    all_rows: list[dict[str, Any]] = []
    for scene in scenes:
        print(f"[replay] {scene.instance}/{scene.scene}: {len(scene.frames)} frames", flush=True)
        all_rows.extend(replay_scene(scene, configs))
    aggregate = aggregate_rows(all_rows, configs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "condition_metrics.csv", [*all_rows, *aggregate])
    summary = build_summary(all_rows, aggregate, d1_root=d1_root, output_dir=output_dir, scene_count=len(scenes))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("d1_root", nargs="?", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--consecutive-k", default="1,2,3", help="comma-separated K values")
    parser.add_argument("--bf-threshold", default="1,3", help="comma-separated prior-removed BF thresholds")
    parser.add_argument("--min-strength", type=float, default=1.0e-6)
    parser.add_argument("--inactive-to-active-prior", type=probability, default=0.01)
    parser.add_argument("--active-to-inactive-prior", type=probability, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    k_values = parse_csv_numbers(args.consecutive_k, positive_int)
    bf_values = parse_csv_numbers(args.bf_threshold, positive_float)
    if args.min_strength < 0.0 or not math.isfinite(args.min_strength):
        raise SystemExit("--min-strength must be finite and non-negative")
    configs = build_configs(
        consecutive_k=k_values,
        bf_threshold=bf_values,
        min_strength=float(args.min_strength),
        p01_prior=float(args.inactive_to_active_prior),
        p10_prior=float(args.active_to_inactive_prior),
    )
    summary = run_replay(args.d1_root, args.output_dir, configs)
    best = summary.get("best_by_repeated_transition_extras") or {}
    print(
        "[replay] wrote",
        args.output_dir,
        "best=",
        best.get("condition"),
        "repeated_extras=",
        best.get("repeated_transition_extras"),
        flush=True,
    )


if __name__ == "__main__":
    main()
