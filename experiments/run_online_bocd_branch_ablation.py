"""Matched real-stream detector ablation for MAP-reset versus beam-2 BOCD.

This is an experimental adapter around
``experiments.run_online_bayesian_lifespan_thaw``. It deliberately leaves the
validated production runner unchanged while enabling one additional detector
mode for the first branch-preservation experiment. Both modes run detector-only
with identical evidence, poses, thresholds, and stream order.

Pass ordinary runner arguments after ``--``. Example::

    python -m experiments.run_online_bocd_branch_ablation \
      --output-root outputs/bocd_branch_ablation -- \
      --source-path data/Instance_1/scene_change1_2_3 \
      --skip-post-inference-evaluation
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import csv
import json
from pathlib import Path
from typing import Iterator, Sequence

import torch

from experiments import run_online_bayesian_lifespan_thaw as runner
from temporal.beam2_bocd import (
    BeamTwoBernoulliFilter,
    beam2_persistent_state_bytes,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/escd_bocd_branch_ablation")
MODES = ("map_reset", "beam2")
FORBIDDEN_RUNNER_ARGUMENTS = {
    "--bocd-mode",
    "--output-dir",
    "--detector-only-smoke",
}


@contextmanager
def experimental_beam2_runner_support() -> Iterator[None]:
    """Temporarily allow ``beam2`` without altering the baseline runner file."""

    original_validate = runner.validate_run_config
    original_estimate = runner.estimated_bocd_state_bytes
    original_factory = runner.make_bocd_filter

    def validate(config):
        if config.bocd_mode == "beam2":
            # All non-mode invariants are identical to map_reset. The actual
            # filter factory still receives beam2 below.
            original_validate(replace(config, bocd_mode="map_reset"))
            return
        original_validate(config)

    def estimate(mode, gaussian_count, max_run_length, dtype=torch.float32):
        if mode == "beam2":
            return beam2_persistent_state_bytes(gaussian_count, dtype)
        return original_estimate(mode, gaussian_count, max_run_length, dtype)

    def factory(
        mode, num_gaussians, config=None, *, device=None, dtype=torch.float32
    ):
        if mode == "beam2":
            return BeamTwoBernoulliFilter(
                num_gaussians, config, device=device, dtype=dtype
            )
        return original_factory(
            mode, num_gaussians, config, device=device, dtype=dtype
        )

    runner.validate_run_config = validate
    runner.estimated_bocd_state_bytes = estimate
    runner.make_bocd_filter = factory
    try:
        yield
    finally:
        runner.validate_run_config = original_validate
        runner.estimated_bocd_state_bytes = original_estimate
        runner.make_bocd_filter = original_factory


def sanitize_runner_args(values: Sequence[str]) -> list[str]:
    args = list(values)
    if args and args[0] == "--":
        args = args[1:]
    for token in args:
        if token in FORBIDDEN_RUNNER_ARGUMENTS:
            raise ValueError(
                f"{token} is controlled by this ablation; remove it from runner args"
            )
    return args


def build_mode_args(
    runner_args: Sequence[str],
    *,
    mode: str,
    output_dir: Path,
) -> argparse.Namespace:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    # The baseline parser does not list experimental modes. Parse through its
    # validated map_reset choice, then replace only the mode field.
    parsed = runner.parse_args(
        [
            *sanitize_runner_args(runner_args),
            "--detector-only",
            "--bocd-mode",
            "map_reset",
            "--output-dir",
            str(output_dir),
        ]
    )
    parsed.bocd_mode = mode
    parsed.detector_only = True
    parsed.detector_only_smoke = False
    parsed.output_dir = output_dir
    return parsed


def read_mode_summary(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    event_counter: Counter[str] = Counter()
    event_path = output_dir / "lifecycle_events.jsonl"
    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                event_counter[json.loads(line)["action"]] += 1

    frame_totals = Counter()
    frame_path = output_dir / "frame_metrics.csv"
    if frame_path.exists():
        with frame_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                for key in (
                    "open_count",
                    "keep_count",
                    "close_count",
                    "uncertain_count",
                ):
                    frame_totals[key] += int(row.get(key, 0) or 0)

    metrics = summary.get("metrics", {})
    return {
        "algorithm": summary.get("algorithm"),
        "frames": summary.get("frames"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "persistent_state_gib": summary.get(
            "bocd_persistent_state_estimated_gib"
        ),
        "event_action_counts": dict(event_counter),
        "frame_action_totals": dict(frame_totals),
        "reopen_count": summary.get("reopen_count"),
        "boundary_diagnostics": summary.get(
            "boundary_diagnostics_after_inference"
        ),
        "transition_diagnostics": summary.get(
            "transition_diagnostics_after_inference"
        ),
        "mean_frame_iou": metrics.get("mean_frame_iou"),
        "mean_frame_f1": metrics.get("mean_frame_f1"),
        "aggregate_iou": metrics.get("aggregate_iou"),
        "aggregate_f1": metrics.get("aggregate_f1"),
        "output_dir": str(output_dir),
    }


def write_comparison(output_root: Path, results: dict[str, dict]) -> None:
    payload = {
        "schema_version": 1,
        "contract": "matched_detector_only_map_reset_vs_beam2",
        "modes": results,
        "decision_rules": {
            "primary_success": (
                "beam2 produces CLOSE and REOPEN events that map_reset misses"
            ),
            "evidence_failure": (
                "beam2 also has CLOSE=0; negative alpha-T evidence is insufficient "
                "or does not persist on the affected Gaussian rows"
            ),
            "over_sensitive_failure": (
                "beam2 produces many transition events far from scene transitions"
            ),
        },
    }
    (output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    map_result = results.get("map_reset", {})
    beam_result = results.get("beam2", {})
    lines = [
        "# MAP-reset vs beam-2 detector-only ablation",
        "",
        "| Metric | MAP-reset | Beam-2 |",
        "|---|---:|---:|",
        f"| OPEN events | {map_result.get('event_action_counts', {}).get('OPEN', 0)} | {beam_result.get('event_action_counts', {}).get('OPEN', 0)} |",
        f"| CLOSE events | {map_result.get('event_action_counts', {}).get('CLOSE', 0)} | {beam_result.get('event_action_counts', {}).get('CLOSE', 0)} |",
        f"| REOPEN | {map_result.get('reopen_count')} | {beam_result.get('reopen_count')} |",
        f"| Mean-frame IoU | {map_result.get('mean_frame_iou')} | {beam_result.get('mean_frame_iou')} |",
        f"| Mean-frame F1 | {map_result.get('mean_frame_f1')} | {beam_result.get('mean_frame_f1')} |",
        f"| Runtime (s) | {map_result.get('runtime_seconds')} | {beam_result.get('runtime_seconds')} |",
        f"| BOCD state (GiB) | {map_result.get('persistent_state_gib')} | {beam_result.get('persistent_state_gib')} |",
        "",
        "Interpretation:",
        "",
        "- Beam-2 CLOSE/REOPEN with MAP-reset CLOSE=0 supports the branch-discard diagnosis.",
        "- CLOSE=0 in both modes moves the blocker to evidence observability/scale.",
        "- Many off-boundary CLOSE/OPEN events indicate that hazard/threshold calibration is too permissive.",
    ]
    (output_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to run_online_bayesian_lifespan_thaw after --",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    with experimental_beam2_runner_support():
        for mode in args.modes:
            output_dir = args.output_root / mode
            mode_args = build_mode_args(
                args.runner_args, mode=mode, output_dir=output_dir
            )
            summary = runner.run_online(mode_args)
            if summary.get("algorithm") != mode:
                raise RuntimeError(
                    f"requested {mode}, runner reported {summary.get('algorithm')}"
                )
            results[mode] = read_mode_summary(output_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_comparison(args.output_root, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
