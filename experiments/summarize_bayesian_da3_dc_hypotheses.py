"""Summarize the controlled Bayesian+DA3 DC hypothesis suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.evaluate_bayesian_da3_dc_hypotheses import DEFAULT_OUTPUT_ROOT


PRIMARY_CONDITIONS = (
    "A_baseline",
    "B_h1_amplitude2",
    "C_h2_seed_local",
    "D_h4_current_dc",
    "E_h1_h2",
    "F_h1_h4",
    "G_all",
)

COMPARISONS = (
    ("H1_amplitude_only", "B_h1_amplitude2", "A_baseline"),
    ("H2_seed_local_only", "C_h2_seed_local", "A_baseline"),
    ("H4_current_dc_only", "D_h4_current_dc", "A_baseline"),
    ("H2_after_H1", "E_h1_h2", "B_h1_amplitude2"),
    ("H4_after_H1", "F_h1_h4", "B_h1_amplitude2"),
    ("H4_after_H1_H2", "G_all", "E_h1_h2"),
    ("H2_after_H1_H4", "G_all", "F_h1_h4"),
    ("all_vs_baseline", "G_all", "A_baseline"),
)

METRICS: dict[str, tuple[str, ...]] = {
    "gt_mean_frame_iou": ("metrics", "gt", "mean_frame_iou"),
    "gt_mean_frame_f1": ("metrics", "gt", "mean_frame_f1"),
    "gt_precision": ("metrics", "gt", "precision"),
    "gt_recall": ("metrics", "gt", "recall"),
    "cue_mean_frame_iou": ("metrics", "cue_mask", "mean_frame_iou"),
    "cue_high_mean": (
        "metrics",
        "cue_diagnostics",
        "learned_mean_on_high_cue",
    ),
    "cue_high_positive_fraction": (
        "metrics",
        "cue_diagnostics",
        "learned_positive_fraction_on_high_cue",
    ),
    "cue_low_mean": (
        "metrics",
        "cue_diagnostics",
        "learned_mean_on_low_cue",
    ),
    "cue_mae": (
        "metrics",
        "cue_diagnostics",
        "mean_absolute_error_to_q",
    ),
    "learned_to_white_ceiling_ratio": (
        "metrics",
        "cue_diagnostics",
        "learned_to_white_ceiling_mass_ratio_on_high_cue",
    ),
    "base_open_rgb_mean": ("final_intrinsic_dc", "base_open", "rgb_mean"),
    "seed_open_rgb_mean": ("final_intrinsic_dc", "seed_open", "rgb_mean"),
    "base_open_rgb_fraction_ge_0_5": (
        "final_intrinsic_dc",
        "base_open",
        "rgb_fraction_ge_0_5",
    ),
    "seed_open_rgb_fraction_ge_0_5": (
        "final_intrinsic_dc",
        "seed_open",
        "rgb_fraction_ge_0_5",
    ),
}
GRADIENT_METRICS = (
    "base_gradient_whitening_fraction",
    "base_gradient_absolute_mean",
    "seed_gradient_whitening_fraction",
    "seed_gradient_absolute_mean",
)


def nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def metric_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        name: nested(payload, path)
        for name, path in METRICS.items()
    }


def aggregate_gradient_csv(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, float | int] = {}
    for branch in ("base", "seed"):
        total_rows = sum(int(row[f"{branch}_gradient_rows"]) for row in rows)
        whitening_mass = sum(
            float(row[f"{branch}_gradient_whitening_fraction"])
            * int(row[f"{branch}_gradient_rows"])
            for row in rows
        )
        absolute_mass = sum(
            float(row[f"{branch}_gradient_absolute_mean"])
            * int(row[f"{branch}_gradient_rows"])
            for row in rows
        )
        result[f"{branch}_gradient_rows"] = total_rows
        result[f"{branch}_gradient_whitening_fraction"] = (
            whitening_mass / total_rows if total_rows else 0.0
        )
        result[f"{branch}_gradient_absolute_mean"] = (
            absolute_mass / total_rows if total_rows else 0.0
        )
    return result


def compare_controlled_state(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    discrete_names = (
        "base_current_state_index",
        "base_num_states",
        "base_open_timestamp",
        "seed_start",
        "seed_end",
        "seed_geometry_update_counts",
        "accepted_da3_source_rows",
        "accepted_da3_birth_global",
    )
    geometry_names = (
        "seed_xyz",
        "seed_opacity",
        "seed_scaling",
        "seed_rotation",
    )
    discrete_equal = {
        name: (
            tuple(reference[name].shape) == tuple(candidate[name].shape)
            and torch.equal(reference[name], candidate[name])
        )
        for name in discrete_names
    }
    geometry_max_absolute_difference: dict[str, float | None] = {}
    for name in geometry_names:
        if tuple(reference[name].shape) != tuple(candidate[name].shape):
            geometry_max_absolute_difference[name] = None
        elif reference[name].numel() == 0:
            geometry_max_absolute_difference[name] = 0.0
        else:
            geometry_max_absolute_difference[name] = float(
                (reference[name] - candidate[name]).abs().max().item()
            )

    reference_ids = reference["accepted_da3_source_rows"].tolist()
    candidate_ids = candidate["accepted_da3_source_rows"].tolist()
    if len(reference_ids) != len(set(reference_ids)) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError("accepted DA3 source rows must be unique")
    reference_lookup = {int(value): index for index, value in enumerate(reference_ids)}
    candidate_lookup = {int(value): index for index, value in enumerate(candidate_ids)}
    common_ids = sorted(reference_lookup.keys() & candidate_lookup.keys())
    union_count = len(reference_lookup.keys() | candidate_lookup.keys())
    reference_common = torch.as_tensor(
        [reference_lookup[value] for value in common_ids], dtype=torch.long
    )
    candidate_common = torch.as_tensor(
        [candidate_lookup[value] for value in common_ids], dtype=torch.long
    )
    common_geometry_max_absolute_difference: dict[str, float] = {}
    for name in geometry_names:
        if not common_ids:
            common_geometry_max_absolute_difference[name] = 0.0
        else:
            common_geometry_max_absolute_difference[name] = float(
                (
                    reference[name].index_select(0, reference_common)
                    - candidate[name].index_select(0, candidate_common)
                )
                .abs()
                .max()
                .item()
            )
    common_discrete_equal_fraction: dict[str, float] = {}
    for name in (
        "seed_start",
        "seed_end",
        "seed_geometry_update_counts",
        "accepted_da3_birth_global",
    ):
        if not common_ids:
            common_discrete_equal_fraction[name] = 1.0
        else:
            common_discrete_equal_fraction[name] = float(
                (
                    reference[name].index_select(0, reference_common)
                    == candidate[name].index_select(0, candidate_common)
                )
                .float()
                .mean()
                .item()
            )
    return {
        "all_discrete_equal": all(discrete_equal.values()),
        "discrete_equal": discrete_equal,
        "geometry_max_absolute_difference": geometry_max_absolute_difference,
        "accepted_seed_count_reference": len(reference_ids),
        "accepted_seed_count_candidate": len(candidate_ids),
        "accepted_seed_count_delta": len(candidate_ids) - len(reference_ids),
        "accepted_source_rows_common": len(common_ids),
        "accepted_source_rows_symmetric_difference": (
            union_count - len(common_ids)
        ),
        "accepted_source_rows_jaccard": (
            len(common_ids) / union_count if union_count else 1.0
        ),
        "common_discrete_equal_fraction": common_discrete_equal_fraction,
        "common_geometry_max_absolute_difference": (
            common_geometry_max_absolute_difference
        ),
    }


def build_summary(root: Path) -> dict[str, Any]:
    payloads = {
        name: json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        for name in PRIMARY_CONDITIONS
    }
    rows = {
        name: {
            "condition": name,
            **metric_row(payload),
            **aggregate_gradient_csv(root / name / "frame_metrics.csv"),
            **{f"event_{key}": value for key, value in payload["events"].items()},
            "runtime_seconds": payload["runtime_seconds"],
            "peak_cuda_bytes": payload["peak_cuda_bytes"],
        }
        for name, payload in payloads.items()
    }
    comparisons: dict[str, Any] = {}
    for label, left, right in COMPARISONS:
        comparisons[label] = {
            "left": left,
            "right": right,
            "delta": {
                metric: float(rows[left][metric]) - float(rows[right][metric])
                for metric in (*METRICS, *GRADIENT_METRICS)
                if rows[left][metric] is not None and rows[right][metric] is not None
            },
        }

    reference_state = root / "A_baseline" / "controlled_state.pt"
    controls = {
        name: compare_controlled_state(
            reference_state,
            root / name / "controlled_state.pt",
        )
        for name in PRIMARY_CONDITIONS
    }
    baseline_diag = payloads["A_baseline"]["metrics"]["cue_diagnostics"]
    invariant_hash_fields = (
        "immutable_reference_sha256",
        "base_detector_lifecycle_sha256",
        "seed_discrete_topology_lifecycle_sha256",
    )
    hash_invariants = {
        key: len(
            {
                payloads[name]["invariant_hashes"][key]
                for name in PRIMARY_CONDITIONS
            }
        )
        == 1
        for key in invariant_hash_fields
    }
    best_iou = max(PRIMARY_CONDITIONS, key=lambda name: rows[name]["gt_mean_frame_iou"])
    best_cue = max(PRIMARY_CONDITIONS, key=lambda name: rows[name]["cue_mean_frame_iou"])
    result = {
        "contract": "bayesian_da3_dc_hypothesis_suite",
        "conditions": list(PRIMARY_CONDITIONS),
        "condition_metrics": rows,
        "comparisons": comparisons,
        "coverage_hypothesis_from_baseline": {
            "white_ceiling_mean_on_high_cue": baseline_diag[
                "white_ceiling_mean_on_high_cue"
            ],
            "open_only_ceiling_mean_on_high_cue": baseline_diag[
                "open_only_ceiling_mean_on_high_cue"
            ],
            "coverage_limited_fraction_on_high_cue": baseline_diag[
                "coverage_limited_fraction_on_high_cue"
            ],
            "black_occlusion_mean_penalty_on_high_cue": baseline_diag[
                "black_occlusion_mean_penalty_on_high_cue"
            ],
            "learned_to_white_ceiling_mass_ratio_on_high_cue": baseline_diag[
                "learned_to_white_ceiling_mass_ratio_on_high_cue"
            ],
            "gt_positive_coverage_limited_fraction": baseline_diag[
                "gt_positive_coverage_limited_fraction"
            ],
        },
        "controlled_state_vs_baseline": controls,
        "hash_invariants": hash_invariants,
        "gradient_audits": {
            name: payloads[name]["gradient_finite_difference_audit"]
            for name in PRIMARY_CONDITIONS
        },
        "best_gt_mean_frame_iou": best_iou,
        "best_cue_mean_frame_iou": best_cue,
        "total_runtime_seconds": sum(
            float(payload["runtime_seconds"]) for payload in payloads.values()
        ),
    }
    repeat_summary_path = root / "R_baseline_repeat" / "summary.json"
    if repeat_summary_path.is_file():
        repeat_payload = json.loads(repeat_summary_path.read_text(encoding="utf-8"))
        baseline_metrics = rows["A_baseline"]
        repeat_metrics = {
            **metric_row(repeat_payload),
            **aggregate_gradient_csv(
                root / "R_baseline_repeat" / "frame_metrics.csv"
            ),
        }
        repeat_delta = {
            metric: float(repeat_metrics[metric]) - float(baseline_metrics[metric])
            for metric in (*METRICS, *GRADIENT_METRICS)
            if repeat_metrics[metric] is not None
            and baseline_metrics[metric] is not None
        }
        result["baseline_repeat_noise"] = {
            "metrics": repeat_metrics,
            "delta_vs_baseline": repeat_delta,
            "absolute_delta_vs_baseline": {
                metric: abs(value) for metric, value in repeat_delta.items()
            },
            "controlled_state_vs_baseline": compare_controlled_state(
                reference_state,
                root / "R_baseline_repeat" / "controlled_state.pt",
            ),
            "events": repeat_payload["events"],
            "gradient_audit": repeat_payload["gradient_finite_difference_audit"],
            "runtime_seconds": repeat_payload["runtime_seconds"],
        }
        result["total_runtime_seconds"] += float(repeat_payload["runtime_seconds"])

    negative_path = root / "N_seed_only_ssf_renderer_control" / "summary.json"
    if negative_path.is_file():
        negative = json.loads(negative_path.read_text(encoding="utf-8"))
        result["naive_seed_only_renderer_negative_control"] = {
            "frames": negative["frames"],
            "gt": negative["metrics"]["gt"],
            "cue_mask": negative["metrics"]["cue_mask"],
            "final_seed_open_dc": negative["final_intrinsic_dc"]["seed_open"],
            "gradient_audit": negative["gradient_finite_difference_audit"],
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_summary(args.experiment_root)
    (args.experiment_root / "all_conditions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = list(summary["condition_metrics"].values())
    with (args.experiment_root / "condition_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
