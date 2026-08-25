"""Causal lifespans over one persistent mutable R_change Gaussian bank.

The immutable reference bank still owns alpha-T detector evidence.  A separate
mutable change bank stores one direct parameter set per Gaussian:

``change DC, xyz, SH-rest, opacity, scaling, rotation``.

Lifespan slots contain interval metadata only.  CLOSE hides a row but neither
its value nor Adam moments are reset; REOPEN allocates a new interval and
resumes the same learned representation.  This isolates the user's proposed
persistent-memory representation from the existing state-local delta model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import time

import numpy as np
import torch

from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL
from experiments.run_online_binary_state_lifespan_thaw import (
    LifecycleEvent,
    RunConfig,
    _prediction_from_package,
    base_drift,
    base_snapshots,
    event_diagnostics,
    evaluate_after_inference,
    make_controller,
    make_filter,
    posterior_run_diagnostics,
    same_scene_repeated_transition_diagnostics,
    update_binary_lifecycle_chunks,
)
from experiments.run_ref_sc1_change_cue_density import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    DEFAULT_SOURCE,
    SCOPE_LABELS,
    SCOPE_MAX_FRAMES,
    select_scope_records,
)


SCOPES = ("scene_change1", "scene_change2", "scene_change3")
DIRECT_PARAMETER_NAMES = (
    "dc",
    "xyz",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _parameter_drift(
    before: Mapping[str, torch.Tensor], model: Any
) -> dict[str, Any]:
    parameters = dict(model.persistent_parameter_items())
    per: dict[str, float] = {}
    changed_rows: dict[str, int] = {}
    maximum = 0.0
    for name, expected in before.items():
        actual = parameters[name].detach().cpu()
        difference = (actual - expected).abs()
        value = float(difference.max().item()) if difference.numel() else 0.0
        per[name] = value
        changed_rows[name] = int(
            difference.reshape(difference.shape[0], -1).ne(0).any(dim=1).sum().item()
        )
        maximum = max(maximum, value)
    return {
        "max_abs": maximum,
        "per_tensor_max_abs": per,
        "changed_rows": changed_rows,
    }


def _optimizer_row_snapshot(
    model: Any, optimizer: Any, rows: torch.Tensor
) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in model.persistent_parameter_items():
        values = {"parameter": parameter.detach()[rows].cpu().clone()}
        state = optimizer.state.get(parameter, {})
        for state_name, state_value in state.items():
            if not isinstance(state_value, torch.Tensor):
                continue
            if state_value.ndim >= 1 and state_value.shape[0] == parameter.shape[0]:
                values[state_name] = state_value.detach()[rows].cpu().clone()
        snapshot[name] = values
    return snapshot


class PersistentClosedRowAudit:
    """Verify exact stability while rows remain CLOSED, then allow REOPEN."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.total_closed_rows = 0
        self.total_reopened_rows = 0
        self.max_abs = 0.0
        self.per_state: dict[str, float] = {}

    def add(
        self,
        model: Any,
        optimizer: Any,
        rows: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        del slots
        rows = rows.detach().flatten().long().unique(sorted=True)
        if rows.numel() == 0:
            return
        self.entries.append(
            {
                "rows": rows.cpu().clone(),
                "snapshot": _optimizer_row_snapshot(model, optimizer, rows),
            }
        )
        self.total_closed_rows += int(rows.numel())

    @torch.no_grad()
    def _compare_positions(
        self,
        model: Any,
        optimizer: Any,
        entry: dict[str, Any],
        positions: torch.Tensor,
    ) -> None:
        if positions.numel() == 0:
            return
        parameters = dict(model.persistent_parameter_items())
        rows = entry["rows"][positions].to(device=model.state_valid.device)
        for name, values in entry["snapshot"].items():
            parameter = parameters[name]
            for state_name, expected_all in values.items():
                expected = expected_all[positions].cpu()
                if state_name == "parameter":
                    actual = parameter.detach()[rows].cpu()
                else:
                    actual = optimizer.state[parameter][state_name].detach()[rows].cpu()
                difference = (
                    float((actual - expected).abs().max().item())
                    if expected.numel()
                    else 0.0
                )
                key = f"{name}.{state_name}"
                self.per_state[key] = max(self.per_state.get(key, 0.0), difference)
                self.max_abs = max(self.max_abs, difference)

    @torch.no_grad()
    def release_reopened(
        self, model: Any, optimizer: Any, reopened_rows: torch.Tensor
    ) -> None:
        reopened = reopened_rows.detach().flatten().long().cpu().unique(sorted=True)
        if reopened.numel() == 0:
            return
        retained: list[dict[str, Any]] = []
        released = 0
        for entry in self.entries:
            selected = torch.isin(entry["rows"], reopened)
            positions = torch.nonzero(selected, as_tuple=False).flatten()
            self._compare_positions(model, optimizer, entry, positions)
            released += int(positions.numel())
            keep = ~selected
            if bool(keep.any()):
                entry["rows"] = entry["rows"][keep]
                for values in entry["snapshot"].values():
                    for state_name in tuple(values):
                        values[state_name] = values[state_name][keep]
                retained.append(entry)
        self.entries = retained
        self.total_reopened_rows += released

    @torch.no_grad()
    def verify(self, model: Any, optimizer: Any) -> dict[str, Any]:
        for entry in self.entries:
            positions = torch.arange(entry["rows"].numel(), dtype=torch.long)
            self._compare_positions(model, optimizer, entry, positions)
        return {
            "passed": self.max_abs == 0.0,
            "max_abs": float(self.max_abs),
            "total_closed_rows": int(self.total_closed_rows),
            "reopened_rows_verified_before_resume": int(self.total_reopened_rows),
            "remaining_closed_rows_verified_at_end": int(
                sum(int(entry["rows"].numel()) for entry in self.entries)
            ),
            "per_parameter_and_optimizer_state_max_abs": dict(self.per_state),
        }

    def status(self) -> dict[str, Any]:
        return {
            "passed": None,
            "max_abs": float(self.max_abs),
            "total_closed_rows": int(self.total_closed_rows),
        }


def _inactive_gradient_audit(
    model: Any, selected_rows: torch.Tensor
) -> dict[str, float | int]:
    violations = 0
    maximum = 0.0
    for _name, parameter in model.persistent_parameter_items():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().reshape(parameter.shape[0], -1)
        inactive = gradient[~selected_rows]
        nonzero = int(torch.count_nonzero(inactive).item())
        violations += nonzero
        if nonzero:
            maximum = max(maximum, float(inactive.abs().max().item()))
    return {"count": int(violations), "max_abs": float(maximum)}


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        bayes_cue_mode="binary",
        bayes_cue_threshold=float(args.bayes_cue_threshold),
        bayes_cue_scale=1.0,
        evidence_count_mode="capped",
        evidence_mass_saturation=float(args.evidence_mass_saturation),
        min_evidence_mass=float(args.min_evidence_mass),
        state_emission_reliability=float(args.state_emission_reliability),
        inactive_to_active_prior=float(args.inactive_to_active_prior),
        active_to_inactive_prior=float(args.active_to_inactive_prior),
        initial_active_probability=float(args.initial_active_probability),
        filter_chunk_size=int(args.filter_chunk_size),
        lifecycle_controller="view_consistent",
        open_probability=0.6,
        close_probability=0.4,
        transition_confirmation_views=int(args.transition_confirmation_views),
        min_transition_bayes_factor=float(args.min_transition_bayes_factor),
        min_transition_evidence_strength=float(args.min_transition_evidence_strength),
        max_states=int(args.max_states),
        # Used only by the shared detector/controller dataclass.  This runner's
        # direct optimizer owns the actual six-parameter list above.
        thaw_parameters=("dc",),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=False,
        evaluation_threshold=float(args.evaluation_threshold),
        seed=int(args.seed),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import (
        file_checksum,
        oscd_positive_sparsity_loss,
        seed_everything,
    )
    from gaussian_renderer import render_change_temporal
    from scene import GaussianModel
    from temporal import PersistentGaussianLifespanModel
    from temporal.change_evidence import accumulate_change_evidence
    from temporal.masked_optimizer import MaskedRowAdam

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.scope not in SCOPES:
        raise ValueError("unsupported independent scope")
    if args.max_frames is not None and args.max_frames > SCOPE_MAX_FRAMES[args.scope]:
        raise ValueError("max_frames exceeds the selected independent scope")
    config = _config(args)
    seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records, _ = build_causal_records(args.source_path)
    frame_limit = int(args.max_frames or SCOPE_MAX_FRAMES[args.scope])
    records = select_scope_records(all_records, scope=args.scope, max_frames=frame_limit)
    if [record.global_index for record in records] != list(range(len(records))):
        raise RuntimeError("independent records must be causally reindexed")
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)

    reference = GaussianModel(sh_degree=3, active_sh_degree=0)
    reference.load_ply_change(str(base_ply))
    mutable = GaussianModel(sh_degree=3, active_sh_degree=0)
    mutable.load_ply_change(str(base_ply))
    reference_before = base_snapshots(reference)
    initial_mutable = {
        "dc": mutable._features_dc.detach().cpu().clone(),
        "xyz": mutable._xyz.detach().cpu().clone(),
        "features_rest": mutable._features_rest.detach().cpu().clone(),
        "opacity": mutable._opacity.detach().cpu().clone(),
        "scaling": mutable._scaling.detach().cpu().clone(),
        "rotation": mutable._rotation.detach().cpu().clone(),
    }
    model = PersistentGaussianLifespanModel.from_gaussians(
        mutable, max_states=config.max_states
    )
    model.reset_all_lifespans_closed()
    optimizer = MaskedRowAdam(
        dict(model.persistent_parameter_items()),
        thaw_names=DIRECT_PARAMETER_NAMES,
        lrs={
            "dc": float(args.dc_lr),
            "xyz": float(args.xyz_lr),
            "features_rest": float(args.features_rest_lr),
            "opacity": float(args.opacity_lr),
            "scaling": float(args.scaling_lr),
            "rotation": float(args.rotation_lr),
        },
        eps=float(args.adam_eps),
    )
    tracker = make_filter(
        model.state_valid.shape[0],
        config,
        device=reference._xyz.device,
        dtype=reference._xyz.dtype,
    )
    controller = make_controller(model, config)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device=reference._xyz.device, dtype=reference._xyz.dtype)
    closed_audit = PersistentClosedRowAudit()

    predictions: list[np.ndarray] = []
    pre_predictions: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    lifecycle_events: list[LifecycleEvent] = []
    inactive_gradient_violations = 0
    inactive_gradient_max_abs = 0.0
    max_features_rest_gradient = 0.0

    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        evidence = accumulate_change_evidence(
            view,
            reference,
            pipe,
            background,
            view.candidate_map,
            cue_mode="binary",
            cue_threshold=config.bayes_cue_threshold,
            count_mode="capped",
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
        )
        binary = update_binary_lifecycle_chunks(
            tracker,
            controller,
            model,
            optimizer,
            evidence.delta_a,
            evidence.delta_b,
            evidence.total_mass,
            timestamp=timestamp,
            min_evidence_mass=config.min_evidence_mass,
            chunk_size=config.filter_chunk_size,
            closed_audit=closed_audit,
        )
        frame_events = binary["events"]
        lifecycle_events.extend(frame_events)
        reopened_rows = [
            event.gaussian_index
            for event in frame_events
            if event.action == "OPEN" and event.new_current_slot > 0
        ]
        if reopened_rows:
            closed_audit.release_reopened(
                model,
                optimizer,
                torch.tensor(reopened_rows, device=reference._xyz.device),
            )

        optimizer.zero_grad(set_to_none=True)
        pre_package = render_change_temporal(
            view, model, pipe, background, timestamp=float(timestamp)
        )
        pre_prediction, _ = _prediction_from_package(
            pre_package, config.evaluation_threshold
        )
        package = pre_package
        active = model.current_state_index >= 0
        selected_any = torch.zeros_like(active)
        for update_index in range(config.updates_per_frame):
            selected = active & (package["radii"].detach() > 0)
            if not bool(selected.any()):
                break
            selected_any |= selected
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = oscd_positive_sparsity_loss(
                view.training_target, package["render"]
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at timestamp {timestamp}")
            loss.backward()
            rest_gradient = model.features_rest.grad
            if rest_gradient is not None and rest_gradient.numel():
                max_features_rest_gradient = max(
                    max_features_rest_gradient,
                    float(rest_gradient.detach().abs().max().item()),
                )
            if update_index == 0:
                audit = _inactive_gradient_audit(model, selected)
                inactive_gradient_violations += int(audit["count"])
                inactive_gradient_max_abs = max(
                    inactive_gradient_max_abs, float(audit["max_abs"])
                )
            optimizer.step(selected)
            package = render_change_temporal(
                view, model, pipe, background, timestamp=float(timestamp)
            )

        post_prediction, _ = _prediction_from_package(
            package, config.evaluation_threshold
        )
        pre_predictions.append(pre_prediction)
        predictions.append(post_prediction)
        counts = binary["action_counts"]
        frame_rows.append(
            {
                "timestamp": timestamp,
                "frame": record.name,
                "scope": args.scope,
                "observed_gaussian_count": int(binary["observed_count"]),
                "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
                "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
                "open_count": int(counts["OPEN"]),
                "keep_count": int(counts["KEEP"]),
                "close_count": int(counts["CLOSE"]),
                "none_count": int(counts["NONE"]),
                "uncertain_count": int(counts.get("UNCERTAIN", counts.get("HOLD", 0))),
                "reopen_count": int(len(reopened_rows)),
                "p_active_stats": binary["p_active_stats"],
                "p_flip_stats": binary["p_flip_stats"],
                "p_01_stats": binary["p_01_stats"],
                "p_10_stats": binary["p_10_stats"],
                "q_stats": binary["q_stats"],
                "active_gaussian_count": int(active.sum().item()),
                "optimizer_selected_active_visible_rows": int(selected_any.sum().item()),
                "pre_predicted_positive_fraction": float(pre_prediction.mean()),
                "post_predicted_positive_fraction": float(post_prediction.mean()),
                "frame_runtime_seconds": time.time() - frame_started,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "pre_iou": None,
                "pre_f1": None,
                "iou": None,
                "f1": None,
                "precision": None,
                "recall": None,
            }
        )

    metrics = evaluate_after_inference(
        args.source_path,
        records,
        pre_predictions,
        predictions,
        frame_rows,
    )
    closed_result = closed_audit.verify(model, optimizer)
    reference_result = base_drift(reference_before, reference)
    mutable_result = _parameter_drift(initial_mutable, model)
    if not reference_result["bitwise_equal"]:
        raise RuntimeError(f"immutable detector reference drifted: {reference_result}")
    if not closed_result["passed"]:
        raise RuntimeError(f"persistent rows drifted while CLOSED: {closed_result}")
    if inactive_gradient_violations:
        raise RuntimeError(
            f"inactive gradients detected: {inactive_gradient_violations}"
        )
    if not model.validate_lifecycle():
        raise RuntimeError("lifecycle validation failed")
    event_diag = event_diagnostics(lifecycle_events)
    if event_diag["active_to_active_false_split_count"] or event_diag["reused_slot_violations"]:
        raise RuntimeError(f"lifecycle invariant failed: {event_diag}")

    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_persistent_gaussian_lifespan.py",
        "contract": "immutable_detector_lifespan_visibility_persistent_direct_rchange_bank",
        "scope": SCOPE_LABELS[args.scope],
        "scope_key": args.scope,
        "frames": len(records),
        "run_config": asdict(config),
        "detector": {
            "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
            "evidence": "immutable-reference alpha-T capped pseudo-counts",
            "controller": "view-consistent K-view/Bayes-factor confirmation",
            "representation_feedback": False,
        },
        "representation": {
            "parameter_storage": "one persistent direct raw parameter set per Gaussian",
            "parameters": list(DIRECT_PARAMETER_NAMES),
            "state_local_trainable_parameters": False,
            "close_behavior": "opacity-gated hidden; values and Adam moments preserved",
            "reopen_behavior": "new interval slot; same values and Adam moments resumed",
            "optimizer_mask": "currently OPEN and visible in current view",
            "features_rest_note": "render_change uses SH degree 0; retained as a direct parameter but expected gradient is zero",
        },
        "metrics": metrics,
        **event_diag,
        **same_scene_repeated_transition_diagnostics(lifecycle_events, ()),
        "posterior_diagnostics": posterior_run_diagnostics(frame_rows),
        "final_active_gs": int((model.current_state_index >= 0).sum().item()),
        "immutable_reference_drift": reference_result,
        "mutable_parameter_drift": mutable_result,
        "closed_interval_persistence_audit": closed_result,
        "inactive_gradient_violations": int(inactive_gradient_violations),
        "inactive_gradient_max_abs": float(inactive_gradient_max_abs),
        "features_rest_gradient_max_abs": float(max_features_rest_gradient),
        "gt_used_in_causal_loop": False,
        "gt_loaded_after_inference_only": True,
        "manual_boundaries_used": False,
        "fixed_topology": True,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cue_cache_metadata": cue_metadata,
        "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
        "run_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    with (args.output_dir / "lifecycle_events.jsonl").open("w", encoding="utf-8") as file:
        for event in lifecycle_events:
            file.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    if not args.skip_checkpoint:
        torch.save(
            _cpu_tree(
                {
                    "summary": summary,
                    "model_state": model.state_dict(),
                    "filter_state": tracker.state_dict(),
                    "controller_state": controller.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
            ),
            args.output_dir / "checkpoint.pt",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS))
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--bayes-cue-threshold", type=float, default=0.5)
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=nonnegative_float, default=1e-6)
    parser.add_argument("--state-emission-reliability", type=float, default=0.9)
    parser.add_argument("--inactive-to-active-prior", type=float, default=0.01)
    parser.add_argument("--active-to-inactive-prior", type=float, default=0.01)
    parser.add_argument("--initial-active-probability", type=float, default=0.5)
    parser.add_argument("--filter-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--transition-confirmation-views", type=positive_int, default=3)
    parser.add_argument("--min-transition-bayes-factor", type=nonnegative_float, default=3.0)
    parser.add_argument("--min-transition-evidence-strength", type=nonnegative_float, default=1e-6)
    parser.add_argument("--max-states", type=positive_int, default=16)
    parser.add_argument("--evaluation-threshold", type=float, default=0.5)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--features-rest-lr", type=nonnegative_float, default=0.000125)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=1e-15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "scope": summary["scope_key"],
                "mIoU": summary["metrics"]["mean_frame_iou"],
                "F1": summary["metrics"]["mean_frame_f1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
