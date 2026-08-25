"""Causal binary lifespans with active all-geometry and K-view density control.

The immutable reference prefix drives alpha-T Bayesian observations.  A second
frozen render-anchor copy owns the temporal sidecar; only OPEN row-slot pairs
visible in the current training view are optimized.  FastGS-style gradient AND
multi-view cue importance may append residual children, while cue-consistency
pruning is restricted to currently OPEN residual children.  Reference-prefix
rows and CLOSED historical rows are never pruned.

This is an independent ablation.  The BOCD, direct-binary, and fixed-topology
production runners remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import time

import numpy as np
import torch

from experiments.run_online_binary_state_lifespan_thaw import (
    ExactClosedPairArchive,
    RunConfig,
    _audit_new_open_slots,
    _inactive_gradient_audit,
    _prediction_from_package,
    base_checksums,
    base_drift,
    base_snapshots,
    combined_checksum,
    current_pair_mask,
    event_diagnostics,
    evaluate_after_inference,
    make_controller,
    make_filter,
    make_optimizer,
    posterior_run_diagnostics,
    same_scene_repeated_transition_diagnostics,
    update_binary_lifecycle_chunks,
)
from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL
from experiments.run_ref_sc1_change_cue_density import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    DEFAULT_SOURCE,
    SCOPE_LABELS,
    SCOPE_MAX_FRAMES,
    causal_random_view_indices,
    reference_scene_extent,
    select_scope_records,
)
from temporal.active_density_topology import (
    TemporalDensityResult,
    TemporalTopologyManager,
    compute_temporal_multiview_score,
)


CONDITIONS = ("baseline", "active_cue_vcd_vcp")
SCOPES = ("scene_change1", "scene_change2", "scene_change3")


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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _prefix_snapshot(base: Any, count: int) -> dict[str, torch.Tensor]:
    return {
        name: getattr(base, name).detach()[:count].cpu().clone()
        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        )
    }


def _prefix_drift(before: dict[str, torch.Tensor], base: Any) -> dict[str, Any]:
    per: dict[str, float] = {}
    exact = True
    maximum = 0.0
    for name, expected in before.items():
        actual = getattr(base, name).detach()[: expected.shape[0]].cpu()
        same = torch.equal(actual, expected)
        exact &= same
        difference = (
            float((actual - expected).abs().max().item()) if expected.numel() else 0.0
        )
        per[name] = difference
        maximum = max(maximum, difference)
    return {"bitwise_equal": bool(exact), "max_abs": maximum, "per_tensor_max_abs": per}


def _quantiles(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().flatten().float()
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"count": 0, "mean": None, "q05": None, "q50": None, "q95": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "q05": float(torch.quantile(values, 0.05).item()),
        "q50": float(torch.quantile(values, 0.50).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def _run_config(args: argparse.Namespace) -> RunConfig:
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
        thaw_parameters=("dc", "xyz", "opacity", "scaling", "rotation"),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=False,
        evaluation_threshold=float(args.evaluation_threshold),
        seed=int(args.seed),
    )


def _density_event_row(
    result: TemporalDensityResult,
    score: Any,
    records: Sequence[Any],
    timestamp: int,
) -> dict[str, Any]:
    observed = score.visible_view_count > 0
    return {
        "timestamp": int(timestamp),
        "selected_view_indices": list(score.view_indices),
        "selected_view_names": [records[index].name for index in score.view_indices],
        "selected_view_count": len(score.view_indices),
        "future_view_access_count": int(sum(index > timestamp for index in score.view_indices)),
        "initial_gaussian_count": int(result.initial_count),
        "final_gaussian_count": int(result.final_count),
        "importance_candidate_count": int(result.importance_count),
        "clone_child_count": int(result.clone_count),
        "split_source_count": int(result.split_source_count),
        "split_child_count": int(result.split_child_count),
        "vcp_pruned_count": int(result.vcp_pruned_count),
        "split_residual_removed_count": int(result.split_residual_removed_count),
        "removed_source_count": int(result.removed_count),
        "importance": _quantiles(score.importance_score),
        "change_ratio": _quantiles(score.change_ratio[observed]),
        "visible_view_count": _quantiles(score.visible_view_count.float()),
        "support_view_count": _quantiles(score.support_view_count.float()),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("unknown condition")
    if args.scope not in SCOPES:
        raise ValueError("only independent SC1/SC2/SC3 scopes are supported")
    if args.max_frames is not None and args.max_frames > SCOPE_MAX_FRAMES[args.scope]:
        raise ValueError("max_frames exceeds the selected scope")
    if not 0 <= args.densify_update_index < args.updates_per_frame:
        raise ValueError("densify_update_index must be inside the update schedule")
    if args.min_densify_views > args.k_views or args.min_prune_views > args.k_views:
        raise ValueError("density view thresholds cannot exceed K")
    if args.max_prune_support_views < 0:
        raise ValueError("max prune support must be nonnegative")
    if args.max_prune_support_views > args.min_prune_views:
        raise ValueError("max prune support cannot exceed required visible views")


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
    from temporal import TemporalGeometryChangeModel
    from temporal.change_evidence import accumulate_change_evidence
    from temporal.masked_optimizer import active_visible_pair_mask

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = _run_config(args)
    seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records, _ = build_causal_records(args.source_path)
    max_frames = int(args.max_frames or SCOPE_MAX_FRAMES[args.scope])
    records = select_scope_records(all_records, scope=args.scope, max_frames=max_frames)
    if [record.global_index for record in records] != list(range(len(records))):
        raise RuntimeError("independent scope was not reindexed causally")
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)

    reference = GaussianModel(sh_degree=3, active_sh_degree=0)
    reference.load_ply_change(str(base_ply))
    render_anchor = GaussianModel(sh_degree=3, active_sh_degree=0)
    render_anchor.load_ply_change(str(base_ply))
    immutable_count = int(reference.get_xyz.shape[0])
    if int(render_anchor.get_xyz.shape[0]) != immutable_count:
        raise RuntimeError("reference and render-anchor topology differ")
    reference_before = base_snapshots(reference)
    reference_checksum = combined_checksum(base_checksums(reference))
    render_prefix_before = _prefix_snapshot(render_anchor, immutable_count)

    model = TemporalGeometryChangeModel.from_gaussians(
        render_anchor, max_states=config.max_states
    )
    model.reset_all_lifespans_closed()
    optimizer = make_optimizer(model, config, args)
    if optimizer is None:
        raise RuntimeError("all-geometry optimizer was not constructed")
    topology = TemporalTopologyManager(model, optimizer, immutable_count)
    tracker = make_filter(
        immutable_count,
        config,
        device=reference._xyz.device,
        dtype=reference._xyz.dtype,
    )
    controller = make_controller(model, config)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device=reference._xyz.device, dtype=reference._xyz.dtype)
    extent = reference_scene_extent(args.source_path / "reference_reconstruction/cameras.json")
    closed_prefix_audit = ExactClosedPairArchive()

    processed_views: list[Any] = []
    predictions: list[np.ndarray] = []
    pre_predictions: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    lifecycle_events: list[Any] = []
    density_events: list[dict[str, Any]] = []
    total_clone_children = 0
    total_split_children = 0
    total_vcp_pruned = 0
    total_split_residual_removed = 0
    total_closed_descendants = 0
    inactive_gradient_violations = 0
    inactive_gradient_max_abs = 0.0
    open_audit_totals = {
        "zero_initialized_parameter_violations": 0,
        "zero_initialized_optimizer_state_violations": 0,
        "allocation_contract_violations": 0,
    }

    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        processed_views.append(view)

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
            closed_audit=closed_prefix_audit,
        )
        frame_events = binary["events"]
        lifecycle_events.extend(frame_events)
        open_audit = _audit_new_open_slots(model, optimizer, frame_events)
        for key, value in open_audit.items():
            open_audit_totals[key] += int(value)
        close_events = [event for event in frame_events if event.action == "CLOSE"]
        if close_events:
            total_closed_descendants += topology.close_descendants(
                torch.tensor(
                    [event.gaussian_index for event in close_events],
                    device=reference._xyz.device,
                    dtype=torch.long,
                ),
                torch.tensor(
                    [event.old_slot for event in close_events],
                    device=reference._xyz.device,
                    dtype=torch.long,
                ),
                timestamp=timestamp,
            )

        active_pairs = current_pair_mask(model)
        optimizer.zero_grad(set_to_none=True)
        pre_package = render_change_temporal(
            view, model, pipe, background, timestamp=float(timestamp)
        )
        pre_prediction, _ = _prediction_from_package(
            pre_package, config.evaluation_threshold
        )
        package = pre_package
        max_selected_rows = 0
        frame_density: dict[str, Any] | None = None
        for update_index in range(config.updates_per_frame):
            active_pairs = current_pair_mask(model)
            update_pairs = active_visible_pair_mask(active_pairs, package["radii"])
            selected_rows = int(update_pairs.any(dim=1).sum().item())
            max_selected_rows = max(max_selected_rows, selected_rows)
            if not bool(update_pairs.any()):
                break
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = oscd_positive_sparsity_loss(
                view.training_target, package["render"]
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at timestamp {timestamp}")
            loss.backward()
            if update_index == 0:
                audit = _inactive_gradient_audit(model, update_pairs)
                inactive_gradient_violations += int(audit["count"])
                inactive_gradient_max_abs = max(
                    inactive_gradient_max_abs, float(audit["max_abs"])
                )
            if args.condition == "active_cue_vcd_vcp":
                topology.add_gradient_stats(
                    package["viewspace_points"],
                    package["radii"],
                    active_pairs.any(dim=1),
                )
            optimizer.step(update_pairs)

            if (
                args.condition == "active_cue_vcd_vcp"
                and update_index == int(args.densify_update_index)
            ):
                selected_indices = causal_random_view_indices(
                    timestamp, int(args.k_views), seed=int(args.seed)
                )
                if max(selected_indices) > timestamp:
                    raise RuntimeError("future density view accessed")
                score = compute_temporal_multiview_score(
                    processed_views,
                    selected_indices,
                    model,
                    pipe,
                    background,
                    current_timestamp=timestamp,
                    cue_scale=float(args.density_cue_scale),
                    min_mass=float(args.density_min_mass),
                    support_threshold=float(args.density_support_threshold),
                )
                density = topology.apply_density_control(
                    score,
                    timestamp=timestamp,
                    scene_extent=extent,
                    importance_threshold=float(args.importance_threshold),
                    grad_threshold=float(args.fastgs_grad_threshold),
                    grad_abs_threshold=float(args.fastgs_grad_abs_threshold),
                    dense_fraction=float(args.fastgs_dense_fraction),
                    min_densify_views=int(args.min_densify_views),
                    min_prune_views=int(args.min_prune_views),
                    max_prune_support_views=int(args.max_prune_support_views),
                    prune_grace_frames=int(args.prune_grace_frames),
                )
                total_clone_children += density.clone_count
                total_split_children += density.split_child_count
                total_vcp_pruned += density.vcp_pruned_count
                total_split_residual_removed += density.split_residual_removed_count
                frame_density = _density_event_row(
                    density, score, records, timestamp
                )
                density_events.append(frame_density)
                if topology.count - immutable_count > int(args.max_residual_gaussians):
                    raise RuntimeError(
                        "residual Gaussian safety limit exceeded: "
                        f"{topology.count - immutable_count} > {args.max_residual_gaussians}"
                    )
            package = render_change_temporal(
                view, model, pipe, background, timestamp=float(timestamp)
            )

        post_prediction, _ = _prediction_from_package(
            package, config.evaluation_threshold
        )
        pre_predictions.append(pre_prediction)
        predictions.append(post_prediction)
        active_reference = int(
            (model.current_state_index[:immutable_count] >= 0).sum().item()
        )
        active_residual = int(
            (model.current_state_index[immutable_count:] >= 0).sum().item()
        )
        density_values = frame_density or {
            "selected_view_count": 0,
            "importance_candidate_count": 0,
            "clone_child_count": 0,
            "split_child_count": 0,
            "vcp_pruned_count": 0,
            "split_residual_removed_count": 0,
            "removed_source_count": 0,
            "future_view_access_count": 0,
        }
        counts = binary["action_counts"]
        frame_rows.append(
            {
                "timestamp": timestamp,
                "frame": record.name,
                "scope": args.scope,
                "condition": args.condition,
                "observed_gaussian_count": int(binary["observed_count"]),
                "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
                "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
                "open_count": int(counts["OPEN"]),
                "keep_count": int(counts["KEEP"]),
                "close_count": int(counts["CLOSE"]),
                "none_count": int(counts["NONE"]),
                "uncertain_count": int(counts.get("UNCERTAIN", counts.get("HOLD", 0))),
                "reopen_count": int(
                    sum(
                        event.action == "OPEN" and event.new_current_slot > 0
                        for event in frame_events
                    )
                ),
                "p_active_stats": binary["p_active_stats"],
                "p_flip_stats": binary["p_flip_stats"],
                "p_01_stats": binary["p_01_stats"],
                "p_10_stats": binary["p_10_stats"],
                "q_stats": binary["q_stats"],
                "active_reference_gaussians": active_reference,
                "active_residual_gaussians": active_residual,
                "total_render_gaussians": topology.count,
                "residual_gaussians": topology.count - immutable_count,
                "optimizer_selected_active_visible_rows": max_selected_rows,
                "density_selected_view_count": density_values["selected_view_count"],
                "importance_candidate_count": density_values["importance_candidate_count"],
                "clone_child_count": density_values["clone_child_count"],
                "split_child_count": density_values["split_child_count"],
                "vcp_pruned_count": density_values["vcp_pruned_count"],
                "split_residual_removed_count": density_values["split_residual_removed_count"],
                "removed_source_count": density_values["removed_source_count"],
                "future_view_access_count": density_values["future_view_access_count"],
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
    prefix_closed_audit = closed_prefix_audit.verify(model, optimizer)
    residual_closed_audit = topology.verify_closed_residuals()
    topology.validate()
    reference_audit = base_drift(reference_before, reference)
    render_prefix_audit = _prefix_drift(render_prefix_before, render_anchor)
    if not reference_audit["bitwise_equal"]:
        raise RuntimeError(f"immutable detector reference drifted: {reference_audit}")
    if not render_prefix_audit["bitwise_equal"]:
        raise RuntimeError(f"render-anchor prefix drifted: {render_prefix_audit}")
    if not prefix_closed_audit["passed"] or not residual_closed_audit["passed"]:
        raise RuntimeError(
            "closed temporal parameters drifted: "
            f"prefix={prefix_closed_audit}, residual={residual_closed_audit}"
        )
    if inactive_gradient_violations:
        raise RuntimeError(
            f"inactive gradients detected: {inactive_gradient_violations}"
        )
    if any(open_audit_totals.values()):
        raise RuntimeError(f"OPEN initialization audit failed: {open_audit_totals}")
    event_diag = event_diagnostics(lifecycle_events)
    if event_diag["active_to_active_false_split_count"] or event_diag["reused_slot_violations"]:
        raise RuntimeError(f"lifecycle invariant failed: {event_diag}")

    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_binary_lifespan_active_density.py",
        "contract": "immutable_reference_binary_lifespan_active_geometry_dynamic_residual_density",
        "condition": args.condition,
        "scope": SCOPE_LABELS[args.scope],
        "scope_key": args.scope,
        "frames": len(records),
        "run_config": asdict(config),
        "detector": {
            "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
            "cue": "binary candidate_map > threshold",
            "evidence": "immutable-reference alpha-T capped pseudo-counts",
            "controller": "view-consistent committed-state K-view/Bayes-factor confirmation",
            "representation_feedback": False,
        },
        "representation": {
            "parameters": ["dc", "xyz", "opacity", "scaling", "rotation"],
            "optimizer_mask": "current OPEN row-slot AND current-view radius > 0",
            "reference_prefix_rows_prunable": False,
            "closed_rows_prunable": False,
            "active_residual_rows_prunable": args.condition == "active_cue_vcd_vcp",
        },
        "density": {
            "enabled": args.condition == "active_cue_vcd_vcp",
            "k": int(args.k_views),
            "view_sampling": "current plus K-1 deterministic-random already processed views",
            "cue": "soft raw ref-inf O-SCD pixel+SAM candidate cue",
            "importance_equation": "sum_j sum_p alpha_i T_i C_j / K",
            "densify": "OPEN AND gradient qualifier AND multi-view importance",
            "prune": "OPEN residual only AND age grace AND >=min visible views AND <=max supporting views",
            "fastgs_difference": "fixed reference-prefix split sources are retained; two residual children are appended",
            "future_view_access_count": int(
                sum(event["future_view_access_count"] for event in density_events)
            ),
        },
        "metrics": metrics,
        "initial_gaussian_count": immutable_count,
        "final_gaussian_count": topology.count,
        "final_residual_gaussian_count": topology.count - immutable_count,
        "total_clone_children": int(total_clone_children),
        "total_split_children": int(total_split_children),
        "total_vcp_pruned": int(total_vcp_pruned),
        "total_split_residual_removed": int(total_split_residual_removed),
        "total_closed_descendants": int(total_closed_descendants),
        **event_diag,
        **same_scene_repeated_transition_diagnostics(lifecycle_events, ()),
        "posterior_diagnostics": posterior_run_diagnostics(frame_rows),
        "reference_checksum": reference_checksum,
        "immutable_reference_drift": reference_audit,
        "render_anchor_prefix_drift": render_prefix_audit,
        "closed_prefix_slot_audit": prefix_closed_audit,
        "closed_residual_slot_audit": residual_closed_audit,
        "inactive_gradient_violations": int(inactive_gradient_violations),
        "inactive_gradient_max_abs": float(inactive_gradient_max_abs),
        "open_zero_initialization_audit": open_audit_totals,
        "topology_integrity": bool(topology.validate()),
        "gt_used_in_causal_loop": False,
        "gt_loaded_after_inference_only": True,
        "manual_boundaries_used": False,
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
    with (args.output_dir / "density_events.jsonl").open("w", encoding="utf-8") as file:
        for event in density_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    with (args.output_dir / "residual_lineage.jsonl").open("w", encoding="utf-8") as file:
        for event in topology.lineage_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    if not args.skip_checkpoint:
        torch.save(
            _cpu_tree(
                {
                    "summary": summary,
                    "temporal_state": model.state_dict(),
                    "render_anchor": {
                        name: getattr(render_anchor, name)
                        for name in (
                            "_xyz",
                            "_features_dc",
                            "_features_rest",
                            "_opacity",
                            "_scaling",
                            "_rotation",
                        )
                    },
                    "filter_state": tracker.state_dict(),
                    "controller_state": controller.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "topology_state": topology.state_dict(),
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
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--densify-update-index", type=int, default=4)
    parser.add_argument("--k-views", type=positive_int, default=10)
    parser.add_argument("--density-cue-scale", type=float, default=1.0)
    parser.add_argument("--density-min-mass", type=nonnegative_float, default=1e-6)
    parser.add_argument("--density-support-threshold", type=float, default=0.5)
    parser.add_argument("--importance-threshold", type=nonnegative_float, default=5.0)
    parser.add_argument("--fastgs-grad-threshold", type=nonnegative_float, default=2e-4)
    parser.add_argument("--fastgs-grad-abs-threshold", type=nonnegative_float, default=1.2e-3)
    parser.add_argument("--fastgs-dense-fraction", type=float, default=1e-3)
    parser.add_argument("--min-densify-views", type=positive_int, default=3)
    parser.add_argument("--min-prune-views", type=positive_int, default=8)
    parser.add_argument("--max-prune-support-views", type=int, default=2)
    parser.add_argument("--prune-grace-frames", type=positive_int, default=10)
    parser.add_argument("--max-residual-gaussians", type=positive_int, default=100_000)
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
                "condition": summary["condition"],
                "mIoU": summary["metrics"]["mean_frame_iou"],
                "final_residual_gaussians": summary["final_residual_gaussian_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
