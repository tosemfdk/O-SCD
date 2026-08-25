"""Equal-status dynamic ``R_change`` with active-only O-SCD density control.

The current mutable Gaussian bank is both detector support and representation:

``mutable alpha-T evidence -> binary ACTIVE/INACTIVE -> selective training``

The default remains OPEN-only rendering/training.  Two explicit occlusion
ablations can additionally keep NEVER_OPEN rows in the compositor: fixed
zero-DC occluders, or DC/opacity-plastic occluders with frozen geometry.

At local update four of a 16-update online schedule, ACTIVE rows alone may be
cloned/split by the original O-SCD screen-space gradient rule.  The same event
also applies the explicitly requested ACTIVE-only opacity/size pruning.  There
is no immutable prefix, root/residual distinction, FastGS VCD/VCP, or K-view
importance window.  New rows copy the source Bayesian and lifespan state once,
then evolve independently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import time

import numpy as np
import torch

from experiments.run_online_bayesian_lifespan_thaw import BASE_PLY_REL
from experiments.run_online_binary_state_lifespan_thaw import (
    LifecycleEvent,
    RunConfig,
    _prediction_from_package,
    event_diagnostics,
    evaluate_after_inference,
    make_controller,
    make_filter,
    posterior_run_diagnostics,
    same_scene_repeated_transition_diagnostics,
    update_binary_lifecycle_chunks,
)
from experiments.run_online_persistent_gaussian_lifespan import (
    DIRECT_PARAMETER_NAMES,
    _cpu_tree,
    _write_csv,
)
from experiments.run_ref_sc1_change_cue_density import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    DEFAULT_SOURCE,
    SCOPE_LABELS,
    SCOPE_MAX_FRAMES,
    reference_scene_extent,
    select_scope_records,
)


SCOPES = ("scene_change1", "scene_change2", "scene_change3", "continuous")
DENSITY_POLICIES = ("none", "active_oscd")
RENDER_SUPPORT_MODES = (
    "open_only",
    "open_or_never_open",
    "open_or_never_open_dc_opacity",
)


def _includes_never_open(mode: str) -> bool:
    return mode in {"open_or_never_open", "open_or_never_open_dc_opacity"}


def _trains_never_open_appearance(mode: str) -> bool:
    return mode == "open_or_never_open_dc_opacity"


def _optimizer_row_masks(
    mode: str,
    *,
    active_visible: torch.Tensor,
    never_open_visible: torch.Tensor,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if not _trains_never_open_appearance(mode):
        return active_visible
    appearance = active_visible | never_open_visible
    return {
        name: appearance if name in {"dc", "opacity"} else active_visible
        for name in DIRECT_PARAMETER_NAMES
    }


def _gradient_audit(
    model: Any,
    masks: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, float | int]:
    violations = 0
    maximum = 0.0
    for name, parameter in model.persistent_parameter_items():
        if parameter.grad is None:
            continue
        selected = masks if isinstance(masks, torch.Tensor) else masks[name]
        gradient = parameter.grad.detach().reshape(parameter.shape[0], -1)
        outside = gradient[~selected]
        nonzero = int(torch.count_nonzero(outside).item())
        violations += nonzero
        if nonzero:
            maximum = max(maximum, float(outside.abs().max().item()))
    return {"count": int(violations), "max_abs": float(maximum)}


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


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


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
        thaw_parameters=("dc",),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=False,
        evaluation_threshold=float(args.evaluation_threshold),
        seed=int(args.seed),
    )


def _optimizer_row_snapshot(
    model: Any, optimizer: Any, rows: torch.Tensor
) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for name, parameter in model.persistent_parameter_items():
        values = {"parameter": parameter.detach()[rows].cpu().clone()}
        for state_name, state_value in optimizer.state.get(parameter, {}).items():
            if (
                isinstance(state_value, torch.Tensor)
                and state_value.ndim >= 1
                and state_value.shape[0] == parameter.shape[0]
            ):
                values[state_name] = state_value.detach()[rows].cpu().clone()
        snapshot[name] = values
    return snapshot


class StableClosedRowAudit:
    """Audit CLOSED rows by stable ID across unrelated topology mutations."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.entries: list[dict[str, Any]] = []
        self.total_closed_rows = 0
        self.reopened_rows = 0
        self.max_abs = 0.0
        self.missing_stable_ids = 0
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
                "stable_ids": self.manager.stable_id[rows].detach().cpu().clone(),
                "snapshot": _optimizer_row_snapshot(model, optimizer, rows),
            }
        )
        self.total_closed_rows += int(rows.numel())

    def _positions(self, stable_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.manager.stable_id
        query = stable_ids.to(device=current.device)
        positions = torch.searchsorted(current, query)
        valid = positions < current.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = current[positions[valid]] == query[valid]
        return positions, matched

    @torch.no_grad()
    def _compare(
        self,
        entry: dict[str, Any],
        selected: torch.Tensor,
    ) -> None:
        if selected.numel() == 0:
            return
        stable_ids = entry["stable_ids"][selected]
        rows, matched = self._positions(stable_ids)
        self.missing_stable_ids += int((~matched).sum().item())
        if not bool(matched.any()):
            return
        rows = rows[matched]
        selected = selected[matched.cpu()]
        parameters = dict(self.manager.model.persistent_parameter_items())
        for name, values in entry["snapshot"].items():
            parameter = parameters[name]
            for state_name, expected_all in values.items():
                expected = expected_all[selected].cpu()
                if state_name == "parameter":
                    actual = parameter.detach()[rows].cpu()
                else:
                    actual = self.manager.optimizer.state[parameter][state_name].detach()[rows].cpu()
                difference = (
                    float((actual - expected).abs().max().item())
                    if expected.numel()
                    else 0.0
                )
                key = f"{name}.{state_name}"
                self.per_state[key] = max(self.per_state.get(key, 0.0), difference)
                self.max_abs = max(self.max_abs, difference)

    @torch.no_grad()
    def release_reopened_rows(self, rows: torch.Tensor) -> None:
        rows = rows.detach().flatten().long().unique(sorted=True)
        if rows.numel() == 0:
            return
        reopened_ids = self.manager.stable_id[rows].detach().cpu()
        retained: list[dict[str, Any]] = []
        for entry in self.entries:
            selected_mask = torch.isin(entry["stable_ids"], reopened_ids)
            selected = torch.nonzero(selected_mask, as_tuple=False).flatten()
            self._compare(entry, selected)
            self.reopened_rows += int(selected.numel())
            keep = ~selected_mask
            if bool(keep.any()):
                entry["stable_ids"] = entry["stable_ids"][keep]
                for values in entry["snapshot"].values():
                    for state_name in tuple(values):
                        values[state_name] = values[state_name][keep]
                retained.append(entry)
        self.entries = retained

    @torch.no_grad()
    def verify(self) -> dict[str, Any]:
        for entry in self.entries:
            self._compare(
                entry,
                torch.arange(entry["stable_ids"].numel(), dtype=torch.long),
            )
        return {
            "passed": self.max_abs == 0.0 and self.missing_stable_ids == 0,
            "max_abs": float(self.max_abs),
            "total_closed_rows": int(self.total_closed_rows),
            "reopened_rows_verified_before_resume": int(self.reopened_rows),
            "remaining_closed_rows_verified_at_end": int(
                sum(entry["stable_ids"].numel() for entry in self.entries)
            ),
            "missing_stable_ids": int(self.missing_stable_ids),
            "per_parameter_and_optimizer_state_max_abs": dict(self.per_state),
        }


def _stable_lifecycle_events(
    events: Sequence[LifecycleEvent], stable_ids: torch.Tensor
) -> list[LifecycleEvent]:
    return [
        replace(event, gaussian_index=int(stable_ids[event.gaussian_index].item()))
        for event in events
    ]


def _sample_training_view(
    processed_views: Sequence[Any],
    rng: np.random.Generator,
    current_probability: float,
) -> tuple[Any, int]:
    if not processed_views:
        raise ValueError("processed view bank must be nonempty")
    if float(rng.random()) < float(current_probability):
        index = len(processed_views) - 1
    else:
        index = int(rng.integers(0, len(processed_views)))
    return processed_views[index], index


def _validate_args(args: argparse.Namespace) -> None:
    if args.scope not in SCOPES:
        raise ValueError("unknown independent/continuous ESCD scope")
    if args.density_policy not in DENSITY_POLICIES:
        raise ValueError("unknown density policy")
    if args.render_support_mode not in RENDER_SUPPORT_MODES:
        raise ValueError("unknown render support mode")
    if args.max_frames is not None and args.max_frames > SCOPE_MAX_FRAMES[args.scope]:
        raise ValueError("max_frames exceeds the selected independent scope")
    if not 0 <= args.densify_update_index < args.updates_per_frame:
        raise ValueError("densify_update_index must be inside the update schedule")


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
    from temporal import (
        DynamicGaussianTopologyManager,
        PersistentGaussianLifespanModel,
    )
    from temporal.change_evidence import accumulate_change_evidence
    from temporal.masked_optimizer import MaskedRowAdam

    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = _config(args)
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
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
    extent = reference_scene_extent(
        args.source_path / "reference_reconstruction/cameras.json"
    )

    mutable = GaussianModel(sh_degree=3, active_sh_degree=0)
    mutable.load_ply_change(str(base_ply))
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
        device=model.xyz.device,
        dtype=model.xyz.dtype,
    )
    controller = make_controller(model, config)
    topology = DynamicGaussianTopologyManager(
        model,
        optimizer,
        tracker,
        controller,
        percent_dense=float(args.percent_dense),
    )
    initial_count = topology.count
    closed_audit = StableClosedRowAudit(topology)
    pipe = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    background = torch.zeros(3, device=model.xyz.device, dtype=model.xyz.dtype)

    processed_views: list[Any] = []
    predictions: list[np.ndarray] = []
    pre_predictions: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    lifecycle_events: list[LifecycleEvent] = []
    density_rows: list[dict[str, Any]] = []
    inactive_gradient_violations = 0
    inactive_gradient_max_abs = 0.0
    max_features_rest_gradient = 0.0
    total_clones = 0
    total_split_sources = 0
    total_split_children = 0
    total_pruned = 0
    total_opacity_pruned = 0
    total_size_pruned = 0
    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        current_view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        processed_views.append(current_view)

        # Every current row is observable through its current raw geometry and
        # opacity, irrespective of lifespan opacity gating.  This intentionally
        # makes detector support follow the equal-status mutable topology.
        evidence = accumulate_change_evidence(
            current_view,
            model.base,
            pipe,
            background,
            current_view.candidate_map,
            cue_mode="binary",
            cue_threshold=config.bayes_cue_threshold,
            count_mode="capped",
            mass_saturation=config.evidence_mass_saturation,
            min_evidence_mass=config.min_evidence_mass,
        )
        stable_before_decision = topology.stable_id.detach().clone()
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
        raw_events = list(binary["events"])
        reopened_rows = torch.tensor(
            [
                event.gaussian_index
                for event in raw_events
                if event.action == "OPEN" and event.new_current_slot > 0
            ],
            device=model.xyz.device,
            dtype=torch.long,
        )
        if reopened_rows.numel():
            closed_audit.release_reopened_rows(reopened_rows)
        frame_events = _stable_lifecycle_events(raw_events, stable_before_decision)
        lifecycle_events.extend(frame_events)

        optimizer.zero_grad(set_to_none=True)
        pre_package = render_change_temporal(
            current_view,
            model,
            pipe,
            background,
            timestamp=float(timestamp),
            include_never_open_occluders=(
                _includes_never_open(args.render_support_mode)
            ),
            train_never_open_dc_opacity=_trains_never_open_appearance(
                args.render_support_mode
            ),
        )
        pre_prediction, _ = _prediction_from_package(
            pre_package, config.evaluation_threshold
        )
        pre_predictions.append(pre_prediction)

        selected_stable_ids: list[torch.Tensor] = []
        selected_never_open_stable_ids: list[torch.Tensor] = []
        sampled_view_indices: list[int] = []
        density_result = None
        density_event = {
            "timestamp": timestamp,
            "initial_gaussian_count": topology.count,
            "final_gaussian_count": topology.count,
            "clone_source_count": 0,
            "clone_child_count": 0,
            "split_source_count": 0,
            "split_child_count": 0,
            "split_source_removed_count": 0,
            "opacity_pruned_count": 0,
            "size_pruned_count": 0,
            "total_removed_count": 0,
        }
        for update_index in range(config.updates_per_frame):
            train_view, sampled_index = _sample_training_view(
                processed_views, rng, float(args.current_view_probability)
            )
            sampled_view_indices.append(sampled_index)
            package = render_change_temporal(
                train_view,
                model,
                pipe,
                background,
                timestamp=float(timestamp),
                include_never_open_occluders=(
                    _includes_never_open(args.render_support_mode)
                ),
                train_never_open_dc_opacity=_trains_never_open_appearance(
                    args.render_support_mode
                ),
            )
            active = topology.active_mask()
            visible = package["radii"].detach() > 0
            selected = active & visible
            never_open_selected = (model.num_states == 0) & visible
            optimizer_masks = _optimizer_row_masks(
                args.render_support_mode,
                active_visible=selected,
                never_open_visible=never_open_selected,
            )
            any_selected = selected | (
                never_open_selected
                if _trains_never_open_appearance(args.render_support_mode)
                else False
            )
            if bool(any_selected.any()):
                if bool(selected.any()):
                    selected_stable_ids.append(
                        topology.stable_id[selected].detach().clone()
                    )
                if (
                    _trains_never_open_appearance(args.render_support_mode)
                    and bool(never_open_selected.any())
                ):
                    selected_never_open_stable_ids.append(
                        topology.stable_id[never_open_selected].detach().clone()
                    )
                optimizer.zero_grad(set_to_none=True)
                loss, _parts = oscd_positive_sparsity_loss(
                    train_view.training_target, package["render"]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite loss at timestamp {timestamp}, update {update_index}"
                    )
                loss.backward()
                rest_gradient = model.features_rest.grad
                if rest_gradient is not None and rest_gradient.numel():
                    max_features_rest_gradient = max(
                        max_features_rest_gradient,
                        float(rest_gradient.detach().abs().max().item()),
                    )
                audit = _gradient_audit(model, optimizer_masks)
                inactive_gradient_violations += int(audit["count"])
                inactive_gradient_max_abs = max(
                    inactive_gradient_max_abs, float(audit["max_abs"])
                )
                topology.add_gradient_stats(
                    package["viewspace_points"], package["radii"], active
                )
                optimizer.step(optimizer_masks)

            if (
                args.density_policy == "active_oscd"
                and update_index == int(args.densify_update_index)
            ):
                density_result = topology.apply_active_oscd_density_control(
                    timestamp=timestamp,
                    scene_extent=extent,
                    grad_threshold=float(args.oscd_grad_threshold),
                    min_opacity=float(args.min_opacity),
                    max_screen_size=(
                        None
                        if float(args.max_screen_size) == 0.0
                        else float(args.max_screen_size)
                    ),
                )
                density_event.update(
                    {
                        "initial_gaussian_count": density_result.initial_count,
                        "final_gaussian_count": density_result.final_count,
                        "clone_source_count": density_result.clone_source_count,
                        "clone_child_count": density_result.clone_child_count,
                        "split_source_count": density_result.split_source_count,
                        "split_child_count": density_result.split_child_count,
                        "split_source_removed_count": density_result.split_source_removed_count,
                        "opacity_pruned_count": density_result.opacity_pruned_count,
                        "size_pruned_count": density_result.size_pruned_count,
                        "total_removed_count": density_result.total_removed_count,
                    }
                )
                total_clones += density_result.clone_child_count
                total_split_sources += density_result.split_source_count
                total_split_children += density_result.split_child_count
                total_pruned += density_result.total_removed_count
                total_opacity_pruned += density_result.opacity_pruned_count
                total_size_pruned += density_result.size_pruned_count

        density_event["sampled_training_view_indices"] = sampled_view_indices
        density_event["future_training_view_access_count"] = int(
            sum(index > timestamp for index in sampled_view_indices)
        )
        density_rows.append(density_event)
        if density_event["future_training_view_access_count"]:
            raise RuntimeError("online replay accessed a future view")

        post_package = render_change_temporal(
            current_view,
            model,
            pipe,
            background,
            timestamp=float(timestamp),
            include_never_open_occluders=(
                _includes_never_open(args.render_support_mode)
            ),
            train_never_open_dc_opacity=_trains_never_open_appearance(
                args.render_support_mode
            ),
        )
        post_prediction, _ = _prediction_from_package(
            post_package, config.evaluation_threshold
        )
        predictions.append(post_prediction)
        selected_count = (
            int(torch.unique(torch.cat(selected_stable_ids)).numel())
            if selected_stable_ids
            else 0
        )
        never_open_selected_count = (
            int(torch.unique(torch.cat(selected_never_open_stable_ids)).numel())
            if selected_never_open_stable_ids
            else 0
        )
        counts = binary["action_counts"]
        active_count = int(topology.active_mask().sum().item())
        frame_rows.append(
            {
                "timestamp": timestamp,
                "frame": record.name,
                "scope": args.scope,
                "gaussian_count": topology.count,
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
                        for event in raw_events
                    )
                ),
                "p_active_stats": binary["p_active_stats"],
                "p_flip_stats": binary["p_flip_stats"],
                "p_01_stats": binary["p_01_stats"],
                "p_10_stats": binary["p_10_stats"],
                "q_stats": binary["q_stats"],
                "active_gaussian_count": active_count,
                "optimizer_selected_active_visible_rows": selected_count,
                "optimizer_selected_never_open_visible_rows": never_open_selected_count,
                "clone_count": int(density_event["clone_child_count"]),
                "split_source_count": int(density_event["split_source_count"]),
                "split_child_count": int(density_event["split_child_count"]),
                "pruned_count": int(density_event["total_removed_count"]),
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
        args.source_path, records, pre_predictions, predictions, frame_rows
    )
    closed_result = closed_audit.verify()
    topology.validate()
    if not closed_result["passed"]:
        raise RuntimeError(f"CLOSED rows drifted under dynamic topology: {closed_result}")
    if inactive_gradient_violations:
        raise RuntimeError(
            f"inactive/off-view parameter gradients detected: {inactive_gradient_violations}"
        )
    event_diag = event_diagnostics(lifecycle_events)
    if event_diag["active_to_active_false_split_count"] or event_diag["reused_slot_violations"]:
        raise RuntimeError(f"lifespan invariant failed: {event_diag}")

    diagnostic_boundaries = (95, 199) if args.scope == "continuous" else ()
    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_dynamic_active_oscd_density.py",
        "contract": (
            "equal_status_mutable_rchange_never_open_appearance"
            if _trains_never_open_appearance(args.render_support_mode)
            else "equal_status_mutable_rchange_active_only_oscd_density"
        ),
        "scope": SCOPE_LABELS[args.scope],
        "scope_key": args.scope,
        "frames": len(records),
        "run_config": asdict(config),
        "detector": {
            "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
            "evidence": "current mutable-bank alpha-T capped pseudo-counts without lifespan opacity gating",
            "representation_feedback": True,
            "inactive_rows_observable": True,
            "new_row_state": "one-time copy of source posterior/count/timestamp and controller counters",
            "post_birth_behavior": "independent evidence and posterior updates",
        },
        "representation": {
            "gaussian_classes": "none; every row has equal mutable status",
            "render_support_mode": args.render_support_mode,
            "render_support_contract": (
                "OPEN union NEVER_OPEN; NEVER_OPEN DC/opacity trainable; CLOSED hidden"
                if _trains_never_open_appearance(args.render_support_mode)
                else (
                    "OPEN union fixed zero-DC NEVER_OPEN; CLOSED hidden"
                    if _includes_never_open(args.render_support_mode)
                    else "OPEN only"
                )
            ),
            "lifecycle_render_distinction": (
                "OPEN vs NEVER_OPEN vs CLOSED"
                if _includes_never_open(args.render_support_mode)
                else "OPEN vs non-OPEN"
            ),
            "parameters": list(DIRECT_PARAMETER_NAMES),
            "optimizer_mask": (
                "OPEN-visible all attributes; NEVER_OPEN-visible DC/opacity only"
                if _trains_never_open_appearance(args.render_support_mode)
                else "currently ACTIVE and visible in sampled causal training view"
            ),
            "closed_behavior": "hidden and exact parameter/Adam preservation",
            "features_rest_note": "SH degree 0 keeps features_rest gradient exactly zero",
        },
        "density_control": {
            "policy": args.density_policy,
            "updates_per_frame": int(args.updates_per_frame),
            "local_update_index": int(args.densify_update_index),
            "densification": "original online O-SCD gradient-only clone/split restricted to ACTIVE rows",
            "pruning_enabled": bool(
                float(args.min_opacity) > 0.0 or float(args.max_screen_size) > 0.0
            ),
            "pruning": (
                "requested extension: O-SCD opacity/size criteria restricted to ACTIVE rows"
                if float(args.min_opacity) > 0.0 or float(args.max_screen_size) > 0.0
                else "disabled; split sources are still replaced by their children"
            ),
            "online_oscd_pruning_source_note": "oscd.py 16-step online loop itself clone/splits but does not prune",
            "fastgs_vcd_vcp": False,
            "k_view_importance": None,
            "replay": "current with configured probability; otherwise uniform already-processed view",
            "initial_gaussian_count": int(initial_count),
            "final_gaussian_count": int(topology.count),
            "total_clone_children": int(total_clones),
            "total_split_sources": int(total_split_sources),
            "total_split_children": int(total_split_children),
            "total_removed": int(total_pruned),
            "total_opacity_pruned": int(total_opacity_pruned),
            "total_size_pruned": int(total_size_pruned),
        },
        "metrics": metrics,
        **event_diag,
        **same_scene_repeated_transition_diagnostics(
            lifecycle_events, diagnostic_boundaries
        ),
        "posterior_diagnostics": posterior_run_diagnostics(frame_rows),
        "final_active_gs": int(topology.active_mask().sum().item()),
        "closed_row_persistence_audit": closed_result,
        "inactive_gradient_violations": int(inactive_gradient_violations),
        "inactive_gradient_max_abs": float(inactive_gradient_max_abs),
        "features_rest_gradient_max_abs": float(max_features_rest_gradient),
        "topology_integrity": True,
        "gt_used_in_causal_loop": False,
        "gt_loaded_after_inference_only": True,
        "manual_boundaries_used": False,
        "manual_boundaries_used_for_posthoc_diagnostics_only": list(
            diagnostic_boundaries
        ),
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cue_cache_metadata": cue_metadata,
        "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
        "source_ply_sha256": file_checksum(base_ply),
        "run_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(args.output_dir / "density_events.csv", density_rows)
    with (args.output_dir / "lifecycle_events.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for event in lifecycle_events:
            file.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    with (args.output_dir / "topology_events.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for event in topology.lineage_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    if not args.skip_checkpoint:
        torch.save(
            _cpu_tree(
                {
                    "summary": summary,
                    "model_state": model.state_dict(),
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
    parser.add_argument(
        "--fixed-cameras-json", type=Path, default=Path(DEFAULT_FIXED_CAMERAS)
    )
    parser.add_argument("--cue-cache-root", type=Path, default=Path(DEFAULT_CUE_CACHE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--density-policy", choices=DENSITY_POLICIES, default="active_oscd")
    parser.add_argument(
        "--render-support-mode",
        choices=RENDER_SUPPORT_MODES,
        default="open_only",
    )
    parser.add_argument("--updates-per-frame", type=positive_int, default=16)
    parser.add_argument("--densify-update-index", type=int, default=4)
    parser.add_argument("--oscd-grad-threshold", type=nonnegative_float, default=0.001)
    parser.add_argument("--percent-dense", type=float, default=0.01)
    parser.add_argument("--min-opacity", type=probability, default=0.4)
    parser.add_argument("--max-screen-size", type=nonnegative_float, default=0.0)
    parser.add_argument("--current-view-probability", type=probability, default=0.33)
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
                "density_policy": args.density_policy,
                "mIoU": summary["metrics"]["mean_frame_iou"],
                "F1": summary["metrics"]["mean_frame_f1"],
                "initial_gs": summary["density_control"]["initial_gaussian_count"],
                "final_gs": summary["density_control"]["final_gaussian_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
