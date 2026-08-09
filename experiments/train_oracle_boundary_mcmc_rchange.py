"""Unified oracle-boundary fixed-capacity MCMC R_change runner.

A0 ``dc_only`` preserves the existing hard-gated TemporalChangeModel renderer.
A1-A5 and the separate ``geo_conservative_reloc`` ablation use
FixedCapacityChangeState + OracleBoundaryStateManager + StateArchive with no
GT-mask reads, fixed capacity, optional relocation, online predictions, and
deterministic archive/replay audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_cue_temporal_rchange import (  # noqa: E402
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    physical_frame_name,
    serializable_arguments,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import (  # noqa: E402
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    boundaries_from_manifest,
    compute_state_support,
    file_checksum,
    list_images,
    make_training_schedule,
    read_manifest,
    seed_everything,
    segment_id,
    segment_ranges,
    should_audit_gradient,
    summarize_losses,
    tensor_checksum,
    training_schedule_audit,
    training_target,
    validate_boundaries,
    validate_exact_dataset_contract,
)
from gaussian_renderer import render_change, render_change_temporal  # noqa: E402
from scene import GaussianModel  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from temporal import TemporalChangeModel  # noqa: E402
from temporal.conservative_relocation import (  # noqa: E402
    ConservativeRowSnapshot,
    conservative_acceptance,
    initialize_row_local_adam,
    residual_candidates_from_views,
    restore_rows_,
    row_local_adam_step_,
    select_conservative_sources,
    select_residual_destinations,
    source_deactivation_safety,
    transport_sources_,
)
from temporal.mcmc_dynamics import (  # noqa: E402
    activated_change_opacity,
    apply_sgld_xyz_noise_,
    dynamics_audit,
    relocate_dead_gaussians_,
    reset_adam_rows_,
)
from temporal.mcmc_energy import MCMCEnergy  # noqa: E402
from temporal.mcmc_state import (  # noqa: E402
    FixedCapacityChangeState,
    OracleBoundaryStateManager,
    StateArchive,
)

DEFAULT_OUTPUT = Path("outputs/oracle_boundary_mcmc_rchange")
MODES = (
    "dc_only",
    "dc_opacity",
    "geo_adam",
    "geo_sgld",
    "geo_mcmc",
    "geo_mcmc_anchor",
    "geo_conservative_reloc",
)
PROTOCOLS = ("matched_exact", "oracle_stream")
INIT_POLICIES = ("base_zero", "previous")
CONSERVATIVE_RELOCATION_MODES = {"geo_conservative_reloc"}
RELOCATION_MODES = {"geo_mcmc", "geo_mcmc_anchor", *CONSERVATIVE_RELOCATION_MODES}
SGLD_MODES = {"geo_sgld", "geo_mcmc", "geo_mcmc_anchor"}
UNSUPPORTED_OPACITY = 0.001
SUPPORTED_OPACITY = 0.1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def probability_logit(probability: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eps = torch.finfo(dtype).eps
    p = torch.tensor(float(probability), device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def optional_mcmc_api_status() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for name in ("temporal.mcmc_state", "temporal.mcmc_energy", "temporal.mcmc_dynamics"):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            missing.append({"module": name, "error": repr(exc)})
    return {
        "available": not missing,
        "missing_or_failed": missing,
        "loaded_symbols": {
            name: sorted(symbol for symbol in dir(module) if not symbol.startswith("_"))[:80]
            for name, module in modules.items()
        },
    }


def validate_expected_input_hashes(actual: Mapping[str, Any], expected_spec: str | None) -> dict[str, Any]:
    if expected_spec is None:
        return {"checked": False, "matched": None}
    path = Path(expected_spec)
    expected = json.loads(path.read_text(encoding="utf-8") if path.exists() else expected_spec)
    if expected != actual:
        raise RuntimeError("Input hash validation failed; aborting before training")
    return {"checked": True, "matched": True}


def build_no_gt_frame_records(
    source_path: Path,
    boundaries: tuple[int, ...],
    *,
    prefix_frames: int | None = None,
    frames_per_state: int | None = None,
) -> tuple[list[FrameRecord], list[str]]:
    names = list_images(source_path / "inference_scene" / "images")
    validate_boundaries(len(names), boundaries)
    if prefix_frames is not None and frames_per_state is not None:
        raise ValueError("--prefix-frames and --frames-per-state are mutually exclusive")
    if frames_per_state is not None:
        selected_indices: list[int] = []
        for start, end in segment_ranges(len(names), boundaries):
            selected_indices.extend(range(start, min(end, start + frames_per_state)))
    else:
        selected_indices = list(range(len(names) if prefix_frames is None else min(prefix_frames, len(names))))
    if not selected_indices:
        raise ValueError("No frames selected")
    records = [
        FrameRecord(
            global_index=idx,
            segment_id=segment_id(idx, boundaries),
            name=names[idx],
            image_path=str(source_path / "inference_scene" / "images" / names[idx]),
            mask_path="",
        )
        for idx in selected_indices
    ]
    return records, names


def assert_no_gt_training_records(records: Iterable[FrameRecord]) -> None:
    """Fail closed if a training record carries any mask path.

    The cue loader needs RGB/cue paths only. Keeping this check immediately
    after record construction prevents an evaluation/oracle record builder from
    being substituted without an explicit failure.
    """
    offenders = [record.name for record in records if str(record.mask_path).strip()]
    if offenders:
        raise RuntimeError(f"GT/mask paths are forbidden in this training runner: {offenders[:8]}")


def input_hashes(args: argparse.Namespace, records: Iterable[FrameRecord], all_names: list[str]) -> dict[str, Any]:
    base_ply = (Path(args.source_path) / BASE_PLY_REL).resolve()
    cue_root = Path(args.cue_cache_root)
    frame_hashes = []
    for record in records:
        stem = Path(record.name).stem
        cue_path = cue_root / "cues" / f"{physical_frame_name(stem)}.pt"
        frame_hashes.append(
            {
                "global_index": int(record.global_index),
                "name": record.name,
                "image_sha256": file_checksum(Path(record.image_path)),
                "cue_sha256": file_checksum(cue_path),
            }
        )
    return {
        "source_path": str(Path(args.source_path).resolve()),
        "all_frame_names_sha256": sha256_payload(all_names),
        "manifest_sha256": file_checksum(Path(args.source_path) / "manifest.json"),
        "base_ply_sha256": file_checksum(base_ply),
        "fixed_cameras_sha256": file_checksum(Path(args.fixed_cameras_json)),
        "cue_cache_metadata_sha256": file_checksum(cue_root / "metadata.json"),
        "selected_frame_hashes": frame_hashes,
    }


def slice_gaussian_model_(model: GaussianModel, capacity: int | None) -> dict[str, Any]:
    original_n = int(model.get_xyz.shape[0])
    if capacity is None or int(capacity) >= original_n:
        slot_ids = torch.arange(original_n, dtype=torch.long)
        return {
            "enabled": False,
            "initial_N": original_n,
            "N_t": original_n,
            "policy": "all",
            "slot_ids_hash": tensor_checksum(slot_ids),
        }
    n = int(capacity)
    slot_ids = torch.arange(n, device=model.get_xyz.device, dtype=torch.long)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "max_radii2D", "xyz_gradient_accum", "denom"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == original_n:
            setattr(model, name, value[:n].detach().clone().requires_grad_(bool(getattr(value, "requires_grad", False))))
    return {
        "enabled": True,
        "policy": "deterministic_first_n",
        "initial_N": original_n,
        "N_t": n,
        "slot_id_range": [0, n - 1],
        "slot_ids_hash": tensor_checksum(slot_ids),
        "tensor_hashes": {
            name: tensor_checksum(getattr(model, name))
            for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
        },
    }


def build_temporal_dc_model(base: GaussianModel, boundaries: tuple[int, ...], total_frames: int) -> TemporalChangeModel:
    model = TemporalChangeModel.from_gaussians(base, max_states=len(boundaries) + 1, initial_time=0.0)
    ranges = segment_ranges(total_frames, boundaries)
    with torch.no_grad():
        model.state_change_dc.zero_()
        for state, (start, end) in enumerate(ranges):
            model.state_change_dc[:, state].copy_(base._features_dc.detach())
            model.state_start[:, state].fill_(float(start))
            model.state_end[:, state].fill_(float("inf") if state == len(ranges) - 1 else float(end))
            model.state_valid[:, state].fill_(True)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(model.base, name, None)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(False)
    model.state_change_dc.requires_grad_(True)
    return model


def mcmc_parameter_items(state: FixedCapacityChangeState, mode: str) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    items = dict(state.current_parameter_items())
    names = ("features_dc", "raw_change_opacity") if mode == "dc_opacity" else ("xyz", "features_dc", "raw_change_opacity", "scaling", "rotation")
    return tuple((name, items[name]) for name in names)


def configure_mcmc_trainable(state: FixedCapacityChangeState, mode: str) -> None:
    selected = {id(parameter) for _name, parameter in mcmc_parameter_items(state, mode)}
    for _name, parameter in state.named_parameters():
        parameter.requires_grad_(id(parameter) in selected)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(state.base, name, None)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(False)


def mcmc_optimizer(state: FixedCapacityChangeState, args: argparse.Namespace) -> torch.optim.Adam:
    lr = {"xyz": args.xyz_lr, "features_dc": args.dc_lr, "raw_change_opacity": args.opacity_lr, "scaling": args.scaling_lr, "rotation": args.rotation_lr}
    return torch.optim.Adam(
        [{"params": [parameter], "lr": float(lr[name]), "name": name} for name, parameter in mcmc_parameter_items(state, args.mode)],
        lr=0.0,
        eps=1e-15,
    )


def set_unsupported_opacity_(state: FixedCapacityChangeState, mask: torch.Tensor | None = None) -> None:
    with torch.no_grad():
        target = probability_logit(UNSUPPORTED_OPACITY, device=state.current_raw_change_opacity.device, dtype=state.current_raw_change_opacity.dtype)
        if mask is None:
            state.current_raw_change_opacity.fill_(target)
        else:
            mask = mask.to(device=state.current_raw_change_opacity.device, dtype=torch.bool)
            if bool(mask.any()):
                state.current_raw_change_opacity[mask, 0] = target


def promote_supported_opacity_(state: FixedCapacityChangeState, mask: torch.Tensor) -> int:
    mask = mask.to(device=state.support_mask.device, dtype=torch.bool)
    newly = mask & ~state.support_mask
    if not bool(newly.any()):
        return 0
    with torch.no_grad():
        target = probability_logit(
            SUPPORTED_OPACITY,
            device=state.current_raw_change_opacity.device,
            dtype=state.current_raw_change_opacity.dtype,
        )
        state.support_mask[newly] = True
        # A previous-state warm start may already have stronger opacity. Cue
        # support initializes dead capacity, but must not reduce live evidence.
        state.current_raw_change_opacity[newly, 0] = torch.maximum(
            state.current_raw_change_opacity[newly, 0], target
        )
    return int(newly.sum().item())


def reset_state_support_for_new_segment_(
    state: FixedCapacityChangeState,
    *,
    initialize_dead_opacity: bool,
) -> None:
    state.deactivate_slots()
    if initialize_dead_opacity:
        set_unsupported_opacity_(state)


def activate_all_slots(state: FixedCapacityChangeState) -> None:
    """Document the MCMC render contract without mutating support or opacity.

    Every fixed-capacity slot is already returned as active by
    ``FixedCapacityChangeState.get_active_render_attributes``.  This explicit
    no-op is kept as a guardrail for callers/tests that want to assert that
    enabling the pool never promotes cue support or resets opacity.
    """
    if not isinstance(state, FixedCapacityChangeState):
        raise TypeError("state must be a FixedCapacityChangeState")


def current_render_overrides(state: FixedCapacityChangeState) -> dict[str, torch.Tensor]:
    return {
        "dc": state.current_features_dc,
        "opacity": state.base.opacity_activation(state.current_raw_change_opacity),
        "xyz": state.current_xyz,
        "scaling": state.base.scaling_activation(state.current_scaling),
        "rotation": state.base.rotation_activation(state.current_rotation),
    }


def mcmc_render(
    state: FixedCapacityChangeState,
    view: Camera,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    *,
    get_flag: bool | None = None,
    metric_map: torch.Tensor | None = None,
    support_projection: bool = False,
) -> dict[str, torch.Tensor]:
    attrs = current_render_overrides(state)
    opacity = state.base.get_opacity.detach() if support_projection else attrs["opacity"]
    return render_change(
        view,
        state.base,
        pipe,
        background,
        override_dc=attrs["dc"].detach() if support_projection else attrs["dc"],
        override_opacity=opacity,
        override_xyz=attrs["xyz"].detach() if support_projection else attrs["xyz"],
        override_scaling=attrs["scaling"].detach() if support_projection else attrs["scaling"],
        override_rotation=attrs["rotation"].detach() if support_projection else attrs["rotation"],
        get_flag=get_flag,
        metric_map=metric_map,
    )


def slot_evidence_from_view(
    state: FixedCapacityChangeState,
    view: Camera,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal visibility and positive-cue overlap for every fixed slot."""

    support = getattr(view, "support_map", None)
    if not isinstance(support, torch.Tensor):
        raise AttributeError("view.support_map must be a tensor for cue support updates")
    metric_map = (support[0] > float(threshold)).flatten().int()
    with torch.no_grad():
        package = mcmc_render(state, view, pipe, background, get_flag=True, metric_map=metric_map, support_projection=True)
    visible = (package["radii"] > 0).to(device=state.support_mask.device)
    supported = package["accum_metric_counts"].to(device=state.support_mask.device) > 0
    # A positive raster contribution is itself visibility evidence even if a
    # backend reports a zero radius for a numerically tiny footprint.
    visible |= supported
    return visible, supported


def cue_support_mask_from_view(state: FixedCapacityChangeState, view: Camera, pipe: SimpleNamespace, background: torch.Tensor, *, threshold: float) -> torch.Tensor:
    _visible, supported = slot_evidence_from_view(
        state,
        view,
        pipe,
        background,
        threshold=threshold,
    )
    return supported


def update_dc_support_from_view(
    model: TemporalChangeModel,
    view: Camera,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    *,
    support_threshold: int,
    map_threshold: float,
) -> int:
    """Causally union one observed cue frame into the A0 hard support gate."""
    state_id = int(view.segment_id)
    support = getattr(view, "support_map", None)
    if not isinstance(support, torch.Tensor):
        raise AttributeError("view.support_map must be a tensor for cue support updates")
    metric_map = (support[0] > float(map_threshold)).flatten().int()
    with torch.no_grad():
        package = render_change(
            view,
            model.base,
            pipe,
            background,
            get_flag=True,
            metric_map=metric_map,
            override_dc=model.state_change_dc[:, state_id].detach(),
            override_opacity=model.base.get_opacity.detach(),
        )
        frame_counts = package["accum_metric_counts"].to(torch.int32)
        frame_valid = frame_counts >= int(support_threshold)
        if not bool(frame_valid.any()):
            frame_valid = frame_counts > 0
        newly = frame_valid & ~model.state_valid[:, state_id]
        model.state_valid[:, state_id] |= frame_valid
    return int(newly.sum().item())


def mcmc_energy_value(energy: MCMCEnergy, state: FixedCapacityChangeState, view: Camera, pipe: SimpleNamespace, background: torch.Tensor, unsupported_anchor: Mapping[str, torch.Tensor] | None) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    package = mcmc_render(state, view, pipe, background)
    total, terms = energy(training_target(view), package["render"], state=state, unsupported_anchor=unsupported_anchor)
    return total, terms, {"render": package["render"], "active_gaussians_rendered": int((package["radii"] > 0).sum().detach().item())}


def tensor_terms_to_float(terms: Mapping[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in terms.items():
        detached = value.detach().float()
        out[key] = float(detached.item()) if detached.numel() == 1 else float(detached.mean().item())
    return out


def views_by_state(views: Iterable[Camera], max_states: int) -> dict[int, list[Camera]]:
    grouped = {state: [] for state in range(max_states)}
    for view in views:
        grouped[int(view.segment_id)].append(view)
    return grouped


def stream_schedule(views: list[Camera], updates_per_frame: int) -> list[tuple[int, int, int, Camera]]:
    schedule: list[tuple[int, int, int, Camera]] = []
    for protocol_step, current in enumerate(views, start=1):
        state = int(current.segment_id)
        for local_step in range(updates_per_frame):
            schedule.append((protocol_step, state, local_step, current))
    return schedule


def conservative_stream_schedule(
    views: list[Camera],
    updates_per_frame: int,
    *,
    replay_buffer_size: int,
) -> list[tuple[int, int, int, Camera]]:
    """Causal same-state replay with the arriving view used every other update."""

    if replay_buffer_size < 1:
        raise ValueError("replay_buffer_size must be positive")
    schedule: list[tuple[int, int, int, Camera]] = []
    buffers: dict[int, list[Camera]] = {}
    for protocol_step, current in enumerate(views, start=1):
        state = int(current.segment_id)
        buffer = buffers.setdefault(state, [])
        buffer.append(current)
        del buffer[:-replay_buffer_size]
        for local_step in range(updates_per_frame):
            if local_step % 2 == 0 or len(buffer) == 1:
                training_view = current
            else:
                training_view = buffer[(local_step // 2) % len(buffer)]
            schedule.append((protocol_step, state, local_step, training_view))
    return schedule


def make_protocol_schedule(views: list[Camera], max_states: int, args: argparse.Namespace, dataset_audit: dict[str, Any] | None) -> tuple[list[tuple[int, int, int, Camera]], dict[str, Any]]:
    if args.protocol == "oracle_stream":
        conservative = args.mode in CONSERVATIVE_RELOCATION_MODES
        schedule = (
            conservative_stream_schedule(
                views,
                args.updates_per_frame,
                replay_buffer_size=args.replay_buffer_size,
            )
            if conservative
            else stream_schedule(views, args.updates_per_frame)
        )
        counts = {view.image_name: 0 for view in views}
        training_counts = {view.image_name: 0 for view in views}
        for protocol_step, _state, _local_step, view in schedule:
            counts[views[protocol_step - 1].image_name] += 1
            training_counts[view.image_name] += 1
        return schedule, {
            "mode": "oracle_stream_causal_multiview_replay" if conservative else "oracle_stream_current_frame_online_updates",
            "requested_updates_per_frame": int(args.updates_per_frame),
            "actual_total_updates": len(schedule),
            "causal_current_frame_only": not conservative,
            "causal_prefix_replay": conservative,
            "replay_buffer_size": int(args.replay_buffer_size) if conservative else 0,
            "prefix_support_updates": True,
            "per_frame_update_counts": counts,
            "training_view_update_counts": training_counts,
            "per_frame_update_counts_checksum": sha256_payload(counts),
            "min_updates_per_frame": int(min(counts.values())) if counts else 0,
            "max_updates_per_frame": int(max(counts.values())) if counts else 0,
            "exact_all_images_guarantee": False,
        }
    grouped = views_by_state(views, max_states)
    steps = make_training_schedule(grouped, max_states, args)
    audit = training_schedule_audit(views, steps, args, dataset_audit)
    return [(i, step.state, step.local_step, grouped[step.state][step.view_index]) for i, step in enumerate(steps, start=1)], audit


class ArtifactWriters:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.energy_path = output_dir / "training_energy.csv"
        self.counts_path = output_dir / "gaussian_counts.csv"
        self.relocation_jsonl_path = output_dir / "relocation_events.jsonl"
        self.relocation_audit_path = output_dir / "relocation_audit.csv"
        self._energy_file = self.energy_path.open("w", newline="", encoding="utf-8")
        self._counts_file = self.counts_path.open("w", newline="", encoding="utf-8")
        self._audit_file = self.relocation_audit_path.open("w", newline="", encoding="utf-8")
        self._jsonl_file = self.relocation_jsonl_path.open("w", encoding="utf-8")
        self.energy = csv.DictWriter(self._energy_file, fieldnames=[
            "global_step", "protocol_step", "state_id", "frame", "timestamp", "mode",
            "ssf", "opacity_regularizer", "scale_regularizer", "unsupported_anchor_regularizer", "energy",
            "pre_relocation_energy", "post_relocation_energy", "noise_mean_abs", "noise_p50_abs", "noise_p95_abs", "noise_rms", "noise_max_abs", "noise_opacity_bin_stats", "lr_max",
        ])
        self.counts = csv.DictWriter(self._counts_file, fieldnames=[
            "global_step", "protocol_step", "state_id", "frame", "timestamp", "support_count", "rendered_count", "capacity", "N_t", "dead_count", "live_count", "selected_dead_count", "selected_live_count", "relocation_assignments", "relocation_applied", "relocation_deferred",
        ])
        self.audit = csv.DictWriter(self._audit_file, fieldnames=[
            "global_step", "protocol_step", "state_id", "frame", "event_type",
            "source_index", "target_index", "group_size", "source_target_distance",
            "delta_ssf", "delta_opacity_reg", "delta_scale_reg",
            "relative_delta_total", "relative_delta_total_energy",
            "delta_render_l1", "delta_render_linf",
            "attempted_post_energy", "attempted_delta_energy",
            "attempted_relative_delta_total_energy",
            "attempted_delta_render_l1", "attempted_delta_render_linf",
            "pre_coverage", "post_coverage", "coverage_gain",
            "attempted_post_coverage", "attempted_coverage_gain",
            "pre_negative_mass", "post_negative_mass", "negative_mass_delta",
            "attempted_post_negative_mass", "attempted_negative_mass_delta",
            "burnin_steps", "removal_protected_count",
            "attempted", "applied", "deferred", "pre_energy", "post_energy",
            "delta_energy", "pre_rendered_count", "post_rendered_count", "reason",
        ])
        for writer in (self.energy, self.counts, self.audit):
            writer.writeheader()

    def close(self) -> None:
        for handle in (self._energy_file, self._counts_file, self._audit_file, self._jsonl_file):
            handle.close()

    def write_energy(self, row: Mapping[str, Any]) -> None:
        self.energy.writerow(row)

    def write_counts(self, row: Mapping[str, Any]) -> None:
        self.counts.writerow(row)

    def write_relocation(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, default=str) + "\n"
        self._jsonl_file.write(line)
        self.audit.writerow({key: event.get(key, "") for key in self.audit.fieldnames})


def compact_relocation_audit(audit: Mapping[str, Any], *, preview: int = 16) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in audit.items():
        if key.endswith("_indices") and isinstance(value, list):
            out[f"{key}_count"] = len(value)
            out[f"{key}_head"] = value[:preview]
        elif key == "events" and isinstance(value, list):
            out["events_count"] = len(value)
            out["events_head"] = value[:preview]
        else:
            out[key] = value
    return out


def live_dead_counts(state: FixedCapacityChangeState, threshold: float) -> dict[str, int]:
    opacity = activated_change_opacity(state.current_raw_change_opacity)
    dead = int((opacity < float(threshold)).sum().item())
    live = int(opacity.numel() - dead)
    return {"dead_count": dead, "live_count": live, "N_t": int(opacity.numel())}


def assert_fixed_capacity(state: FixedCapacityChangeState, initial_n: int, parameter_ids: Mapping[str, int]) -> None:
    if int(state.capacity) != int(initial_n):
        raise RuntimeError(f"capacity drift: {state.capacity} != {initial_n}")
    for name, parameter in mcmc_parameter_items(state, "geo_mcmc"):
        if parameter.shape[0] != initial_n:
            raise RuntimeError(f"parameter {name} shape drift: {parameter.shape[0]} != {initial_n}")
        if parameter_ids.get(name) != id(parameter):
            raise RuntimeError(f"parameter identity drift for {name}")


def archive_record_file(manager: OracleBoundaryStateManager, archive_index: int, output_dir: Path) -> dict[str, Any]:
    record = manager.archive.records[archive_index]
    tensors = manager.archive.tensors(archive_index)
    archive_dir = output_dir / "state_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"state_{record.state_id:02d}_{int(record.start_time):06d}_{int(record.end_time):06d}.pt"
    torch.save({"record": asdict(record), "tensors": tensors}, path)
    tensor_hashes = {name: tensor_checksum(value) for name, value in tensors.items() if isinstance(value, torch.Tensor)}
    return {"state_id": record.state_id, "start_time": record.start_time, "end_time": record.end_time, "checksum": record.checksum, "path": str(path), "file_sha256": file_checksum(path), "tensor_hashes": tensor_hashes, "metadata": dict(record.metadata)}


def load_archived_state(base: GaussianModel, tensors: Mapping[str, torch.Tensor]) -> FixedCapacityChangeState:
    state = FixedCapacityChangeState.from_gaussians(base, capacity=int(tensors["xyz"].shape[0]), base_zero=True)
    state.reset_current(warm_start=tensors, base_zero=True)
    return state


def render_checksum_for_state(state: FixedCapacityChangeState, views: Sequence[Camera], pipe: SimpleNamespace, background: torch.Tensor) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for view in views:
            rendered = mcmc_render(state, view, pipe, background)["render"].detach().cpu().contiguous()
            digest.update(rendered.numpy().tobytes())
    return digest.hexdigest()


def archive_render_audit(manager: OracleBoundaryStateManager, archive_index: int, views: Sequence[Camera], base: GaussianModel, pipe: SimpleNamespace, background: torch.Tensor) -> dict[str, Any]:
    tensors = manager.archive.tensors(archive_index)
    state = load_archived_state(base, tensors)
    parameter_checksum = sha256_payload({name: tensor_checksum(value) for name, value in tensors.items() if isinstance(value, torch.Tensor)})
    return {"archive_index": archive_index, "render_checksum": render_checksum_for_state(state, views, pipe, background), "parameter_checksum": parameter_checksum, "view_names": [v.image_name for v in views]}


def audit_archive_replay(archive_audits: list[dict[str, Any]], manager: OracleBoundaryStateManager, base: GaussianModel, pipe: SimpleNamespace, background: torch.Tensor, views_by_sid: Mapping[int, list[Camera]]) -> list[dict[str, Any]]:
    results = []
    for item in archive_audits:
        idx = int(item["archive_index"])
        sid = int(manager.archive.records[idx].state_id)
        by_name = {view.image_name: view for view in views_by_sid.get(sid, [])}
        missing = [name for name in item["view_names"] if name not in by_name]
        if missing:
            raise KeyError(f"archive replay views missing for state {sid}: {missing}")
        views = [by_name[name] for name in item["view_names"]]
        replay = archive_render_audit(manager, idx, views, base, pipe, background)
        results.append({**item, "replay_render_checksum": replay["render_checksum"], "replay_parameter_checksum": replay["parameter_checksum"], "render_drift_zero": replay["render_checksum"] == item["render_checksum"], "parameter_drift_zero": replay["parameter_checksum"] == item["parameter_checksum"]})
    return results


def geometry_diagnostics_from_tensors(tensors: Mapping[str, torch.Tensor], base: GaussianModel, threshold: float) -> dict[str, Any]:
    xyz = tensors["xyz"].float()
    base_xyz = base._xyz.detach().cpu().float()[: xyz.shape[0]]
    displacement = (xyz - base_xyz).norm(dim=1)
    opacity = torch.sigmoid(tensors["raw_change_opacity"].float()).flatten()
    def stats(values: torch.Tensor) -> dict[str, float | int]:
        if values.numel() == 0:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {"count": int(values.numel()), "mean": float(values.mean()), "p50": float(torch.quantile(values, 0.5)), "p95": float(torch.quantile(values, 0.95)), "max": float(values.max())}
    max_displacement = float(displacement.max().item()) if displacement.numel() else 0.0
    if max_displacement > 0.0:
        hist_counts, hist_edges = torch.histogram(displacement, bins=64, range=(0.0, max_displacement))
    else:
        hist_counts = torch.tensor([int(displacement.numel())], dtype=torch.float32)
        hist_edges = torch.tensor([0.0, 1.0], dtype=torch.float32)
    return {
        "xyz_displacement_vs_base": {
            **stats(displacement),
            "histogram_counts": hist_counts.to(torch.int64).tolist(),
            "histogram_edges": hist_edges.tolist(),
        },
        "opacity": {"live_count": int((opacity >= threshold).sum()), "dead_count": int((opacity < threshold).sum()), "stats": stats(opacity)},
    }


def relocation_due(args: argparse.Namespace, state_local_step: int) -> bool:
    return args.mode in RELOCATION_MODES and state_local_step > args.relocation_warmup_steps and state_local_step % args.relocation_interval == 0


def aggregate_audit_metrics(energy: MCMCEnergy, state: FixedCapacityChangeState, views: Sequence[Camera], pipe: SimpleNamespace, background: torch.Tensor, anchor: Mapping[str, torch.Tensor] | None) -> dict[str, Any]:
    totals = []
    ssfs = []
    opacity_regs = []
    scale_regs = []
    renders = []
    rendered_counts = []
    with torch.no_grad():
        for view in views:
            total, terms, stats = mcmc_energy_value(energy, state, view, pipe, background, anchor)
            vals = tensor_terms_to_float(terms)
            totals.append(float(total.item()))
            ssfs.append(vals.get("ssf", 0.0))
            opacity_regs.append(vals.get("opacity", 0.0))
            scale_regs.append(vals.get("scale", 0.0))
            renders.append(stats["render"].detach().cpu())
            rendered_counts.append(stats["active_gaussians_rendered"])
    return {"total": float(np.mean(totals)), "ssf": float(np.mean(ssfs)), "opacity_reg": float(np.mean(opacity_regs)), "scale_reg": float(np.mean(scale_regs)), "renders": renders, "rendered_count": int(np.mean(rendered_counts)) if rendered_counts else 0}


def aggregate_cue_coverage(
    renders: Sequence[torch.Tensor],
    views: Sequence[Camera],
    *,
    cue_threshold: float,
) -> float:
    """Mean rendered mass on positive cue pixels for a fixed causal replay set."""

    values: list[float] = []
    for rendered, view in zip(renders, views):
        cue = training_target(view).detach().float().cpu()
        prediction = rendered.detach().float().cpu().mean(dim=0, keepdim=True).clamp(0.0, 1.0)
        positive = cue >= float(cue_threshold)
        cue_mass = float(cue[positive].sum().item()) if bool(positive.any()) else 0.0
        covered = float(prediction[positive].sum().item()) if bool(positive.any()) else 0.0
        values.append(covered / cue_mass if cue_mass > 0.0 else 0.0)
    return float(np.mean(values)) if values else 0.0


def aggregate_negative_render_mass(
    renders: Sequence[torch.Tensor],
    views: Sequence[Camera],
    *,
    cue_threshold: float,
) -> float:
    """Mean direct R_change mass where the observed cue says no change.

    Unlike positive coverage, this detects zero-DC alpha occluders: removing a
    useful foreground occluder can expose a bright change Gaussian behind it
    and increase mass on cue-negative pixels.
    """

    values: list[float] = []
    for rendered, view in zip(renders, views):
        cue = training_target(view).detach().float().cpu()
        prediction = rendered.detach().float().cpu().mean(dim=0, keepdim=True).clamp(0.0, 1.0)
        negative = cue < float(cue_threshold)
        values.append(float(prediction[negative].mean().item()) if bool(negative.any()) else 0.0)
    return float(np.mean(values)) if values else 0.0


def conservative_burnin_view_split(
    views: Sequence[Camera],
) -> tuple[list[Camera], list[Camera]]:
    """Deterministically separate causal proposal/train and validation views."""

    values = list(views)
    if len(values) < 4:
        return values, values
    return values[::2], values[1::2]


def run_row_local_conservative_burnin(
    state: FixedCapacityChangeState,
    optimizer: torch.optim.Optimizer,
    energy: MCMCEnergy,
    views: Sequence[Camera],
    sources: torch.Tensor,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    anchor: Mapping[str, torch.Tensor] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train only relocated rows; the first half keeps geometry frozen."""

    steps = int(args.conservative_burnin_steps)
    if steps <= 0 or sources.numel() == 0:
        return {"steps": 0, "first_energy": None, "last_energy": None, "max_grad_norm": 0.0, "finite": True}
    train_views = list(views)
    if not train_views:
        raise ValueError("row-local burn-in requires at least one causal view")
    parameters = dict(state.current_parameter_items())
    local_adam = initialize_row_local_adam(parameters, sources)
    multiplier = float(args.conservative_burnin_lr_multiplier)
    learning_rates = {
        "xyz": args.xyz_lr * multiplier,
        "features_dc": args.dc_lr * multiplier,
        "raw_change_opacity": args.opacity_lr * multiplier,
        "scaling": args.scaling_lr * multiplier,
        "rotation": args.rotation_lr * multiplier,
    }
    attribute_only = ("features_dc", "raw_change_opacity")
    all_names = tuple(parameters)
    energies: list[float] = []
    max_grad_norm = 0.0
    optimizer.zero_grad(set_to_none=True)
    for burnin_step in range(steps):
        view = train_views[burnin_step % len(train_views)]
        total, _terms, _render = mcmc_energy_value(energy, state, view, pipe, background, anchor)
        if not bool(torch.isfinite(total)):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite conservative burn-in energy at step {burnin_step}")
        total.backward()
        allowed = attribute_only if burnin_step < math.ceil(steps * 0.5) else all_names
        grad_norms = row_local_adam_step_(
            parameters,
            sources,
            learning_rates,
            local_adam,
            allowed_names=allowed,
        )
        if grad_norms:
            max_grad_norm = max(max_grad_norm, max(grad_norms.values()))
        energies.append(float(total.detach().item()))
        optimizer.zero_grad(set_to_none=True)
    for name, parameter in parameters.items():
        rows = parameter.detach().index_select(0, sources)
        if not bool(torch.isfinite(rows).all()):
            raise FloatingPointError(f"non-finite conservative burn-in parameter rows for {name}")
    return {
        "steps": steps,
        "first_energy": energies[0],
        "last_energy": energies[-1],
        "max_grad_norm": max_grad_norm,
        "finite": True,
        "attribute_only_steps": int(math.ceil(steps * 0.5)),
        "geometry_steps": int(steps - math.ceil(steps * 0.5)),
        "train_view_names": [view.image_name for view in train_views],
    }


def unexplained_residual_maps(
    state: FixedCapacityChangeState,
    views: Sequence[Camera],
    pipe: SimpleNamespace,
    background: torch.Tensor,
    *,
    cue_threshold: float,
) -> list[torch.Tensor]:
    """Positive SSF residual only; negative/false-positive pixels are not destinations."""

    residuals: list[torch.Tensor] = []
    with torch.no_grad():
        for view in views:
            rendered = mcmc_render(state, view, pipe, background)["render"]
            # R_change is rendered directly in [0, 1].  Treating it as a logit
            # would turn a fully explained pixel (1.0) into a false residual
            # of roughly 0.27 and continually propose already-covered regions.
            probability = rendered.mean(dim=0, keepdim=True).clamp(0.0, 1.0)
            cue = training_target(view)
            residual = (cue - probability).clamp_min(0.0)
            residual = residual * (cue >= float(cue_threshold)).to(residual.dtype)
            residuals.append(residual)
    return residuals


def conservative_scene_bounds(
    xyz: torch.Tensor,
    *,
    quantile: float,
    margin_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Robust reference-scene bounds used to reject implausible ray intersections."""

    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must have shape [N, 3] and be nonempty")
    q = float(quantile)
    if not 0.0 <= q < 0.5:
        raise ValueError("quantile must be in [0, 0.5)")
    if float(margin_ratio) < 0.0:
        raise ValueError("margin_ratio must be nonnegative")
    detached = xyz.detach()
    lower = torch.quantile(detached, q, dim=0)
    upper = torch.quantile(detached, 1.0 - q, dim=0)
    extent = (upper - lower).clamp_min(torch.finfo(detached.dtype).eps)
    lower = lower - float(margin_ratio) * extent
    upper = upper + float(margin_ratio) * extent
    diagonal = float((upper - lower).norm().item())
    return lower, upper, diagonal


def conservative_relocation_audit_and_apply(
    manager: OracleBoundaryStateManager,
    optimizer: torch.optim.Optimizer,
    energy: MCMCEnergy,
    audit_views: Sequence[Camera],
    pipe: SimpleNamespace,
    background: torch.Tensor,
    anchor: Mapping[str, torch.Tensor] | None,
    writers: ArtifactWriters,
    args: argparse.Namespace,
    *,
    global_step: int,
    protocol_step: int,
    state_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Occlusion-safe relocation with optional row-local multi-view burn-in."""

    state = manager.current_state
    proposal_views, validation_views = conservative_burnin_view_split(audit_views)
    pre_all = aggregate_audit_metrics(energy, state, audit_views, pipe, background, anchor)
    pre = aggregate_audit_metrics(energy, state, validation_views, pipe, background, anchor)
    pre_coverage = aggregate_cue_coverage(pre["renders"], validation_views, cue_threshold=args.support_map_threshold)
    pre_negative_mass = aggregate_negative_render_mass(pre["renders"], validation_views, cue_threshold=args.support_map_threshold)
    pre_all_negative_mass = aggregate_negative_render_mass(pre_all["renders"], audit_views, cue_threshold=args.support_map_threshold)
    protected = state.carryover_protected_mask | state.removal_protected_mask
    sources = select_conservative_sources(
        observation_count=state.slot_observation_count,
        cue_support_count=state.slot_cue_support_count,
        protected_mask=protected,
        tentative_mask=state.tentative_mask,
        min_observations=args.conservative_min_observations,
        max_sources=args.conservative_max_relocations,
        seed=args.seed + global_step,
    )
    eligible_source_count = int(sources.numel())
    residuals = unexplained_residual_maps(
        state,
        proposal_views,
        pipe,
        background,
        cue_threshold=args.support_map_threshold,
    )
    lower, upper, scene_diagonal = conservative_scene_bounds(
        state.base._xyz,
        quantile=args.conservative_scene_quantile,
        margin_ratio=args.conservative_scene_margin,
    )
    candidates = residual_candidates_from_views(
        proposal_views,
        residuals,
        per_view_topk=args.conservative_residual_topk,
        residual_threshold=args.conservative_residual_threshold,
        min_support_views=args.conservative_min_support_views,
        max_ray_distance=max(scene_diagonal * args.conservative_max_ray_distance_ratio, torch.finfo(state.current_xyz.dtype).eps),
        scene_bounds=(lower, upper),
        max_candidates=args.conservative_max_candidates,
    )
    destination_rows = select_residual_destinations(
        candidates,
        count=min(int(sources.numel()), args.conservative_max_relocations),
        min_support_views=args.conservative_min_support_views,
        seed=args.seed + 1000000 + global_step,
    )
    count = min(int(sources.numel()), int(destination_rows.numel()))
    sources = sources[:count]
    destination_rows = destination_rows[:count]

    reason = "proposed"
    accepted = False
    decision = None
    deactivation_decision = None
    rollback_restored_exact: bool | None = None
    post = pre
    post_coverage = pre_coverage
    post_negative_mass = pre_negative_mass
    deactivated_all = pre_all
    transported_all = pre_all
    burnin = {"steps": 0, "first_energy": None, "last_energy": None, "max_grad_norm": 0.0, "finite": True}
    snapshot = None
    source_xyz_before = state.current_xyz.detach().index_select(0, sources).clone() if count else state.current_xyz.new_empty((0, 3))
    destination_xyz = candidates.xyz.index_select(0, destination_rows) if count else state.current_xyz.new_empty((0, 3))
    if len(proposal_views) < args.conservative_min_support_views:
        reason = "insufficient_views"
    elif eligible_source_count == 0:
        reason = "no_safe_source"
    elif int(candidates.xyz.shape[0]) == 0 or count == 0:
        reason = "no_multiview_destination"
    else:
        snapshot = ConservativeRowSnapshot.capture(state, sources)
        # First test whether removing these rows exposes false change or hurts
        # the current-state energy. DC=0 alone is not evidence that a row is free.
        transport_sources_(state, sources, source_xyz_before, tentative_opacity=args.conservative_transport_opacity, global_step=global_step)
        deactivated_all = aggregate_audit_metrics(energy, state, audit_views, pipe, background, anchor)
        transported_all = deactivated_all
        deactivated_validation = aggregate_audit_metrics(energy, state, validation_views, pipe, background, anchor)
        deactivated_negative_mass = aggregate_negative_render_mass(
            deactivated_all["renders"], audit_views, cue_threshold=args.support_map_threshold
        )
        deactivation_decision = source_deactivation_safety(
            pre_energy=pre_all["total"],
            deactivated_energy=deactivated_all["total"],
            pre_negative_mass=pre_all_negative_mass,
            deactivated_negative_mass=deactivated_negative_mass,
            max_relative_energy_increase=args.conservative_max_deactivation_relative_energy_increase,
            max_negative_mass_increase=args.conservative_max_negative_mass_increase,
        )
        if not deactivation_decision.safe:
            post = deactivated_validation
            post_coverage = aggregate_cue_coverage(post["renders"], validation_views, cue_threshold=args.support_map_threshold)
            post_negative_mass = aggregate_negative_render_mass(post["renders"], validation_views, cue_threshold=args.support_map_threshold)
            reason = deactivation_decision.reason
            restore_rows_(state, snapshot)
            rollback_restored_exact = snapshot.matches(state)
            if not rollback_restored_exact:
                raise RuntimeError("source deactivation rollback failed exact row restoration")
            with torch.no_grad():
                state.removal_protected_mask[sources] = True
        else:
            # Transport while contribution is negligible, then activate and
            # adapt only the moved rows on proposal-side causal replay views.
            with torch.no_grad():
                state.current_xyz.index_copy_(0, sources, destination_xyz)
            transported_all = aggregate_audit_metrics(energy, state, audit_views, pipe, background, anchor)
            with torch.no_grad():
                tentative_raw = probability_logit(
                    args.conservative_tentative_opacity,
                    device=state.current_raw_change_opacity.device,
                    dtype=state.current_raw_change_opacity.dtype,
                )
                state.current_raw_change_opacity.index_copy_(0, sources, tentative_raw.reshape(1, 1).expand(sources.numel(), 1))
            try:
                burnin = run_row_local_conservative_burnin(
                    state, optimizer, energy, proposal_views, sources, pipe, background, anchor, args
                )
                post = aggregate_audit_metrics(energy, state, validation_views, pipe, background, anchor)
                post_coverage = aggregate_cue_coverage(post["renders"], validation_views, cue_threshold=args.support_map_threshold)
                post_negative_mass = aggregate_negative_render_mass(post["renders"], validation_views, cue_threshold=args.support_map_threshold)
                decision = conservative_acceptance(
                    pre_energy=pre["total"],
                    post_energy=post["total"],
                    pre_coverage=pre_coverage,
                    post_coverage=post_coverage,
                    max_relative_energy_increase=args.conservative_max_relative_energy_increase,
                    min_coverage_gain=args.conservative_min_coverage_gain,
                    pre_negative_mass=pre_negative_mass,
                    post_negative_mass=post_negative_mass,
                    max_negative_mass_increase=args.conservative_max_negative_mass_increase,
                )
                accepted = decision.accepted
                reason = decision.reason
            except FloatingPointError:
                accepted = False
                reason = "nonfinite_burnin"
                burnin = {**burnin, "finite": False}
            if accepted:
                reset_adam_rows_(optimizer, dict(state.current_parameter_items()), sources)
            else:
                restore_rows_(state, snapshot)
                rollback_restored_exact = snapshot.matches(state)
                if not rollback_restored_exact:
                    raise RuntimeError("conservative relocation rollback failed exact row restoration")

    attempted_post = post
    attempted_post_coverage = post_coverage
    attempted_post_negative_mass = post_negative_mass
    attempted_render_l1 = float(np.mean([(a - b).abs().mean().item() for a, b in zip(pre["renders"], attempted_post["renders"])])) if pre["renders"] else 0.0
    attempted_render_linf = float(max(((a - b).abs().max().item() for a, b in zip(pre["renders"], attempted_post["renders"])), default=0.0))
    attempted_relative_delta = (attempted_post["total"] - pre["total"]) / max(abs(pre["total"]), 1e-12)
    # The ordinary post/delta fields describe the state that actually remains
    # after this transaction.  Rejected proposal diagnostics are retained
    # separately under attempted_* instead of pretending a rolled-back move
    # is still present in the model.
    final_post = attempted_post if accepted else pre
    final_post_coverage = attempted_post_coverage if accepted else pre_coverage
    final_render_l1 = attempted_render_l1 if accepted else 0.0
    final_render_linf = attempted_render_linf if accepted else 0.0
    final_relative_delta = (final_post["total"] - pre["total"]) / max(abs(pre["total"]), 1e-12)
    transport_render_l1 = float(np.mean([(a - b).abs().mean().item() for a, b in zip(deactivated_all["renders"], transported_all["renders"])])) if deactivated_all["renders"] else 0.0
    counts = {
        "selected_dead_count": int(sources.numel()),
        "selected_live_count": int(candidates.xyz.shape[0]),
        "assignment_count": count,
        "applied_count": count if accepted else 0,
        "deferred_count": count if count and not accepted else 0,
    }
    aggregate = {
        "event_type": "aggregate",
        "relocation_kind": "causal_multiview_residual_guided_fixed_capacity",
        "global_step": global_step,
        "protocol_step": protocol_step,
        "state_id": state_id,
        "frame": audit_views[-1].image_name if audit_views else "",
        "timestamp": float(audit_views[-1].timestamp) if audit_views else 0.0,
        "attempted": count,
        "applied": counts["applied_count"],
        "deferred": counts["deferred_count"],
        "accepted": accepted,
        "rollback": bool(count and not accepted),
        "rollback_restored_exact": rollback_restored_exact,
        "pre_energy": pre["total"],
        "post_energy": final_post["total"],
        "delta_energy": final_post["total"] - pre["total"],
        "relative_delta_total": final_relative_delta,
        "relative_delta_total_energy": final_relative_delta,
        "delta_ssf": final_post["ssf"] - pre["ssf"],
        "delta_opacity_reg": final_post["opacity_reg"] - pre["opacity_reg"],
        "delta_scale_reg": final_post["scale_reg"] - pre["scale_reg"],
        "delta_render_l1": final_render_l1,
        "delta_render_linf": final_render_linf,
        "attempted_post_energy": attempted_post["total"],
        "attempted_delta_energy": attempted_post["total"] - pre["total"],
        "attempted_relative_delta_total_energy": attempted_relative_delta,
        "attempted_delta_ssf": attempted_post["ssf"] - pre["ssf"],
        "attempted_delta_opacity_reg": attempted_post["opacity_reg"] - pre["opacity_reg"],
        "attempted_delta_scale_reg": attempted_post["scale_reg"] - pre["scale_reg"],
        "attempted_delta_render_l1": attempted_render_l1,
        "attempted_delta_render_linf": attempted_render_linf,
        "transport_only_render_l1": transport_render_l1,
        "pre_coverage": pre_coverage,
        "post_coverage": final_post_coverage,
        "coverage_gain": final_post_coverage - pre_coverage,
        "attempted_post_coverage": attempted_post_coverage,
        "attempted_coverage_gain": attempted_post_coverage - pre_coverage,
        "pre_negative_mass": pre_negative_mass,
        "post_negative_mass": pre_negative_mass if not accepted else attempted_post_negative_mass,
        "negative_mass_delta": 0.0 if not accepted else attempted_post_negative_mass - pre_negative_mass,
        "attempted_post_negative_mass": attempted_post_negative_mass,
        "attempted_negative_mass_delta": attempted_post_negative_mass - pre_negative_mass,
        "pre_rendered_count": pre["rendered_count"],
        "post_rendered_count": final_post["rendered_count"],
        "attempted_post_rendered_count": attempted_post["rendered_count"],
        "reason": reason,
        "source_policy": "observed_zero_positive_support_and_removal_safe",
        "candidate_source_indices": sources.detach().cpu().tolist(),
        "candidate_destination_indices": destination_rows.detach().cpu().tolist(),
        "candidate_destination_xyz": destination_xyz.detach().cpu().tolist(),
        "candidate_scores": candidates.scores.index_select(0, destination_rows).detach().cpu().tolist() if count else [],
        "candidate_support_views": candidates.support_views.index_select(0, destination_rows).detach().cpu().tolist() if count else [],
        "candidate_ray_distance": candidates.ray_distance.index_select(0, destination_rows).detach().cpu().tolist() if count else [],
        "eligible_source_count_before_destination_cap": eligible_source_count,
        "candidate_count": int(candidates.xyz.shape[0]),
        "scene_diagonal": scene_diagonal,
        "decision": None if decision is None else decision.__dict__,
        "deactivation_decision": None if deactivation_decision is None else deactivation_decision.__dict__,
        "deactivation_pre_energy": pre_all["total"],
        "deactivation_post_energy": deactivated_all["total"],
        "deactivation_pre_negative_mass": pre_all_negative_mass,
        "deactivation_post_negative_mass": aggregate_negative_render_mass(deactivated_all["renders"], audit_views, cue_threshold=args.support_map_threshold),
        "burnin": burnin,
        "burnin_steps": int(burnin.get("steps", 0)),
        "proposal_view_names": [view.image_name for view in proposal_views],
        "validation_view_names": [view.image_name for view in validation_views],
        "removal_protected_count": int(state.removal_protected_mask.sum().item()),
    }
    writers.write_relocation(aggregate)
    source_xyz_after = state.current_xyz.detach().index_select(0, sources).clone() if count else state.current_xyz.new_empty((0, 3))
    for row_index, source in enumerate(sources.detach().cpu().tolist()):
        event = {
            **aggregate,
            "event_type": "source",
            "source_index": int(source),
            "source_slot_id": int(source),
            "target_index": -1,
            "target_slot_id": -1,
            "source_xyz_before": source_xyz_before[row_index].detach().cpu().tolist(),
            "source_xyz_after": source_xyz_after[row_index].detach().cpu().tolist(),
            "destination_xyz": destination_xyz[row_index].detach().cpu().tolist(),
            "proposed_destination_xyz": destination_xyz[row_index].detach().cpu().tolist(),
            "proposal_applied": accepted,
            "source_target_distance": float((source_xyz_before[row_index] - source_xyz_after[row_index]).norm().item()),
            "proposed_source_target_distance": float((source_xyz_before[row_index] - destination_xyz[row_index]).norm().item()),
            "optimizer_source_moment_reset": accepted,
            "representation_capacity_lineage_not_object_motion": True,
        }
        writers.write_relocation(event)
    return aggregate, counts


def relocation_audit_and_apply(manager: OracleBoundaryStateManager, optimizer: torch.optim.Optimizer, energy: MCMCEnergy, audit_views: Sequence[Camera], pipe: SimpleNamespace, background: torch.Tensor, anchor: Mapping[str, torch.Tensor] | None, writers: ArtifactWriters, args: argparse.Namespace, *, global_step: int, protocol_step: int, state_id: int, state_local_step: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not relocation_due(args, state_local_step):
        return None, {"selected_dead_count": 0, "selected_live_count": 0, "assignment_count": 0, "applied_count": 0, "deferred_count": 0}
    if args.mode in CONSERVATIVE_RELOCATION_MODES:
        return conservative_relocation_audit_and_apply(
            manager,
            optimizer,
            energy,
            audit_views,
            pipe,
            background,
            anchor,
            writers,
            args,
            global_step=global_step,
            protocol_step=protocol_step,
            state_id=state_id,
        )
    current = manager.current_state
    pre = aggregate_audit_metrics(energy, current, audit_views, pipe, background, anchor)
    with torch.no_grad():
        xyz_before = current.current_xyz.detach().clone()
        opacity_before = activated_change_opacity(current.current_raw_change_opacity).clone()
        scale_before = current.base.scaling_activation(current.current_scaling).detach().clone()
    result = relocate_dead_gaussians_(
        manager.current_state,
        optimizers=optimizer,
        opacity_threshold=args.dead_opacity_threshold,
        relocation_count=args.relocation_count,
        seed=args.seed + global_step,
        max_group_total=args.relocation_max_group_total,
    )
    post = aggregate_audit_metrics(energy, manager.current_state, audit_views, pipe, background, anchor)
    audit = result.audit()
    render_l1 = float(np.mean([(a - b).abs().mean().item() for a, b in zip(pre["renders"], post["renders"])])) if pre["renders"] else 0.0
    render_linf = float(max(((a - b).abs().max().item() for a, b in zip(pre["renders"], post["renders"])), default=0.0))
    reason = "applied" if result.applied_count else ("no_live" if audit["selected_live_count"] == 0 else "no_dead")
    relative_delta = (post["total"] - pre["total"]) / max(abs(pre["total"]), 1e-12)
    aggregate = {
        "event_type": "aggregate", "global_step": global_step, "protocol_step": protocol_step, "state_id": state_id, "frame": audit_views[0].image_name if audit_views else "", "timestamp": float(audit_views[0].timestamp) if audit_views else 0.0,
        "attempted": int(audit["assignment_count"]), "applied": int(result.applied_count), "deferred": int(result.deferred_count), "pre_energy": pre["total"], "post_energy": post["total"], "delta_energy": post["total"] - pre["total"], "relative_delta_total": relative_delta, "relative_delta_total_energy": relative_delta, "delta_ssf": post["ssf"] - pre["ssf"], "delta_opacity_reg": post["opacity_reg"] - pre["opacity_reg"], "delta_scale_reg": post["scale_reg"] - pre["scale_reg"], "delta_render_l1": render_l1, "delta_render_linf": render_linf, "pre_rendered_count": pre["rendered_count"], "post_rendered_count": post["rendered_count"], "reason": reason, "pre_terms": {"ssf": pre["ssf"], "opacity_reg": pre["opacity_reg"], "scale_reg": pre["scale_reg"]}, "post_terms": {"ssf": post["ssf"], "opacity_reg": post["opacity_reg"], "scale_reg": post["scale_reg"]}, "relocation": compact_relocation_audit(audit),
    }
    writers.write_relocation(aggregate)
    assignment_by_source = {a.source_index: a.target_index for a in result.assignments}
    group_by_source = {
        source: event.group_total
        for event in result.events
        for source in event.source_indices
        if event.applied
    }
    opacity_after = activated_change_opacity(current.current_raw_change_opacity)
    scale_after = current.base.scaling_activation(current.current_scaling).detach()
    for source in result.applied_source_indices:
        target = int(assignment_by_source.get(source, -1))
        row = {
            "event_type": "source",
            "state_id": state_id,
            "global_step": global_step,
            "global_iteration": global_step,
            "protocol_step": protocol_step,
            "frame": aggregate["frame"],
            "frame_id": aggregate["frame"],
            "timestamp": aggregate["timestamp"],
            "source_index": int(source),
            "source_slot_id": int(source),
            "target_index": target,
            "target_slot_id": target,
            "group_size": int(group_by_source.get(source, 0)),
            "source_xyz_before": xyz_before[source].cpu().tolist(),
            "target_xyz_before": xyz_before[target].cpu().tolist(),
            "source_xyz_after": current.current_xyz[source].detach().cpu().tolist(),
            "source_target_distance": float((xyz_before[source] - xyz_before[target]).norm().item()),
            "opacity_before": float(opacity_before[source].item()),
            "opacity_after": float(opacity_after[source].item()),
            "scale_before": scale_before[source].cpu().tolist(),
            "scale_after": scale_after[source].cpu().tolist(),
            "optimizer_target_moment_reset": target in result.reset_target_indices,
            "optimizer_source_moment_retained": True,
            "delta_total_energy": aggregate["delta_energy"],
            "pre_energy": aggregate["pre_energy"],
            "post_energy": aggregate["post_energy"],
            "delta_energy": aggregate["delta_energy"],
            "relative_delta_total": aggregate["relative_delta_total"],
            "relative_delta_total_energy": aggregate["relative_delta_total_energy"],
            "delta_ssf": aggregate["delta_ssf"],
            "delta_opacity_reg": aggregate["delta_opacity_reg"],
            "delta_scale_reg": aggregate["delta_scale_reg"],
            "delta_render_l1": aggregate["delta_render_l1"],
            "delta_render_linf": aggregate["delta_render_linf"],
            "pre_rendered_count": aggregate["pre_rendered_count"],
            "post_rendered_count": aggregate["post_rendered_count"],
            "reason": aggregate["reason"],
            "attempted": 1,
            "applied": 1,
            "deferred": 0,
        }
        writers.write_relocation(row)
    return aggregate, audit


def save_online_prediction(state_or_model: FixedCapacityChangeState | TemporalChangeModel, view: Camera, pipe: SimpleNamespace, background: torch.Tensor, output_dir: Path, *, protocol_step: int, global_step: int, state_id: int) -> dict[str, Any]:
    pred_dir = output_dir / "online_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if isinstance(state_or_model, FixedCapacityChangeState):
            rendered = mcmc_render(state_or_model, view, pipe, background)["render"]
        else:
            rendered = render_change_temporal(view, state_or_model, pipe, background, timestamp=float(view.timestamp))["render"]
        mean_map = rendered.mean(dim=0).detach().to(torch.float16).cpu()
    path = pred_dir / f"{int(view.timestamp):06d}_{view.image_name}.pt"
    torch.save(mean_map, path)
    relative_path = path.relative_to(output_dir)
    row = {"state_id": int(state_id), "frame": view.image_name, "global_index": int(view.timestamp), "protocol_step": int(protocol_step), "global_step": int(global_step), "path": str(relative_path), "checksum": tensor_checksum(mean_map)}
    with (pred_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def train_dc_only(model: TemporalChangeModel, views: list[Camera], background: torch.Tensor, pipe: SimpleNamespace, args: argparse.Namespace, dataset_audit: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if args.protocol == "oracle_stream":
        schedule, schedule_audit = make_protocol_schedule(views, model.max_states, args, dataset_audit)
    else:
        grouped = views_by_state(views, model.max_states)
        steps = make_training_schedule(grouped, model.max_states, args)
        schedule_audit = training_schedule_audit(views, steps, args, dataset_audit)
        schedule = [(i, step.state, step.local_step, grouped[step.state][step.view_index]) for i, step in enumerate(steps, start=1)]
    if args.protocol == "matched_exact" and not schedule_audit.get("exact_all_images_guarantee"):
        raise RuntimeError(f"dc_only matched_exact schedule failed: {schedule_audit}")
    optimizer: torch.optim.Adam | None = None
    optimizer_state: int | None = None
    logs: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    last_support_protocol_step: int | None = None
    support_updates = support_promotions = 0
    writers = ArtifactWriters(Path(args.output_dir))
    try:
        for global_step, (protocol_step, state, local_step, view) in enumerate(schedule, start=1):
            if optimizer_state != state:
                optimizer = torch.optim.Adam([model.state_change_dc], lr=args.dc_lr, eps=1e-15)
                optimizer_state = state
            assert optimizer is not None
            if args.protocol == "oracle_stream" and last_support_protocol_step != protocol_step:
                support_promotions += update_dc_support_from_view(
                    model,
                    view,
                    pipe,
                    background,
                    support_threshold=args.support_threshold,
                    map_threshold=args.support_map_threshold,
                )
                support_updates += 1
                last_support_protocol_step = protocol_step
            optimizer.zero_grad(set_to_none=True)
            package = render_change_temporal(view, model, pipe, background, timestamp=float(view.timestamp))
            loss, terms = MCMCEnergy(regularizer_offset=args.regularizer_offset)(training_target(view), package["render"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite A0 energy at step {global_step}")
            loss.backward()
            optimizer.step()
            if args.protocol == "oracle_stream" and local_step == args.updates_per_frame - 1:
                predictions.append(save_online_prediction(model, view, pipe, background, Path(args.output_dir), protocol_step=protocol_step, global_step=global_step, state_id=state))
            term_values = tensor_terms_to_float(terms)
            live = int(model.state_valid[:, state].sum().item())
            capacity = int(model.state_valid.shape[0])
            writers.write_energy({
                "global_step": global_step, "protocol_step": protocol_step, "state_id": state,
                "frame": view.image_name, "timestamp": float(view.timestamp), "mode": args.mode,
                "ssf": term_values.get("ssf", float(loss.detach().item())),
                "opacity_regularizer": 0.0, "scale_regularizer": 0.0,
                "unsupported_anchor_regularizer": 0.0, "energy": float(loss.detach().item()),
                "pre_relocation_energy": "", "post_relocation_energy": "",
                "noise_mean_abs": "", "noise_p50_abs": "", "noise_p95_abs": "",
                "noise_rms": "", "noise_max_abs": "", "noise_opacity_bin_stats": "",
                "lr_max": args.dc_lr,
            })
            writers.write_counts({
                "global_step": global_step, "protocol_step": protocol_step, "state_id": state,
                "frame": view.image_name, "timestamp": float(view.timestamp), "support_count": live,
                "rendered_count": int((package["radii"] > 0).sum().detach().item()),
                "capacity": capacity, "N_t": capacity, "dead_count": capacity - live,
                "live_count": live, "selected_dead_count": 0, "selected_live_count": 0,
                "relocation_assignments": 0, "relocation_applied": 0, "relocation_deferred": 0,
            })
            if global_step in {1, len(schedule)} or global_step % args.progress_interval == 0:
                print(f"[dc_only/{args.protocol}] step {global_step}/{len(schedule)} state={state} frame={view.image_name} energy={float(loss.detach()):.6f}", flush=True)
            if len(logs) < 20 or global_step in {1, len(schedule)}:
                logs.append({"global_step": global_step, "protocol_step": protocol_step, "state": state, "frame": view.image_name, "energy": float(loss.detach().item()), **term_values})
    finally:
        writers.close()
    return logs, {
        "optimized_steps": len(schedule),
        "optimizer_reset_on_state_boundary": True,
        "scheduler_policy": "constant_lr_no_scheduler",
        "prefix_support_updates": support_updates if args.protocol == "oracle_stream" else None,
        "support_promotions": support_promotions,
        "all_n_rendered_regardless_support": False,
        "topology_ops": {"densification": False, "pruning": False, "relocation": False},
        "topology_mutation": False,
        "fixed_capacity_relocation": False,
        "online_predictions": predictions,
    }, schedule_audit, []


def train_mcmc_state(manager: OracleBoundaryStateManager, views: list[Camera], background: torch.Tensor, pipe: SimpleNamespace, args: argparse.Namespace, dataset_audit: dict[str, Any] | None, base: GaussianModel, config_hash: str, input_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    schedule, schedule_audit = make_protocol_schedule(views, len(args.boundaries) + 1, args, dataset_audit)
    if args.protocol == "matched_exact" and not schedule_audit.get("exact_all_images_guarantee"):
        raise RuntimeError(f"matched_exact schedule failed: {schedule_audit}")
    energy = MCMCEnergy(
        opacity_weight=(
            args.conservative_opacity_reg_weight
            if args.mode in CONSERVATIVE_RELOCATION_MODES
            else args.opacity_reg_weight
        ),
        scale_weight=args.scale_reg_weight,
        unsupported_anchor_weight=args.unsupported_anchor_weight,
        beta_scale=args.anchor_beta_scale,
        beta_rotation=args.anchor_beta_rotation,
        reduction=args.regularizer_reduction,
        regularizer_offset=args.regularizer_offset,
    )
    writers = ArtifactWriters(Path(args.output_dir))
    optimizer: torch.optim.Adam | None = None
    optimizer_state: int | None = None
    logs: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    archive_audits: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    views_by_sid = views_by_state(views, len(args.boundaries) + 1)
    audit_views_by_sid = {sid: state_views[: args.relocation_audit_views] for sid, state_views in views_by_sid.items()}
    seen_audit_views_by_sid: dict[int, list[Camera]] = {sid: [] for sid in views_by_sid}
    state_local_steps: dict[int, int] = {}
    last_support_protocol_step: int | None = None
    support_updates = support_promotions = relocation_attempts = relocation_applied = relocation_no_live = noise_steps = 0
    relocation_burnin_steps = source_protection_events = 0
    last_noise: dict[str, Any] | None = None
    initial_n = int(manager.current_state.capacity)
    param_ids = {name: id(param) for name, param in mcmc_parameter_items(manager.current_state, "geo_mcmc")}
    initial_snapshots: dict[int, dict[str, str]] = {}
    start = time.time()
    unsupported_anchor = manager.current_state.snapshot(detach=True, cpu=False)
    try:
        for global_step, (protocol_step, scheduled_state, local_step, view) in enumerate(schedule, start=1):
            arrival_view = (
                views[protocol_step - 1]
                if args.protocol == "oracle_stream" and args.mode in CONSERVATIVE_RELOCATION_MODES
                else view
            )
            if args.protocol == "oracle_stream":
                before_len = len(manager.archive)
                state_id = manager.ensure_state(float(arrival_view.timestamp), metadata={"config_hash": config_hash, "input_hash": input_hash, "frame": arrival_view.image_name, "global_step": global_step})
                if len(manager.archive) > before_len:
                    archive_idx = len(manager.archive) - 1
                    archives.append(archive_record_file(manager, archive_idx, Path(args.output_dir)))
                    closed_sid = manager.archive.records[archive_idx].state_id
                    archive_audits.append(archive_render_audit(manager, archive_idx, seen_audit_views_by_sid.get(closed_sid, []), base, pipe, background))
                    reset_state_support_for_new_segment_(
                        manager.current_state,
                        initialize_dead_opacity=args.init_policy == "base_zero",
                    )
                    unsupported_anchor = manager.current_state.snapshot(detach=True, cpu=False)
                if optimizer is None or optimizer_state != state_id:
                    optimizer = mcmc_optimizer(manager.current_state, args)
                    optimizer_state = state_id
            else:
                state_id = int(scheduled_state)
                if optimizer is None or optimizer_state != state_id:
                    before_len = len(manager.archive)
                    boundary_timestamp = float(0 if state_id == 0 else args.boundaries[state_id - 1])
                    actual_state_id = manager.ensure_state(
                        boundary_timestamp,
                        metadata={
                            "config_hash": config_hash,
                            "input_hash": input_hash,
                            "scheduled_state_change": state_id > 0,
                            "global_step": global_step,
                        },
                    )
                    if actual_state_id != state_id:
                        raise RuntimeError(f"oracle manager state mismatch: {actual_state_id} != {state_id}")
                    if len(manager.archive) > before_len:
                        archive_idx = len(manager.archive) - 1
                        archives.append(archive_record_file(manager, archive_idx, Path(args.output_dir)))
                        archive_audits.append(archive_render_audit(manager, archive_idx, audit_views_by_sid.get(manager.archive.records[archive_idx].state_id, []), base, pipe, background))
                        reset_state_support_for_new_segment_(
                            manager.current_state,
                            initialize_dead_opacity=args.init_policy == "base_zero",
                        )
                    unsupported_anchor = manager.current_state.snapshot(detach=True, cpu=False)
                    optimizer = mcmc_optimizer(manager.current_state, args)
                    optimizer_state = state_id
            assert optimizer is not None
            assert_fixed_capacity(manager.current_state, initial_n, param_ids)
            if args.protocol == "oracle_stream" and last_support_protocol_step != protocol_step:
                if args.mode in CONSERVATIVE_RELOCATION_MODES:
                    visible, mask = slot_evidence_from_view(
                        manager.current_state,
                        arrival_view,
                        pipe,
                        background,
                        threshold=args.support_map_threshold,
                    )
                    support_promotions += promote_supported_opacity_(manager.current_state, mask)
                    manager.current_state.record_slot_evidence(visible, mask)
                else:
                    mask = cue_support_mask_from_view(manager.current_state, arrival_view, pipe, background, threshold=args.support_map_threshold)
                    support_promotions += promote_supported_opacity_(manager.current_state, mask)
                support_updates += 1
                last_support_protocol_step = protocol_step
                if args.mode in CONSERVATIVE_RELOCATION_MODES:
                    buffer = seen_audit_views_by_sid[state_id]
                    buffer.append(arrival_view)
                    del buffer[:-args.relocation_audit_views]
                elif len(seen_audit_views_by_sid[state_id]) < args.relocation_audit_views:
                    seen_audit_views_by_sid[state_id].append(arrival_view)
            elif args.protocol == "matched_exact" and state_id not in initial_snapshots:
                state_mask = torch.zeros_like(manager.current_state.support_mask)
                for support_view in views_by_sid.get(state_id, []):
                    if args.mode in CONSERVATIVE_RELOCATION_MODES:
                        visible, supported = slot_evidence_from_view(
                            manager.current_state,
                            support_view,
                            pipe,
                            background,
                            threshold=args.support_map_threshold,
                        )
                        support_promotions += promote_supported_opacity_(manager.current_state, supported)
                        manager.current_state.record_slot_evidence(visible, supported)
                        state_mask |= supported
                    else:
                        state_mask |= cue_support_mask_from_view(
                            manager.current_state,
                            support_view,
                            pipe,
                            background,
                            threshold=args.support_map_threshold,
                        )
                if args.mode not in CONSERVATIVE_RELOCATION_MODES:
                    support_promotions += promote_supported_opacity_(manager.current_state, state_mask)
                support_updates += len(views_by_sid.get(state_id, []))
            if state_id not in initial_snapshots:
                initial_snapshots[state_id] = {name: tensor_checksum(value) for name, value in manager.current_state.snapshot(detach=True, cpu=True).items() if isinstance(value, torch.Tensor)}
            state_local_steps[state_id] = state_local_steps.get(state_id, 0) + 1
            state_local_step = state_local_steps[state_id]
            audit_views = (
                seen_audit_views_by_sid.get(state_id, [])
                if args.protocol == "oracle_stream"
                else audit_views_by_sid.get(state_id, [])
            ) or [view]
            relocation_event, relocation_counts = relocation_audit_and_apply(manager, optimizer, energy, audit_views, pipe, background, unsupported_anchor, writers, args, global_step=global_step, protocol_step=protocol_step, state_id=state_id, state_local_step=state_local_step)
            if relocation_event is not None:
                relocation_attempts += 1
                relocation_applied += int(relocation_event["applied"])
                relocation_no_live += int(relocation_event["reason"] == "no_live")
                relocation_burnin_steps += int(relocation_event.get("burnin_steps", 0))
                source_protection_events += int(relocation_event["reason"] in {"source_occlusion_responsibility", "source_energy_responsibility"})
            optimizer.zero_grad(set_to_none=True)
            total, terms, render_stats = mcmc_energy_value(energy, manager.current_state, view, pipe, background, unsupported_anchor)
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"non-finite MCMC energy at step {global_step}")
            total.backward()
            optimizer.step()
            noise_stats = None
            if args.mode in SGLD_MODES:
                noise_stats = apply_sgld_xyz_noise_(
                    manager.current_state,
                    xyz_lr=args.xyz_lr,
                    noise_scale=args.noise_lr * args.sgld_noise_scale,
                    opacity_gate_k=args.sgld_opacity_gate_k,
                    opacity_gate_threshold=args.sgld_opacity_gate_threshold,
                    seed=args.seed + 100000 + global_step,
                )
                last_noise = {**noise_stats.__dict__}
                noise_steps += 1
            if args.protocol == "oracle_stream" and local_step == args.updates_per_frame - 1:
                predictions.append(save_online_prediction(manager.current_state, arrival_view, pipe, background, Path(args.output_dir), protocol_step=protocol_step, global_step=global_step, state_id=state_id))
            counts = live_dead_counts(manager.current_state, args.dead_opacity_threshold)
            term_values = tensor_terms_to_float(terms)
            writers.write_energy({"global_step": global_step, "protocol_step": protocol_step, "state_id": state_id, "frame": view.image_name, "timestamp": float(view.timestamp), "mode": args.mode, "ssf": term_values.get("ssf", 0.0), "opacity_regularizer": term_values.get("opacity", 0.0), "scale_regularizer": term_values.get("scale", 0.0), "unsupported_anchor_regularizer": term_values.get("unsupported_anchor", 0.0), "energy": float(total.detach().item()), "pre_relocation_energy": "" if relocation_event is None else relocation_event["pre_energy"], "post_relocation_energy": "" if relocation_event is None else relocation_event["post_energy"], "noise_mean_abs": "" if noise_stats is None else noise_stats.noise_mean_abs, "noise_p50_abs": "" if noise_stats is None else noise_stats.noise_p50_abs, "noise_p95_abs": "" if noise_stats is None else noise_stats.noise_p95_abs, "noise_rms": "" if noise_stats is None else noise_stats.noise_rms, "noise_max_abs": "" if noise_stats is None else noise_stats.noise_max_abs, "noise_opacity_bin_stats": "" if noise_stats is None else json.dumps(noise_stats.opacity_bin_stats, sort_keys=True), "lr_max": max(float(group.get("lr", 0.0)) for group in optimizer.param_groups)})
            writers.write_counts({"global_step": global_step, "protocol_step": protocol_step, "state_id": state_id, "frame": view.image_name, "timestamp": float(view.timestamp), "support_count": int(manager.current_state.support_mask.sum().item()), "rendered_count": render_stats["active_gaussians_rendered"], "capacity": int(manager.current_state.capacity), **counts, "selected_dead_count": relocation_counts.get("selected_dead_count", 0), "selected_live_count": relocation_counts.get("selected_live_count", 0), "relocation_assignments": relocation_counts.get("assignment_count", 0), "relocation_applied": relocation_counts.get("applied_count", 0), "relocation_deferred": relocation_counts.get("deferred_count", 0)})
            if global_step in {1, len(schedule)} or global_step % args.progress_interval == 0:
                print(f"[{args.mode}/{args.protocol}] step {global_step}/{len(schedule)} state={state_id} frame={view.image_name} energy={float(total.detach()):.6f} live={counts['live_count']} dead={counts['dead_count']} reloc_applied={relocation_applied} elapsed={time.time()-start:.1f}s", flush=True)
            if len(logs) < 20 or global_step in {1, len(schedule)}:
                logs.append({"global_step": global_step, "protocol_step": protocol_step, "state_id": state_id, "frame": view.image_name, "timestamp": float(view.timestamp), "energy": float(total.detach().item()), "terms": term_values, "counts": counts, "relocation": relocation_counts, "dynamics": dynamics_audit(noise_stats, None)})
        if manager.current_start_time is not None:
            end_time = float(max(getattr(view, "timestamp", 0.0) for view in views) + 1.0)
            manager.close(end_time, metadata={"config_hash": config_hash, "input_hash": input_hash, "closed_by_runner": True, "global_step": len(schedule)})
            archive_idx = len(manager.archive) - 1
            archives.append(archive_record_file(manager, archive_idx, Path(args.output_dir)))
            final_sid = manager.archive.records[archive_idx].state_id
            final_audit_views = seen_audit_views_by_sid.get(final_sid, []) if args.protocol == "oracle_stream" else audit_views_by_sid.get(final_sid, [])
            archive_audits.append(archive_render_audit(manager, archive_idx, final_audit_views, base, pipe, background))
        manager.archive.save(Path(args.output_dir) / "state_archives" / "state_archive_all.pt")
    finally:
        writers.close()
    replay = audit_archive_replay(archive_audits, manager, base, pipe, background, views_by_sid)
    geometry = {str(i): geometry_diagnostics_from_tensors(manager.archive.tensors(i), base, args.dead_opacity_threshold) for i in range(len(manager.archive))}
    if args.mode in RELOCATION_MODES and relocation_attempts > 0 and relocation_applied == 0:
        print(f"[relocation] no relocation applied; audited_attempts={relocation_attempts} no_live_events={relocation_no_live}", flush=True)
    conservative_attempt_logged = args.mode in CONSERVATIVE_RELOCATION_MODES and relocation_attempts > 0
    return logs, {"optimized_steps": len(schedule), "optimizer_reset_on_state_boundary": True, "scheduler_policy": "constant_lr_no_scheduler", "prefix_support_updates": support_updates if args.protocol == "oracle_stream" else None, "support_promotions": support_promotions, "all_n_rendered_regardless_support": True, "initial_state_snapshots": initial_snapshots, "archive_replay_audit": replay, "geometry_diagnostics": geometry, "online_predictions": predictions, "mcmc_dynamics": {"noise_steps": noise_steps, "last_noise_stats": last_noise, "relocation_attempts": relocation_attempts, "relocation_applied_count": relocation_applied, "relocation_no_live_events": relocation_no_live, "row_local_burnin_steps": relocation_burnin_steps, "source_protection_events": source_protection_events, "required_smoke_relocation_or_no_live_logged": bool(relocation_applied > 0 or relocation_no_live > 0 or conservative_attempt_logged or args.mode not in RELOCATION_MODES)}, "topology_ops": {"densification": False, "pruning": False, "relocation": args.mode in RELOCATION_MODES}, "topology_mutation": False, "fixed_capacity_relocation": args.mode in RELOCATION_MODES}, schedule_audit, archives


def support_summary(model_or_state: Any, views: list[Camera]) -> dict[str, Any]:
    if isinstance(model_or_state, FixedCapacityChangeState):
        return {
            "support_policy": "cue_support_mask_metadata_only",
            "unsupported_opacity": UNSUPPORTED_OPACITY,
            "promoted_opacity": SUPPORTED_OPACITY,
            "capacity": int(model_or_state.capacity),
            "support_count_at_summary": int(model_or_state.support_mask.sum().item()),
            "sufficient_observation_metadata": True,
            "observed_slot_count": int((model_or_state.slot_observation_count > 0).sum().item()),
            "carryover_protected_count": int(model_or_state.carryover_protected_mask.sum().item()),
            "removal_protected_count": int(model_or_state.removal_protected_mask.sum().item()),
            "tentative_slot_count": int(model_or_state.tentative_mask.sum().item()),
            "relocated_slot_count": int((model_or_state.slot_relocation_count > 0).sum().item()),
        }
    valid = model_or_state.state_valid.detach()
    return {"support_policy": "corrected_temporal_state_valid_hard_gate", "valid_gaussians_per_state": valid.sum(dim=0).cpu().tolist()}


def save_outputs(model_or_state: TemporalChangeModel | FixedCapacityChangeState, manager: OracleBoundaryStateManager | None, args: argparse.Namespace, manifest: dict[str, Any], records: list[FrameRecord], poses: dict[str, Any], intrinsics: np.ndarray, cue_metadata: dict[str, Any], inputs: dict[str, Any], support: dict[str, Any], loss_summary: dict[str, Any], train_log: list[dict[str, Any]], training_audit: dict[str, Any], schedule_audit: dict[str, Any], archives: list[dict[str, Any]], subset: dict[str, Any] | None = None, config_hash: str = "", input_hash: str = "") -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    subset = dict(subset or {"enabled": False, "policy": "unspecified"})
    if not config_hash:
        config_hash = sha256_payload({"mode": args.mode, "protocol": args.protocol, "subset": subset})
    if not input_hash:
        input_hash = sha256_payload(inputs)
    ckpt_path = output_dir / "temporal_rchange_checkpoint.pt"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    config_path = output_dir / "config.json"
    input_hashes_path = output_dir / "input_hashes.json"
    base_ply = (Path(args.source_path) / BASE_PLY_REL).resolve()
    contract = f"fixed_topology_oracle_boundary_{args.mode}_{args.protocol}_no_gt"
    if isinstance(model_or_state, FixedCapacityChangeState):
        state_payload = model_or_state.snapshot(detach=True, cpu=True)
        state_dict = {name: value.detach().cpu() for name, value in model_or_state.state_dict().items()}
        archive_state_dict = None if manager is None else manager.archive.state_dict()
        parameter_checksums = {key: tensor_checksum(value) for key, value in state_payload.items() if isinstance(value, torch.Tensor)}
        gaussian_count = int(model_or_state.capacity)
    else:
        state_payload = None
        state_dict = {name: value.detach().cpu() for name, value in model_or_state.state_dict().items()}
        archive_state_dict = None
        parameter_checksums = {"state_change_dc": tensor_checksum(model_or_state.state_change_dc)}
        gaussian_count = int(model_or_state.state_change_dc.shape[0])
    metadata = {"schema_version": 3, "contract": contract, "mode": args.mode, "protocol": args.protocol, "config_hash": config_hash, "input_hash": input_hash, "input_hashes": inputs, "subset": subset, "gt_used_for_training": False, "topology_ops": training_audit["topology_ops"], "schedule": schedule_audit, "training_audit": training_audit}
    torch.save({"state_dict": state_dict, "current_snapshot": state_payload, "state_archive": archive_state_dict, "base_ply": str(base_ply), "boundaries": list(args.boundaries), "contract": contract, "metadata": metadata}, ckpt_path)
    config_path.write_text(json.dumps(serializable_arguments(args), indent=2) + "\n", encoding="utf-8")
    input_hashes_path.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    run_manifest = {"schema_version": 3, "script": "experiments/train_oracle_boundary_mcmc_rchange.py", "created_at_unix": time.time(), "contract": contract, "config_hash": config_hash, "input_hash": input_hash, "run_arguments": serializable_arguments(args), "input_hashes": inputs, "subset": subset, "artifacts": {"checkpoint": str(ckpt_path), "summary": str(summary_path), "config": str(config_path), "input_hashes": str(input_hashes_path), "training_energy_csv": str(output_dir / "training_energy.csv"), "gaussian_counts_csv": str(output_dir / "gaussian_counts.csv"), "relocation_events_jsonl": str(output_dir / "relocation_events.jsonl"), "relocation_audit_csv": str(output_dir / "relocation_audit.csv"), "state_archives": archives}}
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    summary = {"script": "experiments/train_oracle_boundary_mcmc_rchange.py", "contract": contract, "created_at_unix": time.time(), "runtime_seconds": time.time() - args.started_at, "runtime_per_frame_seconds": (time.time() - args.started_at) / max(1, len(records)), "peak_cuda_memory_bytes": int(peak_memory), "mode": args.mode, "protocol": args.protocol, "init_policy": args.init_policy, "target_switch_known": True, "topology_mutation": False, "fixed_capacity_relocation": bool(training_audit["topology_ops"].get("relocation", False)), "source_path": str(args.source_path), "supervision": cue_metadata.get("candidate_map_definition", "cached cue"), "oracle_supervision": False, "gt_used_for_training": False, "gt_mask_pixels_loaded": 0, "no_gt_path_audit": {"gt_mask_paths_required": False, "gt_masks_opened": 0, "training_signal": "cached_oscd_cues"}, "manual_boundaries": True, "bocd": False, "fixed_gaussian_topology": True, "densify_prune": False, "topology_ops": training_audit["topology_ops"], "optimized_parameters": [name for name, _ in (mcmc_parameter_items(model_or_state, args.mode) if isinstance(model_or_state, FixedCapacityChangeState) else (("state_change_dc", model_or_state.state_change_dc),))], "resolution": float(args.resolution), "boundaries": list(args.boundaries), "run_arguments": serializable_arguments(args), "config_hash": config_hash, "input_hash": input_hash, "input_hashes": inputs, "input_hash_validation": getattr(args, "input_hash_validation", {"checked": False, "matched": None}), "subset": subset, "manifest_counts": manifest.get("counts"), "train_frames": [asdict(record) for record in records], "pose_policy": "O-SCD fixed canonical pose reused from fixed-pose protocol", "pose_results": {name: asdict(result) for name, result in poses.items()}, "camera_intrinsics": intrinsics.tolist(), "support": support, "loss_summary": loss_summary, "train_log": train_log, "training_schedule": schedule_audit, "training_audit": training_audit, "gaussian_count": gaussian_count, "parameter_checksums": parameter_checksums, "state_archives": archives, "base_ply_sha256": file_checksum(base_ply), "checkpoint_sha256": "", "artifact_sha256": {}}
    summary["checkpoint_sha256"] = file_checksum(ckpt_path)
    artifact_paths = [ckpt_path, manifest_path, config_path, input_hashes_path]
    for name in ("training_energy.csv", "gaussian_counts.csv", "relocation_events.jsonl", "relocation_audit.csv"):
        path = output_dir / name
        if path.exists():
            artifact_paths.append(path)
    summary["artifact_sha256"] = {path.name: file_checksum(path) for path in artifact_paths}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return ckpt_path, summary_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified fixed-topology oracle-boundary MCMC R_change runner")
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--mode", choices=MODES, default="geo_mcmc")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="oracle_stream")
    parser.add_argument("--init-policy", choices=INIT_POLICIES, default="previous")
    parser.add_argument("--state-init", dest="init_policy", choices=INIT_POLICIES, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--updates-per-frame", type=positive_int, default=None)
    parser.add_argument("--iterations-per-frame", dest="updates_per_frame", type=positive_int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--prefix-frames", type=positive_int, default=None)
    parser.add_argument("--frames-per-state", type=positive_int, default=None)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--capacity", type=positive_int, default=None)
    parser.add_argument("--dc-lr", type=float, default=0.0025)
    parser.add_argument("--xyz-lr", type=float, default=1.6e-4)
    parser.add_argument("--opacity-lr", type=float, default=0.05)
    parser.add_argument("--scaling-lr", type=float, default=0.005)
    parser.add_argument("--rotation-lr", type=float, default=0.001)
    parser.add_argument("--opacity-reg-weight", type=float, default=0.01)
    parser.add_argument("--scale-reg-weight", type=float, default=0.01)
    parser.add_argument("--unsupported-anchor-weight", type=float, default=None)
    parser.add_argument("--regularizer-offset", type=float, default=1.0)
    parser.add_argument("--regularizer-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--noise-lr", type=float, default=5e5)
    parser.add_argument("--sgld-noise-lr", dest="noise_lr", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--sgld-noise-scale", type=float, default=1.0)
    parser.add_argument("--sgld-opacity-gate-k", type=float, default=100.0)
    parser.add_argument("--sgld-opacity-gate-threshold", type=float, default=0.005)
    parser.add_argument("--relocation-warmup-steps", type=nonnegative_int, default=500)
    parser.add_argument("--mcmc-warmup", dest="relocation_warmup_steps", type=nonnegative_int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--relocation-interval", type=positive_int, default=100)
    parser.add_argument("--relocation-count", type=nonnegative_int, default=10000)
    parser.add_argument("--max-relocations-per-step", dest="relocation_count", type=nonnegative_int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--dead-opacity-threshold", dest="dead_opacity_threshold", type=float, default=0.005)
    parser.add_argument("--relocation-opacity-threshold", dest="dead_opacity_threshold", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--relocation-max-group-total", type=positive_int, default=50)
    parser.add_argument("--relocation-audit-views", type=positive_int, default=8)
    parser.add_argument("--replay-buffer-size", type=positive_int, default=8)
    parser.add_argument("--conservative-min-observations", type=positive_int, default=3)
    parser.add_argument("--conservative-max-relocations", type=nonnegative_int, default=256)
    parser.add_argument("--conservative-residual-topk", type=positive_int, default=32)
    parser.add_argument("--conservative-residual-threshold", type=float, default=0.05)
    parser.add_argument("--conservative-min-support-views", type=positive_int, default=2)
    parser.add_argument("--conservative-max-ray-distance-ratio", type=float, default=0.02)
    parser.add_argument("--conservative-max-candidates", type=positive_int, default=1024)
    parser.add_argument("--conservative-scene-quantile", type=float, default=0.01)
    parser.add_argument("--conservative-scene-margin", type=float, default=0.10)
    parser.add_argument("--conservative-transport-opacity", type=float, default=1e-6)
    parser.add_argument("--conservative-tentative-opacity", type=float, default=0.005)
    parser.add_argument("--conservative-max-relative-energy-increase", type=float, default=0.0)
    parser.add_argument("--conservative-min-coverage-gain", type=float, default=1e-6)
    parser.add_argument("--conservative-max-deactivation-relative-energy-increase", type=float, default=0.0)
    parser.add_argument("--conservative-max-negative-mass-increase", type=float, default=0.0)
    parser.add_argument("--conservative-burnin-steps", type=nonnegative_int, default=0)
    parser.add_argument("--conservative-burnin-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--conservative-opacity-reg-weight", type=float, default=0.0)
    parser.add_argument("--anchor-beta", type=float, default=None)
    parser.add_argument("--anchor-beta-xyz", type=float, default=1.0)
    parser.add_argument("--anchor-beta-scale", type=float, default=1.0)
    parser.add_argument("--anchor-beta-rotation", type=float, default=1.0)
    parser.add_argument("--anchor-beta-opacity", type=float, default=1.0)
    parser.add_argument("--support-map-threshold", type=float, default=0.5)
    parser.add_argument("--support-threshold", type=positive_int, default=1)
    parser.add_argument("--expected-input-hashes", type=str, default=None)
    parser.add_argument("--gradient-audit-interval", type=positive_int, default=100)
    parser.add_argument("--progress-interval", type=positive_int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-full-stream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.updates_per_frame is None:
        args.updates_per_frame = 16 if args.protocol == "oracle_stream" else 120
    return args


def main() -> None:
    args = parse_args()
    args.started_at = time.time()
    if args.prefix_frames is not None and args.frames_per_state is not None:
        raise ValueError("--prefix-frames and --frames-per-state are mutually exclusive")
    if args.unsupported_anchor_weight is None:
        args.unsupported_anchor_weight = 0.01 if args.mode == "geo_mcmc_anchor" else 0.0
    if args.anchor_beta is None:
        args.anchor_beta = args.unsupported_anchor_weight
    if args.mode in CONSERVATIVE_RELOCATION_MODES:
        if args.conservative_min_support_views > args.relocation_audit_views:
            raise ValueError("conservative min-support views cannot exceed relocation audit views")
        for name in ("conservative_transport_opacity", "conservative_tentative_opacity"):
            value = float(getattr(args, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be inside (0, 1)")
        if args.conservative_transport_opacity >= args.conservative_tentative_opacity:
            raise ValueError("transport opacity must be smaller than tentative opacity")
        if args.conservative_residual_threshold < 0.0:
            raise ValueError("conservative residual threshold must be nonnegative")
        if args.conservative_max_ray_distance_ratio <= 0.0:
            raise ValueError("conservative max ray-distance ratio must be positive")
        if not 0.0 <= args.conservative_scene_quantile < 0.5:
            raise ValueError("conservative scene quantile must be in [0, 0.5)")
        if args.conservative_scene_margin < 0.0:
            raise ValueError("conservative scene margin must be nonnegative")
        if args.conservative_max_relative_energy_increase < 0.0:
            raise ValueError("conservative max relative energy increase must be nonnegative")
        if args.conservative_min_coverage_gain < 0.0:
            raise ValueError("conservative minimum coverage gain must be nonnegative")
        if args.conservative_max_deactivation_relative_energy_increase < 0.0:
            raise ValueError("conservative max deactivation energy increase must be nonnegative")
        if args.conservative_max_negative_mass_increase < 0.0:
            raise ValueError("conservative max negative mass increase must be nonnegative")
        if args.conservative_burnin_lr_multiplier <= 0.0:
            raise ValueError("conservative burn-in LR multiplier must be positive")
        if args.conservative_opacity_reg_weight < 0.0:
            raise ValueError("conservative opacity regularizer weight must be nonnegative")
    if args.protocol == "oracle_stream" and args.prefix_frames is None and args.frames_per_state is None and not args.allow_full_stream:
        raise ValueError("oracle_stream supports bounded runs by default; pass --prefix-frames, --frames-per-state, or --allow-full-stream")
    if args.protocol == "matched_exact" and (args.prefix_frames is not None or args.frames_per_state is not None):
        raise ValueError("--prefix-frames/--frames-per-state are only supported by oracle_stream")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing Gaussian renderer")
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {args.output_dir}; pass --overwrite for this run directory")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.source_path)
    args.boundaries = tuple(args.boundaries if args.boundaries is not None else boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES))
    records, all_names = build_no_gt_frame_records(args.source_path, args.boundaries, prefix_frames=args.prefix_frames, frames_per_state=args.frames_per_state)
    assert_no_gt_training_records(records)
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    views, poses, intrinsics = build_fixed_cue_views(records, load_fixed_camera_index(args.fixed_cameras_json), args.cue_cache_root, args.resolution)
    dataset_audit = validate_exact_dataset_contract(manifest, all_names, records, views, args.updates_per_frame) if args.protocol == "matched_exact" else None
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    subset = slice_gaussian_model_(base, args.capacity)
    args.capacity = int(subset["N_t"])
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    inputs = input_hashes(args, records, all_names)
    input_hash = sha256_payload(inputs)
    config_payload = {**serializable_arguments(args), "subset": subset}
    config_hash = sha256_payload(config_payload)
    args.input_hash_validation = validate_expected_input_hashes(inputs, args.expected_input_hashes)
    if args.mode == "dc_only":
        model = build_temporal_dc_model(base, args.boundaries, len(all_names))
        if args.protocol == "matched_exact":
            support = compute_state_support(
                model,
                views,
                background,
                pipe,
                args.support_threshold,
                map_threshold=args.support_map_threshold,
            )
        else:
            with torch.no_grad():
                model.state_valid.zero_()
            support = {"support_policy": "causal_prefix_temporal_state_valid_hard_gate", "support_threshold": args.support_threshold, "support_map_threshold": args.support_map_threshold}
        pre_loss = summarize_losses(model, views, background, pipe) if args.protocol == "matched_exact" else None
        train_log, training_audit, schedule_audit, archives = train_dc_only(model, views, background, pipe, args, dataset_audit)
        post_loss = summarize_losses(model, views, background, pipe) if args.protocol == "matched_exact" else None
        support = {**support, **support_summary(model, views), "prefix_support_updates": training_audit.get("prefix_support_updates")}
        manager = None
        saved: TemporalChangeModel | FixedCapacityChangeState = model
    else:
        # S0 always starts from reference geometry with zero DC and dead change
        # opacity. ``previous`` controls only later oracle-boundary warm starts.
        current = FixedCapacityChangeState.from_gaussians(base, capacity=args.capacity, base_zero=True)
        configure_mcmc_trainable(current, args.mode)
        reset_state_support_for_new_segment_(current, initialize_dead_opacity=True)
        manager = OracleBoundaryStateManager(current, boundaries=args.boundaries, archive=StateArchive(metadata={"mode": args.mode, "protocol": args.protocol, "config_hash": config_hash, "input_hash": input_hash, "subset": subset}), warm_start=args.init_policy == "previous", base_zero=True)
        train_log, training_audit, schedule_audit, archives = train_mcmc_state(manager, views, background, pipe, args, dataset_audit, base, config_hash, input_hash)
        support = {**support_summary(current, views), "prefix_support_updates": training_audit.get("prefix_support_updates"), "support_promotions": training_audit.get("support_promotions")}
        pre_loss = post_loss = None
        saved = current
    ckpt, summary, run_manifest = save_outputs(saved, manager, args, manifest, records, poses, intrinsics, cue_metadata, inputs, support, {"pre_train": pre_loss, "post_train": post_loss}, train_log, training_audit, schedule_audit, archives, subset, config_hash, input_hash)
    print(json.dumps({"checkpoint": str(ckpt), "summary": str(summary), "manifest": str(run_manifest), "frames": len(views), "updates": schedule_audit["actual_total_updates"], "gt_used_for_training": False, "topology_ops": training_audit["topology_ops"], "relocation_applied_count": training_audit.get("mcmc_dynamics", {}).get("relocation_applied_count"), "relocation_no_live_events": training_audit.get("mcmc_dynamics", {}).get("relocation_no_live_events")}, indent=2))


if __name__ == "__main__":
    main()
