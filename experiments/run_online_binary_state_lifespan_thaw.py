"""Causal online direct binary-state lifespans with thawed temporal slots.

This runner is an independent ablation from the BOCD runners.  It keeps the
same immutable-reference alpha-T evidence path and temporal representation, but
replaces run-length inference with a direct two-state Bayesian filter over
``P(z_t=active | D_1:t)``.  GT masks and manual boundaries are used only after
the frame-major causal loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from temporal.binary_state_filter import (
        BinaryStateFilter,
        BinaryStateFilterConfig,
        BinaryStateFilterUpdate,
    )
    from temporal.binary_state_lifespan_controller import (
        BinaryLifespanAction,
        BinaryStateLifespanController,
        BinaryStateLifespanControllerConfig,
    )
    from temporal.view_consistent_binary_lifespan_controller import (
        ViewConsistentBinaryLifespanController,
        ViewConsistentBinaryLifespanControllerConfig,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - package __init__ imports renderer-only deps in unit env.
    import importlib.util
    import sys
    import types
    if (exc.name or "").split(".")[0] not in {
        "plyfile",
        "simple_knn",
        "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_fastgs",
    }:
        raise
    temporal_pkg = types.ModuleType("temporal")
    temporal_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "temporal")]
    sys.modules.setdefault("temporal", temporal_pkg)
    root = Path(__file__).resolve().parents[1] / "temporal"
    spec = importlib.util.spec_from_file_location("temporal.binary_state_filter", root / "binary_state_filter.py")
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["temporal.binary_state_filter"] = module
    spec.loader.exec_module(module)
    spec2 = importlib.util.spec_from_file_location("temporal.binary_state_lifespan_controller", root / "binary_state_lifespan_controller.py")
    if spec2 is None or spec2.loader is None:
        raise
    module2 = importlib.util.module_from_spec(spec2)
    sys.modules["temporal.binary_state_lifespan_controller"] = module2
    spec2.loader.exec_module(module2)
    BinaryStateFilter = module.BinaryStateFilter
    BinaryStateFilterConfig = module.BinaryStateFilterConfig
    BinaryStateFilterUpdate = module.BinaryStateFilterUpdate
    BinaryLifespanAction = module2.BinaryLifespanAction
    BinaryStateLifespanController = module2.BinaryStateLifespanController
    BinaryStateLifespanControllerConfig = module2.BinaryStateLifespanControllerConfig
    spec3 = importlib.util.spec_from_file_location("temporal.view_consistent_binary_lifespan_controller", root / "view_consistent_binary_lifespan_controller.py")
    if spec3 is None or spec3.loader is None:
        raise
    module3 = importlib.util.module_from_spec(spec3)
    sys.modules["temporal.view_consistent_binary_lifespan_controller"] = module3
    spec3.loader.exec_module(module3)
    ViewConsistentBinaryLifespanController = module3.ViewConsistentBinaryLifespanController
    ViewConsistentBinaryLifespanControllerConfig = module3.ViewConsistentBinaryLifespanControllerConfig

try:
    from temporal.change_evidence import evidence_counts
except ModuleNotFoundError as exc:  # pragma: no cover - renderer-only deps unavailable in unit env.
    if (exc.name or "").split(".")[0] not in {
        "plyfile",
        "simple_knn",
        "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_fastgs",
    }:
        raise
    def evidence_counts(positive, negative, *, mode="capped", mass_saturation=1.0, min_evidence_mass=1e-6):
        if mode == "raw":
            return positive, negative, positive, negative
        mass = positive + negative
        scale = (mass / float(mass_saturation)).clamp(max=1.0) / mass.clamp_min(torch.finfo(mass.dtype).eps)
        scale = torch.where(mass >= float(min_evidence_mass), scale, torch.zeros_like(scale))
        return positive * scale, negative * scale, positive, negative


try:
    from experiments.run_online_bayesian_lifespan_thaw import (
        BASE_PLY_REL, DEFAULT_BOUNDARIES, DEFAULT_CUE_CACHE, DEFAULT_FIXED_CAMERAS, DEFAULT_SOURCE,
        GEOMETRY_NAMES, STAT_KEYS, ClosedPairAudit, base_checksums, base_drift, base_snapshots,
        binary_metrics, build_causal_records, capture_pair_audit, combined_checksum, compare_pair_audit,
        current_pair_mask, nonnegative_float, nonnegative_int, parse_thaw_parameters, positive_int,
        quantile_summary, serializable_arguments, summarize_binary_metric_rows, transition_diagnostics,
        write_visualization_artifacts,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-light synthetic tests.
    if (exc.name or "").split(".")[0] not in {
        "plyfile",
        "simple_knn",
        "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_fastgs",
    }:
        raise
    import hashlib
    BASE_PLY_REL = "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
    DEFAULT_BOUNDARIES = (95, 199)
    DEFAULT_CUE_CACHE = Path("/home/rvl/workspace/github/O-SCD/artifacts/escd_396ref/fixed_pose_cues_res4_v1")
    DEFAULT_FIXED_CAMERAS = Path("/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/scene_change1_2_3/cameras_fixed.json")
    DEFAULT_SOURCE = "data/Instance_1/scene_change1_2_3"
    GEOMETRY_NAMES = ("dc", "xyz", "opacity", "scaling", "rotation")
    STAT_KEYS = ("count", "mean", "q05", "q50", "q95")

    class ClosedPairAudit:
        def __init__(self, max_pairs: int):
            self.max_pairs = int(max_pairs); self.rows = []; self.slots = []; self.total_closed_pairs = 0
        def add(self, rows, slots):
            rv = rows.detach().cpu().tolist(); sv = slots.detach().cpu().tolist(); self.total_closed_pairs += len(rv)
            avail = max(0, self.max_pairs - len(self.rows)); self.rows.extend(int(v) for v in rv[:avail]); self.slots.extend(int(v) for v in sv[:avail])
        def coordinates(self, device):
            return torch.tensor(self.rows, device=device, dtype=torch.long), torch.tensor(self.slots, device=device, dtype=torch.long)
        @property
        def exhaustive(self):
            return self.total_closed_pairs <= self.max_pairs

    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0: raise argparse.ArgumentTypeError("value must be positive")
        return parsed
    def nonnegative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0: raise argparse.ArgumentTypeError("value must be nonnegative")
        return parsed
    def nonnegative_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0: raise argparse.ArgumentTypeError("value must be finite and nonnegative")
        return parsed
    def parse_thaw_parameters(value):
        parts = tuple(p.strip() for p in value.split(",") if p.strip()) if isinstance(value, str) else tuple(value)
        if not parts or len(set(parts)) != len(parts) or any(p not in GEOMETRY_NAMES for p in parts): raise argparse.ArgumentTypeError("bad thaw parameters")
        if tuple(n for n in GEOMETRY_NAMES if n in parts) != parts or parts[0] != "dc": raise argparse.ArgumentTypeError("bad thaw parameter order")
        return parts
    def serializable_arguments(args):
        return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    def _tensor_hash(t):
        a=t.detach().cpu().contiguous().numpy(); h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape,dtype=np.int64).tobytes()); h.update(a.tobytes()); return h.hexdigest()
    def base_snapshots(base):
        return {n: getattr(base,n).detach().cpu().clone() for n in ("_xyz","_features_dc","_features_rest","_opacity","_scaling","_rotation") if isinstance(getattr(base,n,None), torch.Tensor)}
    def base_checksums(base): return {n: _tensor_hash(v) for n,v in base_snapshots(base).items()}
    def combined_checksum(cs):
        h=hashlib.sha256()
        for n in sorted(cs): h.update(n.encode()); h.update(cs[n].encode())
        return h.hexdigest()
    def base_drift(before, base):
        maxv=0.0; equal=True; per={}
        for n,e in before.items():
            a=getattr(base,n).detach().cpu(); equal = equal and torch.equal(a,e); d=float((a-e).abs().max().item()) if e.numel() else 0.0; per[n]=d; maxv=max(maxv,d)
        return {"bitwise_equal": bool(equal), "max_abs": maxv, "per_tensor_max_abs": per}
    def quantile_summary(values):
        values=values.detach().flatten().float().cpu(); values=values[torch.isfinite(values)]
        if values.numel()==0: return {"count":0,"mean":None,"q05":None,"q50":None,"q95":None}
        return {"count":int(values.numel()),"mean":float(values.mean()),"q05":float(torch.quantile(values,0.05)),"q50":float(torch.quantile(values,0.5)),"q95":float(torch.quantile(values,0.95))}
    def binary_metrics(prediction, target):
        p=prediction.astype(bool,copy=False); t=target.astype(bool,copy=False); tp=int(np.logical_and(p,t).sum()); tn=int(np.logical_and(~p,~t).sum()); fp=int(np.logical_and(p,~t).sum()); fn=int(np.logical_and(~p,t).sum()); prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"iou":tp/(tp+fp+fn) if tp+fp+fn else 0.0,"f1":2*prec*rec/(prec+rec) if prec+rec else 0.0,"precision":prec,"recall":rec}
    def current_pair_mask(model):
        m=torch.zeros_like(model.state_valid,dtype=torch.bool); rows=torch.arange(m.shape[0],device=m.device); slots=model.current_state_index.long(); active=slots>=0; m[rows[active],slots[active]]=True; return m
    def capture_pair_audit(model, optimizer, rows, slots):
        return {n:{"parameter":p.detach()[rows,slots].cpu().clone()} for n,p in model.state_parameter_items()}
    def compare_pair_audit(before, model, optimizer, rows, slots, audit):
        maxv=0.0; per={}; params=dict(model.state_parameter_items())
        for n,vals in before.items():
            d=float((params[n].detach()[rows,slots].cpu()-vals["parameter"]).abs().max().item()) if vals["parameter"].numel() else 0.0; per[f"{n}.parameter"]=d; maxv=max(maxv,d)
        return {"passed": maxv==0.0, "max_abs": maxv, "audited_pair_count": len(audit.rows), "total_closed_pair_count": audit.total_closed_pairs, "exhaustive": audit.exhaustive, "per_parameter_and_optimizer_state_max_abs": per}
    def build_causal_records(source_path, *, max_frames=None): raise RuntimeError("heavy dependencies unavailable")
    def summarize_binary_metric_rows(rows): return {}
    def transition_diagnostics(frame_rows, boundaries, *, radius=3): return []
    def write_visualization_artifacts(**kwargs): return None


DEFAULT_OUTPUT = Path("outputs/instance1_scene_change1_2_3_online_binary_state_lifespan_thaw")
OUTPUT_FILES = (
    "summary.json",
    "frame_metrics.csv",
    "lifecycle_events.jsonl",
    "per_frame_binary_state_stats.npz",
    "checkpoint.pt",
)


@dataclass(frozen=True)
class RunConfig:
    bayes_cue_mode: str = "binary"
    bayes_cue_threshold: float = 0.5
    bayes_cue_scale: float = 1.0
    evidence_count_mode: str = "capped"
    evidence_mass_saturation: float = 1.0
    min_evidence_mass: float = 1e-6
    state_emission_reliability: float = 0.9
    inactive_to_active_prior: float = 0.01
    active_to_inactive_prior: float = 0.01
    initial_active_probability: float = 0.5
    filter_chunk_size: int = 65536
    lifecycle_controller: str = "posterior_hysteresis"
    open_probability: float = 0.6
    close_probability: float = 0.4
    transition_confirmation_views: int = 2
    min_transition_bayes_factor: float = 3.0
    min_transition_evidence_strength: float = 1e-6
    max_states: int = 8
    thaw_parameters: tuple[str, ...] = ("dc",)
    updates_per_frame: int = 120
    detector_only: bool = False
    evaluation_threshold: float = 0.5
    seed: int = 0


def validate_cue_camera_checksum(
    cue_metadata: Mapping[str, Any], actual_camera_checksum: str
) -> None:
    cached_camera_checksum = cue_metadata.get("fixed_cameras_sha256")
    if (
        cached_camera_checksum is not None
        and cached_camera_checksum != actual_camera_checksum
    ):
        raise ValueError(
            "cue cache/fixed camera checksum mismatch: "
            f"{cached_camera_checksum} != {actual_camera_checksum}"
        )


@dataclass(frozen=True)
class LifecycleEvent:
    gaussian_index: int
    decision_timestamp: int
    old_binary_label: int
    new_binary_label: int
    action: str
    old_slot: int
    new_current_slot: int
    p_active: float
    p_01: float
    p_10: float
    p_flip: float
    visible_observation_count: int


class ExactClosedPairArchive:
    """Snapshot every CLOSED row-slot pair and verify it after all future steps."""

    def __init__(self) -> None:
        self._entries: list[
            tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]
        ] = []
        self.total_closed_pairs = 0

    def add(
        self,
        model: Any,
        optimizer: Any | None,
        rows: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        rows = rows.detach().flatten().long()
        slots = slots.detach().flatten().long()
        if rows.shape != slots.shape or bool((slots < 0).any()):
            raise ValueError("closed audit requires matching valid row-slot pairs")
        if rows.numel() == 0:
            return
        snapshot = capture_pair_audit(model, optimizer, rows, slots)
        self._entries.append((rows.cpu().clone(), slots.cpu().clone(), snapshot))
        self.total_closed_pairs += int(rows.numel())

    def status(self) -> dict[str, Any]:
        return {
            "passed": None,
            "max_abs": None,
            "audited_pair_count": int(self.total_closed_pairs),
            "total_closed_pair_count": int(self.total_closed_pairs),
            "exhaustive": True,
            "verification_stage": "deferred_until_all_future_updates_finish",
            "per_parameter_and_optimizer_state_max_abs": {},
        }

    @torch.no_grad()
    def verify(self, model: Any, optimizer: Any | None) -> dict[str, Any]:
        maximum = 0.0
        per_state: dict[str, float] = {}
        parameters = dict(model.state_parameter_items())
        device = model.state_valid.device
        for rows_cpu, slots_cpu, snapshot in self._entries:
            rows = rows_cpu.to(device=device)
            slots = slots_cpu.to(device=device)
            for name, values in snapshot.items():
                parameter = parameters[name]
                for state_name, expected in values.items():
                    if state_name == "parameter":
                        actual = parameter.detach()[rows, slots].cpu()
                    else:
                        if optimizer is None or parameter not in optimizer.state:
                            raise RuntimeError("closed-pair optimizer audit state disappeared")
                        actual = optimizer.state[parameter][state_name].detach()[rows, slots].cpu()
                    difference = (
                        float((actual - expected).abs().max().item())
                        if expected.numel()
                        else 0.0
                    )
                    key = f"{name}.{state_name}"
                    per_state[key] = max(per_state.get(key, 0.0), difference)
                    maximum = max(maximum, difference)
        return {
            "passed": maximum == 0.0,
            "max_abs": maximum,
            "audited_pair_count": int(self.total_closed_pairs),
            "total_closed_pair_count": int(self.total_closed_pairs),
            "exhaustive": True,
            "verification_stage": "final_after_all_future_updates",
            "per_parameter_and_optimizer_state_max_abs": per_state,
        }


def validate_run_config(config: RunConfig) -> None:
    if config.bayes_cue_mode not in {"binary", "soft"}:
        raise ValueError("bayes_cue_mode must be binary|soft")
    if config.evidence_count_mode not in {"raw", "capped"}:
        raise ValueError("evidence_count_mode must be raw|capped")
    if not 0.0 < float(config.bayes_cue_threshold) < 1.0:
        raise ValueError("bayes_cue_threshold must be in (0, 1)")
    if not math.isfinite(float(config.bayes_cue_scale)) or config.bayes_cue_scale <= 0:
        raise ValueError("bayes_cue_scale must be finite and positive")
    if not math.isfinite(float(config.evidence_mass_saturation)) or config.evidence_mass_saturation <= 0:
        raise ValueError("evidence_mass_saturation must be finite and positive")
    if not math.isfinite(float(config.min_evidence_mass)) or config.min_evidence_mass < 0:
        raise ValueError("min_evidence_mass must be finite and nonnegative")
    for name in ("state_emission_reliability", "inactive_to_active_prior", "active_to_inactive_prior", "initial_active_probability", "open_probability", "close_probability", "evaluation_threshold"):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if not (0.5 < float(config.state_emission_reliability) < 1.0):
        raise ValueError("state_emission_reliability must be in (0.5, 1)")
    if not (0.0 < float(config.inactive_to_active_prior) < 1.0):
        raise ValueError("inactive_to_active_prior must be in (0, 1)")
    if not (0.0 < float(config.active_to_inactive_prior) < 1.0):
        raise ValueError("active_to_inactive_prior must be in (0, 1)")
    if config.lifecycle_controller not in {"posterior_hysteresis", "view_consistent"}:
        raise ValueError("lifecycle_controller must be posterior_hysteresis or view_consistent")
    if not (0.0 < float(config.close_probability) < float(config.open_probability) < 1.0):
        raise ValueError("close_probability/open_probability must satisfy 0 < close < open < 1")
    if isinstance(config.transition_confirmation_views, bool) or int(config.transition_confirmation_views) < 1:
        raise ValueError("transition_confirmation_views must be positive")
    if not math.isfinite(float(config.min_transition_bayes_factor)) or float(config.min_transition_bayes_factor) <= 0:
        raise ValueError("min_transition_bayes_factor must be finite and positive")
    if not math.isfinite(float(config.min_transition_evidence_strength)) or float(config.min_transition_evidence_strength) < 0:
        raise ValueError("min_transition_evidence_strength must be finite and nonnegative")
    if not 0.0 < float(config.evaluation_threshold) < 1.0:
        raise ValueError("evaluation_threshold must be in (0, 1)")
    if min(config.filter_chunk_size, config.max_states, config.updates_per_frame) < 1:
        raise ValueError("chunk/state/update sizes must be positive")
    parse_thaw_parameters(config.thaw_parameters)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        bayes_cue_mode=args.bayes_cue_mode,
        bayes_cue_threshold=float(args.bayes_cue_threshold),
        bayes_cue_scale=float(args.bayes_cue_scale),
        evidence_count_mode=args.evidence_count_mode,
        evidence_mass_saturation=float(args.evidence_mass_saturation),
        min_evidence_mass=float(args.min_evidence_mass),
        state_emission_reliability=float(args.state_emission_reliability),
        inactive_to_active_prior=float(args.inactive_to_active_prior),
        active_to_inactive_prior=float(args.active_to_inactive_prior),
        initial_active_probability=float(args.initial_active_probability),
        filter_chunk_size=int(args.filter_chunk_size),
        lifecycle_controller=str(args.lifecycle_controller),
        open_probability=float(args.open_probability),
        close_probability=float(args.close_probability),
        transition_confirmation_views=int(args.transition_confirmation_views),
        min_transition_bayes_factor=float(args.min_transition_bayes_factor),
        min_transition_evidence_strength=float(args.min_transition_evidence_strength),
        max_states=int(args.max_states),
        thaw_parameters=parse_thaw_parameters(args.thaw_parameters),
        updates_per_frame=int(args.updates_per_frame),
        detector_only=bool(args.detector_only),
        evaluation_threshold=float(args.evaluation_threshold),
        seed=int(args.seed),
    )
    validate_run_config(config)
    return config


def filter_config(config: RunConfig) -> BinaryStateFilterConfig:
    return BinaryStateFilterConfig(
        emission_reliability=config.state_emission_reliability,
        inactive_to_active_prior=config.inactive_to_active_prior,
        active_to_inactive_prior=config.active_to_inactive_prior,
        initial_p_active=config.initial_active_probability,
        min_observation_mass=config.min_evidence_mass,
    )


def make_filter(gaussian_count: int, config: RunConfig, *, device, dtype):
    return BinaryStateFilter(
        gaussian_count,
        filter_config(config),
        device=device,
        dtype=dtype,
    )


def make_controller(model: Any, config: RunConfig):
    if config.lifecycle_controller == "view_consistent":
        lifecycle_cfg = ViewConsistentBinaryLifespanControllerConfig(
            inactive_to_active_prior=config.inactive_to_active_prior,
            active_to_inactive_prior=config.active_to_inactive_prior,
            min_transition_bayes_factor=config.min_transition_bayes_factor,
            confirmation_views=config.transition_confirmation_views,
            min_evidence_strength=config.min_transition_evidence_strength,
        )
        return ViewConsistentBinaryLifespanController(
            model,
            lifecycle_cfg,
            initialization="zero",
        )
    lifecycle_cfg = BinaryStateLifespanControllerConfig(
        active_threshold=config.open_probability,
        inactive_threshold=config.close_probability,
    )
    return BinaryStateLifespanController(
        model,
        lifecycle_cfg,
        initialization="zero",
    )


def _call_filter_update(tracker: Any, delta_a: torch.Tensor, delta_b: torch.Tensor, total_mass: torch.Tensor, rows: torch.Tensor, timestamp: int):
    return tracker.update(
        delta_a,
        delta_b,
        total_mass=total_mass,
        indices=rows,
        timestamp=timestamp,
    )


def controller_events(decision) -> list[LifecycleEvent]:
    positions = torch.nonzero(decision.event_mask, as_tuple=False).flatten().tolist()
    records: list[LifecycleEvent] = []
    for pos in positions:
        records.append(LifecycleEvent(
            gaussian_index=int(decision.indices[pos].item()),
            decision_timestamp=int(decision.decision_timestamp),
            old_binary_label=int(decision.old_binary_label[pos].item()),
            new_binary_label=int(decision.new_binary_label[pos].item()),
            action=BinaryLifespanAction(int(decision.action[pos].item())).name,
            old_slot=int(decision.old_slot[pos].item()),
            new_current_slot=int(decision.current_slot[pos].item()),
            p_active=float((decision.posterior_probability if hasattr(decision, "posterior_probability") else decision.p_active)[pos].item()),
            p_01=float(decision.p_01[pos].item()),
            p_10=float(decision.p_10[pos].item()),
            p_flip=float(decision.p_flip[pos].item()),
            visible_observation_count=int(decision.visible_observations[pos].item()),
        ))
    return records


def _concat(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([v.detach().flatten().float().cpu() for v in values])


def update_binary_lifecycle_chunks(tracker: Any, controller: Any, model: Any, optimizer: Any | None, delta_a: torch.Tensor, delta_b: torch.Tensor, total_mass: torch.Tensor, *, timestamp: int, min_evidence_mass: float, chunk_size: int, closed_audit: ExactClosedPairArchive) -> dict[str, Any]:
    observed = (
        (total_mass > 0)
        & (total_mass >= float(min_evidence_mass))
        & ((delta_a + delta_b) > 0)
    )
    rows_all = torch.nonzero(observed, as_tuple=False).flatten()
    action_counts = {a.name: 0 for a in BinaryLifespanAction}
    p_active_values: list[torch.Tensor] = []
    p_flip_values: list[torch.Tensor] = []
    p01_values: list[torch.Tensor] = []
    p10_values: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    open_bf_values: list[torch.Tensor] = []
    close_bf_values: list[torch.Tensor] = []
    events: list[LifecycleEvent] = []
    for start in range(0, int(rows_all.numel()), int(chunk_size)):
        rows = rows_all[start:start + int(chunk_size)]
        update = _call_filter_update(tracker, delta_a[rows], delta_b[rows], total_mass[rows], rows, timestamp)
        decision = controller.update(update, timestamp=timestamp, optimizer=optimizer)
        counts = torch.bincount(decision.action.to(torch.long), minlength=len(BinaryLifespanAction))
        for action in BinaryLifespanAction:
            action_counts[action.name] += int(counts[int(action)].item())
        p_active_values.append(update.p_active)
        p_flip_values.append(update.p_flip)
        p01_values.append(update.p_01)
        p10_values.append(update.p_10)
        q_values.append(update.q)
        if hasattr(decision, "open_bayes_factor"):
            open_bf_values.append(decision.open_bayes_factor)
        if hasattr(decision, "close_bayes_factor"):
            close_bf_values.append(decision.close_bayes_factor)
        events.extend(controller_events(decision))
        close_pos = torch.nonzero(decision.action == int(BinaryLifespanAction.CLOSE), as_tuple=False).flatten()
        if close_pos.numel():
            closed_audit.add(
                model,
                optimizer,
                decision.indices[close_pos],
                decision.old_slot[close_pos],
            )
    return {
        "observed": observed,
        "observed_count": int(rows_all.numel()),
        "action_counts": action_counts,
        "p_active_stats": quantile_summary(_concat(p_active_values)),
        "p_flip_stats": quantile_summary(_concat(p_flip_values)),
        "p_01_stats": quantile_summary(_concat(p01_values)),
        "p_10_stats": quantile_summary(_concat(p10_values)),
        "q_stats": quantile_summary(_concat(q_values)),
        "open_bayes_factor_stats": quantile_summary(_concat(open_bf_values)),
        "close_bayes_factor_stats": quantile_summary(_concat(close_bf_values)),
        "events": events,
    }


def _evidence_namespace(delta_a, delta_b, total_mass):
    return SimpleNamespace(delta_a=delta_a, delta_b=delta_b, e_plus=delta_a, e_minus=delta_b, total_mass=total_mass)


def frame_diagnostics(*, timestamp: int, frame_name: str, evidence: Any, binary: Mapping[str, Any], model: Any, geometry_thawed_rows: int, base_checksum: str, inactive_audit: Mapping[str, Any], frame_runtime_seconds: float, cuda_peak_memory_bytes: int, pre_predicted_positive_fraction: float, post_predicted_positive_fraction: float) -> dict[str, Any]:
    counts = binary["action_counts"]
    return {
        "timestamp": int(timestamp),
        "frame": frame_name,
        "algorithm": "direct_binary_state_filter",
        "observed_gaussian_count": int(binary["observed_count"]),
        "positive_pseudocount_mass": float(evidence.delta_a.sum().item()),
        "negative_pseudocount_mass": float(evidence.delta_b.sum().item()),
        "raw_alpha_t_mass": float(evidence.total_mass.sum().item()),
        "p_active_stats": dict(binary["p_active_stats"]),
        "p_flip_stats": dict(binary["p_flip_stats"]),
        "p_01_stats": dict(binary["p_01_stats"]),
        "p_10_stats": dict(binary["p_10_stats"]),
        "q_stats": dict(binary["q_stats"]),
        "open_bayes_factor_stats": dict(binary["open_bayes_factor_stats"]),
        "close_bayes_factor_stats": dict(binary["close_bayes_factor_stats"]),
        "open_count": int(counts["OPEN"]),
        "keep_count": int(counts["KEEP"]),
        "close_count": int(counts["CLOSE"]),
        "none_count": int(counts["NONE"]),
        "uncertain_count": int(counts.get("UNCERTAIN", counts.get("HOLD", 0))),
        "reopen_count": int(sum(1 for e in binary["events"] if e.action == "OPEN" and e.new_current_slot > 0)),
        "active_lifespan_count": int((model.current_state_index >= 0).sum().item()),
        "final_active_gs": int((model.current_state_index >= 0).sum().item()),
        "number_of_geometry_thawed_rows": int(geometry_thawed_rows),
        "base_checksum": base_checksum,
        "inactive_parameter_drift_audit": dict(inactive_audit),
        "pre_predicted_positive_fraction": float(pre_predicted_positive_fraction),
        "post_predicted_positive_fraction": float(post_predicted_positive_fraction),
        "predicted_positive_fraction": float(post_predicted_positive_fraction),
        "frame_runtime_seconds": float(frame_runtime_seconds),
        "cuda_peak_memory_bytes": int(cuda_peak_memory_bytes),
        "pre_tp": None, "pre_tn": None, "pre_fp": None, "pre_fn": None, "pre_iou": None, "pre_f1": None, "pre_precision": None, "pre_recall": None,
        "tp": None, "tn": None, "fp": None, "fn": None, "iou": None, "f1": None, "precision": None, "recall": None,
    }


def compact_npz_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    def stat_matrix(name: str) -> np.ndarray:
        return np.asarray([[float("nan") if row[name][k] is None else float(row[name][k]) for k in STAT_KEYS] for row in rows], dtype=np.float64)
    return {
        "timestamp": np.asarray([r["timestamp"] for r in rows], dtype=np.int64),
        "observed": np.asarray([r["observed_gaussian_count"] for r in rows], dtype=np.int64),
        "p_active": stat_matrix("p_active_stats"),
        "p_flip": stat_matrix("p_flip_stats"),
        "p_01": stat_matrix("p_01_stats"),
        "p_10": stat_matrix("p_10_stats"),
        "q": stat_matrix("q_stats"),
        "open_bayes_factor": stat_matrix("open_bayes_factor_stats"),
        "close_bayes_factor": stat_matrix("close_bayes_factor_stats"),
        "open_count": np.asarray([r["open_count"] for r in rows], dtype=np.int64),
        "close_count": np.asarray([r["close_count"] for r in rows], dtype=np.int64),
        "keep_count": np.asarray([r["keep_count"] for r in rows], dtype=np.int64),
        "uncertain_count": np.asarray([r["uncertain_count"] for r in rows], dtype=np.int64),
        "reopen_count": np.asarray([r["reopen_count"] for r in rows], dtype=np.int64),
        "active_lifespan_count": np.asarray([r["active_lifespan_count"] for r in rows], dtype=np.int64),
        "pre_predicted_positive_fraction": np.asarray([r["pre_predicted_positive_fraction"] for r in rows], dtype=np.float64),
        "post_predicted_positive_fraction": np.asarray([r["post_predicted_positive_fraction"] for r in rows], dtype=np.float64),
        "frame_runtime_seconds": np.asarray([r["frame_runtime_seconds"] for r in rows], dtype=np.float64),
        "cuda_peak_memory_bytes": np.asarray([r["cuda_peak_memory_bytes"] for r in rows], dtype=np.int64),
        "pre_iou": np.asarray([float("nan") if r.get("pre_iou") is None else r["pre_iou"] for r in rows], dtype=np.float64),
        "post_iou": np.asarray([float("nan") if r.get("iou") is None else r["iou"] for r in rows], dtype=np.float64),
        "pre_f1": np.asarray([float("nan") if r.get("pre_f1") is None else r["pre_f1"] for r in rows], dtype=np.float64),
        "post_f1": np.asarray([float("nan") if r.get("f1") is None else r["f1"] for r in rows], dtype=np.float64),
        "open_frame_iou_delta": np.asarray([float("nan") if r.get("open_frame_iou_delta") is None else r["open_frame_iou_delta"] for r in rows], dtype=np.float64),
        "open_frame_f1_delta": np.asarray([float("nan") if r.get("open_frame_f1_delta") is None else r["open_frame_f1_delta"] for r in rows], dtype=np.float64),
    }


def _json_cell(value: Any) -> Any:
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    frame_metrics: list[dict[str, Any]],
    events: Sequence[LifecycleEvent],
    checkpoint: Mapping[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_tmp = output_dir / ".summary.json.tmp"
    summary_path.unlink(missing_ok=True)
    summary_tmp.unlink(missing_ok=True)
    with (output_dir / "lifecycle_events.jsonl").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")
    rows = [{k: _json_cell(v) for k, v in row.items()} for row in frame_metrics]
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["timestamp"])
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output_dir / "per_frame_binary_state_stats.npz", **compact_npz_stats(frame_metrics))
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint is None:
        checkpoint_path.unlink(missing_ok=True)
    else:
        torch.save(_cpu_tree(dict(checkpoint)), checkpoint_path)
    summary_tmp.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)


def evaluate_after_inference(source_path: Path, records: Sequence[Any], pre_predictions: Sequence[np.ndarray], post_predictions: Sequence[np.ndarray], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import cv2
    if not (
        len(records) == len(pre_predictions) == len(post_predictions) == len(frame_rows)
    ):
        raise ValueError("post-inference evaluation inputs have different lengths")
    aggregate = {k: 0 for k in ("tp", "tn", "fp", "fn")}
    pre_aggregate = {k: 0 for k in ("tp", "tn", "fp", "fn")}
    open_deltas: list[dict[str, float]] = []
    for record, pre, post, row in zip(records, pre_predictions, post_predictions, frame_rows):
        mask_path = source_path / "gt_mask" / f"{Path(record.name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != post.shape:
            mask = cv2.resize(mask, (post.shape[1], post.shape[0]), interpolation=cv2.INTER_NEAREST)
        target = mask >= 128
        pre_m = binary_metrics(pre, target)
        post_m = binary_metrics(post, target)
        row.update(post_m)
        for key, value in pre_m.items():
            row[f"pre_{key}"] = value
        for key, value in post_m.items():
            row[f"post_{key}"] = value
        is_open_frame = int(row.get("open_count", 0)) > 0
        row["open_frame_iou_delta"] = (
            float(post_m["iou"] - pre_m["iou"]) if is_open_frame else None
        )
        row["open_frame_f1_delta"] = (
            float(post_m["f1"] - pre_m["f1"]) if is_open_frame else None
        )
        for k in aggregate:
            aggregate[k] += int(post_m[k]); pre_aggregate[k] += int(pre_m[k])
        if is_open_frame:
            open_deltas.append({"pre_iou": float(pre_m["iou"]), "post_iou": float(post_m["iou"]), "delta_iou": float(post_m["iou"] - pre_m["iou"]), "pre_f1": float(pre_m["f1"]), "post_f1": float(post_m["f1"]), "delta_f1": float(post_m["f1"] - pre_m["f1"])})
    def summarize(agg, prefix=""):
        precision = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
        recall = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
        union = agg["tp"] + agg["fp"] + agg["fn"]
        return {**agg, "precision": precision, "recall": recall, "aggregate_iou": agg["tp"] / union if union else 0.0, "aggregate_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0, "mean_frame_iou": float(np.mean([float(r[f"{prefix}iou"]) for r in frame_rows])), "mean_frame_f1": float(np.mean([float(r[f"{prefix}f1"]) for r in frame_rows]))}
    segment_rows: dict[str, list[Mapping[str, Any]]] = {}
    for record, row in zip(records, frame_rows):
        label = str(getattr(record, "segment_name", Path(record.name).stem))
        segment_rows.setdefault(label, []).append(row)
    return {
        "evaluated": True,
        "frames": len(frame_rows),
        **summarize(aggregate),
        "pre_opt": summarize(pre_aggregate, "pre_"),
        "post_opt": summarize(aggregate),
        "segments": [
            {"segment": label, **summarize_binary_metric_rows(segment)}
            for label, segment in segment_rows.items()
        ],
        "open_frame_pre_post": {
            "count": len(open_deltas),
            "mean_pre_iou": float(np.mean([d["pre_iou"] for d in open_deltas])) if open_deltas else None,
            "mean_post_iou": float(np.mean([d["post_iou"] for d in open_deltas])) if open_deltas else None,
            "mean_delta_iou": float(np.mean([d["delta_iou"] for d in open_deltas])) if open_deltas else None,
            "mean_pre_f1": float(np.mean([d["pre_f1"] for d in open_deltas])) if open_deltas else None,
            "mean_post_f1": float(np.mean([d["post_f1"] for d in open_deltas])) if open_deltas else None,
            "mean_delta_f1": float(np.mean([d["delta_f1"] for d in open_deltas])) if open_deltas else None,
        },
    }


def event_diagnostics(events: Sequence[LifecycleEvent]) -> dict[str, Any]:
    false_splits = sum(1 for e in events if e.action in {"OPEN", "CLOSE"} and e.old_binary_label == e.new_binary_label == 1)
    seen_slots: dict[int, set[int]] = {}
    reused = 0
    per_gaussian_events: dict[int, int] = {}
    for event in events:
        per_gaussian_events[event.gaussian_index] = per_gaussian_events.get(event.gaussian_index, 0) + 1
        if event.action != "OPEN":
            continue
        allocated = seen_slots.setdefault(event.gaussian_index, set())
        if event.new_current_slot in allocated:
            reused += 1
        allocated.add(event.new_current_slot)
    return {
        "open_count": sum(e.action == "OPEN" for e in events),
        "close_count": sum(e.action == "CLOSE" for e in events),
        "reopen_count": sum(e.action == "OPEN" and e.new_current_slot > 0 for e in events),
        "active_to_active_false_split_count": int(false_splits),
        "reused_slot_violations": int(reused),
        "gaussians_with_repeated_transitions": int(
            sum(count > 1 for count in per_gaussian_events.values())
        ),
        "repeated_transition_event_count": int(
            sum(max(0, count - 1) for count in per_gaussian_events.values())
        ),
    }


def same_scene_repeated_transition_diagnostics(
    events: Sequence[LifecycleEvent],
    boundaries: Sequence[int],
) -> dict[str, int]:
    """Count repeated flips inside evaluation-only manual-boundary segments."""

    ordered_boundaries = tuple(sorted(int(value) for value in boundaries))
    per_gaussian_segment: dict[tuple[int, int], int] = {}
    for event in events:
        segment = sum(
            int(event.decision_timestamp >= boundary)
            for boundary in ordered_boundaries
        )
        key = (int(event.gaussian_index), segment)
        per_gaussian_segment[key] = per_gaussian_segment.get(key, 0) + 1
    repeated_pairs = {
        key: count for key, count in per_gaussian_segment.items() if count > 1
    }
    return {
        "same_scene_repeated_transition_event_count": int(
            sum(count - 1 for count in repeated_pairs.values())
        ),
        "same_scene_repeated_gaussian_segment_count": int(len(repeated_pairs)),
        "same_scene_repeated_gaussian_count": int(
            len({gaussian_index for gaussian_index, _segment in repeated_pairs})
        ),
    }


def posterior_run_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("q", "p_active", "p_flip", "p_01", "p_10"):
        key = f"{name}_stats"
        means = [row[key]["mean"] for row in rows if row[key]["mean"] is not None]
        result[name] = {
            "last_observed_frame": next(
                (dict(row[key]) for row in reversed(rows) if row[key]["count"]),
                quantile_summary(torch.empty(0)),
            ),
            "distribution_of_frame_means": quantile_summary(
                torch.as_tensor(means, dtype=torch.float64)
            ),
        }
    return result


def make_optimizer(model: Any, config: RunConfig, args: argparse.Namespace) -> Any | None:
    from temporal.masked_optimizer import MaskedRowSlotAdam

    if config.detector_only:
        for _, p in model.state_parameter_items():
            p.requires_grad_(False)
        return None
    enabled = set(config.thaw_parameters)
    for name, p in model.state_parameter_items():
        p.requires_grad_(name in enabled)
    return MaskedRowSlotAdam(dict(model.state_parameter_items()), thaw_names=config.thaw_parameters, lrs={"dc": args.dc_lr, "xyz": args.xyz_lr, "opacity": args.opacity_lr, "scaling": args.scaling_lr, "rotation": args.rotation_lr}, eps=args.adam_eps)


def _prediction_from_package(package: Mapping[str, torch.Tensor], threshold: float) -> tuple[np.ndarray, torch.Tensor]:
    score = package["render"].detach().mean(dim=0).clamp(0.0, 1.0).cpu()
    return score.numpy() >= float(threshold), score


class SyntheticTemporalModel:
    def __init__(self, base: Any, n: int, max_states: int):
        self.base = base; self.max_states = max_states
        self.state_valid = torch.zeros(n, max_states, dtype=torch.bool)
        self.state_start = torch.zeros(n, max_states)
        self.state_end = torch.full((n, max_states), float("inf"))
        self.state_status = torch.zeros(n, max_states, dtype=torch.int8)
        self.num_states = torch.zeros(n, dtype=torch.long)
        self.current_state_index = torch.full((n,), -1, dtype=torch.long)
        self.state_change_dc = torch.zeros(n, max_states, 1, 3)
        self.state_xyz_delta = torch.zeros(n, max_states, 3)
        self.state_opacity_delta = torch.zeros(n, max_states, 1)
        self.state_scaling_delta = torch.zeros(n, max_states, 3)
        self.state_rotation_delta = torch.zeros(n, max_states, 4)

    def state_parameter_items(self):
        return (("dc", self.state_change_dc), ("xyz", self.state_xyz_delta), ("opacity", self.state_opacity_delta), ("scaling", self.state_scaling_delta), ("rotation", self.state_rotation_delta))

    def open_rows(self, rows, timestamp, initialization="zero", optimizer=None):
        rows = torch.as_tensor(rows, dtype=torch.long).flatten()
        if bool((self.current_state_index[rows] >= 0).any()):
            raise RuntimeError("row already open")
        slots = self.num_states[rows].clone()
        if bool((slots >= self.max_states).any()):
            raise RuntimeError("temporal state capacity exceeded")
        for _name, param in self.state_parameter_items():
            param[rows, slots].zero_()
        self.state_valid[rows, slots] = True; self.state_status[rows, slots] = 1
        self.state_start[rows, slots] = float(timestamp); self.state_end[rows, slots] = float("inf")
        self.current_state_index[rows] = slots; self.num_states[rows] = slots + 1
        return slots

    def close_rows(self, rows, timestamp):
        rows = torch.as_tensor(rows, dtype=torch.long).flatten()
        slots = self.current_state_index[rows].clone(); active = slots >= 0
        rows, slots = rows[active], slots[active]
        self.state_end[rows, slots] = float(timestamp); self.state_status[rows, slots] = 2; self.current_state_index[rows] = -1
        return slots

    def validate_lifecycle(self):
        return True


def _synthetic_base(n=1):
    return SimpleNamespace(_xyz=torch.zeros(n, 3), _features_dc=torch.zeros(n, 1, 3), _features_rest=torch.zeros(n, 0, 3), _opacity=torch.zeros(n, 1), _scaling=torch.zeros(n, 3), _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1))


def q_sequence_to_evidence(q_values: Sequence[float | None], strength: float = 1.0) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = []
    for q in q_values:
        if q is None:
            out.append((torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])))
        else:
            qq = float(q); out.append((torch.tensor([qq * strength]), torch.tensor([(1.0 - qq) * strength]), torch.tensor([strength])))
    return out


def run_detector_sequence(evidence: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], model: Any, config: RunConfig) -> dict[str, Any]:
    validate_run_config(config)
    tracker = make_filter(model.state_valid.shape[0], config, device=model.state_valid.device, dtype=model.state_start.dtype)
    controller = make_controller(model, config)
    checksums = base_checksums(model.base); before = base_snapshots(model.base)
    closed_audit = ExactClosedPairArchive()
    frames: list[dict[str, Any]] = []; events: list[LifecycleEvent] = []
    for timestamp, (pos, neg, mass_raw) in enumerate(evidence):
        pos = pos.to(model.state_start); neg = neg.to(model.state_start); mass_raw = mass_raw.to(model.state_start)
        da, db, _, _ = evidence_counts(pos, neg, mode=config.evidence_count_mode, mass_saturation=config.evidence_mass_saturation, min_evidence_mass=config.min_evidence_mass)
        binary = update_binary_lifecycle_chunks(tracker, controller, model, None, da, db, mass_raw, timestamp=timestamp, min_evidence_mass=config.min_evidence_mass, chunk_size=config.filter_chunk_size, closed_audit=closed_audit)
        events.extend(binary["events"])
        frames.append(frame_diagnostics(timestamp=timestamp, frame_name=f"synthetic_{timestamp:03d}", evidence=_evidence_namespace(da, db, mass_raw), binary=binary, model=model, geometry_thawed_rows=0, base_checksum=combined_checksum(checksums), inactive_audit=closed_audit.status(), frame_runtime_seconds=0.0, cuda_peak_memory_bytes=0, pre_predicted_positive_fraction=0.0, post_predicted_positive_fraction=0.0))
    return {"frame_metrics": frames, "events": events, "binary_stats": compact_npz_stats(frames), "base_drift": base_drift(before, model.base), "closed_slot_audit": closed_audit.verify(model, None), "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"), "filter_state": tracker.state_dict() if hasattr(tracker, "state_dict") else {}}


def run_detector_only_synthetic_smoke() -> dict[str, Any]:
    model = SyntheticTemporalModel(_synthetic_base(), 1, 4)
    cfg = RunConfig(evidence_count_mode="raw", min_evidence_mass=0.1, inactive_to_active_prior=0.2, active_to_inactive_prior=0.2, initial_active_probability=0.1, max_states=4, detector_only=True)
    result = run_detector_sequence(q_sequence_to_evidence([0.05, 0.08, 0.10, 0.90, 0.92, 0.88, 0.07, 0.05, 0.10, 0.91, 0.94, 0.90]), model, cfg)
    result["final_num_states"] = model.num_states.tolist(); result["final_current_state_index"] = model.current_state_index.tolist()
    return result


def _inactive_gradient_audit(model: Any, active_pairs: torch.Tensor) -> dict[str, Any]:
    maximum = 0.0
    violation_count = 0
    for _name, p in model.state_parameter_items():
        if p.grad is None:
            continue
        g = p.grad.detach()
        view = g.reshape(g.shape[0], g.shape[1], -1)
        mask = active_pairs.to(device=g.device)
        total_nonzero = int(torch.count_nonzero(view).item())
        active_nonzero = int(torch.count_nonzero(view[mask]).item()) if bool(mask.any()) else 0
        inactive_nonzero = total_nonzero - active_nonzero
        violation_count += inactive_nonzero
        if inactive_nonzero:
            audit_values = view.detach().abs().clone()
            audit_values[mask] = 0
            maximum = max(maximum, float(audit_values.max().item()))
    return {"count": int(violation_count), "max_abs": float(maximum)}


@torch.no_grad()
def _audit_new_open_slots(
    model: Any,
    optimizer: Any | None,
    events: Sequence[LifecycleEvent],
) -> dict[str, int]:
    opened = [event for event in events if event.action == "OPEN"]
    if not opened:
        return {
            "zero_initialized_parameter_violations": 0,
            "zero_initialized_optimizer_state_violations": 0,
            "allocation_contract_violations": 0,
        }
    device = model.state_valid.device
    rows = torch.tensor([event.gaussian_index for event in opened], device=device, dtype=torch.long)
    slots = torch.tensor([event.new_current_slot for event in opened], device=device, dtype=torch.long)
    parameter_violations = 0
    optimizer_violations = 0
    allocation_violations = int(
        (
            (slots < 0)
            | (model.current_state_index[rows] != slots)
            | ((model.num_states[rows] - 1) != slots)
        ).sum().item()
    )
    for _name, parameter in model.state_parameter_items():
        values = parameter.detach()[rows, slots].reshape(rows.numel(), -1)
        parameter_violations += int((values != 0).any(dim=1).sum().item())
        if optimizer is None or parameter not in optimizer.state:
            continue
        for state_value in optimizer.state[parameter].values():
            if not isinstance(state_value, torch.Tensor):
                continue
            if state_value.shape == parameter.shape:
                selected = state_value.detach()[rows, slots].reshape(rows.numel(), -1)
            elif tuple(state_value.shape) == tuple(parameter.shape[:2]):
                selected = state_value.detach()[rows, slots].reshape(rows.numel(), -1)
            else:
                continue
            optimizer_violations += int((selected != 0).any(dim=1).sum().item())
    return {
        "zero_initialized_parameter_violations": int(parameter_violations),
        "zero_initialized_optimizer_state_violations": int(optimizer_violations),
        "allocation_contract_violations": int(allocation_violations),
    }


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    import cv2  # noqa: F401
    from temporal import TemporalGeometryChangeModel
    from temporal.change_evidence import accumulate_change_evidence
    from experiments.train_cue_temporal_rchange import build_fixed_cue_views, load_fixed_camera_index, validate_cue_cache
    from experiments.train_real_temporal_rchange import (
        file_checksum,
        oscd_positive_sparsity_loss,
        seed_everything,
    )
    from gaussian_renderer import render_change_temporal
    from scene import GaussianModel

    config = run_config_from_args(args)
    skip_checkpoint = bool(getattr(args, "skip_checkpoint", False))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    seed_everything(config.seed); torch.cuda.reset_peak_memory_stats(); started = time.time()
    records, names = build_causal_records(args.source_path, max_frames=args.max_frames)
    timestamps = [int(record.global_index) for record in records]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise RuntimeError("views are not in strict global timestamp order")
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    validate_cue_camera_checksum(
        cue_metadata,
        file_checksum(args.fixed_cameras_json),
    )
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    base = GaussianModel(sh_degree=3, active_sh_degree=0); base.load_ply_change(str(base_ply))
    base_before = base_snapshots(base); checksums = base_checksums(base); checksum = combined_checksum(checksums)
    model = TemporalGeometryChangeModel.from_gaussians(base, max_states=config.max_states); model.reset_all_lifespans_closed()
    optimizer = make_optimizer(model, config, args)
    tracker = make_filter(model.state_valid.shape[0], config, device=base._xyz.device, dtype=base._xyz.dtype)
    controller = make_controller(model, config)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=base._xyz.dtype, device=base._xyz.device)
    closed_audit = ExactClosedPairArchive()
    rows: list[dict[str, Any]] = []; events: list[LifecycleEvent] = []
    pre_predictions: list[np.ndarray] = []; post_predictions: list[np.ndarray] = []; score_maps: list[np.ndarray] = []; raw_render_maps: list[np.ndarray] = []
    intrinsics = None
    inactive_gradient_violation_count = 0
    inactive_gradient_max_abs = 0.0
    open_audit_totals = {
        "zero_initialized_parameter_violations": 0,
        "zero_initialized_optimizer_state_violations": 0,
        "allocation_contract_violations": 0,
    }
    for record in records:
        frame_started = time.time(); timestamp = int(record.global_index)
        views, _pose, current_intrinsics = build_fixed_cue_views([record], cameras, args.cue_cache_root, args.resolution)
        view = views[0]
        if intrinsics is None:
            intrinsics = current_intrinsics
        elif not np.array_equal(intrinsics, current_intrinsics):
            raise ValueError("fixed views do not share one intrinsic matrix")
        evidence = accumulate_change_evidence(view, base, pipe, background, view.candidate_map, cue_mode=config.bayes_cue_mode, cue_threshold=config.bayes_cue_threshold, cue_scale=config.bayes_cue_scale, count_mode=config.evidence_count_mode, mass_saturation=config.evidence_mass_saturation, min_evidence_mass=config.min_evidence_mass)
        binary = update_binary_lifecycle_chunks(tracker, controller, model, optimizer, evidence.delta_a, evidence.delta_b, evidence.total_mass, timestamp=timestamp, min_evidence_mass=config.min_evidence_mass, chunk_size=config.filter_chunk_size, closed_audit=closed_audit)
        events.extend(binary["events"])
        open_audit = _audit_new_open_slots(model, optimizer, binary["events"])
        for key, value in open_audit.items():
            open_audit_totals[key] += int(value)
        active_pairs = current_pair_mask(model); active_count = int((model.current_state_index >= 0).sum().item())
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        pre_package = render_change_temporal(view, model, pipe, background, timestamp=float(timestamp))
        pre_pred, pre_score = _prediction_from_package(pre_package, config.evaluation_threshold)
        package = pre_package
        if optimizer is not None and active_count:
            for update_index in range(config.updates_per_frame):
                optimizer.zero_grad(set_to_none=True)
                loss, _parts = oscd_positive_sparsity_loss(view.training_target, package["render"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite online loss at timestamp {timestamp}")
                loss.backward()
                if update_index == 0:
                    gradient_audit = _inactive_gradient_audit(model, active_pairs)
                    inactive_gradient_violation_count += int(gradient_audit["count"])
                    inactive_gradient_max_abs = max(
                        inactive_gradient_max_abs,
                        float(gradient_audit["max_abs"]),
                    )
                optimizer.step(active_pairs)
                package = render_change_temporal(view, model, pipe, background, timestamp=float(timestamp))
        inactive_audit = closed_audit.status()
        post_pred, post_score = _prediction_from_package(package, config.evaluation_threshold)
        pre_predictions.append(pre_pred); post_predictions.append(post_pred)
        if args.visualization_dir is not None:
            score_maps.append(post_score.mul(255).round().to(torch.uint8).numpy())
            raw_render_maps.append(package["render"].detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        rows.append(frame_diagnostics(timestamp=timestamp, frame_name=record.name, evidence=evidence, binary=binary, model=model, geometry_thawed_rows=active_count if any(n != "dc" for n in config.thaw_parameters) else 0, base_checksum=checksum, inactive_audit=inactive_audit, frame_runtime_seconds=time.time() - frame_started, cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(), pre_predicted_positive_fraction=float(pre_pred.mean()), post_predicted_positive_fraction=float(post_pred.mean())))
    closed_slot_audit = closed_audit.verify(model, optimizer)
    if not closed_slot_audit["passed"]:
        raise RuntimeError(f"closed slot drift after future updates: {closed_slot_audit}")
    boundaries = tuple(int(v) for v in (() if args.disable_boundary_diagnostics else args.boundaries if args.boundaries is not None else DEFAULT_BOUNDARIES))
    metrics = {"evaluated": False, "reason": "--skip-post-inference-evaluation"} if args.skip_post_inference_evaluation else evaluate_after_inference(args.source_path, records, pre_predictions, post_predictions, rows)
    evdiag = event_diagnostics(events)
    same_scene_diag = same_scene_repeated_transition_diagnostics(events, boundaries)
    drift = base_drift(base_before, base)
    if not drift["bitwise_equal"]:
        raise RuntimeError(f"immutable base tensor drift detected: {drift}")
    if evdiag["active_to_active_false_split_count"] != 0 or evdiag["reused_slot_violations"] != 0:
        raise RuntimeError(f"lifecycle invariant violation: {evdiag}")
    if any(open_audit_totals.values()):
        raise RuntimeError(f"OPEN zero-init/allocation invariant violation: {open_audit_totals}")
    if inactive_gradient_violation_count:
        raise RuntimeError(
            "inactive temporal gradients were nonzero: "
            f"count={inactive_gradient_violation_count}, max_abs={inactive_gradient_max_abs}"
        )
    visualization_summary = None
    if args.visualization_dir is not None:
        visualization_summary = write_visualization_artifacts(source_path=args.source_path, cue_cache_root=args.cue_cache_root, records=records, predictions=post_predictions, score_maps=score_maps, raw_render_maps=raw_render_maps, frame_rows=rows, output_dir=args.visualization_dir, panel_width=int(args.visualization_panel_width), sample_count=int(args.visualization_sample_count), frame_indices=args.visualization_frame_indices, boundaries=boundaries, all_frames=bool(args.visualization_all_frames), gif_width=int(args.visualization_gif_width), gif_duration_ms=int(args.visualization_gif_duration_ms))
    summary = {
        "schema_version": 1,
        "script": "experiments/run_online_binary_state_lifespan_thaw.py",
        "contract": "direct_binary_state_filter_lifespan_active_geometry",
        "runtime_seconds": time.time() - started,
        "run_config": asdict(config),
        "run_arguments": serializable_arguments(args),
        "algorithm": getattr(tracker, "algorithm", "direct_binary_state_filter"),
        "transition_equation": "normalize [(1-b)(1-p01)L0, (1-b)p01L1, bp10L0, b(1-p10)L1]; p_active=P01+P11; p_flip=P01+P10",
        "lifecycle_controller": config.lifecycle_controller,
        "lifecycle_controller_equation": (
            "posterior hysteresis on p_active"
            if config.lifecycle_controller == "posterior_hysteresis"
            else "committed-state branch BF confirmation: inactive uses (P01/P00)/(p01/(1-p01)); active uses (P10/P11)/(p10/(1-p10)); p_active is diagnostic only"
        ),
        "gt_used_for_training": False,
        "manual_boundaries_used_for_inference": False,
        "gt_loaded_after_inference_only": not args.skip_post_inference_evaluation,
        "metrics": metrics,
        "posterior_diagnostics": posterior_run_diagnostics(rows),
        "transition_diagnostics_after_inference": transition_diagnostics(rows, boundaries),
        "frames": len(rows),
        "processed_frame_count": len(names),
        "event_count": len(events),
        **evdiag,
        **same_scene_diag,
        "keep_count": int(sum(int(row["keep_count"]) for row in rows)),
        "uncertain_count": int(sum(int(row["uncertain_count"]) for row in rows)),
        "none_count": int(sum(int(row["none_count"]) for row in rows)),
        "base_tensor_drift": drift,
        "closed_slot_audit": closed_slot_audit,
        "closed_slot_max_drift": float(closed_slot_audit["max_abs"]),
        "inactive_gradient_first_step_violations": int(
            inactive_gradient_violation_count
        ),
        "inactive_gradient_audit_scope": "first_optimizer_step_per_frame",
        "inactive_gradient_max_abs": float(inactive_gradient_max_abs),
        "open_zero_initialization_audit": open_audit_totals,
        "zero_init_violations": int(
            open_audit_totals["zero_initialized_parameter_violations"]
            + open_audit_totals["zero_initialized_optimizer_state_violations"]
        ),
        "reopen_allocation_contract_violations": int(
            open_audit_totals["allocation_contract_violations"]
        ),
        "base_max_drift": float(drift["max_abs"]),
        "final_active_gs": int((model.current_state_index >= 0).sum().item()),
        "fixed_topology": True,
        "optimized_parameters": list(config.thaw_parameters) if not config.detector_only else [],
        "detector_only": config.detector_only,
        "cue_cache_metadata": cue_metadata,
        "camera_intrinsics": None if intrinsics is None else intrinsics.tolist(),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "binary_filter_resident_state_bytes": int(
            tracker.p_active.numel() * tracker.p_active.element_size()
            + tracker.visible_observations.numel() * tracker.visible_observations.element_size()
            + tracker.last_timestamp.numel() * tracker.last_timestamp.element_size()
        ),
        "output_files": list(OUTPUT_FILES[:-1] if skip_checkpoint else OUTPUT_FILES),
        "checkpoint_saved": not skip_checkpoint,
        "visualization": visualization_summary,
    }
    checkpoint = None
    if not skip_checkpoint:
        checkpoint = {"schema_version": 1, "contract": summary["contract"], "base_ply": str(base_ply), "state_dict": model.state_dict(), "binary_filter_state": tracker.state_dict() if hasattr(tracker, "state_dict") else {}, "controller_state": controller.state_dict() if hasattr(controller, "state_dict") else {}, "optimizer_state": None if optimizer is None else optimizer.state_dict(), "metadata": summary}
    write_outputs(args.output_dir, summary, rows, events, checkpoint)
    return summary


def parse_visualization_frame_indices(value: str | None) -> tuple[int, ...] | None:
    from experiments.run_online_bayesian_lifespan_thaw import parse_visualization_frame_indices as parse
    return parse(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--disable-boundary-diagnostics", action="store_true")
    parser.add_argument("--skip-post-inference-evaluation", action="store_true")
    parser.add_argument("--evaluation-threshold", type=float, default=0.5)
    parser.add_argument("--visualization-dir", type=Path, default=None)
    parser.add_argument("--visualization-sample-count", "--visualization-max-panels", dest="visualization_sample_count", type=nonnegative_int, default=9)
    parser.add_argument("--visualization-panel-width", type=positive_int, default=320)
    parser.add_argument("--visualization-frame-indices", type=parse_visualization_frame_indices, default=None)
    parser.add_argument("--visualization-all-frames", action="store_true")
    parser.add_argument("--visualization-gif-width", type=positive_int, default=960)
    parser.add_argument("--visualization-gif-duration-ms", type=positive_int, default=160)
    parser.add_argument("--bayes-cue-mode", choices=("binary", "soft"), default="binary")
    parser.add_argument("--bayes-cue-threshold", type=float, default=0.5)
    parser.add_argument("--bayes-cue-scale", type=float, default=1.0)
    parser.add_argument("--evidence-count-mode", choices=("raw", "capped"), default="capped")
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=float, default=1e-6)
    parser.add_argument("--state-emission-reliability", type=float, default=0.9)
    parser.add_argument("--inactive-to-active-prior", type=float, default=0.01)
    parser.add_argument("--active-to-inactive-prior", type=float, default=0.01)
    parser.add_argument("--initial-active-probability", type=float, default=0.5)
    parser.add_argument("--filter-chunk-size", type=positive_int, default=65536)
    parser.add_argument("--lifecycle-controller", choices=("posterior_hysteresis", "view_consistent"), default="posterior_hysteresis")
    parser.add_argument("--open-probability", type=float, default=0.6)
    parser.add_argument("--close-probability", type=float, default=0.4)
    parser.add_argument("--transition-confirmation-views", type=positive_int, default=2)
    parser.add_argument("--min-transition-bayes-factor", type=nonnegative_float, default=3.0)
    parser.add_argument("--min-transition-evidence-strength", type=nonnegative_float, default=1e-6)
    parser.add_argument("--max-states", type=positive_int, default=8)
    parser.add_argument("--thaw-parameters", type=parse_thaw_parameters, default=("dc",))
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--detector-only-smoke", action="store_true")
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--adam-eps", type=float, default=1e-15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Keep textual diagnostics but omit the large training checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detector_only_smoke:
        result = run_detector_only_synthetic_smoke()
        summary = {"algorithm": result["algorithm"], "frames": len(result["frame_metrics"]), "event_count": len(result["events"]), "active_to_active_false_split_count": event_diagnostics(result["events"])["active_to_active_false_split_count"], "reused_slot_violations": event_diagnostics(result["events"])["reused_slot_violations"], "base_tensor_drift": result["base_drift"], "output_files": list(OUTPUT_FILES[:-1] if args.skip_checkpoint else OUTPUT_FILES), "checkpoint_saved": not bool(args.skip_checkpoint), "detector_only_smoke": True}
        checkpoint = None if args.skip_checkpoint else {"summary": summary, "filter_state": result["filter_state"]}
        write_outputs(args.output_dir, summary, result["frame_metrics"], result["events"], checkpoint)
        print(json.dumps(summary, indent=2)); return 0
    summary = run_online(args)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "frames": summary["frames"], "algorithm": summary["algorithm"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
