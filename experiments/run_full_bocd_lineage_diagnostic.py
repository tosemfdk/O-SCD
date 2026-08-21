"""Full BOCD versus protected Beam-2 lineage diagnostic.

This script is intentionally detector-only: it compares exact Beta-Bernoulli
BOCD and the protected two-branch approximation on the same causal per-Gaussian
alpha-T evidence.  It does not instantiate a TemporalChangeModel, lifecycle
controller, renderer for current masks, GT masks, or geometry optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


DEFAULT_SOURCE = Path("data/Instance_1/scene_change1_2_3")
DEFAULT_BOUNDARIES = (95, 199)
BASE_PLY_REL = "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"
DEFAULT_FIXED_CAMERAS = Path(
    "/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/"
    "scene_change1_2_3/cameras_fixed.json"
)
DEFAULT_CUE_CACHE = Path(
    "/home/rvl/workspace/github/O-SCD/artifacts/escd_396ref/"
    "fixed_pose_cues_res4_v1"
)
DEFAULT_OUTPUT = Path("outputs/full_bocd_lineage_diagnostic")
OUTPUT_FILES = (
    "summary.json",
    "frame_metrics.csv",
    "start_mass_stats.npz",
    "lineage_recovery.json",
    "beam_commit_comparison.jsonl",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    exact_recurrence: str = "prior_reset"
    cue_mode: str = "binary"
    cue_threshold: float = 0.5
    cue_scale: float = 1.0
    evidence_count_mode: str = "capped"
    evidence_mass_saturation: float = 1.0
    min_evidence_mass: float = 1e-6
    prior_a: float = 1.0
    prior_b: float = 1.0
    expected_run_length: float | None = 100.0
    hazard: float | None = None
    max_run_length: int = 304
    chunk_size: int = 65536
    changepoint_probability: float = 0.5
    min_run_evidence: float = 0.0
    min_visible_observations: int = 1
    recent_window: int = 5
    start_cluster_radii: tuple[int, ...] = (0, 1, 2, 3, 5)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def config_from_args(args: argparse.Namespace) -> DiagnosticConfig:
    cfg = DiagnosticConfig(
        exact_recurrence=args.exact_recurrence,
        cue_mode=args.cue_mode,
        cue_threshold=float(args.cue_threshold),
        cue_scale=float(args.cue_scale),
        evidence_count_mode=args.evidence_count_mode,
        evidence_mass_saturation=float(args.evidence_mass_saturation),
        min_evidence_mass=float(args.min_evidence_mass),
        prior_a=float(args.prior_a),
        prior_b=float(args.prior_b),
        expected_run_length=None if args.hazard is not None else float(args.expected_run_length),
        hazard=None if args.hazard is None else float(args.hazard),
        max_run_length=int(args.max_run_length),
        chunk_size=int(args.chunk_size),
        changepoint_probability=float(args.changepoint_probability),
        min_run_evidence=float(args.min_run_evidence),
        min_visible_observations=int(args.min_visible_observations),
        recent_window=int(args.recent_window),
        start_cluster_radii=tuple(sorted(set(args.start_cluster_radii))),
    )
    validate_config(cfg)
    return cfg


def validate_config(config: DiagnosticConfig) -> None:
    if config.exact_recurrence not in {"prior_reset", "adams_mackay"}:
        raise ValueError("exact_recurrence must be prior_reset|adams_mackay")
    if config.cue_mode not in {"binary", "soft"}:
        raise ValueError("cue_mode must be binary|soft")
    if config.cue_scale <= 0:
        raise ValueError("cue_scale must be positive")
    if config.evidence_count_mode not in {"raw", "capped"}:
        raise ValueError("evidence_count_mode must be raw|capped")
    if config.evidence_mass_saturation <= 0:
        raise ValueError("evidence_mass_saturation must be positive")
    if config.min_evidence_mass < 0:
        raise ValueError("min_evidence_mass must be nonnegative")
    if min(config.max_run_length, config.chunk_size, config.recent_window) < 1:
        raise ValueError("max_run_length, chunk_size and recent_window must be positive")
    if not config.start_cluster_radii or min(config.start_cluster_radii) < 0:
        raise ValueError("start_cluster_radii must be nonempty and nonnegative")
    if not 0.0 <= config.changepoint_probability <= 1.0:
        raise ValueError("changepoint_probability must be in [0,1]")
    _ = bocd_config(config)


def load_bocd_classes():
    """Load BOCD classes without importing renderer-heavy temporal.__init__ in tests."""
    try:
        from temporal.bernoulli_bocd import (
            AdamsMacKayBetaBernoulliBOCD,
            BernoulliBOCDConfig,
            BetaBernoulliBOCD,
        )
        from temporal.beam2_bocd import BeamTwoBernoulliFilter
        return (
            BernoulliBOCDConfig,
            BetaBernoulliBOCD,
            AdamsMacKayBetaBernoulliBOCD,
            BeamTwoBernoulliFilter,
        )
    except ModuleNotFoundError:
        import importlib.util
        import sys
        import types

        root = Path(__file__).resolve().parents[1]
        package_name = "_oscd_light_temporal"
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(root / "temporal")]
            sys.modules[package_name] = package

        def load(name: str):
            fullname = f"{package_name}.{name}"
            if fullname in sys.modules:
                return sys.modules[fullname]
            spec = importlib.util.spec_from_file_location(fullname, root / "temporal" / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)
            return module

        bernoulli = load("bernoulli_bocd")
        beam2 = load("beam2_bocd")
        return (
            bernoulli.BernoulliBOCDConfig,
            bernoulli.BetaBernoulliBOCD,
            bernoulli.AdamsMacKayBetaBernoulliBOCD,
            beam2.BeamTwoBernoulliFilter,
        )


def bocd_config(config: DiagnosticConfig):
    (
        BernoulliBOCDConfig,
        _BetaBernoulliBOCD,
        _AdamsMacKayBetaBernoulliBOCD,
        _BeamTwoBernoulliFilter,
    ) = load_bocd_classes()
    return BernoulliBOCDConfig(
        prior_a=config.prior_a,
        prior_b=config.prior_b,
        expected_run_length=config.expected_run_length,
        hazard=config.hazard,
        max_run_length=config.max_run_length,
        min_evidence_mass=config.min_evidence_mass,
        changepoint_probability=config.changepoint_probability,
        min_run_evidence=config.min_run_evidence,
        min_visible_observations=config.min_visible_observations,
    )


def validate_full_run_max_r(max_run_length: int, processed_frame_count: int) -> None:
    """Require enough run-length slots to avoid truncation for processed frames."""
    if max_run_length < processed_frame_count:
        raise ValueError(
            "full BOCD diagnostic requires --max-run-length >= processed frame count "
            f"({max_run_length} < {processed_frame_count})"
        )


def estimated_exact_state_bytes(gaussian_count: int, max_run_length: int, dtype: torch.dtype = torch.float32) -> int:
    if gaussian_count < 1 or max_run_length < 1:
        raise ValueError("gaussian_count and max_run_length must be positive")
    float_bytes = torch.empty((), dtype=dtype).element_size()
    long_bytes = torch.empty((), dtype=torch.long).element_size()
    run_count = int(max_run_length) + 1
    return int(gaussian_count) * (
        run_count * (4 * float_bytes + 2 * long_bytes) + float_bytes + 3 * long_bytes
    )



def estimated_beam2_state_bytes(gaussian_count: int, dtype: torch.dtype = torch.float32) -> int:
    if gaussian_count < 1:
        raise ValueError("gaussian_count must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating")
    float_bytes = torch.empty((), dtype=dtype).element_size()
    long_bytes = torch.empty((), dtype=torch.long).element_size()
    # Resident state includes the serialized 9-float/8-int64 contract plus
    # four floating diagnostics and one candidate-start timestamp.
    return int(gaussian_count) * (13 * float_bytes + 9 * long_bytes)


def build_cohort_mapping(
    num_gaussians: int,
    cohort_indices: Sequence[int] | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Map global Gaussian rows to compact detector-local rows.

    Event-selected diagnostics must not allocate an ``[all N, R]`` exact BOCD
    state.  ``None`` denotes the identity mapping used by the unbiased all-row
    mode; an explicit cohort receives dense local indices ``[0, M)``.
    """
    if num_gaussians < 1:
        raise ValueError("num_gaussians must be positive")
    if cohort_indices is None:
        return None, None, int(num_gaussians)
    unique = sorted({int(index) for index in cohort_indices})
    if not unique:
        raise ValueError("event-selected cohort is empty")
    if unique[0] < 0 or unique[-1] >= int(num_gaussians):
        raise ValueError("cohort contains Gaussian index outside the base model")
    global_rows = torch.tensor(unique, device=device, dtype=torch.long)
    global_to_local = torch.full(
        (int(num_gaussians),), -1, device=device, dtype=torch.long
    )
    global_to_local[global_rows] = torch.arange(
        global_rows.numel(), device=device, dtype=torch.long
    )
    return global_rows, global_to_local, int(global_rows.numel())

def parse_action_list(value: str) -> set[str]:
    actions = {part.strip().upper() for part in value.split(",") if part.strip()}
    if not actions:
        raise argparse.ArgumentTypeError("at least one cohort action is required")
    return actions


def parse_cohort_events(path: Path, actions: Iterable[str]) -> list[int]:
    """Parse unique Gaussian indices from lifecycle_events.jsonl by action."""
    wanted = {action.upper() for action in actions}
    indices: list[int] = []
    seen: set[int] = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            action = str(record.get("action", "")).upper()
            if action not in wanted:
                continue
            if "gaussian_index" not in record:
                raise ValueError(f"missing gaussian_index at {path}:{line_number}")
            index = int(record["gaussian_index"])
            if index < 0:
                raise ValueError(f"negative gaussian_index at {path}:{line_number}")
            if index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def tensor_quantile_or_zero(values: torch.Tensor, q: float) -> float:
    values = values.detach().flatten().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, q).item())


def posterior_mass_by_start(
    probabilities: torch.Tensor,
    run_start: torch.Tensor,
    *,
    frame_count: int,
) -> torch.Tensor:
    """Return per-row posterior mass binned by global run-start timestamp.

    ``run_start`` entries outside ``[0, frame_count)`` (for example the Beam-2
    missing-candidate timestamp ``-1``) are ignored.
    """
    if probabilities.shape != run_start.shape:
        raise ValueError("probabilities and run_start must have the same shape")
    if probabilities.ndim != 2:
        raise ValueError("probabilities and run_start must be rank-2")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    probs = probabilities.float()
    starts = run_start.long()
    out = torch.zeros((probs.shape[0], int(frame_count)), device=probs.device, dtype=probs.dtype)
    valid = (starts >= 0) & (starts < int(frame_count)) & torch.isfinite(probs)
    if bool(valid.any()):
        out.scatter_add_(1, starts.clamp(0, int(frame_count) - 1), torch.where(valid, probs, torch.zeros_like(probs)))
    return out


def start_mass_stats(mass_by_start: torch.Tensor, *, current_t: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    if mass_by_start.ndim != 2:
        raise ValueError("mass_by_start must be rank-2")
    mean = np.zeros((frame_count,), dtype=np.float32)
    q95 = np.zeros((frame_count,), dtype=np.float32)
    if mass_by_start.numel() == 0:
        return mean, q95
    upto = min(int(current_t), int(frame_count) - 1)
    values = mass_by_start[:, : upto + 1].detach().float()
    if values.shape[0] > 0:
        mean[: upto + 1] = values.mean(dim=0).cpu().numpy().astype(np.float32)
        q95[: upto + 1] = torch.quantile(values, 0.95, dim=0).cpu().numpy().astype(np.float32)
    return mean, q95


def recent_start_mass(mass_by_start: torch.Tensor, *, current_t: int, recent_window: int) -> torch.Tensor:
    if recent_window < 1:
        raise ValueError("recent_window must be positive")
    if mass_by_start.ndim != 2:
        raise ValueError("mass_by_start must be rank-2")
    lo = max(0, int(current_t) - int(recent_window) + 1)
    hi = min(int(current_t) + 1, mass_by_start.shape[1])
    if hi <= lo:
        return torch.zeros((mass_by_start.shape[0],), device=mass_by_start.device, dtype=mass_by_start.dtype)
    return mass_by_start[:, lo:hi].sum(dim=1)


def centered_start_cluster_mass(
    mass_by_start: torch.Tensor,
    centers: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Sum posterior mass within ``center +/- radius`` on the start axis."""
    if mass_by_start.ndim != 2:
        raise ValueError("mass_by_start must be rank-2")
    centers = torch.as_tensor(
        centers, device=mass_by_start.device, dtype=torch.long
    ).flatten()
    if centers.numel() != mass_by_start.shape[0]:
        raise ValueError("centers must contain one start timestamp per row")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    offsets = torch.arange(
        -int(radius), int(radius) + 1, device=mass_by_start.device
    )
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < mass_by_start.shape[1])
    gathered = mass_by_start.gather(
        1, indices.clamp(0, mass_by_start.shape[1] - 1)
    )
    return torch.where(valid, gathered, torch.zeros_like(gathered)).sum(dim=1)


def record_initial_lineage_mass(
    target: torch.Tensor,
    row_indices: torch.Tensor,
    mass_by_start: torch.Tensor,
    *,
    timestamp: int,
    exact_recurrence: str,
    first_observation: torch.Tensor | None = None,
) -> None:
    """Record each start hypothesis at the update that creates it.

    The existing prior-reset recurrence creates ``start=t`` after consuming
    observation ``t``. Literal Adams--MacKay Algorithm 1 instead creates an
    empty ``r_t=0`` branch for ``start=t+1``; its first evidence arrives on a
    later observation. Rows observed for the first time also create their
    initial established run at ``start=t``.
    """
    if target.ndim != 2 or mass_by_start.ndim != 2:
        raise ValueError("target and mass_by_start must be rank-2")
    rows = torch.as_tensor(
        row_indices, device=target.device, dtype=torch.long
    ).flatten()
    if rows.numel() != mass_by_start.shape[0]:
        raise ValueError("row_indices and mass_by_start must have equal rows")
    if target.shape[1] != mass_by_start.shape[1]:
        raise ValueError("target and mass_by_start must share the start axis")
    t = int(timestamp)
    if not 0 <= t < target.shape[1]:
        raise ValueError("timestamp is outside the start axis")
    if exact_recurrence == "prior_reset":
        target[rows, t] = mass_by_start[:, t]
        return
    if exact_recurrence != "adams_mackay":
        raise ValueError("exact_recurrence must be prior_reset|adams_mackay")
    if first_observation is None:
        raise ValueError("first_observation is required for adams_mackay")
    first = torch.as_tensor(
        first_observation, device=target.device, dtype=torch.bool
    ).flatten()
    if first.numel() != rows.numel():
        raise ValueError("first_observation must contain one flag per row")
    if bool(first.any()):
        target[rows[first], t] = mass_by_start[first, t]
    if t + 1 < target.shape[1]:
        target[rows, t + 1] = mass_by_start[:, t + 1]


def map_start_summary(mass_by_start: torch.Tensor) -> tuple[float | None, float | None]:
    if mass_by_start.numel() == 0 or mass_by_start.shape[0] == 0:
        return None, None
    probs, starts = mass_by_start.max(dim=1)
    return float(starts.float().mean().item()), float(probs.float().mean().item())


def summarize_boundary_lineages(
    mean_matrix: np.ndarray,
    q95_matrix: np.ndarray,
    *,
    boundaries: Sequence[int],
) -> list[dict[str, Any]]:
    """Posthoc lineage summaries from precomputed causal start-mass matrices."""
    if mean_matrix.shape != q95_matrix.shape or mean_matrix.ndim != 2:
        raise ValueError("mean and q95 matrices must have equal [T,T] shape")
    frame_count = mean_matrix.shape[0]
    summaries: list[dict[str, Any]] = []
    for boundary in boundaries:
        b = int(boundary)
        if not 0 <= b < frame_count:
            summaries.append({"boundary": b, "in_range": False})
            continue
        lineage = []
        best_mean = -1.0
        best_t = None
        crosses_half_at = None
        for t in range(b, frame_count):
            mean = float(mean_matrix[t, b])
            q95 = float(q95_matrix[t, b])
            value = {"timestamp": t, "age": t - b, "mean": mean, "q95": q95}
            lineage.append(value)
            if mean > best_mean:
                best_mean = mean
                best_t = t
            if crosses_half_at is None and mean >= 0.5:
                crosses_half_at = t
        summaries.append(
            {
                "boundary": b,
                "in_range": True,
                "peak_mean": float(best_mean),
                "peak_mean_timestamp": best_t,
                "mean_crosses_0_5_at": crosses_half_at,
                "lineage": lineage,
            }
        )
    return summaries


def summarize_lineage_recoveries(
    initial_mass: torch.Tensor,
    first_cross_timestamp: torch.Tensor,
    *,
    threshold: float,
    birth_timestamp_offset: int = 0,
) -> dict[str, Any]:
    """Summarize low-initial-probability start hypotheses that later dominate."""
    if initial_mass.shape != first_cross_timestamp.shape or initial_mass.ndim != 2:
        raise ValueError("lineage recovery tensors must have equal [rows, starts] shape")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    per_start: list[dict[str, Any]] = []
    total_low = 0
    total_recovered = 0
    for start in range(initial_mass.shape[1]):
        values = initial_mass[:, start]
        finite = torch.isfinite(values)
        positive = finite & (values > 0)
        low = positive & (values < float(threshold))
        first_cross = first_cross_timestamp[:, start].long()
        birth_timestamp = int(start) + int(birth_timestamp_offset)
        recovered = low & (first_cross > birth_timestamp)
        low_count = int(low.sum().item())
        recovered_count = int(recovered.sum().item())
        delays = first_cross[recovered] - int(start)
        total_low += low_count
        total_recovered += recovered_count
        per_start.append(
            {
                "start_timestamp": start,
                "evaluated_count": int(finite.sum().item()),
                "positive_initial_count": int(positive.sum().item()),
                "low_initial_count": low_count,
                "recovered_above_threshold_count": recovered_count,
                "recovery_rate": recovered_count / low_count if low_count else None,
                "mean_recovery_delay": float(delays.float().mean().item()) if delays.numel() else None,
                "min_recovery_delay": int(delays.min().item()) if delays.numel() else None,
                "max_recovery_delay": int(delays.max().item()) if delays.numel() else None,
            }
        )
    return {
        "threshold": float(threshold),
        "birth_timestamp_offset_from_start": int(birth_timestamp_offset),
        "low_initial_lineage_count": total_low,
        "recovered_lineage_count": total_recovered,
        "recovery_rate": total_recovered / total_low if total_low else None,
        "per_start": per_start,
    }


def _rows_in_cohort(observed_rows: torch.Tensor, cohort_mask: torch.Tensor | None) -> torch.Tensor:
    if cohort_mask is None:
        return observed_rows
    return observed_rows[cohort_mask[observed_rows]]


def detector_local_rows(
    global_rows: torch.Tensor,
    global_to_local: torch.Tensor | None,
) -> torch.Tensor:
    if global_to_local is None:
        return global_rows
    local_rows = global_to_local[global_rows]
    if bool((local_rows < 0).any()):
        raise RuntimeError("observed rows contain a Gaussian outside the detector cohort")
    return local_rows


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_frame_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "timestamp",
        "frame",
        "observed_cohort_count",
        "lineage_population_count",
        "raw_observed_count",
        "exact_p_run_zero_mean",
        "exact_p_run_zero_q95",
        "beam_p_run_zero_mean",
        "beam_p_run_zero_q95",
        "exact_recent_start_mass_mean",
        "exact_recent_start_mass_q95",
        "beam_recent_start_mass_mean",
        "beam_recent_start_mass_q95",
        "exact_map_start_mean",
        "exact_map_probability_mean",
        "beam_map_start_mean",
        "beam_map_probability_mean",
        "frame_runtime_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def snapshot_filter_start_masses(
    exact,
    beam,
    local_rows: torch.Tensor,
    *,
    frame_count: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snapshot start-time posterior masses without advancing either filter."""
    exact_chunks: list[torch.Tensor] = []
    beam_chunks: list[torch.Tensor] = []
    for start in range(0, int(local_rows.numel()), int(chunk_size)):
        rows = local_rows[start : start + int(chunk_size)]
        exact_chunks.append(
            posterior_mass_by_start(
                exact.log_run_probs[rows].exp(),
                exact.run_start[rows],
                frame_count=frame_count,
            )
        )
        branch_probability = torch.stack(
            (
                beam.log_incumbent_weight[rows].exp(),
                torch.where(
                    torch.isfinite(beam.log_candidate_weight[rows]),
                    beam.log_candidate_weight[rows].exp(),
                    torch.zeros_like(beam.log_candidate_weight[rows]),
                ),
            ),
            dim=1,
        )
        branch_start = torch.stack(
            (beam.run_start[rows], beam.candidate_run_start[rows]), dim=1
        )
        beam_chunks.append(
            posterior_mass_by_start(
                branch_probability, branch_start, frame_count=frame_count
            )
        )
    empty = torch.zeros(
        (0, frame_count), device=local_rows.device, dtype=torch.float32
    )
    return (
        torch.cat(exact_chunks, dim=0) if exact_chunks else empty,
        torch.cat(beam_chunks, dim=0) if beam_chunks else empty.clone(),
    )


def build_causal_records(source_path: Path, *, max_frames: int | None = None):
    # Lazy import prevents tests from importing dataset helpers or renderers.
    from experiments.run_online_bayesian_lifespan_thaw import build_causal_records as _build

    return _build(source_path, max_frames=max_frames)


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    # Heavy imports are intentionally local so CPU helper tests remain light.
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import seed_everything
    from gaussian_renderer import render_change  # noqa: F401  # validates renderer availability through evidence path
    from scene import GaussianModel
    from temporal.change_evidence import accumulate_change_evidence

    config = config_from_args(args)
    if not math.isfinite(float(args.exact_state_memory_limit_gb)) or float(
        args.exact_state_memory_limit_gb
    ) <= 0:
        raise ValueError("--exact-state-memory-limit-gb must be finite and positive")
    seed_everything(int(args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for alpha-T evidence accumulation")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    records, _names = build_causal_records(args.source_path, max_frames=args.max_frames)
    validate_full_run_max_r(config.max_run_length, len(records))
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    num_gaussians = int(base.get_xyz.shape[0])
    dtype = base.get_xyz.dtype
    device = base.get_xyz.device

    cohort_indices: list[int] | None = None
    use_event_cohort = args.cohort_events is not None or args.cohort == "events"
    if use_event_cohort:
        if args.cohort_events is None:
            raise ValueError("--cohort-events is required with --cohort events")
        cohort_indices = parse_cohort_events(args.cohort_events, parse_action_list(args.cohort_actions))
        invalid = [idx for idx in cohort_indices if idx >= num_gaussians]
        if invalid:
            raise ValueError(f"cohort contains gaussian index outside model: {invalid[:5]}")
        cohort_mask = torch.zeros((num_gaussians,), device=device, dtype=torch.bool)
        if cohort_indices:
            cohort_mask[torch.tensor(cohort_indices, device=device, dtype=torch.long)] = True
        cohort_contract = {
            "kind": "posthoc_event_selected_diagnostic",
            "unbiased_whole_population_inference": False,
            "events_path": str(args.cohort_events),
            "actions": sorted(parse_action_list(args.cohort_actions)),
            "size": len(cohort_indices),
        }
    else:
        cohort_mask = None
        cohort_contract = {"kind": "all", "unbiased_whole_population_inference": True, "size": num_gaussians}

    _cohort_global_rows, global_to_local, detector_row_count = build_cohort_mapping(
        num_gaussians,
        cohort_indices,
        device=device,
    )
    cohort_contract["detector_row_count"] = detector_row_count

    cfg = bocd_config(config)
    (
        _BernoulliBOCDConfig,
        BetaBernoulliBOCD,
        AdamsMacKayBetaBernoulliBOCD,
        BeamTwoBernoulliFilter,
    ) = load_bocd_classes()
    exact_state_bytes = estimated_exact_state_bytes(
        detector_row_count, config.max_run_length, dtype
    )
    exact_limit_bytes = int(float(args.exact_state_memory_limit_gb) * 1024**3)
    if exact_state_bytes > exact_limit_bytes:
        raise RuntimeError(
            "estimated compact exact BOCD state exceeds "
            "--exact-state-memory-limit-gb: "
            f"{exact_state_bytes / 1024**3:.3f} GiB > "
            f"{float(args.exact_state_memory_limit_gb):.3f} GiB; "
            "lineage tracking, Beam-2 state, renderer state, and temporaries are extra"
        )
    exact_class = (
        BetaBernoulliBOCD
        if config.exact_recurrence == "prior_reset"
        else AdamsMacKayBetaBernoulliBOCD
    )
    exact = exact_class(detector_row_count, cfg, device=device, dtype=dtype)
    beam = BeamTwoBernoulliFilter(detector_row_count, cfg, device=device, dtype=dtype)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=dtype, device=device)

    frame_count = len(records)
    if frame_count > torch.iinfo(torch.int16).max:
        raise ValueError(
            "lineage recovery timestamps require at most 32767 processed frames"
        )
    exact_mean = np.zeros((frame_count, frame_count), dtype=np.float32)
    exact_q95 = np.zeros((frame_count, frame_count), dtype=np.float32)
    beam_mean = np.zeros((frame_count, frame_count), dtype=np.float32)
    beam_q95 = np.zeros((frame_count, frame_count), dtype=np.float32)
    exact_initial_mass = torch.full(
        (detector_row_count, frame_count),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    beam_initial_mass = torch.full_like(exact_initial_mass, float("nan"))
    exact_first_cross = torch.full(
        (detector_row_count, frame_count),
        -1,
        device=device,
        dtype=torch.int16,
    )
    beam_first_cross = torch.full_like(exact_first_cross, -1)
    frame_rows: list[dict[str, Any]] = []
    commit_path = args.output_dir / "beam_commit_comparison.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if commit_path.exists():
        commit_path.unlink()
    cluster_commit_counts = {
        radius: {
            "beam_commit_count": 0,
            "exact_beam_start_cluster_ge_threshold_count": 0,
            "exact_map_start_cluster_ge_threshold_count": 0,
        }
        for radius in config.start_cluster_radii
    }

    intrinsics = None
    with commit_path.open("a", encoding="utf-8") as commit_file:
        for record in records:
            frame_started = time.time()
            t = int(record.global_index)
            views, _pose, current_intrinsics = build_fixed_cue_views(
                [record], cameras, args.cue_cache_root, args.resolution
            )
            if intrinsics is None:
                intrinsics = current_intrinsics
            elif not np.array_equal(intrinsics, current_intrinsics):
                raise ValueError("fixed views do not share one intrinsic matrix")
            view = views[0]
            evidence = accumulate_change_evidence(
                view,
                base,
                pipe,
                background,
                view.candidate_map,
                cue_mode=config.cue_mode,
                cue_threshold=config.cue_threshold,
                cue_scale=config.cue_scale,
                count_mode=config.evidence_count_mode,
                mass_saturation=config.evidence_mass_saturation,
                min_evidence_mass=config.min_evidence_mass,
            )
            observed_rows_all = torch.nonzero(evidence.total_mass >= config.min_evidence_mass, as_tuple=False).flatten()
            rows_for_frame = _rows_in_cohort(observed_rows_all, cohort_mask)
            local_rows_for_frame = detector_local_rows(rows_for_frame, global_to_local)

            exact_p0_values: list[torch.Tensor] = []
            beam_p0_values: list[torch.Tensor] = []
            exact_recent_values: list[torch.Tensor] = []
            beam_recent_values: list[torch.Tensor] = []
            exact_mass_chunks: list[torch.Tensor] = []
            beam_mass_chunks: list[torch.Tensor] = []
            exact_first_observation_chunks: list[torch.Tensor] = []

            for start in range(0, int(rows_for_frame.numel()), config.chunk_size):
                rows = rows_for_frame[start : start + config.chunk_size]
                local_rows = local_rows_for_frame[start : start + config.chunk_size]
                exact_first_observation_chunks.append(
                    (exact.visible_observations[local_rows] == 0).detach()
                )
                exact_result = exact.update(
                    evidence.delta_a[rows],
                    evidence.delta_b[rows],
                    total_mass=evidence.total_mass[rows],
                    row_indices=local_rows,
                    timestamp=t,
                )
                beam_result = beam.update(
                    evidence.delta_a[rows],
                    evidence.delta_b[rows],
                    total_mass=evidence.total_mass[rows],
                    row_indices=local_rows,
                    timestamp=t,
                )

                exact_starts = exact.run_start[local_rows]
                exact_mass = posterior_mass_by_start(
                    exact_result.run_length_posterior, exact_starts, frame_count=frame_count
                )
                beam_mass = posterior_mass_by_start(
                    beam_result.branch_probability, beam_result.branch_run_start, frame_count=frame_count
                )
                exact_mass_chunks.append(exact_mass.detach())
                beam_mass_chunks.append(beam_mass.detach())

                exact_p0_values.append(exact_result.changepoint_probability.detach().float().cpu())
                beam_p0_values.append(beam_result.run_length_posterior[:, 0].detach().float().cpu())
                exact_recent_values.append(recent_start_mass(exact_mass, current_t=t, recent_window=config.recent_window).detach().cpu())
                beam_recent_values.append(recent_start_mass(beam_mass, current_t=t, recent_window=config.recent_window).detach().cpu())
                exact_rows = torch.arange(rows.numel(), device=device)

                commit = beam_result.candidate_probability >= config.changepoint_probability
                if bool(commit.any()):
                    pos = torch.nonzero(commit, as_tuple=False).flatten()
                    exact_same = torch.zeros((pos.numel(),), device=device, dtype=dtype)
                    cand_starts = beam_result.candidate_run_start[pos].long()
                    valid = (cand_starts >= 0) & (cand_starts < frame_count)
                    if bool(valid.any()):
                        exact_same[valid] = exact_mass[pos[valid], cand_starts[valid]]
                    exact_map_run_prob = exact_result.run_length_posterior[
                        exact_rows, exact_result.map_run_length
                    ]
                    exact_binned_map_prob, exact_binned_map_start = exact_mass.max(
                        dim=1
                    )
                    beam_start_clusters = {
                        radius: centered_start_cluster_mass(
                            exact_mass[pos], cand_starts, radius=radius
                        )
                        for radius in config.start_cluster_radii
                    }
                    exact_map_clusters = {
                        radius: centered_start_cluster_mass(
                            exact_mass[pos],
                            exact_binned_map_start[pos],
                            radius=radius,
                        )
                        for radius in config.start_cluster_radii
                    }
                    for radius in config.start_cluster_radii:
                        counts = cluster_commit_counts[radius]
                        counts["beam_commit_count"] += int(pos.numel())
                        counts[
                            "exact_beam_start_cluster_ge_threshold_count"
                        ] += int(
                            (
                                beam_start_clusters[radius]
                                >= config.changepoint_probability
                            )
                            .sum()
                            .item()
                        )
                        counts[
                            "exact_map_start_cluster_ge_threshold_count"
                        ] += int(
                            (
                                exact_map_clusters[radius]
                                >= config.changepoint_probability
                            )
                            .sum()
                            .item()
                        )
                    for j, p in enumerate(pos.tolist()):
                        record_out = {
                            "gaussian_index": int(rows[p].item()),
                            "decision_timestamp": t,
                            "beam_estimated_start": int(beam_result.candidate_run_start[p].item()),
                            "beam_candidate_probability": float(beam_result.candidate_probability[p].item()),
                            "exact_posterior_mass_on_same_start": float(exact_same[j].item()),
                            "exact_map_run_start": int(
                                exact_result.estimated_run_start[p].item()
                            ),
                            "exact_map_run_probability": float(
                                exact_map_run_prob[p].item()
                            ),
                            "exact_map_start": int(
                                exact_result.estimated_run_start[p].item()
                            ),
                            "exact_map_probability": float(
                                exact_map_run_prob[p].item()
                            ),
                            "exact_binned_map_start": int(
                                exact_binned_map_start[p].item()
                            ),
                            "exact_binned_map_start_probability": float(
                                exact_binned_map_prob[p].item()
                            ),
                            "exact_p_run_zero": float(exact_result.changepoint_probability[p].item()),
                        }
                        for radius in config.start_cluster_radii:
                            record_out[
                                f"exact_beam_start_cluster_mass_r{radius}"
                            ] = float(beam_start_clusters[radius][j].item())
                            record_out[
                                f"exact_map_start_cluster_mass_r{radius}"
                            ] = float(exact_map_clusters[radius][j].item())
                        commit_file.write(json.dumps(record_out, sort_keys=True) + "\n")

            if exact_mass_chunks:
                observed_exact_frame_mass = torch.cat(exact_mass_chunks, dim=0)
                observed_beam_frame_mass = torch.cat(beam_mass_chunks, dim=0)
                exact_first_observation = torch.cat(
                    exact_first_observation_chunks, dim=0
                )
            else:
                observed_exact_frame_mass = torch.zeros(
                    (0, frame_count), device=device, dtype=torch.float32
                )
                observed_beam_frame_mass = torch.zeros_like(
                    observed_exact_frame_mass
                )
                exact_first_observation = torch.zeros(
                    (0,), device=device, dtype=torch.bool
                )
            exact_frame_mass = observed_exact_frame_mass
            beam_frame_mass = observed_beam_frame_mass
            if use_event_cohort:
                lineage_rows = torch.nonzero(
                    exact.visible_observations > 0, as_tuple=False
                ).flatten()
                exact_frame_mass, beam_frame_mass = snapshot_filter_start_masses(
                    exact,
                    beam,
                    lineage_rows,
                    frame_count=frame_count,
                    chunk_size=config.chunk_size,
                )
            else:
                lineage_rows = local_rows_for_frame
            exact_mean[t], exact_q95[t] = start_mass_stats(exact_frame_mass, current_t=t, frame_count=frame_count)
            beam_mean[t], beam_q95[t] = start_mass_stats(beam_frame_mass, current_t=t, frame_count=frame_count)
            if local_rows_for_frame.numel():
                record_initial_lineage_mass(
                    exact_initial_mass,
                    local_rows_for_frame,
                    observed_exact_frame_mass,
                    timestamp=t,
                    exact_recurrence=config.exact_recurrence,
                    first_observation=exact_first_observation,
                )
                record_initial_lineage_mass(
                    beam_initial_mass,
                    local_rows_for_frame,
                    observed_beam_frame_mass,
                    timestamp=t,
                    exact_recurrence="prior_reset",
                )
                exact_upto = min(
                    t + (1 if config.exact_recurrence == "adams_mackay" else 0),
                    frame_count - 1,
                )
                exact_cross_view = exact_first_cross[
                    local_rows_for_frame, : exact_upto + 1
                ]
                beam_cross_view = beam_first_cross[
                    local_rows_for_frame, : t + 1
                ]
                exact_new_cross = (exact_cross_view < 0) & (
                    observed_exact_frame_mass[:, : exact_upto + 1]
                    >= config.changepoint_probability
                )
                beam_new_cross = (beam_cross_view < 0) & (
                    observed_beam_frame_mass[:, : t + 1]
                    >= config.changepoint_probability
                )
                exact_cross_view[exact_new_cross] = int(t)
                beam_cross_view[beam_new_cross] = int(t)
                exact_first_cross[
                    local_rows_for_frame, : exact_upto + 1
                ] = exact_cross_view
                beam_first_cross[
                    local_rows_for_frame, : t + 1
                ] = beam_cross_view

            exact_map_start_mean, exact_map_prob_mean = map_start_summary(exact_frame_mass)
            beam_map_start_mean, beam_map_prob_mean = map_start_summary(beam_frame_mass)
            ex_p0 = torch.cat(exact_p0_values) if exact_p0_values else torch.empty(0)
            be_p0 = torch.cat(beam_p0_values) if beam_p0_values else torch.empty(0)
            ex_recent = torch.cat(exact_recent_values) if exact_recent_values else torch.empty(0)
            be_recent = torch.cat(beam_recent_values) if beam_recent_values else torch.empty(0)
            frame_rows.append(
                {
                    "timestamp": t,
                    "frame": record.name,
                    "observed_cohort_count": int(rows_for_frame.numel()),
                    "lineage_population_count": int(lineage_rows.numel()),
                    "raw_observed_count": int(observed_rows_all.numel()),
                    "exact_p_run_zero_mean": float(ex_p0.mean().item()) if ex_p0.numel() else None,
                    "exact_p_run_zero_q95": tensor_quantile_or_zero(ex_p0, 0.95) if ex_p0.numel() else None,
                    "beam_p_run_zero_mean": float(be_p0.mean().item()) if be_p0.numel() else None,
                    "beam_p_run_zero_q95": tensor_quantile_or_zero(be_p0, 0.95) if be_p0.numel() else None,
                    "exact_recent_start_mass_mean": float(ex_recent.mean().item()) if ex_recent.numel() else None,
                    "exact_recent_start_mass_q95": tensor_quantile_or_zero(ex_recent, 0.95) if ex_recent.numel() else None,
                    "beam_recent_start_mass_mean": float(be_recent.mean().item()) if be_recent.numel() else None,
                    "beam_recent_start_mass_q95": tensor_quantile_or_zero(be_recent, 0.95) if be_recent.numel() else None,
                    "exact_map_start_mean": exact_map_start_mean,
                    "exact_map_probability_mean": exact_map_prob_mean,
                    "beam_map_start_mean": beam_map_start_mean,
                    "beam_map_probability_mean": beam_map_prob_mean,
                    "frame_runtime_seconds": time.time() - frame_started,
                }
            )

    boundaries = tuple(DEFAULT_BOUNDARIES if args.boundaries is None else args.boundaries)
    boundary_summary = {
        "exact": summarize_boundary_lineages(exact_mean, exact_q95, boundaries=boundaries),
        "beam2": summarize_boundary_lineages(beam_mean, beam_q95, boundaries=boundaries),
        "posthoc_only": True,
    }
    lineage_recovery = {
        "definition": (
            "a start hypothesis has 0 < posterior mass at branch birth < "
            "threshold and crosses threshold after its recurrence-specific "
            "birth update"
        ),
        "birth_contract": (
            "prior_reset creates start=t after observation t; adams_mackay "
            "creates empty r_t=0 start=t+1 after observation t"
        ),
        "exact": summarize_lineage_recoveries(
            exact_initial_mass,
            exact_first_cross,
            threshold=config.changepoint_probability,
            birth_timestamp_offset=(
                -1 if config.exact_recurrence == "adams_mackay" else 0
            ),
        ),
        "beam2": summarize_lineage_recoveries(
            beam_initial_mass,
            beam_first_cross,
            threshold=config.changepoint_probability,
        ),
    }
    (args.output_dir / "lineage_recovery.json").write_text(
        json.dumps(lineage_recovery, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    peak_cuda = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    cluster_commit_summary = {}
    for radius, counts in cluster_commit_counts.items():
        total = counts["beam_commit_count"]
        cluster_commit_summary[str(radius)] = {
            **counts,
            "exact_beam_start_cluster_ge_threshold_rate": (
                counts["exact_beam_start_cluster_ge_threshold_count"] / total
                if total
                else None
            ),
            "exact_map_start_cluster_ge_threshold_rate": (
                counts["exact_map_start_cluster_ge_threshold_count"] / total
                if total
                else None
            ),
        }
    summary = {
        "schema_version": 2,
        "algorithm_names": {"exact": exact.algorithm, "beam2": beam.algorithm},
        "contract": "detector_only_full_bocd_start_posterior_diagnostic",
        "recurrence_contract": {
            "exact": config.exact_recurrence,
            "exact_and_beam_share_recurrence": config.exact_recurrence
            == "prior_reset",
            "beam2": "protected_prior_reset_candidate",
            "adams_mackay_note": (
                "Algorithm 1 scores current evidence under each previous-run "
                "predictive for both growth and CP; r=0 resets to the prior for "
                "the next observation"
            ),
        },
        "frames": frame_count,
        "gaussian_count": num_gaussians,
        "cohort": cohort_contract,
        "lineage_aggregation": (
            "posthoc_event_cohort_rows_observed_at_least_once_by_timestamp"
            if use_event_cohort
            else "all_rows_observed_at_current_timestamp"
        ),
        "cue_contract": {
            "mode": config.cue_mode,
            "threshold": config.cue_threshold,
            "scale": config.cue_scale,
            "binary": "candidate_map > threshold" if config.cue_mode == "binary" else None,
            "soft": "clamp(candidate_map / scale, 0, 1) fractional evidence" if config.cue_mode == "soft" else None,
            "count_mode": config.evidence_count_mode,
            "mass_saturation": config.evidence_mass_saturation,
            "min_evidence_mass": config.min_evidence_mass,
        },
        "bocd": {
            "prior_a": config.prior_a,
            "prior_b": config.prior_b,
            "hazard": cfg.resolved_hazard,
            "expected_run_length": config.expected_run_length,
            "max_run_length": config.max_run_length,
            "full_run_validation": "max_run_length >= processed frame count",
            "changepoint_probability": config.changepoint_probability,
            "recent_window": config.recent_window,
            "start_cluster_radii": list(config.start_cluster_radii),
        },
        "beam_commit_cluster_scan": cluster_commit_summary,
        "memory_estimates": {
            "detector_row_count": detector_row_count,
            "exact_resident_bytes": exact_state_bytes,
            "beam2_resident_bytes": estimated_beam2_state_bytes(detector_row_count, dtype),
            "exact_resident_gib": exact_state_bytes / 1024**3,
            "beam2_resident_gib": estimated_beam2_state_bytes(detector_row_count, dtype) / 1024**3,
            "all_gaussian_exact_resident_gib": estimated_exact_state_bytes(
                num_gaussians, config.max_run_length, dtype
            )
            / 1024**3,
            "lineage_tracking_bytes": int(
                detector_row_count
                * frame_count
                * (
                    2 * torch.empty((), dtype=dtype).element_size()
                    + 2 * torch.empty((), dtype=torch.int16).element_size()
                )
            ),
            "start_mass_stats_bytes": int(4 * frame_count * frame_count * np.dtype(np.float32).itemsize),
            "actual_peak_cuda_memory_bytes": int(peak_cuda),
        },
        "runtime_seconds": time.time() - started,
        "boundary_lineage_after_causal_loop": boundary_summary,
        "lineage_recovery": {
            "exact": {
                key: value
                for key, value in lineage_recovery["exact"].items()
                if key != "per_start"
            },
            "beam2": {
                key: value
                for key, value in lineage_recovery["beam2"].items()
                if key != "per_start"
            },
        },
        "ground_truth_masks_loaded": False,
        "manual_boundaries_consumed_after_inference_only": True,
        "cue_cache_metadata": cue_metadata,
        "arguments": serializable_args(args),
        "output_files": list(OUTPUT_FILES),
    }
    np.savez_compressed(
        args.output_dir / "start_mass_stats.npz",
        exact_mean=exact_mean,
        exact_q95=exact_q95,
        beam2_mean=beam_mean,
        beam2_q95=beam_q95,
    )
    write_frame_metrics(args.output_dir / "frame_metrics.csv", frame_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exact-recurrence",
        choices=("prior_reset", "adams_mackay"),
        default="prior_reset",
    )
    parser.add_argument("--cue-mode", choices=("binary", "soft"), default="binary")
    parser.add_argument("--cue-threshold", type=float, default=0.5)
    parser.add_argument("--cue-scale", type=float, default=1.0)
    parser.add_argument("--evidence-count-mode", choices=("raw", "capped"), default="capped")
    parser.add_argument("--evidence-mass-saturation", type=float, default=1.0)
    parser.add_argument("--min-evidence-mass", type=float, default=1e-6)
    parser.add_argument("--prior-a", type=float, default=1.0)
    parser.add_argument("--prior-b", type=float, default=1.0)
    parser.add_argument("--expected-run-length", type=float, default=100.0)
    parser.add_argument("--hazard", type=float, default=None)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--max-run-length", type=positive_int, default=304)
    parser.add_argument(
        "--exact-state-memory-limit-gb",
        "--exact-memory-limit-gb",
        dest="exact_state_memory_limit_gb",
        type=float,
        default=16.0,
        help=(
            "limit for exact BOCD persistent state only; lineage tracking, "
            "Beam-2, renderer state, and temporaries require additional memory"
        ),
    )
    parser.add_argument("--chunk-size", type=positive_int, default=65536)
    parser.add_argument("--changepoint-probability", type=float, default=0.5)
    parser.add_argument("--min-run-evidence", type=nonnegative_float, default=0.0)
    parser.add_argument("--min-visible-observations", type=int, default=1)
    parser.add_argument("--recent-window", type=positive_int, default=5)
    parser.add_argument(
        "--start-cluster-radii",
        nargs="+",
        type=nonnegative_int,
        default=[0, 1, 2, 3, 5],
        help="global-timestamp radii scanned around Beam and exact MAP starts",
    )
    parser.add_argument("--cohort", choices=("all", "events"), default="all", help="use all observed rows; --cohort-events selects a posthoc event cohort")
    parser.add_argument("--cohort-events", type=Path, default=None)
    parser.add_argument("--cohort-actions", type=str, default="CLOSE")
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_diagnostic(args)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "frames": summary["frames"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
