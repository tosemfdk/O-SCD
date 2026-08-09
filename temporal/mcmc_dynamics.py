"""Fixed-topology MCMC dynamics for temporal change Gaussians.

This module is intentionally sidecar-only: it does not append, delete, or
replace ``nn.Parameter`` objects.  Callers pass any Gaussian-like object that
exposes the following trainable tensors/properties:

``xyz`` [N,3], ``change_dc`` [N,...], ``change_opacity_raw`` [N,1],
``scaling_raw`` [N,3], and ``rotation_raw`` [N,4].  The resolver also
accepts the fixed-capacity state's ``current_*`` parameter names.

The implementation follows the 3DGS-MCMC paper's two core moves: covariance- and
opacity-gated xyz SGLD noise (Eq. 8), and dead-to-live relocation with the
composition-preserving opacity/scale update (Eq. 9).  Selection uses activated
change opacity, not base scene opacity, which keeps this suitable for temporal
R_change experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

import torch


GAUSSIAN_FIELDS = (
    "xyz",
    "change_dc",
    "change_opacity_raw",
    "scaling_raw",
    "rotation_raw",
)

FIELD_ALIASES = {
    "xyz": ("xyz", "current_xyz", "_xyz"),
    "change_dc": (
        "change_dc",
        "current_change_dc",
        "current_features_dc",
        "features_dc",
        "_features_dc",
    ),
    "change_opacity_raw": (
        "change_opacity_raw",
        "current_raw_change_opacity",
        "raw_change_opacity",
        "_opacity",
    ),
    "scaling_raw": ("scaling_raw", "current_scaling", "scaling", "_scaling"),
    "rotation_raw": ("rotation_raw", "current_rotation", "rotation", "_rotation"),
}


@dataclass(frozen=True)
class MCMCNoiseStats:
    """Summary of one xyz SGLD perturbation."""

    count: int
    xyz_lr: float
    noise_scale: float
    gate_min: float
    gate_mean: float
    gate_max: float
    noise_mean_abs: float
    noise_p50_abs: float
    noise_p95_abs: float
    noise_rms: float
    noise_max_abs: float
    opacity_bin_stats: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class RelocationAssignment:
    """A precomputed dead-source to live-target relocation decision."""

    source_index: int
    target_index: int
    source_opacity: float
    target_opacity: float


@dataclass(frozen=True)
class RelocationEvent:
    """Audit record for one grouped Eq. 9 relocation application/defer."""

    target_index: int
    source_indices: tuple[int, ...]
    group_total: int
    old_target_opacity: float
    theoretical_new_group_opacity: float | None
    new_group_opacity: float | None
    covariance_scale: float | None
    applied: bool
    deferred_source_indices: tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RelocationResult:
    """Complete audit returned by :func:`relocate_dead_gaussians_`."""

    selected_dead_indices: tuple[int, ...]
    selected_live_indices: tuple[int, ...]
    assignments: tuple[RelocationAssignment, ...]
    events: tuple[RelocationEvent, ...]
    applied_source_indices: tuple[int, ...]
    deferred_source_indices: tuple[int, ...]
    reset_target_indices: tuple[int, ...]

    @property
    def applied_count(self) -> int:
        return len(self.applied_source_indices)

    @property
    def deferred_count(self) -> int:
        return len(self.deferred_source_indices)

    def audit(self) -> dict[str, Any]:
        """Return JSON-friendly counters and index lists."""
        return {
            "selected_dead_count": len(self.selected_dead_indices),
            "selected_live_count": len(self.selected_live_indices),
            "assignment_count": len(self.assignments),
            "applied_count": self.applied_count,
            "deferred_count": self.deferred_count,
            "reset_target_count": len(self.reset_target_indices),
            "selected_dead_indices": list(self.selected_dead_indices),
            "selected_live_indices": list(self.selected_live_indices),
            "applied_source_indices": list(self.applied_source_indices),
            "deferred_source_indices": list(self.deferred_source_indices),
            "reset_target_indices": list(self.reset_target_indices),
            "events": [event_to_dict(event) for event in self.events],
        }


def event_to_dict(event: RelocationEvent) -> dict[str, Any]:
    """Convert an event dataclass to a JSON-friendly dict."""
    return {
        "target_index": event.target_index,
        "source_indices": list(event.source_indices),
        "group_total": event.group_total,
        "old_target_opacity": event.old_target_opacity,
        "theoretical_new_group_opacity": event.theoretical_new_group_opacity,
        "new_group_opacity": event.new_group_opacity,
        "covariance_scale": event.covariance_scale,
        "applied": event.applied,
        "deferred_source_indices": list(event.deferred_source_indices),
        "reason": event.reason,
    }


def dynamics_audit(
    noise_stats: MCMCNoiseStats | None = None,
    relocation_result: RelocationResult | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-friendly audit for a dynamics step."""
    audit: dict[str, Any] = {}
    if noise_stats is not None:
        audit["noise"] = noise_stats.__dict__.copy()
    if relocation_result is not None:
        audit["relocation"] = relocation_result.audit()
    return audit


def _get_tensor(obj: Any, name: str) -> torch.Tensor:
    aliases = FIELD_ALIASES.get(name, (name,))
    for alias in aliases:
        if hasattr(obj, alias):
            value = getattr(obj, alias)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{alias} resolved for {name} must be a torch.Tensor")
            return value
    joined = ", ".join(repr(alias) for alias in aliases)
    raise AttributeError(f"gaussians must expose {name!r} via one of: {joined}")


def get_mcmc_tensors(gaussians: Any) -> dict[str, torch.Tensor]:
    """Return and validate the generic tensor interface used by this module."""
    tensors = {name: _get_tensor(gaussians, name) for name in GAUSSIAN_FIELDS}
    xyz = tensors["xyz"]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must have shape [N, 3] and be nonempty")
    n = xyz.shape[0]
    expected = {
        "change_opacity_raw": (n, 1),
        "scaling_raw": (n, 3),
        "rotation_raw": (n, 4),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if tensors["change_dc"].shape[0] != n:
        raise ValueError("change_dc must have the same leading dimension as xyz")
    for name, tensor in tensors.items():
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must be floating point")
        if tensor.device != xyz.device:
            raise ValueError("all MCMC tensors must be on the same device")
        if tensor.dtype != xyz.dtype:
            raise TypeError("all MCMC tensors must share dtype")
    return tensors


def activated_change_opacity(change_opacity_raw: torch.Tensor) -> torch.Tensor:
    """Return sigmoid-activated change opacity as a flat ``[N]`` tensor."""
    if change_opacity_raw.ndim != 2 or change_opacity_raw.shape[1] != 1:
        raise ValueError("change_opacity_raw must have shape [N, 1]")
    if not torch.is_floating_point(change_opacity_raw):
        raise TypeError("change_opacity_raw must be floating point")
    return torch.sigmoid(change_opacity_raw.detach()).flatten()


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    _validate_seed(seed)
    if seed is None:
        return None
    # CPU generators cannot drive CUDA random kernels; use a device-local
    # generator so seeded CUDA and CPU runs are both reproducible.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _as_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _as_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _empty_noise_bin() -> dict[str, float | int]:
    return {
        "count": 0,
        "opacity_min": float("nan"),
        "opacity_max": float("nan"),
        "noise_mean_abs": 0.0,
        "noise_p50_abs": 0.0,
        "noise_p95_abs": 0.0,
        "noise_rms": 0.0,
        "noise_max_abs": 0.0,
    }


def _noise_summary(abs_noise: torch.Tensor) -> dict[str, float | int]:
    flat = abs_noise.detach().flatten()
    if flat.numel() == 0:
        return _empty_noise_bin()
    return {
        "count": int(flat.numel()),
        "noise_mean_abs": float(flat.mean().item()),
        "noise_p50_abs": float(torch.quantile(flat, 0.50).item()),
        "noise_p95_abs": float(torch.quantile(flat, 0.95).item()),
        "noise_rms": float(torch.sqrt((flat * flat).mean()).item()),
        "noise_max_abs": float(flat.max().item()),
    }


def _opacity_bin_noise_stats(
    opacity: torch.Tensor,
    noise: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    abs_per_row = noise.detach().norm(dim=1)
    bins = {
        "dead_lt_threshold": opacity < threshold,
        "near_threshold_005_05": (opacity >= threshold) & (opacity < 0.05),
        "mid_05_50": (opacity >= 0.05) & (opacity < 0.5),
        "high_ge_50": opacity >= 0.5,
    }
    out: dict[str, dict[str, float | int]] = {}
    for name, mask in bins.items():
        if mask.any():
            values = abs_per_row[mask]
            stats = _noise_summary(values)
            stats["count"] = int(mask.sum().item())
            stats["opacity_min"] = float(opacity[mask].min().item())
            stats["opacity_max"] = float(opacity[mask].max().item())
            out[name] = stats
        else:
            out[name] = _empty_noise_bin()
    return out


def normalize_quaternion(rotation_raw: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize ``[w, x, y, z]`` quaternions without importing project utilities."""
    return rotation_raw / rotation_raw.norm(dim=1, keepdim=True).clamp_min(eps)


def rotation_matrix_from_quaternion(rotation_raw: torch.Tensor) -> torch.Tensor:
    """Return 3x3 rotation matrices from ``[w, x, y, z]`` quaternions."""
    q = normalize_quaternion(rotation_raw)
    w, x, y, z = q.unbind(dim=1)
    matrices = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def covariance_from_scaling_rotation(
    scaling_raw: torch.Tensor,
    rotation_raw: torch.Tensor,
) -> torch.Tensor:
    """Build full 3D covariance matrices from 3DGS raw scale/rotation tensors."""
    if scaling_raw.ndim != 2 or scaling_raw.shape[1] != 3:
        raise ValueError("scaling_raw must have shape [N, 3]")
    if rotation_raw.ndim != 2 or rotation_raw.shape != (scaling_raw.shape[0], 4):
        raise ValueError("rotation_raw must have shape [N, 4]")
    scales = torch.exp(scaling_raw.detach())
    rotation = rotation_matrix_from_quaternion(rotation_raw.detach())
    return rotation @ torch.diag_embed(scales * scales) @ rotation.transpose(1, 2)


def sgld_xyz_noise(
    gaussians: Any,
    *,
    xyz_lr: float,
    noise_scale: float = 1.0,
    opacity_gate_k: float = 100.0,
    opacity_gate_threshold: float = 0.005,
    seed: int | None = None,
) -> tuple[torch.Tensor, MCMCNoiseStats]:
    """Compute official-style Eq. 8 xyz SGLD noise without applying it."""
    tensors = get_mcmc_tensors(gaussians)
    xyz_lr = _as_positive_float("xyz_lr", xyz_lr)
    noise_scale = _as_nonnegative_float("noise_scale", noise_scale)
    opacity_gate_k = _as_positive_float("opacity_gate_k", opacity_gate_k)
    opacity_gate_threshold = _as_nonnegative_float(
        "opacity_gate_threshold", opacity_gate_threshold
    )

    opacity = activated_change_opacity(tensors["change_opacity_raw"])
    # Official gate: sigmoid(k * ((1 - opacity) - 0.995)).  With the
    # default threshold this is sigmoid(k * (0.005 - opacity)), so low-opacity
    # / dead slots receive larger SGLD noise and high-opacity slots are quiet.
    gate = torch.sigmoid(opacity_gate_k * (opacity_gate_threshold - opacity))
    covariance = covariance_from_scaling_rotation(
        tensors["scaling_raw"], tensors["rotation_raw"]
    )
    generator = _make_generator(tensors["xyz"].device, seed)
    eta = torch.randn(
        tensors["xyz"].shape,
        device=tensors["xyz"].device,
        dtype=tensors["xyz"].dtype,
        generator=generator,
    )
    noise = xyz_lr * noise_scale * gate[:, None] * torch.bmm(
        covariance, eta.unsqueeze(-1)
    ).squeeze(-1)
    detached = noise.detach()
    abs_noise = detached.abs()
    flat_abs = abs_noise.flatten()
    stats = MCMCNoiseStats(
        count=int(noise.shape[0]),
        xyz_lr=xyz_lr,
        noise_scale=noise_scale,
        gate_min=float(gate.min().item()),
        gate_mean=float(gate.mean().item()),
        gate_max=float(gate.max().item()),
        noise_mean_abs=float(flat_abs.mean().item()),
        noise_p50_abs=float(torch.quantile(flat_abs, 0.50).item()),
        noise_p95_abs=float(torch.quantile(flat_abs, 0.95).item()),
        noise_rms=float(torch.sqrt((detached * detached).mean()).item()),
        noise_max_abs=float(flat_abs.max().item()),
        opacity_bin_stats=_opacity_bin_noise_stats(
            opacity, detached, threshold=opacity_gate_threshold
        ),
    )
    return noise, stats


def apply_sgld_xyz_noise_(
    gaussians: Any,
    *,
    xyz_lr: float,
    noise_scale: float = 1.0,
    opacity_gate_k: float = 100.0,
    opacity_gate_threshold: float = 0.005,
    seed: int | None = None,
) -> MCMCNoiseStats:
    """Add Eq. 8 xyz SGLD noise in-place, preserving ``nn.Parameter`` identity."""
    tensors = get_mcmc_tensors(gaussians)
    noise, stats = sgld_xyz_noise(
        gaussians,
        xyz_lr=xyz_lr,
        noise_scale=noise_scale,
        opacity_gate_k=opacity_gate_k,
        opacity_gate_threshold=opacity_gate_threshold,
        seed=seed,
    )
    with torch.no_grad():
        tensors["xyz"].add_(noise)
    return stats


def select_dead_live_indices(
    change_opacity_raw: torch.Tensor,
    *,
    opacity_threshold: float = 0.005,
    relocation_count: int | None = None,
    selection_policy: str = "seeded",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select fixed-N dead sources and live targets from activated change opacity.

    Live targets are all rows at/above the threshold.  When capped, dead-source
    selection defaults to a seeded deterministic permutation over all dead rows;
    pass ``selection_policy="lowest_opacity"`` for the earlier deterministic
    lowest-opacity cap.  ``None`` relocation_count returns every dead row.
    """
    opacity_threshold = _as_nonnegative_float("opacity_threshold", opacity_threshold)
    if relocation_count is not None and (
        isinstance(relocation_count, bool)
        or not isinstance(relocation_count, Integral)
        or relocation_count < 0
    ):
        raise ValueError("relocation_count must be a nonnegative integer or None")
    if selection_policy not in {"seeded", "lowest_opacity", "index"}:
        raise ValueError("selection_policy must be 'seeded', 'lowest_opacity', or 'index'")
    opacity = activated_change_opacity(change_opacity_raw)
    dead = torch.nonzero(opacity < opacity_threshold, as_tuple=False).flatten()
    live = torch.nonzero(opacity >= opacity_threshold, as_tuple=False).flatten()
    if live.numel() > 0:
        live = live[torch.argsort(live, stable=True)]
    if dead.numel() == 0:
        return dead, live

    if relocation_count is None or int(relocation_count) >= int(dead.numel()):
        # Stable full-set order is useful for audits when no cap is applied.
        return dead[torch.argsort(dead, stable=True)], live

    cap = int(relocation_count)
    if selection_policy == "seeded":
        generator = _make_generator(change_opacity_raw.device, seed)
        perm = torch.randperm(int(dead.numel()), device=dead.device, generator=generator)
        selected = dead[perm[:cap]]
        dead = selected[torch.argsort(selected, stable=True)]
    elif selection_policy == "lowest_opacity":
        order = torch.argsort(opacity[dead], stable=True)
        dead = dead[order[:cap]]
    else:
        dead = dead[torch.argsort(dead, stable=True)[:cap]]
    return dead, live


def sample_opacity_weighted_assignments(
    change_opacity_raw: torch.Tensor,
    dead_indices: torch.Tensor,
    live_indices: torch.Tensor,
    *,
    seed: int | None = None,
) -> tuple[RelocationAssignment, ...]:
    """Precompute all dead->live assignments via seeded opacity-weighted multinomial."""
    if dead_indices.ndim != 1 or live_indices.ndim != 1:
        raise ValueError("dead_indices and live_indices must be rank-1")
    opacity = activated_change_opacity(change_opacity_raw)
    if dead_indices.numel() == 0 or live_indices.numel() == 0:
        return ()
    weights = opacity[live_indices].clamp_min(0)
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("live opacity weights must be finite with positive sum")
    generator = _make_generator(change_opacity_raw.device, seed)
    sampled = torch.multinomial(
        weights,
        int(dead_indices.numel()),
        replacement=True,
        generator=generator,
    )
    target_indices = live_indices[sampled]
    dead_cpu = dead_indices.detach().cpu().tolist()
    target_cpu = target_indices.detach().cpu().tolist()
    opacity_cpu = opacity.detach().cpu()
    return tuple(
        RelocationAssignment(
            source_index=int(source),
            target_index=int(target),
            source_opacity=float(opacity_cpu[int(source)].item()),
            target_opacity=float(opacity_cpu[int(target)].item()),
        )
        for source, target in zip(dead_cpu, target_cpu)
    )


def group_assignments(
    assignments: Sequence[RelocationAssignment],
) -> dict[int, list[RelocationAssignment]]:
    """Group precomputed assignments by target, preserving assignment order."""
    groups: dict[int, list[RelocationAssignment]] = {}
    for assignment in assignments:
        groups.setdefault(assignment.target_index, []).append(assignment)
    return groups


def _eq9_new_opacity(old_opacity: torch.Tensor, group_total: int) -> torch.Tensor:
    """Theoretical Eq. 9 relocated opacity before writeback clamping."""
    n = int(group_total)
    return 1.0 - torch.pow(1.0 - old_opacity, 1.0 / n)


def _eq9_covariance_scale(
    old_opacity: torch.Tensor,
    theoretical_new_opacity: torch.Tensor,
    group_total: int,
    *,
    eps: float,
) -> torch.Tensor:
    coeff = torch.zeros((), device=old_opacity.device, dtype=old_opacity.dtype)
    for i in range(1, int(group_total) + 1):
        for k in range(i):
            term = math.comb(i - 1, k) * ((-1.0) ** k) / math.sqrt(k + 1.0)
            coeff = coeff + term * torch.pow(theoretical_new_opacity, k + 1)
    return (old_opacity * old_opacity) / (coeff * coeff).clamp_min(eps)


def eq9_relocation_terms(
    old_opacity: torch.Tensor,
    group_total: int,
    *,
    eps: float | None = None,
    min_opacity: float = 0.005,
    return_theoretical: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return official Eq. 9 relocation terms.

    The covariance scale is computed with the theoretical new opacity exactly as
    upstream ``compute_relocation`` does.  The returned writeback opacity is then
    clamped to ``[min_opacity, 1 - finfo(dtype).eps]`` to match the later Python
    parameter update safeguard.  Set ``return_theoretical=True`` to also return
    the unclamped theoretical opacity used for the scale coefficient.
    """
    if isinstance(group_total, bool) or not isinstance(group_total, Integral) or group_total < 1:
        raise ValueError("group_total must be a positive integer")
    if old_opacity.ndim != 0:
        old_opacity = old_opacity.reshape(())
    clamp_eps = _dtype_eps(old_opacity, eps)
    min_opacity = _as_nonnegative_float("min_opacity", min_opacity)
    if min_opacity >= 1.0:
        raise ValueError("min_opacity must be less than 1")
    max_opacity = 1.0 - clamp_eps
    o_old = old_opacity.clamp(min=clamp_eps, max=max_opacity)
    o_new_theoretical = _eq9_new_opacity(o_old, int(group_total))
    cov_scale = _eq9_covariance_scale(
        o_old, o_new_theoretical, int(group_total), eps=clamp_eps
    )
    o_new_write = o_new_theoretical.clamp(min=min_opacity, max=max_opacity)
    if return_theoretical:
        return o_new_write, cov_scale, o_new_theoretical
    return o_new_write, cov_scale




def _validate_principal_scales(principal_scales: Sequence[float]) -> tuple[float, float]:
    if len(principal_scales) != 2:
        raise ValueError("principal_scales must contain exactly two positive values")
    sx = _as_positive_float("principal_scales[0]", principal_scales[0])
    sy = _as_positive_float("principal_scales[1]", principal_scales[1])
    return sx, sy


def _composited_colocated_gaussian_alpha(
    grid_xy: torch.Tensor,
    *,
    opacity: torch.Tensor,
    principal_scales: tuple[float, float],
    covariance_scale: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    sx, sy = principal_scales
    scale = torch.sqrt(covariance_scale)
    x = grid_xy[..., 0] / (sx * scale)
    y = grid_xy[..., 1] / (sy * scale)
    gaussian = torch.exp(-0.5 * (x * x + y * y))
    per_gaussian_alpha = (opacity * gaussian).clamp(min=0.0, max=1.0)
    return 1.0 - torch.pow(1.0 - per_gaussian_alpha, int(count))


def synthetic_relocation_invariance_audit(
    *,
    old_opacity: float,
    group_total: int,
    principal_scales: Sequence[float] = (1.0, 1.0),
    grid_extent: float = 4.0,
    grid_resolution: int = 129,
    dtype: torch.dtype = torch.float64,
    mean_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Return a JSON-friendly CPU audit of Eq. 9 alpha-response invariance.

    The synthetic scene is a deterministic 2D grid observing colocated Gaussian
    responses.  ``original`` is one Gaussian with ``old_opacity`` and the input
    principal scales.  ``mcmc`` is ``group_total`` colocated relocated Gaussians
    using Eq. 9 opacity and covariance rescale.  ``naive`` is ``group_total``
    colocated clones with the original opacity and covariance.
    """
    old_opacity = _as_positive_float("old_opacity", old_opacity)
    if old_opacity >= 1.0:
        raise ValueError("old_opacity must be less than 1")
    if isinstance(group_total, bool) or not isinstance(group_total, Integral) or group_total < 2:
        raise ValueError("group_total must be an integer >= 2")
    scales = _validate_principal_scales(principal_scales)
    grid_extent = _as_positive_float("grid_extent", grid_extent)
    if isinstance(grid_resolution, bool) or not isinstance(grid_resolution, Integral) or grid_resolution < 3:
        raise ValueError("grid_resolution must be an integer >= 3")
    if grid_resolution % 2 == 0:
        raise ValueError("grid_resolution must be odd so the origin is sampled")
    mean_threshold = _as_positive_float("mean_threshold", mean_threshold)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")

    device = torch.device("cpu")
    old = torch.tensor(float(old_opacity), device=device, dtype=dtype)
    written, cov_scale, theoretical = eq9_relocation_terms(
        old, int(group_total), return_theoretical=True
    )
    coords = torch.linspace(
        -grid_extent, grid_extent, int(grid_resolution), device=device, dtype=dtype
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    one = torch.ones((), device=device, dtype=dtype)

    original = _composited_colocated_gaussian_alpha(
        grid, opacity=old, principal_scales=scales, covariance_scale=one, count=1
    )
    mcmc = _composited_colocated_gaussian_alpha(
        grid,
        opacity=written,
        principal_scales=scales,
        covariance_scale=cov_scale,
        count=int(group_total),
    )
    naive = _composited_colocated_gaussian_alpha(
        grid,
        opacity=old,
        principal_scales=scales,
        covariance_scale=one,
        count=int(group_total),
    )

    mcmc_error = (mcmc - original).abs()
    naive_error = (naive - original).abs()
    mcmc_mean = float(mcmc_error.mean().item())
    naive_mean = float(naive_error.mean().item())
    finite = bool(
        torch.isfinite(original).all()
        and torch.isfinite(mcmc).all()
        and torch.isfinite(naive).all()
        and torch.isfinite(cov_scale)
        and torch.isfinite(written)
        and torch.isfinite(theoretical)
    )
    return {
        "old_opacity": float(old_opacity),
        "group_total": int(group_total),
        "principal_scales": [float(scales[0]), float(scales[1])],
        "grid_extent": float(grid_extent),
        "grid_resolution": int(grid_resolution),
        "dtype": str(dtype).replace("torch.", ""),
        "theoretical_new_opacity": float(theoretical.item()),
        "written_new_opacity": float(written.item()),
        "covariance_scale": float(cov_scale.item()),
        "mcmc_mean_abs_error": mcmc_mean,
        "mcmc_max_abs_error": float(mcmc_error.max().item()),
        "mcmc_p95_abs_error": float(torch.quantile(mcmc_error.flatten(), 0.95).item()),
        "naive_mean_abs_error": naive_mean,
        "naive_max_abs_error": float(naive_error.max().item()),
        "naive_p95_abs_error": float(torch.quantile(naive_error.flatten(), 0.95).item()),
        "mean_error_improvement": float(naive_mean / mcmc_mean) if mcmc_mean > 0.0 else float("inf"),
        "mcmc_mean_le_threshold": bool(mcmc_mean <= mean_threshold),
        "mean_threshold": float(mean_threshold),
        "finite": finite,
    }


def _dtype_eps(tensor: torch.Tensor, eps: float | None = None) -> float:
    if eps is not None:
        return _as_positive_float("eps", eps)
    return float(torch.finfo(tensor.dtype).eps)


def inverse_sigmoid(probability: torch.Tensor, *, eps: float | None = None) -> torch.Tensor:
    """Numerically stable logit using dtype-aware epsilon by default."""
    clamp_eps = _dtype_eps(probability, eps)
    p = probability.clamp(min=clamp_eps, max=1.0 - clamp_eps)
    return torch.log(p / (1.0 - p))


def _iter_optimizers(optimizers: Any | None) -> tuple[torch.optim.Optimizer, ...]:
    if optimizers is None:
        return ()
    if isinstance(optimizers, torch.optim.Optimizer):
        return (optimizers,)
    if isinstance(optimizers, Iterable):
        result = tuple(opt for opt in optimizers if opt is not None)
        if not all(isinstance(opt, torch.optim.Optimizer) for opt in result):
            raise TypeError("optimizers must contain torch.optim.Optimizer instances")
        return result
    raise TypeError("optimizers must be an optimizer, an iterable of optimizers, or None")


def reset_adam_rows_(
    optimizers: Any | None,
    parameters: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    rows: torch.Tensor | Sequence[int],
) -> tuple[int, ...]:
    """Zero Adam exp_avg/exp_avg_sq rows for target parameters only.

    The optimizer state remains attached to the same parameter objects.  Missing
    state is ignored, which allows use before the first optimizer step.
    """
    opts = _iter_optimizers(optimizers)
    if len(opts) == 0:
        return ()
    if isinstance(parameters, Mapping):
        params = tuple(parameters.values())
    else:
        params = tuple(parameters)
    if isinstance(rows, torch.Tensor):
        row_tensor = rows.detach().to(dtype=torch.long)
    else:
        row_tensor = torch.tensor(tuple(int(row) for row in rows), dtype=torch.long)
    if row_tensor.numel() == 0:
        return ()
    reset: set[int] = set(int(row) for row in row_tensor.detach().cpu().tolist())
    for optimizer in opts:
        for param in params:
            if param not in optimizer.state:
                continue
            state = optimizer.state[param]
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    index = row_tensor.to(device=value.device)
                    value.index_fill_(0, index, 0.0)
    return tuple(sorted(reset))


def relocate_dead_gaussians_(
    gaussians: Any,
    *,
    optimizers: Any | None = None,
    opacity_threshold: float = 0.005,
    relocation_count: int | None = None,
    seed: int | None = None,
    selection_policy: str = "seeded",
    max_group_total: int = 50,
    eps: float | None = None,
    min_relocated_opacity: float = 0.005,
) -> RelocationResult:
    """Relocate dead change Gaussians onto live ones using Eq. 9 in-place.

    The number of rows stays fixed.  Dead/live selection and multinomial weights
    use ``sigmoid(change_opacity_raw)``.  All assignments are sampled before any
    parameters are modified, then applied by target group.  If more than
    ``max_group_total - 1`` sources choose the same target, the overflow sources
    are deferred and left untouched for a later call.
    """
    tensors = get_mcmc_tensors(gaussians)
    if isinstance(max_group_total, bool) or not isinstance(max_group_total, Integral):
        raise TypeError("max_group_total must be an integer")
    if max_group_total < 2:
        raise ValueError("max_group_total must be at least 2")
    max_group_total = int(max_group_total)
    clamp_eps = _dtype_eps(tensors["change_opacity_raw"], eps)
    min_relocated_opacity = _as_nonnegative_float(
        "min_relocated_opacity", min_relocated_opacity
    )
    if min_relocated_opacity >= 1.0:
        raise ValueError("min_relocated_opacity must be less than 1")

    dead, live = select_dead_live_indices(
        tensors["change_opacity_raw"],
        opacity_threshold=opacity_threshold,
        relocation_count=relocation_count,
        selection_policy=selection_policy,
        seed=seed,
    )
    assignments = sample_opacity_weighted_assignments(
        tensors["change_opacity_raw"], dead, live, seed=seed
    )
    if len(assignments) == 0:
        return RelocationResult(
            selected_dead_indices=tuple(int(i) for i in dead.detach().cpu().tolist()),
            selected_live_indices=tuple(int(i) for i in live.detach().cpu().tolist()),
            assignments=assignments,
            events=(),
            applied_source_indices=(),
            deferred_source_indices=tuple(int(i) for i in dead.detach().cpu().tolist())
            if live.numel() == 0
            else (),
            reset_target_indices=(),
        )

    events: list[RelocationEvent] = []
    applied_sources: list[int] = []
    deferred_sources: list[int] = []
    reset_targets: list[int] = []
    opacity_before = activated_change_opacity(tensors["change_opacity_raw"])

    with torch.no_grad():
        for target_index, group in group_assignments(assignments).items():
            source_indices = [assignment.source_index for assignment in group]
            apply_sources = source_indices[: max_group_total - 1]
            overflow = source_indices[max_group_total - 1 :]
            if overflow:
                deferred_sources.extend(int(i) for i in overflow)
            if not apply_sources:
                events.append(
                    RelocationEvent(
                        target_index=int(target_index),
                        source_indices=(),
                        group_total=1,
                        old_target_opacity=float(opacity_before[target_index].item()),
                        theoretical_new_group_opacity=None,
                        new_group_opacity=None,
                        covariance_scale=None,
                        applied=False,
                        deferred_source_indices=tuple(int(i) for i in overflow),
                        reason="group_overflow",
                    )
                )
                continue

            idx = torch.tensor(
                [target_index, *apply_sources],
                device=tensors["xyz"].device,
                dtype=torch.long,
            )
            group_total = int(idx.numel())
            old_opacity = opacity_before[target_index]
            new_opacity, cov_scale, theoretical_new_opacity = eq9_relocation_terms(
                old_opacity,
                group_total,
                eps=clamp_eps,
                min_opacity=min_relocated_opacity,
                return_theoretical=True,
            )
            target = torch.tensor(
                [target_index], device=tensors["xyz"].device, dtype=torch.long
            )
            raw_opacity = inverse_sigmoid(new_opacity, eps=clamp_eps).reshape(1, 1)
            scaling_offset = 0.5 * torch.log(cov_scale.clamp_min(clamp_eps))
            target_xyz = tensors["xyz"][target].clone()
            target_dc = tensors["change_dc"][target].clone()
            target_rotation = tensors["rotation_raw"][target].clone()
            new_scaling = tensors["scaling_raw"][target].clone() + scaling_offset

            tensors["xyz"].index_copy_(0, idx, target_xyz.expand(group_total, -1))
            target_dc = target_dc.expand(
                group_total, *tensors["change_dc"].shape[1:]
            )
            tensors["change_dc"].index_copy_(0, idx, target_dc)
            tensors["change_opacity_raw"].index_copy_(
                0, idx, raw_opacity.expand(group_total, -1)
            )
            tensors["scaling_raw"].index_copy_(
                0, idx, new_scaling.expand(group_total, -1)
            )
            tensors["rotation_raw"].index_copy_(
                0, idx, target_rotation.expand(group_total, -1)
            )

            applied_sources.extend(int(i) for i in apply_sources)
            reset_targets.append(int(target_index))
            events.append(
                RelocationEvent(
                    target_index=int(target_index),
                    source_indices=tuple(int(i) for i in apply_sources),
                    group_total=group_total,
                    old_target_opacity=float(old_opacity.item()),
                    theoretical_new_group_opacity=float(theoretical_new_opacity.item()),
                    new_group_opacity=float(new_opacity.item()),
                    covariance_scale=float(cov_scale.item()),
                    applied=True,
                    deferred_source_indices=tuple(int(i) for i in overflow),
                )
            )

    if reset_targets:
        row_tensor = torch.tensor(
            sorted(set(reset_targets)), device=tensors["xyz"].device, dtype=torch.long
        )
        reset_adam_rows_(optimizers, tensors, row_tensor)

    return RelocationResult(
        selected_dead_indices=tuple(int(i) for i in dead.detach().cpu().tolist()),
        selected_live_indices=tuple(int(i) for i in live.detach().cpu().tolist()),
        assignments=assignments,
        events=tuple(events),
        applied_source_indices=tuple(applied_sources),
        deferred_source_indices=tuple(deferred_sources),
        reset_target_indices=tuple(sorted(set(reset_targets))),
    )


__all__ = [
    "GAUSSIAN_FIELDS",
    "FIELD_ALIASES",
    "MCMCNoiseStats",
    "RelocationAssignment",
    "RelocationEvent",
    "RelocationResult",
    "activated_change_opacity",
    "apply_sgld_xyz_noise_",
    "covariance_from_scaling_rotation",
    "dynamics_audit",
    "eq9_relocation_terms",
    "_eq9_new_opacity",
    "_eq9_covariance_scale",
    "event_to_dict",
    "get_mcmc_tensors",
    "group_assignments",
    "inverse_sigmoid",
    "normalize_quaternion",
    "relocate_dead_gaussians_",
    "reset_adam_rows_",
    "rotation_matrix_from_quaternion",
    "sample_opacity_weighted_assignments",
    "select_dead_live_indices",
    "sgld_xyz_noise",
    "synthetic_relocation_invariance_audit",
]
