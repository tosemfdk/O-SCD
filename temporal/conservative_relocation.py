"""Conservative fixed-capacity relocation for temporal change Gaussians.

This module deliberately does *not* implement the live-target Eq. 9 move from
3DGS-MCMC.  It supports a separate ablation whose source slots are selected by
causal observation/support metadata and whose destinations are triangulated from
unexplained multi-view change cues.  The move is transactional so the runner can
restore every touched row when replay energy or cue coverage regresses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ResidualCandidates:
    """Triangulated residual destinations and their causal support."""

    xyz: torch.Tensor
    scores: torch.Tensor
    support_views: torch.Tensor
    ray_distance: torch.Tensor

    @classmethod
    def empty(cls, *, device: torch.device, dtype: torch.dtype) -> "ResidualCandidates":
        return cls(
            xyz=torch.empty((0, 3), device=device, dtype=dtype),
            scores=torch.empty((0,), device=device, dtype=dtype),
            support_views=torch.empty((0,), device=device, dtype=torch.long),
            ray_distance=torch.empty((0,), device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class ConservativeAcceptance:
    accepted: bool
    reason: str
    relative_energy_delta: float
    coverage_gain: float
    negative_mass_delta: float = 0.0


@dataclass(frozen=True)
class SourceDeactivationDecision:
    safe: bool
    reason: str
    relative_energy_delta: float
    negative_mass_delta: float


@dataclass
class RowLocalAdamState:
    """Adam moments owned only by the rows tentatively adapted in one proposal."""

    steps: dict[str, int]
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]


_ROW_PARAMETER_NAMES = (
    "current_xyz",
    "current_features_dc",
    "current_raw_change_opacity",
    "current_scaling",
    "current_rotation",
)
_ROW_METADATA_NAMES = (
    "cue_support_mask",
    "slot_observation_count",
    "slot_cue_support_count",
    "carryover_protected_mask",
    "removal_protected_mask",
    "tentative_mask",
    "slot_relocation_count",
    "last_relocation_step",
)


@dataclass(frozen=True)
class ConservativeRowSnapshot:
    """Exact copy of every mutable row touched by one relocation proposal."""

    indices: torch.Tensor
    values: Mapping[str, torch.Tensor]

    @classmethod
    def capture(cls, state: Any, indices: torch.Tensor | Sequence[int]) -> "ConservativeRowSnapshot":
        ids = _coerce_indices(indices, capacity=int(state.capacity), device=state.current_xyz.device)
        values: dict[str, torch.Tensor] = {}
        for name in (*_ROW_PARAMETER_NAMES, *_ROW_METADATA_NAMES):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                values[name] = value.detach().index_select(0, ids).clone()
        return cls(indices=ids.detach().clone(), values=values)

    def matches(self, state: Any) -> bool:
        return all(
            torch.equal(
                getattr(state, name).detach().index_select(0, self.indices.to(getattr(state, name).device)),
                value.to(getattr(state, name).device),
            )
            for name, value in self.values.items()
        )


def _validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_positive_int(name: str, value: int) -> int:
    value = _validate_nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and inside (0, 1)")
    return value


def _coerce_indices(indices: torch.Tensor | Sequence[int], *, capacity: int, device: torch.device) -> torch.Tensor:
    ids = torch.as_tensor(indices, device=device, dtype=torch.long).flatten()
    if ids.numel() and bool(((ids < 0) | (ids >= capacity)).any()):
        raise IndexError("row index is outside fixed Gaussian capacity")
    if ids.unique().numel() != ids.numel():
        raise ValueError("row indices must be unique")
    return ids


def _validate_slot_vectors(*vectors: torch.Tensor) -> int:
    if not vectors:
        raise ValueError("at least one slot vector is required")
    n = int(vectors[0].numel())
    device = vectors[0].device
    for vector in vectors:
        if vector.ndim != 1 or vector.numel() != n:
            raise ValueError("slot evidence tensors must be rank-1 and have equal length")
        if vector.device != device:
            raise ValueError("slot evidence tensors must share a device")
    return n


def select_conservative_sources(
    *,
    observation_count: torch.Tensor,
    cue_support_count: torch.Tensor,
    protected_mask: torch.Tensor,
    tentative_mask: torch.Tensor,
    min_observations: int,
    max_sources: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Choose sufficiently observed, never-supported, unprotected source slots.

    Change opacity is intentionally absent from this interface.  Observation
    count ranks eligible rows only when the caller applies a cap; returned row
    ids are sorted for stable lineage logs.
    """

    del seed  # Selection is deterministic; the seed is retained in the contract.
    _validate_slot_vectors(observation_count, cue_support_count, protected_mask, tentative_mask)
    min_observations = _validate_positive_int("min_observations", min_observations)
    max_sources = _validate_nonnegative_int("max_sources", max_sources)
    eligible = (
        (observation_count >= min_observations)
        & (cue_support_count == 0)
        & ~protected_mask.bool()
        & ~tentative_mask.bool()
    )
    ids = torch.nonzero(eligible, as_tuple=False).flatten()
    if max_sources == 0 or ids.numel() == 0:
        return ids[:0]
    if ids.numel() > max_sources:
        # Reclaim the rows with the strongest evidence of being visible but
        # unsupported, then sort ids to keep serialized events deterministic.
        order = torch.argsort(observation_count[ids], descending=True, stable=True)
        ids = ids[order[:max_sources]]
    return ids[torch.argsort(ids, stable=True)]


def _camera_intrinsics(view: Any, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    width = int(view.image_width)
    height = int(view.image_height)
    if width <= 0 or height <= 0:
        raise ValueError("camera image dimensions must be positive")
    fx = width / (2.0 * math.tan(float(view.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(view.FoVy) * 0.5))
    return (
        torch.as_tensor(fx, device=device, dtype=dtype),
        torch.as_tensor(fy, device=device, dtype=dtype),
        torch.as_tensor(width / 2.0, device=device, dtype=dtype),
        torch.as_tensor(height / 2.0, device=device, dtype=dtype),
    )


def _top_residual_rays(
    view: Any,
    residual: torch.Tensor,
    *,
    topk: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.ndim == 3 and residual.shape[0] == 1:
        residual = residual[0]
    if residual.ndim != 2:
        raise ValueError("residual map must have shape [H, W] or [1, H, W]")
    if tuple(residual.shape) != (int(view.image_height), int(view.image_width)):
        raise ValueError("residual map and camera image dimensions differ")
    flat = residual.detach().float().flatten()
    k = min(_validate_positive_int("per_view_topk", topk), int(flat.numel()))
    values, indices = torch.topk(flat, k=k, largest=True, sorted=True)
    keep = torch.isfinite(values) & (values >= float(threshold))
    values = values[keep]
    indices = indices[keep]
    device = residual.device
    dtype = residual.dtype
    if indices.numel() == 0:
        return (
            torch.empty((0, 3), device=device, dtype=dtype),
            torch.empty((0, 3), device=device, dtype=dtype),
            values.to(dtype=dtype),
        )
    width = int(view.image_width)
    y = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
    x = (indices % width).to(dtype=dtype)
    fx, fy, cx, cy = _camera_intrinsics(view, device=device, dtype=dtype)
    camera_dirs = torch.stack(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)), dim=-1)
    world_view = view.world_view_transform.to(device=device, dtype=dtype)
    camera_to_world = torch.linalg.inv(world_view)
    directions = camera_dirs @ camera_to_world[:3, :3]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    if isinstance(getattr(view, "camera_center", None), torch.Tensor):
        origin = view.camera_center.to(device=device, dtype=dtype)
    else:
        origin = camera_to_world[3, :3]
    origins = origin.reshape(1, 3).expand(directions.shape[0], -1)
    return origins, directions, values.to(dtype=dtype)


def _pairwise_ray_midpoints(
    origins_a: torch.Tensor,
    directions_a: torch.Tensor,
    scores_a: torch.Tensor,
    origins_b: torch.Tensor,
    directions_b: torch.Tensor,
    scores_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if directions_a.numel() == 0 or directions_b.numel() == 0:
        empty = directions_a.new_empty((0,))
        return directions_a.new_empty((0, 3)), empty, empty
    # Each view has one camera origin, but keeping the vectorized form makes the
    # routine valid for generic ray bundles as well.
    d1 = directions_a[:, None, :]
    d2 = directions_b[None, :, :]
    o1 = origins_a[:, None, :]
    o2 = origins_b[None, :, :]
    w0 = o1 - o2
    b = (d1 * d2).sum(dim=-1)
    d = (d1 * w0).sum(dim=-1)
    e = (d2 * w0).sum(dim=-1)
    denom = 1.0 - b.square()
    eps = torch.finfo(directions_a.dtype).eps * 16.0
    valid = denom.abs() > eps
    s = (b * e - d) / denom.clamp_min(eps)
    t = (e - b * d) / denom.clamp_min(eps)
    valid &= (s > 0.0) & (t > 0.0)
    p1 = o1 + s[..., None] * d1
    p2 = o2 + t[..., None] * d2
    midpoint = 0.5 * (p1 + p2)
    distance = (p1 - p2).norm(dim=-1)
    pair_score = torch.sqrt(scores_a[:, None].clamp_min(0.0) * scores_b[None, :].clamp_min(0.0))
    return midpoint[valid], distance[valid], pair_score[valid]


def _project_and_sample(view: Any, xyz: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.ndim == 3:
        residual = residual[0]
    dtype = xyz.dtype
    device = xyz.device
    world_view = view.world_view_transform.to(device=device, dtype=dtype)
    ones = torch.ones((xyz.shape[0], 1), device=device, dtype=dtype)
    camera = torch.cat((xyz, ones), dim=-1) @ world_view
    z = camera[:, 2]
    fx, fy, cx, cy = _camera_intrinsics(view, device=device, dtype=dtype)
    safe_z = z.clamp_min(torch.finfo(dtype).eps)
    x = fx * camera[:, 0] / safe_z + cx
    y = fy * camera[:, 1] / safe_z + cy
    xi = x.round().long()
    yi = y.round().long()
    valid = (
        (z > 0.0)
        & (xi >= 0)
        & (xi < int(view.image_width))
        & (yi >= 0)
        & (yi < int(view.image_height))
    )
    sampled = torch.zeros(xyz.shape[0], device=device, dtype=dtype)
    if bool(valid.any()):
        sampled[valid] = residual.to(device=device, dtype=dtype)[yi[valid], xi[valid]]
    return sampled, valid


def residual_candidates_from_views(
    views: Sequence[Any],
    residual_maps: Sequence[torch.Tensor],
    *,
    per_view_topk: int,
    residual_threshold: float,
    min_support_views: int,
    max_ray_distance: float,
    scene_bounds: tuple[torch.Tensor, torch.Tensor],
    max_candidates: int,
) -> ResidualCandidates:
    """Triangulate and score 3D points supported by unexplained cue pixels."""

    if len(views) != len(residual_maps):
        raise ValueError("views and residual_maps must have equal length")
    min_support_views = _validate_positive_int("min_support_views", min_support_views)
    max_candidates = _validate_positive_int("max_candidates", max_candidates)
    if len(residual_maps) == 0:
        return ResidualCandidates.empty(device=torch.device("cpu"), dtype=torch.float32)
    device = residual_maps[0].device
    dtype = residual_maps[0].dtype
    if len(views) < min_support_views or len(views) < 2:
        return ResidualCandidates.empty(device=device, dtype=dtype)
    if not math.isfinite(float(max_ray_distance)) or float(max_ray_distance) <= 0.0:
        raise ValueError("max_ray_distance must be finite and positive")

    bundles = [
        _top_residual_rays(view, residual, topk=per_view_topk, threshold=residual_threshold)
        for view, residual in zip(views, residual_maps)
    ]
    xyz_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    pair_score_parts: list[torch.Tensor] = []
    for first in range(len(bundles)):
        for second in range(first + 1, len(bundles)):
            xyz, distance, pair_score = _pairwise_ray_midpoints(*bundles[first], *bundles[second])
            keep = torch.isfinite(distance) & (distance <= float(max_ray_distance))
            if bool(keep.any()):
                xyz_parts.append(xyz[keep])
                distance_parts.append(distance[keep])
                pair_score_parts.append(pair_score[keep])
    if not xyz_parts:
        return ResidualCandidates.empty(device=device, dtype=dtype)

    xyz = torch.cat(xyz_parts, dim=0)
    distance = torch.cat(distance_parts, dim=0)
    pair_score = torch.cat(pair_score_parts, dim=0)
    lower, upper = (bound.to(device=device, dtype=dtype).reshape(1, 3) for bound in scene_bounds)
    in_bounds = ((xyz >= lower) & (xyz <= upper)).all(dim=-1)
    finite = torch.isfinite(xyz).all(dim=-1) & torch.isfinite(pair_score)
    keep = in_bounds & finite
    xyz, distance, pair_score = xyz[keep], distance[keep], pair_score[keep]
    if xyz.numel() == 0:
        return ResidualCandidates.empty(device=device, dtype=dtype)

    # Bound the expensive all-view reprojection while retaining the strongest,
    # geometrically closest ray intersections.
    pre_cap = min(int(xyz.shape[0]), max_candidates * 16)
    rank_score = pair_score / (1.0 + distance / float(max_ray_distance))
    pre_order = torch.argsort(rank_score, descending=True, stable=True)[:pre_cap]
    xyz, distance = xyz[pre_order], distance[pre_order]

    sampled_per_view = []
    valid_per_view = []
    for view, residual in zip(views, residual_maps):
        sampled, valid = _project_and_sample(view, xyz, residual)
        sampled_per_view.append(sampled)
        valid_per_view.append(valid)
    samples = torch.stack(sampled_per_view, dim=1)
    valid = torch.stack(valid_per_view, dim=1)
    support = valid & (samples >= float(residual_threshold))
    support_views = support.sum(dim=1)
    scores = (samples * support).sum(dim=1) / support_views.clamp_min(1).to(dtype)
    keep = (support_views >= min_support_views) & torch.isfinite(scores)
    xyz, distance, scores, support_views = xyz[keep], distance[keep], scores[keep], support_views[keep]
    if xyz.numel() == 0:
        return ResidualCandidates.empty(device=device, dtype=dtype)
    order = torch.argsort(scores, descending=True, stable=True)[:max_candidates]
    return ResidualCandidates(
        xyz=xyz[order],
        scores=scores[order],
        support_views=support_views[order],
        ray_distance=distance[order],
    )


def select_residual_destinations(
    candidates: ResidualCandidates | SimpleNamespace,
    *,
    count: int,
    min_support_views: int,
    seed: int | None,
) -> torch.Tensor:
    """Seeded weighted selection without replacement from multi-view candidates."""

    count = _validate_nonnegative_int("count", count)
    min_support_views = _validate_positive_int("min_support_views", min_support_views)
    valid = torch.nonzero(candidates.support_views >= min_support_views, as_tuple=False).flatten()
    if count == 0 or valid.numel() == 0:
        return valid[:0]
    count = min(count, int(valid.numel()))
    weights = candidates.scores[valid].detach().float().clamp_min(0.0)
    if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0.0:
        order = torch.argsort(valid, stable=True)[:count]
        return valid[order]
    generator = None
    if seed is not None:
        generator = torch.Generator(device=weights.device)
        generator.manual_seed(int(seed))
    chosen = torch.multinomial(weights, count, replacement=False, generator=generator)
    return valid[chosen]


def restore_rows_(state: Any, snapshot: ConservativeRowSnapshot) -> None:
    """Restore a rejected proposal without replacing any Parameter object."""

    with torch.no_grad():
        for name, value in snapshot.values.items():
            target = getattr(state, name)
            ids = snapshot.indices.to(target.device)
            target.index_copy_(0, ids, value.to(device=target.device, dtype=target.dtype))


def transport_sources_(
    state: Any,
    source_indices: torch.Tensor | Sequence[int],
    destination_xyz: torch.Tensor,
    *,
    tentative_opacity: float,
    global_step: int,
) -> None:
    """Move source rows to residual destinations with fresh tentative attributes."""

    ids = _coerce_indices(source_indices, capacity=int(state.capacity), device=state.current_xyz.device)
    destination_xyz = destination_xyz.to(device=state.current_xyz.device, dtype=state.current_xyz.dtype)
    if destination_xyz.shape != (int(ids.numel()), 3):
        raise ValueError("destination_xyz must have shape [num_sources, 3]")
    tentative_opacity = _validate_probability("tentative_opacity", tentative_opacity)
    probability = torch.as_tensor(
        tentative_opacity,
        device=state.current_raw_change_opacity.device,
        dtype=state.current_raw_change_opacity.dtype,
    )
    raw_opacity = torch.log(probability / (1.0 - probability))
    with torch.no_grad():
        state.current_xyz.index_copy_(0, ids, destination_xyz)
        state.current_features_dc.index_fill_(0, ids, 0.0)
        state.current_raw_change_opacity.index_copy_(
            0,
            ids,
            raw_opacity.reshape(1, 1).expand(ids.numel(), 1),
        )
        for name in ("cue_support_mask", "carryover_protected_mask"):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                value[ids] = False
        for name in ("slot_observation_count", "slot_cue_support_count"):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        if isinstance(getattr(state, "tentative_mask", None), torch.Tensor):
            state.tentative_mask[ids] = True
        if isinstance(getattr(state, "slot_relocation_count", None), torch.Tensor):
            state.slot_relocation_count[ids] += 1
        if isinstance(getattr(state, "last_relocation_step", None), torch.Tensor):
            state.last_relocation_step[ids] = int(global_step)


def conservative_acceptance(
    *,
    pre_energy: float,
    post_energy: float,
    pre_coverage: float,
    post_coverage: float,
    max_relative_energy_increase: float,
    min_coverage_gain: float,
    pre_negative_mass: float | None = None,
    post_negative_mass: float | None = None,
    max_negative_mass_increase: float = 0.0,
) -> ConservativeAcceptance:
    """Accept only finite proposals satisfying replay energy and coverage gates."""

    values = [pre_energy, post_energy, pre_coverage, post_coverage]
    if pre_negative_mass is not None:
        values.append(pre_negative_mass)
    if post_negative_mass is not None:
        values.append(post_negative_mass)
    if not all(math.isfinite(float(value)) for value in values):
        return ConservativeAcceptance(False, "nonfinite_audit", float("inf"), float("-inf"), float("inf"))
    denom = max(abs(float(pre_energy)), 1e-12)
    relative = (float(post_energy) - float(pre_energy)) / denom
    gain = float(post_coverage) - float(pre_coverage)
    negative_delta = 0.0
    if (pre_negative_mass is None) != (post_negative_mass is None):
        raise ValueError("pre/post negative mass must be provided together")
    if pre_negative_mass is not None and post_negative_mass is not None:
        negative_delta = float(post_negative_mass) - float(pre_negative_mass)
    if relative > float(max_relative_energy_increase):
        return ConservativeAcceptance(False, "energy_increase", relative, gain, negative_delta)
    if gain < float(min_coverage_gain):
        return ConservativeAcceptance(False, "coverage_drop", relative, gain, negative_delta)
    if negative_delta > float(max_negative_mass_increase):
        return ConservativeAcceptance(False, "negative_mass_increase", relative, gain, negative_delta)
    return ConservativeAcceptance(True, "accepted", relative, gain, negative_delta)


def source_deactivation_safety(
    *,
    pre_energy: float,
    deactivated_energy: float,
    pre_negative_mass: float,
    deactivated_negative_mass: float,
    max_relative_energy_increase: float,
    max_negative_mass_increase: float,
) -> SourceDeactivationDecision:
    """Protect rows whose removal hurts SSF or exposes change on negative pixels."""

    values = (pre_energy, deactivated_energy, pre_negative_mass, deactivated_negative_mass)
    if not all(math.isfinite(float(value)) for value in values):
        return SourceDeactivationDecision(False, "nonfinite_deactivation", float("inf"), float("inf"))
    relative = (float(deactivated_energy) - float(pre_energy)) / max(abs(float(pre_energy)), 1e-12)
    negative_delta = float(deactivated_negative_mass) - float(pre_negative_mass)
    if negative_delta > float(max_negative_mass_increase):
        return SourceDeactivationDecision(False, "source_occlusion_responsibility", relative, negative_delta)
    if relative > float(max_relative_energy_increase):
        return SourceDeactivationDecision(False, "source_energy_responsibility", relative, negative_delta)
    return SourceDeactivationDecision(True, "source_removal_safe", relative, negative_delta)


def initialize_row_local_adam(
    parameters: Mapping[str, torch.Tensor],
    indices: torch.Tensor | Sequence[int],
) -> RowLocalAdamState:
    if not parameters:
        raise ValueError("parameters cannot be empty")
    first = next(iter(parameters.values()))
    ids = _coerce_indices(indices, capacity=int(first.shape[0]), device=first.device)
    exp_avg = {
        name: torch.zeros_like(parameter.detach().index_select(0, ids))
        for name, parameter in parameters.items()
    }
    return RowLocalAdamState(
        steps={name: 0 for name in parameters},
        exp_avg=exp_avg,
        exp_avg_sq={name: torch.zeros_like(value) for name, value in exp_avg.items()},
    )


def row_local_adam_step_(
    parameters: Mapping[str, torch.Tensor],
    indices: torch.Tensor | Sequence[int],
    learning_rates: Mapping[str, float],
    state: RowLocalAdamState,
    *,
    allowed_names: Sequence[str] | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Apply one Adam step to selected rows without touching any other row."""

    if not parameters:
        raise ValueError("parameters cannot be empty")
    first = next(iter(parameters.values()))
    ids = _coerce_indices(indices, capacity=int(first.shape[0]), device=first.device)
    allowed = set(parameters) if allowed_names is None else set(allowed_names)
    unknown = allowed - set(parameters)
    if unknown:
        raise KeyError(f"unknown row-local parameter names: {sorted(unknown)}")
    grad_norms: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in parameters.items():
            if name not in allowed or parameter.grad is None or ids.numel() == 0:
                continue
            if name not in learning_rates:
                raise KeyError(f"missing learning rate for {name!r}")
            grad = parameter.grad.detach().index_select(0, ids)
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError(f"non-finite row-local gradient for {name}")
            exp_avg = state.exp_avg[name]
            exp_avg_sq = state.exp_avg_sq[name]
            state.steps[name] += 1
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias1 = 1.0 - beta1 ** state.steps[name]
            bias2 = 1.0 - beta2 ** state.steps[name]
            update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + float(eps))
            rows = parameter.detach().index_select(0, ids)
            parameter.index_copy_(0, ids, rows - float(learning_rates[name]) * update)
            grad_norms[name] = float(grad.norm().item())
    return grad_norms


__all__ = [
    "ConservativeAcceptance",
    "ConservativeRowSnapshot",
    "RowLocalAdamState",
    "ResidualCandidates",
    "SourceDeactivationDecision",
    "conservative_acceptance",
    "initialize_row_local_adam",
    "residual_candidates_from_views",
    "restore_rows_",
    "row_local_adam_step_",
    "select_conservative_sources",
    "select_residual_destinations",
    "source_deactivation_safety",
    "transport_sources_",
]
