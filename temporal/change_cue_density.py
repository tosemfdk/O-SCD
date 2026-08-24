"""Causal soft change-cue density control for a mutable ``R_change`` bank.

FastGS counts binary high-error pixels touched by each Gaussian.  This module
keeps its multi-view aggregation and gradient/importance intersection, but
replaces the integer error map with the existing differentiable alpha-T VJP.
Consequently a cue value in ``[0, 1]`` contributes fractionally instead of
being thresholded or truncated by the FastGS integer metric-map interface.

The helpers in this file operate only on a mutable change bank.  They do not
resize a temporal fixed-topology sidecar or the immutable reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from .change_evidence import alpha_t_evidence_vjp


@dataclass(frozen=True)
class MultiViewChangeScore:
    """Soft alpha-T support aggregated over one causal view sample."""

    view_indices: tuple[int, ...]
    positive_mass: torch.Tensor
    negative_mass: torch.Tensor
    total_mass: torch.Tensor
    change_ratio: torch.Tensor
    importance_score: torch.Tensor


@dataclass(frozen=True)
class DensityCandidateMasks:
    """FastGS gradient qualifiers intersected with cue importance."""

    importance: torch.Tensor
    clone_gradient: torch.Tensor
    split_gradient: torch.Tensor
    clone: torch.Tensor
    split: torch.Tensor


@dataclass(frozen=True)
class DensificationResult:
    """Counts and exact parent masks for one topology mutation."""

    initial_count: int
    final_count: int
    clone_count: int
    split_count: int
    masks: DensityCandidateMasks


def causal_random_view_indices(
    current_timestamp: int,
    k: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Sample current plus up to ``k-1`` already processed timestamps.

    Including the current timestamp makes ``K=1`` exactly the current-view
    score.  Older timestamps are sampled without replacement using a local RNG,
    so the training schedule and global Python RNG are unaffected.
    """
    if isinstance(current_timestamp, bool) or int(current_timestamp) < 0:
        raise ValueError("current_timestamp must be a nonnegative integer")
    if isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    current_timestamp = int(current_timestamp)
    k = min(int(k), current_timestamp + 1)
    if k == 1:
        return (current_timestamp,)
    rng = random.Random(int(seed) + 1_000_003 * current_timestamp + 97 * k)
    past = rng.sample(range(current_timestamp), k - 1)
    return tuple(sorted((*past, current_timestamp)))


def aggregate_soft_change_masses(
    positive_masses: Sequence[torch.Tensor],
    negative_masses: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate per-view alpha-T masses using the FastGS ``sum / K`` form."""
    if not positive_masses or len(positive_masses) != len(negative_masses):
        raise ValueError("positive and negative mass sequences must be nonempty and aligned")
    first = positive_masses[0]
    if not isinstance(first, torch.Tensor) or not torch.is_floating_point(first):
        raise TypeError("evidence masses must be floating-point tensors")
    for positive, negative in zip(positive_masses, negative_masses):
        if not isinstance(positive, torch.Tensor) or not isinstance(negative, torch.Tensor):
            raise TypeError("evidence masses must be tensors")
        if (
            positive.shape != first.shape
            or negative.shape != first.shape
            or positive.device != first.device
            or negative.device != first.device
            or positive.dtype != first.dtype
            or negative.dtype != first.dtype
        ):
            raise ValueError("all evidence tensors must share shape, device, and dtype")
        if not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
            raise ValueError("evidence masses must be finite")
        if bool((positive < 0).any() or (negative < 0).any()):
            raise ValueError("evidence masses must be nonnegative")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    positive = torch.stack(tuple(positive_masses), dim=0).sum(dim=0)
    negative = torch.stack(tuple(negative_masses), dim=0).sum(dim=0)
    total = positive + negative
    ratio = positive / (total + float(eps))
    importance = positive / float(len(positive_masses))
    return positive, negative, total, ratio, importance


def compute_soft_multiview_change_score(
    views: Sequence[Any],
    view_indices: Sequence[int],
    change_bank: Any,
    pipe: SimpleNamespace,
    background: torch.Tensor,
    *,
    cue_scale: float = 1.0,
    eps: float = 1e-8,
) -> MultiViewChangeScore:
    """Probe current mutable-bank footprints against raw ref--inf cues."""
    if not view_indices:
        raise ValueError("view_indices must be nonempty")
    if not math.isfinite(float(cue_scale)) or float(cue_scale) <= 0.0:
        raise ValueError("cue_scale must be finite and positive")
    positive: list[torch.Tensor] = []
    negative: list[torch.Tensor] = []
    normalized_indices = tuple(int(index) for index in view_indices)
    for index in normalized_indices:
        if index < 0 or index >= len(views):
            raise IndexError("view index is outside the already processed causal bank")
        view = views[index]
        cue = getattr(view, "candidate_map", None)
        if not isinstance(cue, torch.Tensor):
            raise AttributeError("each sampled view must expose candidate_map")
        cue = torch.clamp(cue / float(cue_scale), 0.0, 1.0)
        evidence = alpha_t_evidence_vjp(
            view,
            change_bank,
            pipe,
            background,
            cue,
            count_mode="raw",
            min_evidence_mass=0.0,
        )
        positive.append(evidence.e_plus)
        negative.append(evidence.e_minus)
    positive_sum, negative_sum, total, ratio, importance = aggregate_soft_change_masses(
        positive,
        negative,
        eps=eps,
    )
    return MultiViewChangeScore(
        view_indices=normalized_indices,
        positive_mass=positive_sum,
        negative_mass=negative_sum,
        total_mass=total,
        change_ratio=ratio,
        importance_score=importance,
    )


def fastgs_change_candidate_masks(
    change_bank: Any,
    importance_score: torch.Tensor,
    *,
    scene_extent: float,
    importance_threshold: float = 5.0,
    grad_threshold: float = 2e-4,
    grad_abs_threshold: float = 1.2e-3,
    dense_fraction: float = 1e-3,
) -> DensityCandidateMasks:
    """Reproduce FastGS clone/split qualifiers with a soft cue score."""
    n = int(change_bank.get_xyz.shape[0])
    if importance_score.shape != (n,):
        raise ValueError(f"importance_score must have shape ({n},)")
    if not bool(torch.isfinite(importance_score).all()):
        raise ValueError("importance_score must be finite")
    for name, value, allow_zero in (
        ("scene_extent", scene_extent, False),
        ("importance_threshold", importance_threshold, True),
        ("grad_threshold", grad_threshold, True),
        ("grad_abs_threshold", grad_abs_threshold, True),
        ("dense_fraction", dense_fraction, False),
    ):
        if not math.isfinite(float(value)) or (float(value) < 0.0 if allow_zero else float(value) <= 0.0):
            raise ValueError(f"{name} has an invalid value")
    for name in ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom"):
        value = getattr(change_bank, name, None)
        if not isinstance(value, torch.Tensor) or value.shape != (n, 1):
            raise ValueError(f"change_bank.{name} must have shape ({n}, 1)")
    denominator = change_bank.denom.clamp_min(1.0)
    gradient = torch.nan_to_num(change_bank.xyz_gradient_accum / denominator)
    gradient_abs = torch.nan_to_num(change_bank.xyz_gradient_accum_abs / denominator)
    clone_gradient = torch.linalg.vector_norm(gradient, dim=-1) >= float(grad_threshold)
    split_gradient = torch.linalg.vector_norm(gradient_abs, dim=-1) >= float(grad_abs_threshold)
    small = change_bank.get_scaling.max(dim=1).values <= float(dense_fraction) * float(scene_extent)
    important = importance_score > float(importance_threshold)
    clone = important & clone_gradient & small
    split = important & split_gradient & ~small
    return DensityCandidateMasks(
        importance=important,
        clone_gradient=clone_gradient & small,
        split_gradient=split_gradient & ~small,
        clone=clone,
        split=split,
    )


def topology_integrity(change_bank: Any) -> dict[str, Any]:
    """Audit Gaussian tensors, optimizer moments, and density accumulators."""
    n = int(change_bank.get_xyz.shape[0])
    tensor_lengths = {
        name: int(getattr(change_bank, name).shape[0])
        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
            "xyz_gradient_accum",
            "xyz_gradient_accum_abs",
            "denom",
            "max_radii2D",
        )
    }
    optimizer_lengths: dict[str, int] = {}
    optimizers = [getattr(change_bank, "optimizer", None), getattr(change_bank, "shoptimizer", None)]
    for optimizer in (value for value in optimizers if value is not None):
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            if parameter.shape[0] != n:
                optimizer_lengths[f"{group.get('name', 'unnamed')}.parameter"] = int(parameter.shape[0])
            for state_name, state_value in optimizer.state.get(parameter, {}).items():
                if isinstance(state_value, torch.Tensor) and state_value.ndim > 0:
                    optimizer_lengths[f"{group.get('name', 'unnamed')}.{state_name}"] = int(state_value.shape[0])
    all_lengths = tuple(tensor_lengths.values()) + tuple(optimizer_lengths.values())
    passed = all(value == n for value in all_lengths)
    return {
        "passed": bool(passed),
        "gaussian_count": n,
        "tensor_lengths": tensor_lengths,
        "optimizer_lengths": optimizer_lengths,
    }


@torch.no_grad()
def apply_fastgs_change_densification(
    change_bank: Any,
    radii: torch.Tensor,
    importance_score: torch.Tensor,
    *,
    scene_extent: float,
    importance_threshold: float = 5.0,
    grad_threshold: float = 2e-4,
    grad_abs_threshold: float = 1.2e-3,
    dense_fraction: float = 1e-3,
) -> DensificationResult:
    """Apply clone/split only; cue-based pruning is deliberately absent."""
    initial_count = int(change_bank.get_xyz.shape[0])
    if radii.shape != (initial_count,):
        raise ValueError(f"radii must have shape ({initial_count},)")
    masks = fastgs_change_candidate_masks(
        change_bank,
        importance_score,
        scene_extent=scene_extent,
        importance_threshold=importance_threshold,
        grad_threshold=grad_threshold,
        grad_abs_threshold=grad_abs_threshold,
        dense_fraction=dense_fraction,
    )
    clone_count = int(masks.clone.sum().item())
    split_count = int(masks.split.sum().item())
    change_bank.tmp_radii = radii.detach().clone()
    change_bank.densify_and_clone_fastgs(masks.importance, masks.clone_gradient)
    change_bank.densify_and_split_fastgs(masks.importance, masks.split_gradient)
    change_bank.tmp_radii = None
    final_count = int(change_bank.get_xyz.shape[0])
    expected = initial_count + clone_count + split_count
    if final_count != expected:
        raise RuntimeError(
            f"unexpected FastGS topology size {final_count}; expected {expected}"
        )
    audit = topology_integrity(change_bank)
    if not audit["passed"]:
        raise RuntimeError(f"topology integrity failure after densification: {audit}")
    return DensificationResult(
        initial_count=initial_count,
        final_count=final_count,
        clone_count=clone_count,
        split_count=split_count,
        masks=masks,
    )


@torch.no_grad()
def apply_oscd_gradient_only_densification(
    change_bank: Any,
    radii: torch.Tensor,
    *,
    scene_extent: float,
    grad_threshold: float = 1e-3,
) -> DensificationResult:
    """Apply the original online O-SCD clone/split policy without pruning.

    ``oscd.py`` calls vanilla clone followed by split at local update four.  The
    local port predates FastGS absolute-gradient storage, so this wrapper repairs
    that diagnostic tensor after the topology edit without changing selection.
    """
    initial_count = int(change_bank.get_xyz.shape[0])
    if radii.shape != (initial_count,):
        raise ValueError(f"radii must have shape ({initial_count},)")
    if not math.isfinite(float(scene_extent)) or float(scene_extent) <= 0.0:
        raise ValueError("scene_extent must be finite and positive")
    if not math.isfinite(float(grad_threshold)) or float(grad_threshold) < 0.0:
        raise ValueError("grad_threshold must be finite and nonnegative")
    denominator = change_bank.denom.clamp_min(1.0)
    gradients = torch.nan_to_num(change_bank.xyz_gradient_accum / denominator)
    small = change_bank.get_scaling.max(dim=1).values <= float(change_bank.percent_dense) * float(scene_extent)
    gradient_ok = torch.linalg.vector_norm(gradients, dim=-1) >= float(grad_threshold)
    clone = gradient_ok & small
    split = gradient_ok & ~small
    clone_count = int(clone.sum().item())
    split_count = int(split.sum().item())
    masks = DensityCandidateMasks(
        importance=torch.ones(initial_count, device=gradients.device, dtype=torch.bool),
        clone_gradient=clone,
        split_gradient=split,
        clone=clone,
        split=split,
    )
    change_bank.tmp_radii = radii.detach().clone()
    change_bank.densify_and_clone(gradients, float(grad_threshold), float(scene_extent))
    change_bank.densify_and_split(gradients, float(grad_threshold), float(scene_extent))
    change_bank.tmp_radii = None
    final_count = int(change_bank.get_xyz.shape[0])
    # Vanilla O-SCD does not own this FastGS diagnostic. Keep it aligned so the
    # common integrity audit can still compare conditions.
    change_bank.xyz_gradient_accum_abs = torch.zeros(
        (final_count, 1), device=change_bank.get_xyz.device, dtype=change_bank.get_xyz.dtype
    )
    expected = initial_count + clone_count + split_count
    if final_count != expected:
        raise RuntimeError(
            f"unexpected O-SCD topology size {final_count}; expected {expected}"
        )
    audit = topology_integrity(change_bank)
    if not audit["passed"]:
        raise RuntimeError(f"topology integrity failure after densification: {audit}")
    return DensificationResult(
        initial_count=initial_count,
        final_count=final_count,
        clone_count=clone_count,
        split_count=split_count,
        masks=masks,
    )
