"""E4d NEW-only geometry loss, adaptive density control, and causal pruning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from gaussian_renderer import render_change
from temporal.active_new_gaussians import (
    ActiveNewGaussianModel,
    ActiveNewGeometryView,
    FrozenBaseActiveNewView,
)

@dataclass(frozen=True)
class ActiveNewDensityConfig:
    grad_threshold: float = 2.0e-4
    abs_grad_threshold: float = 1.2e-3
    percent_dense: float = 0.01
    max_gaussians: int = 5000
    split_children: int = 2
    preserve_xfeat_anchors: bool = False
    one_shot_anchor_densification: bool = False


@dataclass(frozen=True)
class ActiveNewPruneConfig:
    min_opacity: float = 0.01
    grace_frames: int = 10
    min_observations: int = 8
    min_support_ratio: float = 0.25
    max_screen_radius: float = 100.0
    max_world_scale_ratio: float = 0.10
    min_visible_mass: float = 1.0e-4
    erosion_pixels: int = 1
    preserve_xfeat_anchors: bool = False


def root_anchor_hinge_loss(
    model: ActiveNewGaussianModel,
    *,
    timestamp: int,
    radius_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Penalize only child displacement outside its root-XFeat trust radius."""

    if not math.isfinite(float(radius_multiplier)) or radius_multiplier <= 0.0:
        raise ValueError("radius multiplier must be finite and positive")
    selected = model.active_mask(float(timestamp)) & (model.generation > 0)
    if not bool(selected.any()):
        zero = model._xyz.sum() * 0.0
        return zero, {
            "rows": 0,
            "outside_rows": 0,
            "loss": 0.0,
            "max_distance_ratio": 0.0,
        }
    displacement = torch.linalg.vector_norm(
        model._xyz[selected] - model.root_anchor_xyz[selected], dim=1
    )
    radius = (
        float(radius_multiplier) * model.root_anchor_scale[selected]
    ).clamp_min(torch.finfo(model._xyz.dtype).eps)
    ratio = displacement / radius
    excess = torch.relu(ratio - 1.0)
    loss = excess.square().mean()
    return loss, {
        "rows": int(selected.sum().item()),
        "outside_rows": int((excess > 0).sum().item()),
        "loss": float(loss.detach().item()),
        "max_distance_ratio": float(ratio.detach().amax().item()),
    }


def new_geometry_coverage_loss(
    target: torch.Tensor,
    coverage_render: torch.Tensor,
    *,
    inside_weight: float = 1.0,
    outside_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize missing NEW support and NEW-sidecar footprint leakage."""

    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("target must have shape [1,H,W]")
    if coverage_render.ndim != 3 or coverage_render.shape[1:] != target.shape[1:]:
        raise ValueError("coverage_render must have shape [C,H,W] matching target")
    if inside_weight < 0.0 or outside_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    target = target.to(device=coverage_render.device, dtype=coverage_render.dtype)
    coverage = coverage_render.mean(dim=0, keepdim=True).clamp(0.0, 1.0)
    positive_count = target.sum().clamp_min(1.0)
    negative_count = (1.0 - target).sum().clamp_min(1.0)
    inside = (target * (1.0 - coverage)).sum() / positive_count
    outside = ((1.0 - target) * coverage).sum() / negative_count
    total = float(inside_weight) * inside + float(outside_weight) * outside
    return total, {
        "loss": float(total.detach().item()),
        "inside": float(inside.detach().item()),
        "outside": float(outside.detach().item()),
        "coverage_mean": float(coverage.detach().mean().item()),
    }


def render_active_new_coverage(
    view: Any,
    active_view: ActiveNewGeometryView,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Render differentiable alpha coverage with white NEW override colors."""

    colors = torch.ones(
        (active_view.get_xyz.shape[0], 3),
        device=active_view.get_xyz.device,
        dtype=active_view.get_xyz.dtype,
    )
    return render_change(
        view,
        active_view,
        pipe,
        torch.zeros_like(background),
        override_color=colors,
        clamp_output=False,
    )


def accumulate_density_statistics(
    model: ActiveNewGaussianModel,
    active_view: ActiveNewGeometryView,
    render_package: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    """Accumulate local FastGS signed and absolute screen-space gradients."""

    points = render_package["viewspace_points"]
    radii = render_package["radii"]
    if points.grad is None:
        raise RuntimeError("view-space points did not receive a geometry gradient")
    if points.shape[0] != active_view.global_rows.numel():
        raise RuntimeError("active renderer rows and gradient rows diverged")
    visible = radii > 0
    if not bool(visible.any()):
        return {"visible": 0, "signed_norm": 0.0, "absolute_norm": 0.0}
    rows = active_view.global_rows[visible]
    signed = torch.linalg.vector_norm(points.grad[visible, :2], dim=-1, keepdim=True)
    absolute = torch.linalg.vector_norm(points.grad[visible, 2:], dim=-1, keepdim=True)
    model.xyz_gradient_accum[rows] += signed.detach()
    model.xyz_gradient_accum_abs[rows] += absolute.detach()
    model.gradient_denom[rows] += 1.0
    model.max_radii2d[rows] = torch.maximum(
        model.max_radii2d[rows], radii[visible].detach()
    )
    return {
        "visible": int(rows.numel()),
        "signed_norm": float(signed.mean().item()),
        "absolute_norm": float(absolute.mean().item()),
    }


def _top_rows(mask: torch.Tensor, score: torch.Tensor, limit: int) -> torch.Tensor:
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    if rows.numel() <= limit:
        return rows
    order = torch.argsort(score[rows], descending=True, stable=True)
    return rows[order[:limit]]


def densify_active_new(
    model: ActiveNewGaussianModel,
    optimizer: torch.optim.Optimizer,
    *,
    timestamp: int,
    scene_extent: float,
    config: ActiveNewDensityConfig,
    random_seed: int,
) -> dict[str, Any]:
    """Perform bounded NEW-only gradient clone/split density control."""

    count_before = model.num_gaussians
    available = max(0, int(config.max_gaussians) - count_before)
    denom = model.gradient_denom.clamp_min(1.0)
    signed = (model.xyz_gradient_accum / denom).squeeze(1)
    absolute = (model.xyz_gradient_accum_abs / denom).squeeze(1)
    observed = model.gradient_denom.squeeze(1) > 0
    active = model.active_mask(float(timestamp))
    anchors = model.generation == 0
    if config.one_shot_anchor_densification:
        observed = observed & (~anchors | (model.densification_count == 0))
    scale = model.get_scaling.detach().amax(dim=1)
    small = scale <= float(config.percent_dense) * float(scene_extent)
    clone_mask = active & observed & small & (signed >= float(config.grad_threshold))
    split_mask = active & observed & ~small & (
        absolute >= float(config.abs_grad_threshold)
    )
    # A split replaces one source with N children, consuming N-1 net rows.
    split_net_rows = (
        int(config.split_children)
        if config.preserve_xfeat_anchors
        else max(int(config.split_children) - 1, 1)
    )
    split_capacity = available // split_net_rows
    split_rows = _top_rows(split_mask, absolute, split_capacity)
    available -= int(split_rows.numel()) * split_net_rows
    clone_rows = _top_rows(clone_mask, signed, available)
    clone_ids = model.stable_id[clone_rows].detach().clone()
    clone_scores = signed[clone_rows].detach().clone()
    events: list[dict[str, Any]] = []
    if split_rows.numel():
        preserved_split_anchors = (
            anchors[split_rows]
            if config.preserve_xfeat_anchors
            else torch.zeros_like(split_rows, dtype=torch.bool)
        )
        normal_split_rows = split_rows[~preserved_split_anchors]
        anchor_split_rows = split_rows[preserved_split_anchors]
        if anchor_split_rows.numel():
            anchor_scores = absolute[anchor_split_rows]
            parent_ids = model.stable_id[anchor_split_rows].detach().cpu().tolist()
            model.densification_count[anchor_split_rows] += 1
            model.append_gradient_children(
                parent_rows=anchor_split_rows,
                split=True,
                timestamp=timestamp,
                optimizer=optimizer,
                trigger_scores=anchor_scores,
                children_per_split=config.split_children,
                random_seed=random_seed,
                replace_split_parent=False,
            )
            for parent_id, score in zip(
                parent_ids, anchor_scores.detach().cpu().tolist()
            ):
                events.append(
                    {
                        "timestamp": int(timestamp),
                        "action": "split_preserve_xfeat_anchor",
                        "parent_stable_id": int(parent_id),
                        "children": int(config.split_children),
                        "trigger_score": float(score),
                    }
                )
        split_rows = normal_split_rows
    if split_rows.numel():
        parent_ids = model.stable_id[split_rows].detach().cpu().tolist()
        scores = absolute[split_rows]
        child_rows = model.append_gradient_children(
            parent_rows=split_rows,
            split=True,
            timestamp=timestamp,
            optimizer=optimizer,
            trigger_scores=scores,
            children_per_split=config.split_children,
            random_seed=random_seed,
        )
        for parent_id, score in zip(parent_ids, scores.detach().cpu().tolist()):
            events.append(
                {
                    "timestamp": int(timestamp),
                    "action": "split",
                    "parent_stable_id": int(parent_id),
                    "children": int(config.split_children),
                    "trigger_score": float(score),
                }
            )
        del child_rows
    if clone_rows.numel():
        # Split pruning changes row indices. Stable NEW IDs keep clone sources
        # addressable without ever falling through to reference row indices.
        row_by_id = {
            int(stable_id): row
            for row, stable_id in enumerate(model.stable_id.detach().cpu().tolist())
        }
        clone_rows = torch.tensor(
            [row_by_id[int(stable_id)] for stable_id in clone_ids.cpu().tolist()],
            device=model._xyz.device,
            dtype=torch.long,
        )
        anchor_clones = model.generation[clone_rows] == 0
        if bool(anchor_clones.any()):
            model.densification_count[clone_rows[anchor_clones]] += 1
        model.append_gradient_children(
            parent_rows=clone_rows,
            split=False,
            timestamp=timestamp,
            optimizer=optimizer,
            trigger_scores=clone_scores,
            random_seed=random_seed + 1,
        )
        for parent_id, score in zip(
            clone_ids.detach().cpu().tolist(), clone_scores.detach().cpu().tolist()
        ):
            events.append(
                {
                    "timestamp": int(timestamp),
                    "action": "clone",
                    "parent_stable_id": int(parent_id),
                    "children": 1,
                    "trigger_score": float(score),
                }
            )
    model.reset_density_statistics()
    return {
        "count_before": count_before,
        "count_after": model.num_gaussians,
        "clone_count": int(sum(row["action"] == "clone" for row in events)),
        "split_source_count": int(
            sum(row["action"].startswith("split") for row in events)
        ),
        "events": events,
        "signed_gradient": signed.detach().cpu(),
        "absolute_gradient": absolute.detach().cpu(),
    }


def visibility_mass_with_frozen_reference(
    view: Any,
    base: Any,
    model: ActiveNewGaussianModel,
    *,
    timestamp: int,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return alpha-T responsibility for active NEW rows behind frozen base occlusion."""

    combined = FrozenBaseActiveNewView(
        base, model, timestamp=float(timestamp), detach_new=True
    )
    total = combined.get_xyz.shape[0]
    probe = torch.zeros(
        (total, 3),
        device=combined.get_xyz.device,
        dtype=combined.get_xyz.dtype,
        requires_grad=True,
    )
    package = render_change(
        view,
        combined,
        pipe,
        torch.zeros_like(background),
        override_color=probe,
        clamp_output=False,
    )
    gradient = torch.autograd.grad(
        package["render"],
        probe,
        grad_outputs=torch.ones_like(package["render"]) / 3.0,
        retain_graph=False,
        create_graph=False,
    )[0]
    mass = gradient[combined.base_count :, 0].detach().clamp_min(0.0)
    if not bool(torch.isfinite(mass).all()):
        raise RuntimeError("NEW visibility VJP returned non-finite responsibility")
    return combined.new_global_rows, mass


def _morph(mask: torch.Tensor, pixels: int, *, erode: bool) -> torch.Tensor:
    if pixels <= 0:
        return mask.bool()
    value = mask.float()[None, None]
    kernel = 2 * int(pixels) + 1
    if erode:
        result = 1.0 - F.max_pool2d(1.0 - value, kernel, stride=1, padding=pixels)
    else:
        result = F.max_pool2d(value, kernel, stride=1, padding=pixels)
    return result[0, 0] > 0.5


def _project_centers(view: Any, xyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
    homogeneous = torch.cat(
        (xyz, torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)),
        dim=1,
    )
    camera = homogeneous @ view.world_view_transform
    depth = camera[:, 2]
    fx = float(view.image_width) / (2.0 * math.tan(float(view.FoVx) * 0.5))
    fy = float(view.image_height) / (2.0 * math.tan(float(view.FoVy) * 0.5))
    x = fx * camera[:, 0] / depth.clamp_min(1.0e-6) + float(view.image_width) * 0.5
    y = fy * camera[:, 1] / depth.clamp_min(1.0e-6) + float(view.image_height) * 0.5
    inside = (
        (depth > 0.0)
        & (x >= 0.0)
        & (x < float(view.image_width))
        & (y >= 0.0)
        & (y < float(view.image_height))
    )
    return x, y, depth, inside


def update_causal_new_support(
    model: ActiveNewGaussianModel,
    *,
    decision_timestamp: int,
    observation_timestamp: int,
    view: Any,
    new_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    active_rows: torch.Tensor,
    visibility_mass: torch.Tensor,
    config: ActiveNewPruneConfig,
) -> dict[str, int]:
    """Attribute one causal view as support, contradiction, or no observation."""

    if int(observation_timestamp) > int(decision_timestamp):
        raise ValueError("future observations cannot update NEW pruning state")
    if active_rows.shape != visibility_mass.shape:
        raise ValueError("active rows and visibility mass must align")
    if new_mask.shape != stable_mask.shape or new_mask.ndim != 2:
        raise ValueError("NEW/stable masks must be aligned [H,W] tensors")
    device = model._xyz.device
    active_rows = active_rows.to(device=device, dtype=torch.long)
    visibility_mass = visibility_mass.to(device=device, dtype=model._xyz.dtype)
    new_eroded = _morph(new_mask.to(device=device), config.erosion_pixels, erode=True)
    new_dilated = _morph(new_mask.to(device=device), config.erosion_pixels, erode=False)
    stable_clear = stable_mask.to(device=device).bool() & ~new_dilated
    xyz = model._xyz.detach()[active_rows]
    x, y, _, inside = _project_centers(view, xyz)
    visible = inside & (visibility_mass >= float(config.min_visible_mass))
    columns = x.long().clamp(0, int(view.image_width) - 1)
    rows_px = y.long().clamp(0, int(view.image_height) - 1)
    positive = visible & new_eroded[rows_px, columns]
    contradiction = visible & stable_clear[rows_px, columns]
    accepted_positive = 0
    accepted_contradiction = 0
    for local, row in enumerate(active_rows.tolist()):
        stable_id = int(model.stable_id[row].item())
        seen = model.observed_view_ids.setdefault(stable_id, set())
        if int(observation_timestamp) in seen:
            continue
        if bool(positive[local]):
            model.positive_support[row] += 1
            accepted_positive += 1
            seen.add(int(observation_timestamp))
        elif bool(contradiction[local]):
            model.contradiction_support[row] += 1
            accepted_contradiction += 1
            seen.add(int(observation_timestamp))
    return {
        "positive": accepted_positive,
        "contradiction": accepted_contradiction,
        "visible": int(visible.sum().item()),
    }


def prune_active_new(
    model: ActiveNewGaussianModel,
    optimizer: torch.optim.Optimizer,
    *,
    timestamp: int,
    scene_extent: float,
    config: ActiveNewPruneConfig,
) -> dict[str, Any]:
    """Prune only causally contradicted or pathological NEW-sidecar rows."""

    if model.num_gaussians == 0:
        return {"count_before": 0, "count_after": 0, "events": []}
    age = int(timestamp) - model.birth_frame
    mature = age >= int(config.grace_frames)
    active = model.active_mask(float(timestamp))
    observations = model.positive_support + model.contradiction_support
    support_ratio = model.positive_support.float() / observations.clamp_min(1).float()
    contradicted = (
        mature
        & (observations >= int(config.min_observations))
        & (support_ratio < float(config.min_support_ratio))
    )
    child = model.generation > 0
    low_opacity = mature & child & (
        model.get_opacity.detach().squeeze(1) < float(config.min_opacity)
    )
    huge_screen = mature & (model.max_radii2d > float(config.max_screen_radius))
    huge_world = mature & (
        model.get_scaling.detach().amax(dim=1)
        > float(config.max_world_scale_ratio) * float(scene_extent)
    )
    # Lifespan closure is not topology deletion.  Historical CLOSED rows must
    # remain reconstructable and are outside the current-state pruning policy.
    prune = active & (contradicted | low_opacity | huge_screen | huge_world)
    if config.preserve_xfeat_anchors:
        prune &= model.generation > 0
    events = []
    for row in torch.nonzero(prune, as_tuple=False).flatten().tolist():
        reasons = []
        if bool(contradicted[row]):
            reasons.append("causal_contradiction")
        if bool(low_opacity[row]):
            reasons.append("low_child_opacity")
        if bool(huge_screen[row]):
            reasons.append("excessive_screen_radius")
        if bool(huge_world[row]):
            reasons.append("excessive_world_scale")
        events.append(
            {
                "timestamp": int(timestamp),
                "stable_id": int(model.stable_id[row].item()),
                "birth_kind": model.metadata[row].get("birth_kind"),
                "generation": int(model.generation[row].item()),
                "positive_support": int(model.positive_support[row].item()),
                "contradiction_support": int(model.contradiction_support[row].item()),
                "support_ratio": float(support_ratio[row].item()),
                "reasons": reasons,
            }
        )
    before = model.num_gaussians
    model.prune_rows(prune, optimizer=optimizer, reason="|".join(sorted({reason for event in events for reason in event["reasons"]})))
    return {"count_before": before, "count_after": model.num_gaussians, "events": events}
