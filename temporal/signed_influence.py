"""Signed opacity-removal influence helpers for temporal change Gaussians."""

from __future__ import annotations

import torch


def forced_state_render_attributes(model, state: int) -> dict[str, torch.Tensor]:
    """Return one state's renderer attributes without applying ``state_valid``.

    Influence estimation must start from every fixed-topology Gaussian. Reusing
    the existing validity gate here would make it impossible for a previously
    invalid, dark occluder to become valid.
    """
    if isinstance(state, bool) or not isinstance(state, int):
        raise TypeError("state must be an integer")
    if state < 0 or state >= model.max_states:
        raise ValueError(f"state must be in [0, {model.max_states})")

    model._validate_base_alignment()
    base = model.base
    attributes = {
        "dc": model.state_change_dc[:, state],
        "xyz": base.get_xyz,
        "opacity": base.get_opacity,
        "scaling": base.get_scaling,
        "rotation": base.get_rotation,
    }
    if hasattr(model, "shared_xyz_delta"):
        attributes.update(
            {
                "xyz": base._xyz + model.shared_xyz_delta,
                "opacity": base.opacity_activation(
                    base._opacity + model.shared_opacity_delta
                ),
                "scaling": base.scaling_activation(
                    base._scaling + model.shared_scaling_delta
                ),
                "rotation": base.rotation_activation(
                    base._rotation + model.shared_rotation_delta
                ),
            }
        )
    elif hasattr(model, "state_xyz_delta"):
        attributes.update(
            {
                "xyz": base._xyz + model.state_xyz_delta[:, state],
                "opacity": base.opacity_activation(
                    base._opacity + model.state_opacity_delta[:, state]
                ),
                "scaling": base.scaling_activation(
                    base._scaling + model.state_scaling_delta[:, state]
                ),
                "rotation": base.rotation_activation(
                    base._rotation + model.state_rotation_delta[:, state]
                ),
            }
        )
    return attributes


def opacity_removal_influence(
    rendered_change_rgb: torch.Tensor,
    effective_opacity: torch.Tensor,
    *,
    influence_opacity: torch.Tensor | None = None,
    mask_threshold: float | None = None,
    mask_temperature: float = 0.05,
) -> torch.Tensor:
    """Estimate signed removal influence with one renderer backward pass.

    For fixed depth ordering, ``opacity * d(objective)/d(opacity)`` is the
    integrated first-order effect of changing that opacity. Positive values add
    change evidence; negative values suppress brighter content behind.

    ``influence_opacity`` may supply a nonzero candidate opacity for rows whose
    effective baseline opacity is zero. This evaluates insertion influence for
    currently invalid Gaussians while retaining removal influence for active
    rows, without turning every invalid Gaussian on simultaneously.

    When ``mask_threshold`` is provided, the objective is a differentiable
    approximation of binary-mask area. This concentrates attribution near the
    decision boundary instead of labeling every dark foreground splat that has
    any bright content somewhere behind it.
    """
    if rendered_change_rgb.ndim != 3 or rendered_change_rgb.shape[0] != 3:
        raise ValueError("rendered_change_rgb must have shape [3, H, W]")
    if effective_opacity.ndim != 2 or effective_opacity.shape[1] != 1:
        raise ValueError("effective_opacity must have shape [N, 1]")
    if not effective_opacity.requires_grad:
        raise ValueError("effective_opacity must require gradients")
    if rendered_change_rgb.device != effective_opacity.device:
        raise ValueError("render and opacity must share a device")
    scale_opacity = effective_opacity if influence_opacity is None else influence_opacity
    if scale_opacity.shape != effective_opacity.shape:
        raise ValueError("influence_opacity must match effective_opacity shape")
    if scale_opacity.device != effective_opacity.device:
        raise ValueError("influence_opacity must share the opacity device")

    gray = rendered_change_rgb.mean(dim=0)
    if mask_threshold is None:
        objective = gray.sum()
    else:
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        if mask_temperature <= 0.0:
            raise ValueError("mask_temperature must be positive")
        objective = torch.sigmoid(
            (gray - float(mask_threshold)) / float(mask_temperature)
        ).sum()
    gradient = torch.autograd.grad(
        objective,
        effective_opacity,
        retain_graph=False,
        create_graph=False,
    )[0]
    return scale_opacity.detach().flatten() * gradient.detach().flatten()


def split_signed_influence(
    signed_influence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-negative additive and occluding influence scores."""
    if signed_influence.ndim != 1:
        raise ValueError("signed_influence must be rank-1")
    return signed_influence.clamp_min(0), (-signed_influence).clamp_min(0)


def classify_influence(
    additive: torch.Tensor,
    occluding: torch.Tensor,
    contributing_views: torch.Tensor,
    *,
    min_mean_influence: float,
    min_views: int,
    view_count: int,
) -> dict[str, torch.Tensor]:
    """Classify significant additive, occluding, and combined Gaussian roles."""
    if additive.shape != occluding.shape or additive.shape != contributing_views.shape:
        raise ValueError("influence tensors must have the same shape")
    if additive.ndim != 1:
        raise ValueError("influence tensors must be rank-1")
    if min_mean_influence < 0:
        raise ValueError("min_mean_influence must be non-negative")
    if min_views < 1:
        raise ValueError("min_views must be positive")
    if view_count < 1:
        raise ValueError("view_count must be positive")

    mean_additive = additive / float(view_count)
    mean_occluding = occluding / float(view_count)
    enough_views = contributing_views >= int(min_views)
    additive_mask = enough_views & (mean_additive >= float(min_mean_influence))
    occluding_mask = enough_views & (mean_occluding >= float(min_mean_influence))
    return {
        "mean_additive": mean_additive,
        "mean_occluding": mean_occluding,
        "mean_absolute": mean_additive + mean_occluding,
        "additive_mask": additive_mask,
        "occluding_mask": occluding_mask,
        "valid_mask": additive_mask | occluding_mask,
        "both_mask": additive_mask & occluding_mask,
    }
