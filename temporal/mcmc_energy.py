"""Energy terms for fixed-capacity MCMC current-state optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import compute_ssf_loss


@dataclass(frozen=True)
class EnergyWeights:
    opacity: float = 0.0
    scale: float = 0.0
    unsupported_anchor: float = 0.0
    beta_scale: float = 1.0
    beta_rotation: float = 1.0


class MCMCEnergy(nn.Module):
    """Composable energy for current-state MCMC Gaussian updates.

    The data term is the existing O-SCD SSF loss. Opacity and scale priors apply
    to all fixed-capacity slots. The optional unsupported anchor applies only to
    slots without cue support and penalizes raw xyz drift, activated scale drift,
    and sign-invariant quaternion drift ``1 - abs(dot(q, q_anchor))``.
    """

    def __init__(
        self,
        *,
        opacity_weight: float = 0.0,
        scale_weight: float = 0.0,
        unsupported_anchor_weight: float = 0.0,
        beta_scale: float = 1.0,
        beta_rotation: float = 1.0,
        reduction: str = "mean",
        regularizer_offset: float = 1.0,
    ):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.weights = EnergyWeights(
            opacity=float(opacity_weight),
            scale=float(scale_weight),
            unsupported_anchor=float(unsupported_anchor_weight),
            beta_scale=float(beta_scale),
            beta_rotation=float(beta_rotation),
        )
        self.reduction = reduction
        self.regularizer_offset = float(regularizer_offset)

    @staticmethod
    def _reduce(value: torch.Tensor, reduction: str) -> torch.Tensor:
        if value.numel() == 0:
            return value.sum()
        if reduction == "sum":
            return value.sum()
        return value.mean()

    @staticmethod
    def sign_invariant_quaternion_distance(q: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Return ``1 - abs(dot(q, anchor))`` for normalized quaternions."""
        qn = F.normalize(q, dim=-1)
        an = F.normalize(anchor, dim=-1)
        return 1.0 - (qn * an).sum(dim=-1).abs().clamp(max=1.0)

    @staticmethod
    def _anchor_tensor(anchor, name: str, target: torch.Tensor) -> torch.Tensor:
        if anchor is None:
            raise ValueError("unsupported_anchor was requested but no anchor snapshot was supplied")
        if name not in anchor:
            raise KeyError(f"unsupported_anchor missing {name!r}")
        value = anchor[name].detach().to(device=target.device, dtype=target.dtype)
        if value.shape != target.shape:
            raise ValueError(
                f"unsupported_anchor {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        return value

    def regularization_terms(self, state, *, unsupported_anchor=None) -> dict[str, torch.Tensor]:
        attrs = state.get_active_render_attributes()
        cue_support = attrs.get("cue_support_mask", attrs.get("support_mask"))
        if cue_support is None:
            cue_support = torch.zeros(attrs["opacity"].shape[0], dtype=torch.bool, device=attrs["opacity"].device)
        terms: dict[str, torch.Tensor] = {}

        if self.weights.opacity:
            terms["opacity"] = self._reduce(attrs["opacity"].abs(), self.reduction) * self.weights.opacity
        else:
            terms["opacity"] = attrs["opacity"].sum() * 0.0

        if self.weights.scale:
            terms["scale"] = self._reduce(attrs["scaling"].abs(), self.reduction) * self.weights.scale
        else:
            terms["scale"] = attrs["scaling"].sum() * 0.0

        if self.weights.unsupported_anchor:
            unsupported = ~cue_support
            if unsupported.any():
                xyz_anchor = self._anchor_tensor(unsupported_anchor, "xyz", state.current_xyz)
                scaling_anchor_raw = self._anchor_tensor(unsupported_anchor, "scaling", state.current_scaling)
                rotation_anchor = self._anchor_tensor(unsupported_anchor, "rotation", state.current_rotation)
                scale_anchor = state.base.scaling_activation(scaling_anchor_raw)
                anchor_energy = (
                    (state.current_xyz[unsupported] - xyz_anchor[unsupported]).pow(2).sum(dim=-1)
                    + self.weights.beta_scale
                    * (attrs["scaling"][unsupported] - scale_anchor[unsupported]).pow(2).sum(dim=-1)
                    + self.weights.beta_rotation
                    * self.sign_invariant_quaternion_distance(
                        state.current_rotation[unsupported],
                        rotation_anchor[unsupported],
                    )
                )
                terms["unsupported_anchor"] = self._reduce(anchor_energy, self.reduction) * self.weights.unsupported_anchor
            else:
                terms["unsupported_anchor"] = state.current_xyz.sum() * 0.0
        else:
            terms["unsupported_anchor"] = state.current_xyz.sum() * 0.0
        return terms

    def forward(
        self,
        candidate_map: torch.Tensor,
        rendered_change_rgb: torch.Tensor,
        state=None,
        unsupported_anchor=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ssf_loss, parts = compute_ssf_loss(
            candidate_map,
            rendered_change_rgb,
            regularizer_offset=self.regularizer_offset,
        )
        total = ssf_loss
        out: dict[str, torch.Tensor] = {f"ssf_{key}": value for key, value in parts.items()}
        out["ssf"] = ssf_loss
        if state is not None:
            regs = self.regularization_terms(state, unsupported_anchor=unsupported_anchor)
            for name, value in regs.items():
                total = total + value
                out[name] = value
        out["energy"] = total
        return total, out
