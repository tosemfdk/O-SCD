"""Shared topology primitives used by the existing FastGS port and ablations."""

from __future__ import annotations

from collections.abc import Callable

import torch

from utils.general_utils import build_rotation


def fastgs_split_children(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation_raw: torch.Tensor,
    scaling_inverse_activation: Callable[[torch.Tensor], torch.Tensor],
    *,
    children_per_source: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the official-port FastGS split xyz and raw scaling tensors.

    This extracts the exact sampling/scaling operation previously embedded in
    :meth:`GaussianModel.densify_and_split_fastgs`, so temporal topology code
    can reuse it without creating a second variant.
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if scaling.shape != xyz.shape:
        raise ValueError("scaling must have shape [N,3]")
    if rotation_raw.ndim != 2 or rotation_raw.shape != (xyz.shape[0], 4):
        raise ValueError("rotation_raw must have shape [N,4]")
    if isinstance(children_per_source, bool) or int(children_per_source) < 1:
        raise ValueError("children_per_source must be positive")
    repeat = int(children_per_source)
    repeated_scaling = scaling.repeat(repeat, 1)
    samples = torch.normal(
        mean=torch.zeros_like(repeated_scaling), std=repeated_scaling
    )
    rotations = build_rotation(rotation_raw).repeat(repeat, 1, 1)
    child_xyz = (
        torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
        + xyz.repeat(repeat, 1)
    )
    child_scaling_raw = scaling_inverse_activation(
        repeated_scaling / (0.8 * repeat)
    )
    return child_xyz, child_scaling_raw
