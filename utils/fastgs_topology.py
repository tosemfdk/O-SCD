"""Shared topology primitives used by the existing FastGS port and ablations."""

from __future__ import annotations

from collections.abc import Callable

import torch



def _build_rotation(rotation_raw: torch.Tensor) -> torch.Tensor:
    """Device-preserving quaternion rotation used by topology-only code."""

    quaternion = torch.nn.functional.normalize(rotation_raw, dim=1)
    real, x, y, z = quaternion.unbind(dim=1)
    rotation = torch.zeros(
        (quaternion.shape[0], 3, 3),
        device=quaternion.device,
        dtype=quaternion.dtype,
    )
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - real * z)
    rotation[:, 0, 2] = 2 * (x * z + real * y)
    rotation[:, 1, 0] = 2 * (x * y + real * z)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - real * x)
    rotation[:, 2, 0] = 2 * (x * z - real * y)
    rotation[:, 2, 1] = 2 * (y * z + real * x)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


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
    rotations = _build_rotation(rotation_raw).repeat(repeat, 1, 1)
    child_xyz = (
        torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
        + xyz.repeat(repeat, 1)
    )
    child_scaling_raw = scaling_inverse_activation(
        repeated_scaling / (0.8 * repeat)
    )
    return child_xyz, child_scaling_raw
