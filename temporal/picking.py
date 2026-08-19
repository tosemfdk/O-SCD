"""Ray-based selection helpers for temporal Gaussian inspection."""

from dataclasses import dataclass
from numbers import Real

import torch


@dataclass(frozen=True)
class GaussianRayPick:
    """Nearest truncated Gaussian ellipsoid intersected by a world-space ray."""

    index: int
    entry_depth: float
    center_depth: float
    perpendicular_distance: float
    normalized_miss_distance: float


def _quaternion_rotation_matrices(wxyz: torch.Tensor) -> torch.Tensor:
    """Return local-to-world rotation matrices for normalized ``[N, 4]`` quaternions."""
    q = torch.nn.functional.normalize(wxyz, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def pick_gaussian_along_ray(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    candidate_mask: torch.Tensor,
    ray_origin,
    ray_direction,
    *,
    sigma_radius: float = 3.0,
) -> GaussianRayPick | None:
    """Pick the frontmost candidate intersecting a truncated Gaussian ellipsoid.

    Each Gaussian is approximated by the ellipsoid whose semi-axes are
    ``sigma_radius * scaling``. This matches the finite support used for
    interactive picking more closely than treating every splat as a point.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not torch.is_floating_point(xyz):
        raise ValueError("xyz must be a floating tensor with shape [N, 3]")
    if scaling.shape != xyz.shape or not torch.is_floating_point(scaling):
        raise ValueError("scaling must be a floating tensor with shape [N, 3]")
    if rotation.shape != (xyz.shape[0], 4) or not torch.is_floating_point(rotation):
        raise ValueError("rotation must be a floating tensor with shape [N, 4]")
    if candidate_mask.shape != (xyz.shape[0],) or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be a bool tensor with shape [N]")
    if not (xyz.device == scaling.device == rotation.device == candidate_mask.device):
        raise ValueError("all Gaussian tensors must share a device")
    if isinstance(sigma_radius, bool) or not isinstance(sigma_radius, Real):
        raise TypeError("sigma_radius must be a positive finite number")
    if not torch.isfinite(torch.tensor(float(sigma_radius))) or sigma_radius <= 0:
        raise ValueError("sigma_radius must be a positive finite number")

    indices = torch.where(candidate_mask)[0]
    if indices.numel() == 0:
        return None

    coarse_origin = torch.as_tensor(ray_origin, device=xyz.device, dtype=xyz.dtype)
    coarse_direction = torch.as_tensor(
        ray_direction, device=xyz.device, dtype=xyz.dtype
    )
    if coarse_origin.shape != (3,) or coarse_direction.shape != (3,):
        raise ValueError("ray_origin and ray_direction must each have shape [3]")
    if not torch.isfinite(coarse_origin).all() or not torch.isfinite(
        coarse_direction
    ).all():
        raise ValueError("ray values must be finite")
    coarse_norm = torch.linalg.vector_norm(coarse_direction)
    if coarse_norm <= torch.finfo(coarse_direction.dtype).eps:
        raise ValueError("ray_direction must be non-zero")
    coarse_direction = coarse_direction / coarse_norm

    # A rotated ellipsoid is contained by the sphere whose radius is its
    # largest semi-axis. Reject candidates outside that sphere before the
    # float64 exact test so state-valid sets with hundreds of thousands of
    # Gaussians remain interactive.
    coarse_points = xyz[indices]
    coarse_scales = scaling[indices]
    if not torch.isfinite(coarse_points).all() or not torch.isfinite(
        coarse_scales
    ).all():
        raise ValueError("candidate xyz and scaling values must be finite")
    if torch.any(coarse_scales <= 0):
        raise ValueError("candidate scaling values must be positive")
    coarse_radii = coarse_scales.amax(dim=-1) * float(sigma_radius)
    coarse_relative = coarse_points - coarse_origin
    coarse_depth = coarse_relative @ coarse_direction
    coarse_perpendicular_sq = (
        coarse_relative.square().sum(dim=-1) - coarse_depth.square()
    ).clamp_min(0)
    coarse_hit = (
        (coarse_depth + coarse_radii >= 0)
        & (coarse_perpendicular_sq <= coarse_radii.square())
    )
    indices = indices[coarse_hit]
    if indices.numel() == 0:
        return None

    # Very flat splats can have axis scales near 1e-7. Float32 quadratic
    # discriminants then suffer catastrophic cancellation and may report hits
    # behind the camera. Perform the small candidate-only picking calculation
    # in float64 even when rendering uses float32.
    compute_dtype = torch.float64
    points = xyz[indices].to(dtype=compute_dtype)
    scales = scaling[indices].to(dtype=compute_dtype)
    rotations = rotation[indices].to(dtype=compute_dtype)
    if not torch.isfinite(rotations).all():
        raise ValueError("candidate rotation values must be finite")
    if torch.any(torch.linalg.vector_norm(rotations, dim=-1) == 0):
        raise ValueError("candidate rotations must be non-zero quaternions")
    origin = torch.as_tensor(ray_origin, device=xyz.device, dtype=compute_dtype)
    direction = torch.as_tensor(ray_direction, device=xyz.device, dtype=compute_dtype)
    direction_norm = torch.linalg.vector_norm(direction)
    direction = direction / direction_norm

    rotation_matrices = _quaternion_rotation_matrices(rotations)
    world_to_local = rotation_matrices.transpose(1, 2)
    relative = origin[None, :] - points
    local_origin = torch.bmm(world_to_local, relative.unsqueeze(-1)).squeeze(-1)
    local_direction = torch.bmm(
        world_to_local,
        direction.expand(points.shape[0], -1).unsqueeze(-1),
    ).squeeze(-1)
    radii = scales * float(sigma_radius)
    local_origin = local_origin / radii
    local_direction = local_direction / radii

    a = (local_direction * local_direction).sum(dim=-1)
    b = 2.0 * (local_origin * local_direction).sum(dim=-1)
    c = (local_origin * local_origin).sum(dim=-1) - 1.0
    discriminant = b * b - 4.0 * a * c
    real_hit = discriminant >= 0
    sqrt_discriminant = torch.sqrt(discriminant.clamp_min(0))
    denominator = 2.0 * a.clamp_min(torch.finfo(a.dtype).tiny)
    near_depth = (-b - sqrt_discriminant) / denominator
    far_depth = (-b + sqrt_discriminant) / denominator
    entry_depth = torch.where(near_depth >= 0, near_depth, far_depth)
    center_relative_all = points - origin
    center_depth_all = center_relative_all @ direction
    bounding_radius = radii.amax(dim=-1)
    forward_possible = center_depth_all + bounding_radius >= 0
    hit = real_hit & (entry_depth >= 0) & forward_possible
    if not hit.any():
        return None

    masked_depth = torch.where(hit, entry_depth, torch.full_like(entry_depth, torch.inf))
    local_pick = int(torch.argmin(masked_depth).item())
    selected_index = int(indices[local_pick].item())

    center_relative = center_relative_all[local_pick]
    center_depth = center_depth_all[local_pick]
    perpendicular = torch.linalg.vector_norm(center_relative - center_depth * direction)
    closest_depth = torch.clamp(-b[local_pick] / (2.0 * a[local_pick]), min=0.0)
    closest_local = local_origin[local_pick] + closest_depth * local_direction[local_pick]
    normalized_miss = torch.linalg.vector_norm(closest_local)
    return GaussianRayPick(
        index=selected_index,
        entry_depth=float(entry_depth[local_pick].item()),
        center_depth=float(center_depth.item()),
        perpendicular_distance=float(perpendicular.item()),
        normalized_miss_distance=float(normalized_miss.item()),
    )
