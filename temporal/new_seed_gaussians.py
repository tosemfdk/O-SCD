"""Fixed-geometry, DC-only NEW seed Gaussian sidecar.

This module intentionally keeps NEW seed rows outside the base
``GaussianModel`` topology.  Seed geometry is stored as buffers and the only
trainable tensor is ``seed_dc``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from utils.general_utils import inverse_sigmoid


def _as_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _empty(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(tuple(shape), device=device, dtype=dtype)


def _require_float_tensor(name: str, value: torch.Tensor, shape_tail: tuple[int, ...] | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if shape_tail is not None and tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [N, {', '.join(map(str, shape_tail))}]")


def _broadcast_or_validate_rows(
    name: str,
    value: torch.Tensor | None,
    *,
    rows: int,
    tail: tuple[int, ...],
    fill: float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        if fill is None:
            raise ValueError(f"{name} is required")
        return torch.full((rows, *tail), fill, device=device, dtype=dtype)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    value = value.detach().to(device=device, dtype=dtype)
    if value.shape == tail:
        value = value.expand(rows, *tail).clone()
    if tuple(value.shape) != (rows, *tail):
        raise ValueError(f"{name} must have shape {(rows, *tail)}")
    return value.contiguous()


def _pad_rest_channels(rest: torch.Tensor, channels: int) -> torch.Tensor:
    if rest.shape[1] == channels:
        return rest
    if rest.shape[1] > channels:
        raise ValueError("rest tensor has more channels than requested")
    pad = torch.zeros(
        (rest.shape[0], channels - rest.shape[1], rest.shape[2]),
        device=rest.device,
        dtype=rest.dtype,
    )
    return torch.cat([rest, pad], dim=1)


def _is_active(timestamp: float | torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(timestamp, device=start.device, dtype=start.dtype)
    return (start <= t) & (t < end)


class NewSeedGaussianModel(nn.Module):
    """Sidecar for triangulated NEW Gaussian seeds.

    ``xyz``, ``scaling`` (log scale), ``rotation`` (raw quaternion), ``opacity``
    (logit), and SH rest coefficients are buffers.  The sole trainable
    parameter is ``seed_dc`` with shape ``[N, 1, 3]``.
    """

    def __init__(
        self,
        *,
        sh_degree: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if isinstance(sh_degree, bool) or int(sh_degree) < 0:
            raise ValueError("sh_degree must be a non-negative integer")
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0
        self._rest_channels = (self.max_sh_degree + 1) ** 2 - 1
        dev = _as_device(device)
        self.seed_dc = nn.Parameter(_empty((0, 1, 3), device=dev, dtype=dtype))
        self.register_buffer("_xyz", _empty((0, 3), device=dev, dtype=dtype))
        self.register_buffer("_features_rest", _empty((0, self._rest_channels, 3), device=dev, dtype=dtype))
        self.register_buffer("_opacity", _empty((0, 1), device=dev, dtype=dtype))
        self.register_buffer("_scaling", _empty((0, 3), device=dev, dtype=dtype))
        self.register_buffer("_rotation", _empty((0, 4), device=dev, dtype=dtype))
        self.register_buffer("start", _empty((0,), device=dev, dtype=dtype))
        self.register_buffer("end", _empty((0,), device=dev, dtype=dtype))
        self.metadata: list[dict[str, Any]] = []
        self.setup_functions()

    def setup_functions(self) -> None:
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        # Matches GaussianModel.get_covariance/render_change expectations.
        from utils.general_utils import build_scaling_rotation, strip_symmetric

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            return strip_symmetric(L @ L.transpose(1, 2))

        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def _features_dc(self) -> nn.Parameter:
        return self.seed_dc

    @_features_dc.setter
    def _features_dc(self, value: torch.Tensor) -> None:
        if isinstance(value, nn.Parameter):
            self.seed_dc = value
        else:
            self.seed_dc = nn.Parameter(value)

    @property
    def num_seeds(self) -> int:
        return int(self.seed_dc.shape[0])


    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.seed_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def seed_parameter_items(self) -> list[tuple[str, nn.Parameter]]:
        return [("seed_dc", self.seed_dc)]

    def optimizer_parameter_groups(self, lr: float) -> list[dict[str, Any]]:
        return [{"params": [self.seed_dc], "lr": lr, "name": "seed_dc"}]

    def active_mask(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        return _is_active(timestamp, self.start, self.end)

    def close_active(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        """Close currently-active rows at ``timestamp`` and return the closed mask."""
        active = self.active_mask(timestamp)
        if active.any():
            self.end[active] = torch.as_tensor(timestamp, device=self.end.device, dtype=self.end.dtype)
        return active

    def get_active_render_attributes(self, timestamp: float | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return active seed rows with half-open ``[start, end)`` lifespans."""
        mask = self.active_mask(timestamp)
        return {
            "mask": mask,
            "xyz": self._xyz[mask],
            "dc": self.seed_dc[mask],
            "features_rest": self._features_rest[mask],
            "opacity": self.get_opacity[mask],
            "opacity_logits": self._opacity[mask],
            "scaling": self.get_scaling[mask],
            "scaling_logits": self._scaling[mask],
            "rotation": self.get_rotation[mask],
            "rotation_logits": self._rotation[mask],
            "start": self.start[mask],
            "end": self.end[mask],
        }

    def active_view(self, timestamp: float | torch.Tensor) -> "ActiveNewSeedView":
        """Return a temporary renderer adapter containing active seed rows only."""

        return ActiveNewSeedView(self, timestamp)

    def append(
        self,
        *,
        xyz: torch.Tensor,
        start: float | torch.Tensor,
        scaling: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        opacity: torch.Tensor | float = 0.1,
        dc: torch.Tensor | None = None,
        features_rest: torch.Tensor | None = None,
        end: float | torch.Tensor = float("inf"),
        metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.Tensor:
        """Append fixed-geometry rows and optionally expand an Adam optimizer state.

        Returns the global row indices of the appended seeds.
        """
        _require_float_tensor("xyz", xyz, (3,))
        rows = int(xyz.shape[0])
        if rows == 0:
            return torch.empty((0,), device=self.seed_dc.device, dtype=torch.long)

        device, dtype = self.seed_dc.device, self.seed_dc.dtype
        xyz = xyz.detach().to(device=device, dtype=dtype).contiguous()
        scaling = _broadcast_or_validate_rows("scaling", scaling, rows=rows, tail=(3,), fill=math.log(1.0), device=device, dtype=dtype)
        rotation_default = torch.zeros((rows, 4), device=device, dtype=dtype)
        rotation_default[:, 0] = 1.0
        if rotation is None:
            rotation = rotation_default
        else:
            rotation = _broadcast_or_validate_rows("rotation", rotation, rows=rows, tail=(4,), fill=None, device=device, dtype=dtype)
        if isinstance(opacity, torch.Tensor):
            opacity_tensor = opacity.detach().to(device=device, dtype=dtype)
            if opacity_tensor.ndim == 0:
                opacity_tensor = opacity_tensor.view(1, 1).expand(rows, 1).clone()
            elif tuple(opacity_tensor.shape) == (rows,):
                opacity_tensor = opacity_tensor[:, None]
            elif tuple(opacity_tensor.shape) != (rows, 1):
                raise ValueError(f"opacity must be scalar, [N], or {(rows, 1)}")
        else:
            opacity_tensor = torch.full((rows, 1), float(opacity), device=device, dtype=dtype)
        if (opacity_tensor <= 0).any() or (opacity_tensor >= 1).any():
            # Treat tensor inputs outside probability range as already-logit values.
            opacity_logits = opacity_tensor.contiguous()
        else:
            opacity_logits = self.inverse_opacity_activation(opacity_tensor).contiguous()
        dc = _broadcast_or_validate_rows("dc", dc, rows=rows, tail=(1, 3), fill=0.0, device=device, dtype=dtype)
        features_rest = _broadcast_or_validate_rows(
            "features_rest",
            features_rest,
            rows=rows,
            tail=(self._rest_channels, 3),
            fill=0.0,
            device=device,
            dtype=dtype,
        )
        start_tensor = torch.as_tensor(start, device=device, dtype=dtype).expand(rows).clone()
        end_tensor = torch.as_tensor(end, device=device, dtype=dtype).expand(rows).clone()
        if not torch.all(start_tensor < end_tensor):
            raise ValueError("each seed must satisfy start < end")

        old_param = self.seed_dc
        old_n = self.num_seeds
        new_param = nn.Parameter(torch.cat([old_param.detach(), dc], dim=0))

        self._xyz = torch.cat([self._xyz, xyz], dim=0)
        self._features_rest = torch.cat([self._features_rest, features_rest], dim=0)
        self._opacity = torch.cat([self._opacity, opacity_logits], dim=0)
        self._scaling = torch.cat([self._scaling, scaling], dim=0)
        self._rotation = torch.cat([self._rotation, rotation], dim=0)
        self.start = torch.cat([self.start, start_tensor], dim=0)
        self.end = torch.cat([self.end, end_tensor], dim=0)
        self.seed_dc = new_param
        self._append_metadata(metadata, rows)
        if optimizer is not None:
            self._replace_optimizer_parameter_preserving_state(optimizer, old_param, new_param, old_n)
        return torch.arange(old_n, old_n + rows, device=device, dtype=torch.long)

    def append_densified_children(
        self,
        *,
        xyz: torch.Tensor,
        parent_rows: torch.Tensor,
        start: float | torch.Tensor,
        scaling: torch.Tensor,
        opacity: torch.Tensor | float = 0.1,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.Tensor:
        """Append NEW-only children while inheriting the parent's learned DC.

        This helper cannot address the reference bank: every ``parent_rows``
        entry must be an existing seed-sidecar row.  Geometry remains buffered;
        only the copied child DC participates in subsequent optimization.
        """

        if not isinstance(parent_rows, torch.Tensor):
            raise TypeError("parent_rows must be a tensor")
        parents = parent_rows.detach().to(device=self.seed_dc.device, dtype=torch.long).flatten()
        if parents.numel() != xyz.shape[0]:
            raise ValueError("parent_rows and xyz must have matching rows")
        if parents.numel() and ((parents < 0).any() or (parents >= self.num_seeds).any()):
            raise IndexError("densified children must reference existing NEW seed rows")
        if metadata is not None and len(metadata) != int(parents.numel()):
            raise ValueError("metadata length must match densified child rows")
        child_metadata: list[dict[str, Any]] = []
        for index, parent_row in enumerate(parents.tolist()):
            row = copy.deepcopy(dict(self.metadata[parent_row]))
            parent_seed_id = row.get("seed_id")
            row.update(
                {
                    "birth_kind": "new_only_densified",
                    "parent_seed_row": int(parent_row),
                    "parent_seed_id": parent_seed_id,
                    "seed_id": None,
                    "track_id": None,
                }
            )
            if metadata is not None:
                row.update(copy.deepcopy(dict(metadata[index])))
            child_metadata.append(row)
        child_dc = self.seed_dc.detach()[parents].clone()
        return self.append(
            xyz=xyz,
            start=start,
            scaling=scaling,
            opacity=opacity,
            dc=child_dc,
            metadata=child_metadata,
            optimizer=optimizer,
        )

    def _append_metadata(self, metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None, rows: int) -> None:
        if metadata is None:
            self.metadata.extend({} for _ in range(rows))
            return
        if isinstance(metadata, Mapping):
            if rows != 1:
                raise ValueError("single metadata dict can only be used for one appended seed")
            self.metadata.append(copy.deepcopy(dict(metadata)))
            return
        if len(metadata) != rows:
            raise ValueError("metadata length must match appended rows")
        self.metadata.extend(copy.deepcopy(dict(item)) for item in metadata)

    @staticmethod
    def _replace_optimizer_parameter_preserving_state(
        optimizer: torch.optim.Optimizer,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        old_rows: int,
    ) -> None:
        replaced = False
        for group in optimizer.param_groups:
            params = group.get("params", [])
            for idx, param in enumerate(params):
                if param is old_param:
                    params[idx] = new_param
                    replaced = True
        if not replaced:
            optimizer.add_param_group({"params": [new_param], "name": "seed_dc"})
        old_state = optimizer.state.pop(old_param, {})
        new_state: dict[str, Any] = {}
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor) and value.shape == old_param.shape:
                expanded = torch.zeros_like(new_param.detach())
                if old_rows:
                    expanded[:old_rows].copy_(value.detach())
                new_state[key] = expanded
            elif isinstance(value, torch.Tensor):
                new_state[key] = value.detach().clone()
            else:
                new_state[key] = copy.deepcopy(value)
        optimizer.state[new_param] = new_state

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "sh_degree": self.max_sh_degree,
            "seed_dc": self.seed_dc.detach().clone(),
            "xyz": self._xyz.detach().clone(),
            "features_rest": self._features_rest.detach().clone(),
            "opacity": self._opacity.detach().clone(),
            "scaling": self._scaling.detach().clone(),
            "rotation": self._rotation.detach().clone(),
            "start": self.start.detach().clone(),
            "end": self.end.detach().clone(),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "NewSeedGaussianModel":
        seed_dc = checkpoint["seed_dc"]
        model = cls(
            sh_degree=int(checkpoint.get("sh_degree", 0)),
            device=seed_dc.device,
            dtype=seed_dc.dtype,
        )
        model.seed_dc = nn.Parameter(seed_dc.detach().clone())
        model._xyz = checkpoint["xyz"].detach().clone()
        model._features_rest = checkpoint["features_rest"].detach().clone()
        model._opacity = checkpoint["opacity"].detach().clone()
        model._scaling = checkpoint["scaling"].detach().clone()
        model._rotation = checkpoint["rotation"].detach().clone()
        model.start = checkpoint["start"].detach().clone()
        model.end = checkpoint["end"].detach().clone()
        model.metadata = copy.deepcopy(list(checkpoint.get("metadata", [{} for _ in range(model.num_seeds)])))
        if len(model.metadata) != model.num_seeds:
            raise ValueError("checkpoint metadata length does not match seed rows")
        return model


class ConcatenatedChangeView:
    """Temporary base+active-seed adapter for ``gaussian_renderer.render_change``.

    The adapter does not mutate the base model.  Base tensors are concatenated
    with active seed rows selected by half-open seed lifespans.  By default base
    DC is detached so seed-only optimization cannot update sign-memory rows.
    """

    def __init__(
        self,
        base: Any,
        seeds: NewSeedGaussianModel,
        *,
        timestamp: float | torch.Tensor,
        base_dc: torch.Tensor | None = None,
        base_opacity: torch.Tensor | None = None,
        detach_base_dc: bool = True,
    ) -> None:
        self.base = base
        self.seeds = seeds
        self.timestamp = timestamp
        self.active_sh_degree = 0
        self.max_sh_degree = max(int(getattr(base, "max_sh_degree", 0)), int(seeds.max_sh_degree))
        self.scaling_activation = getattr(base, "scaling_activation", torch.exp)
        self.opacity_activation = getattr(base, "opacity_activation", torch.sigmoid)
        self.rotation_activation = getattr(base, "rotation_activation", torch.nn.functional.normalize)
        self.covariance_activation = getattr(base, "covariance_activation", seeds.covariance_activation)
        attrs = seeds.get_active_render_attributes(timestamp)
        self.seed_active_mask = attrs["mask"]
        base_features_dc = base._features_dc if base_dc is None else base_dc
        if detach_base_dc:
            base_features_dc = base_features_dc.detach()
        base_features_rest = base._features_rest.detach()
        seed_rest = attrs["features_rest"].detach()
        rest_channels = max(base_features_rest.shape[1], seed_rest.shape[1])
        base_features_rest = _pad_rest_channels(base_features_rest, rest_channels)
        seed_rest = _pad_rest_channels(seed_rest, rest_channels)
        self._xyz = torch.cat([base.get_xyz.detach(), attrs["xyz"].detach()], dim=0)
        self._features_dc = torch.cat([base_features_dc, attrs["dc"]], dim=0)
        self._features_rest = torch.cat([base_features_rest, seed_rest], dim=0)
        if base_opacity is None:
            base_opacity_value = base.get_opacity.detach()
        else:
            base_opacity_value = base_opacity.detach() if detach_base_dc else base_opacity
        self._opacity_value = torch.cat([base_opacity_value, attrs["opacity"].detach()], dim=0)
        self._scaling_value = torch.cat([base.get_scaling.detach(), attrs["scaling"].detach()], dim=0)
        self._rotation_value = torch.cat([base.get_rotation.detach(), attrs["rotation"].detach()], dim=0)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity_value

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling_value

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation_value

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)


class ActiveNewSeedView:
    """Temporary active-only adapter for a base-independent NEW coverage loss."""

    def __init__(self, seeds: NewSeedGaussianModel, timestamp: float | torch.Tensor) -> None:
        attrs = seeds.get_active_render_attributes(timestamp)
        self.active_sh_degree = 0
        self.max_sh_degree = seeds.max_sh_degree
        self.scaling_activation = seeds.scaling_activation
        self.opacity_activation = seeds.opacity_activation
        self.rotation_activation = seeds.rotation_activation
        self.covariance_activation = seeds.covariance_activation
        self._xyz = attrs["xyz"].detach()
        self._features_dc = attrs["dc"]
        self._features_rest = attrs["features_rest"].detach()
        self._opacity_value = attrs["opacity"].detach()
        self._scaling_value = attrs["scaling"].detach()
        self._rotation_value = attrs["rotation"].detach()

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._opacity_value

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._scaling_value

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._rotation_value

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)


def build_concatenated_change_view(
    base: Any,
    seeds: NewSeedGaussianModel,
    *,
    timestamp: float | torch.Tensor,
    base_dc: torch.Tensor | None = None,
    base_opacity: torch.Tensor | None = None,
    detach_base_dc: bool = True,
) -> ConcatenatedChangeView:
    return ConcatenatedChangeView(
        base,
        seeds,
        timestamp=timestamp,
        base_dc=base_dc,
        base_opacity=base_opacity,
        detach_base_dc=detach_base_dc,
    )
