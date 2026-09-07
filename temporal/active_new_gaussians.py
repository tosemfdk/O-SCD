"""Trainable NEW-only Gaussian sidecar for the E4d controlled ablation.

The legacy :mod:`temporal.new_seed_gaussians` model intentionally keeps every
geometric attribute fixed.  E4d must not weaken that reproduction contract, so
this module owns a separate dynamic topology whose optimizer can only address
NEW rows.  Reference Gaussian tensors are never stored in this model.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from utils.fastgs_topology import fastgs_split_children
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH


_PARAMETER_NAMES = ("xyz", "dc", "opacity", "scaling", "rotation")


def _device(value: torch.device | str | None) -> torch.device:
    return torch.device("cpu") if value is None else torch.device(value)


def _empty(
    shape: Sequence[int], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(tuple(shape), device=device, dtype=dtype)


def _rows(
    name: str,
    value: torch.Tensor | None,
    *,
    count: int,
    tail: tuple[int, ...],
    fill: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.full((count, *tail), fill, device=device, dtype=dtype)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    result = value.detach().to(device=device, dtype=dtype)
    if tuple(result.shape) == tail:
        result = result.expand(count, *tail).clone()
    if tuple(result.shape) != (count, *tail):
        raise ValueError(f"{name} must have shape {(count, *tail)}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result.contiguous()


def _active(
    timestamp: float | torch.Tensor, start: torch.Tensor, end: torch.Tensor
) -> torch.Tensor:
    current = torch.as_tensor(timestamp, device=start.device, dtype=start.dtype)
    return (start <= current) & (current < end)


class ActiveNewGaussianModel(nn.Module):
    """Dynamic, fully trainable Gaussian bank containing NEW rows only."""

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
        dev = _device(device)
        self._xyz = nn.Parameter(_empty((0, 3), device=dev, dtype=dtype))
        self.new_dc = nn.Parameter(_empty((0, 1, 3), device=dev, dtype=dtype))
        self._opacity = nn.Parameter(_empty((0, 1), device=dev, dtype=dtype))
        self._scaling = nn.Parameter(_empty((0, 3), device=dev, dtype=dtype))
        self._rotation = nn.Parameter(_empty((0, 4), device=dev, dtype=dtype))
        self.register_buffer(
            "_features_rest",
            _empty((0, self._rest_channels, 3), device=dev, dtype=dtype),
        )
        self.register_buffer("start", _empty((0,), device=dev, dtype=dtype))
        self.register_buffer("end", _empty((0,), device=dev, dtype=dtype))
        self.register_buffer("stable_id", torch.empty(0, device=dev, dtype=torch.long))
        self.register_buffer(
            "parent_stable_id", torch.empty(0, device=dev, dtype=torch.long)
        )
        self.register_buffer("generation", torch.empty(0, device=dev, dtype=torch.long))
        self.register_buffer("birth_frame", torch.empty(0, device=dev, dtype=torch.long))
        self.register_buffer("initial_xyz", _empty((0, 3), device=dev, dtype=dtype))
        self.register_buffer("root_anchor_xyz", _empty((0, 3), device=dev, dtype=dtype))
        self.register_buffer("root_anchor_scale", _empty((0,), device=dev, dtype=dtype))
        self.register_buffer(
            "densification_count", torch.empty(0, device=dev, dtype=torch.long)
        )
        self.register_buffer(
            "positive_support", torch.empty(0, device=dev, dtype=torch.long)
        )
        self.register_buffer(
            "contradiction_support", torch.empty(0, device=dev, dtype=torch.long)
        )
        self.register_buffer(
            "xyz_gradient_accum", _empty((0, 1), device=dev, dtype=dtype)
        )
        self.register_buffer(
            "xyz_gradient_accum_abs", _empty((0, 1), device=dev, dtype=dtype)
        )
        self.register_buffer("gradient_denom", _empty((0, 1), device=dev, dtype=dtype))
        self.register_buffer("max_radii2d", _empty((0,), device=dev, dtype=dtype))
        self.metadata: list[dict[str, Any]] = []
        self.observed_view_ids: dict[int, set[int]] = {}
        self.next_stable_id = 0
        self.render_never_open_black = True
        self.setup_functions()

    def setup_functions(self) -> None:
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        from utils.general_utils import build_scaling_rotation, strip_symmetric

        def covariance(scaling, modifier, rotation):
            transform = build_scaling_rotation(modifier * scaling, rotation)
            return strip_symmetric(transform @ transform.transpose(1, 2))

        self.covariance_activation = covariance

    @property
    def _features_dc(self) -> nn.Parameter:
        return self.new_dc

    @_features_dc.setter
    def _features_dc(self, value: torch.Tensor) -> None:
        self.new_dc = value if isinstance(value, nn.Parameter) else nn.Parameter(value)

    @property
    def seed_dc(self) -> nn.Parameter:
        """Compatibility alias for seed PLY/evaluation helpers."""

        return self.new_dc

    @property
    def num_gaussians(self) -> int:
        return int(self._xyz.shape[0])

    @property
    def num_seeds(self) -> int:
        return self.num_gaussians

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.new_dc, self._features_rest), dim=1)

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
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )

    def optimizer_parameter_groups(
        self,
        *,
        xyz_lr: float,
        dc_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
    ) -> list[dict[str, Any]]:
        values = {
            "xyz": (self._xyz, xyz_lr),
            "dc": (self.new_dc, dc_lr),
            "opacity": (self._opacity, opacity_lr),
            "scaling": (self._scaling, scaling_lr),
            "rotation": (self._rotation, rotation_lr),
        }
        groups = []
        for name, (parameter, lr) in values.items():
            if not math.isfinite(float(lr)) or float(lr) < 0.0:
                raise ValueError(f"{name} learning rate must be finite and nonnegative")
            groups.append({"params": [parameter], "lr": float(lr), "name": name})
        return groups

    def zero_xfeat_anchor_xyz_gradient(self) -> int:
        """Remove xyz gradients from generation-zero XFeat anchors exactly."""

        anchors = self.generation == 0
        if self._xyz.grad is not None and bool(anchors.any()):
            self._xyz.grad[anchors] = 0
        return int(anchors.sum().item())

    @torch.no_grad()
    def restore_xfeat_anchor_xyz(
        self, optimizer: torch.optim.Optimizer | None = None
    ) -> int:
        """Restore immutable XFeat positions and clear any Adam xyz momentum."""

        anchors = self.generation == 0
        if not bool(anchors.any()):
            return 0
        self._xyz[anchors] = self.root_anchor_xyz[anchors]
        if optimizer is not None:
            state = optimizer.state.get(self._xyz, {})
            for value in state.values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == self.num_gaussians
                ):
                    value[anchors] = 0
        return int(anchors.sum().item())

    def active_mask(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        return _active(timestamp, self.start, self.end)

    def never_open_mask(self) -> torch.Tensor:
        """Rows proposed geometrically but not yet committed by a detector."""

        return torch.isposinf(self.start) & torch.isposinf(self.end)

    def closed_mask(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        current = torch.as_tensor(
            timestamp, device=self.start.device, dtype=self.start.dtype
        )
        return (~self.never_open_mask()) & (self.end <= current)

    @torch.no_grad()
    def open_rows(
        self, rows: torch.Tensor, timestamp: float | torch.Tensor
    ) -> torch.Tensor:
        selected = rows.to(device=self.start.device, dtype=torch.long).flatten()
        if selected.numel() and (
            bool((selected < 0).any())
            or bool((selected >= self.num_gaussians).any())
        ):
            raise IndexError("NEW rows are out of range")
        if selected.numel() != torch.unique(selected).numel():
            raise ValueError("NEW rows must be unique")
        current = torch.as_tensor(
            timestamp, device=self.start.device, dtype=self.start.dtype
        )
        if selected.numel() and bool(self.active_mask(current)[selected].any()):
            raise ValueError("already-active NEW rows cannot be opened again")
        self.start[selected] = current
        self.end[selected] = float("inf")
        return selected

    @torch.no_grad()
    def close_rows(
        self, rows: torch.Tensor, timestamp: float | torch.Tensor
    ) -> torch.Tensor:
        selected = rows.to(device=self.start.device, dtype=torch.long).flatten()
        if selected.numel() and (
            bool((selected < 0).any())
            or bool((selected >= self.num_gaussians).any())
        ):
            raise IndexError("NEW rows are out of range")
        if selected.numel() != torch.unique(selected).numel():
            raise ValueError("NEW rows must be unique")
        current = torch.as_tensor(
            timestamp, device=self.start.device, dtype=self.start.dtype
        )
        if selected.numel() and not bool(
            self.active_mask(current)[selected].all()
        ):
            raise ValueError("only active NEW rows can be closed")
        self.end[selected] = current
        return selected

    def close_active(self, timestamp: float | torch.Tensor) -> torch.Tensor:
        selected = self.active_mask(timestamp)
        if bool(selected.any()):
            self.end[selected] = torch.as_tensor(
                timestamp, device=self.end.device, dtype=self.end.dtype
            )
        return selected

    def get_active_render_attributes(
        self, timestamp: float | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        selected = self.active_mask(timestamp)
        rows = torch.nonzero(selected, as_tuple=False).flatten()
        return {
            "mask": selected,
            "rows": rows,
            "xyz": self._xyz[selected],
            "dc": self.new_dc[selected],
            "features_rest": self._features_rest[selected],
            "opacity": self.get_opacity[selected],
            "scaling": self.get_scaling[selected],
            "rotation": self.get_rotation[selected],
        }

    def get_all_render_attributes(self) -> dict[str, torch.Tensor]:
        """Return all born rows for a lifespan-agnostic detector probe."""

        selected = torch.ones(
            self.num_gaussians, device=self._xyz.device, dtype=torch.bool
        )
        rows = torch.arange(
            self.num_gaussians, device=self._xyz.device, dtype=torch.long
        )
        return {
            "mask": selected,
            "rows": rows,
            "xyz": self._xyz,
            "dc": self.new_dc,
            "features_rest": self._features_rest,
            "opacity": self.get_opacity,
            "scaling": self.get_scaling,
            "rotation": self.get_rotation,
        }

    def get_lifecycle_render_attributes(
        self, timestamp: float | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return OPEN plus black NEVER_OPEN rows while omitting CLOSED rows."""

        active = self.active_mask(timestamp)
        never_open = self.never_open_mask()
        selected = active | never_open
        rows = torch.nonzero(selected, as_tuple=False).flatten()
        selected_active = active[selected]
        black_dc = RGB2SH(self.new_dc.new_zeros(()))
        dc = torch.where(
            selected_active[:, None, None], self.new_dc[selected], black_dc
        )
        return {
            "mask": selected,
            "rows": rows,
            "xyz": self._xyz[selected],
            "dc": dc,
            "features_rest": self._features_rest[selected],
            "opacity": self.get_opacity[selected],
            "scaling": self.get_scaling[selected],
            "rotation": self.get_rotation[selected],
        }

    def active_view(self, timestamp: float | torch.Tensor) -> "ActiveNewGeometryView":
        return ActiveNewGeometryView(self, timestamp)

    def append_xfeat_anchors(
        self,
        *,
        xyz: torch.Tensor,
        start: float | torch.Tensor,
        scaling: torch.Tensor,
        opacity: float | torch.Tensor = 0.1,
        dc: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        start_active: bool = True,
    ) -> torch.Tensor:
        count = int(xyz.shape[0])
        parent = torch.full(
            (count,), -1, device=self._xyz.device, dtype=torch.long
        )
        generation = torch.zeros(count, device=self._xyz.device, dtype=torch.long)
        rows = self._append(
            xyz=xyz,
            start=start,
            scaling=scaling,
            opacity=opacity,
            dc=dc,
            rotation=rotation,
            parent_stable_id=parent,
            generation=generation,
            birth_kind="xfeat_anchor",
            metadata=metadata,
            optimizer=optimizer,
        )
        if not start_active and rows.numel():
            self.start[rows] = float("inf")
            self.end[rows] = float("inf")
        return rows

    def _append(
        self,
        *,
        xyz: torch.Tensor,
        start: float | torch.Tensor,
        scaling: torch.Tensor,
        opacity: float | torch.Tensor,
        dc: torch.Tensor | None,
        rotation: torch.Tensor | None,
        parent_stable_id: torch.Tensor,
        generation: torch.Tensor,
        birth_kind: str,
        metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        optimizer: torch.optim.Optimizer | None,
        root_anchor_xyz: torch.Tensor | None = None,
        root_anchor_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3]")
        if not torch.is_floating_point(xyz) or not bool(torch.isfinite(xyz).all()):
            raise ValueError("xyz must be a finite floating tensor")
        count = int(xyz.shape[0])
        if count == 0:
            return torch.empty(0, device=self._xyz.device, dtype=torch.long)
        device, dtype = self._xyz.device, self._xyz.dtype
        xyz = xyz.detach().to(device=device, dtype=dtype).contiguous()
        scaling = _rows(
            "scaling",
            scaling,
            count=count,
            tail=(3,),
            fill=0.0,
            device=device,
            dtype=dtype,
        )
        dc = _rows(
            "dc", dc, count=count, tail=(1, 3), fill=0.0, device=device, dtype=dtype
        )
        default_rotation = torch.zeros((count, 4), device=device, dtype=dtype)
        default_rotation[:, 0] = 1.0
        rotation = (
            default_rotation
            if rotation is None
            else _rows(
                "rotation",
                rotation,
                count=count,
                tail=(4,),
                fill=0.0,
                device=device,
                dtype=dtype,
            )
        )
        if isinstance(opacity, torch.Tensor):
            opacity_value = opacity.detach().to(device=device, dtype=dtype)
            if opacity_value.ndim == 0:
                opacity_value = opacity_value.expand(count).reshape(count, 1).clone()
            elif opacity_value.shape == (count,):
                opacity_value = opacity_value[:, None]
            elif opacity_value.shape != (count, 1):
                raise ValueError("opacity must be scalar, [N], or [N,1]")
        else:
            opacity_value = torch.full(
                (count, 1), float(opacity), device=device, dtype=dtype
            )
        if not bool(torch.isfinite(opacity_value).all()):
            raise ValueError("opacity must be finite")
        if bool((opacity_value < 0.0).any() or (opacity_value > 1.0).any()):
            raise ValueError("opacity must be a probability in [0,1]")
        probability_epsilon = torch.finfo(dtype).eps
        opacity_raw = inverse_sigmoid(
            opacity_value.clamp(probability_epsilon, 1.0 - probability_epsilon)
        )
        start_value = torch.as_tensor(start, device=device, dtype=dtype).expand(count).clone()
        end_value = torch.full((count,), float("inf"), device=device, dtype=dtype)
        parent_stable_id = parent_stable_id.detach().to(device=device, dtype=torch.long)
        generation = generation.detach().to(device=device, dtype=torch.long)
        if parent_stable_id.shape != (count,) or generation.shape != (count,):
            raise ValueError("parent IDs and generations must have shape [N]")
        if root_anchor_xyz is None:
            root_anchor_xyz = xyz.clone()
        else:
            root_anchor_xyz = _rows(
                "root_anchor_xyz",
                root_anchor_xyz,
                count=count,
                tail=(3,),
                fill=0.0,
                device=device,
                dtype=dtype,
            )
        if root_anchor_scale is None:
            root_anchor_scale = self.scaling_activation(scaling).amax(dim=1)
        else:
            root_anchor_scale = _rows(
                "root_anchor_scale",
                root_anchor_scale,
                count=count,
                tail=(),
                fill=0.0,
                device=device,
                dtype=dtype,
            )
        if bool((root_anchor_scale <= 0).any()):
            raise ValueError("root anchor scale must be positive")
        old_count = self.num_gaussians
        old_parameters = self._parameter_map()
        new_values = {
            "xyz": torch.cat((self._xyz.detach(), xyz)),
            "dc": torch.cat((self.new_dc.detach(), dc)),
            "opacity": torch.cat((self._opacity.detach(), opacity_raw)),
            "scaling": torch.cat((self._scaling.detach(), scaling)),
            "rotation": torch.cat((self._rotation.detach(), rotation)),
        }
        self._install_parameter_values(new_values, optimizer=optimizer, selector=None)
        # Silence linters and make the state-preservation intent explicit.
        del old_parameters
        self._features_rest = torch.cat(
            (
                self._features_rest,
                torch.zeros(
                    (count, self._rest_channels, 3), device=device, dtype=dtype
                ),
            )
        )
        stable = torch.arange(
            self.next_stable_id,
            self.next_stable_id + count,
            device=device,
            dtype=torch.long,
        )
        self.next_stable_id += count
        self.start = torch.cat((self.start, start_value))
        self.end = torch.cat((self.end, end_value))
        self.stable_id = torch.cat((self.stable_id, stable))
        self.parent_stable_id = torch.cat((self.parent_stable_id, parent_stable_id))
        self.generation = torch.cat((self.generation, generation))
        self.birth_frame = torch.cat((self.birth_frame, start_value.long()))
        self.initial_xyz = torch.cat((self.initial_xyz, xyz.clone()))
        self.root_anchor_xyz = torch.cat((self.root_anchor_xyz, root_anchor_xyz))
        self.root_anchor_scale = torch.cat((self.root_anchor_scale, root_anchor_scale))
        self.densification_count = torch.cat(
            (
                self.densification_count,
                torch.zeros(count, device=device, dtype=torch.long),
            )
        )
        self.positive_support = torch.cat(
            (self.positive_support, torch.zeros(count, device=device, dtype=torch.long))
        )
        self.contradiction_support = torch.cat(
            (
                self.contradiction_support,
                torch.zeros(count, device=device, dtype=torch.long),
            )
        )
        self.xyz_gradient_accum = torch.cat(
            (self.xyz_gradient_accum, torch.zeros((count, 1), device=device, dtype=dtype))
        )
        self.xyz_gradient_accum_abs = torch.cat(
            (
                self.xyz_gradient_accum_abs,
                torch.zeros((count, 1), device=device, dtype=dtype),
            )
        )
        self.gradient_denom = torch.cat(
            (self.gradient_denom, torch.zeros((count, 1), device=device, dtype=dtype))
        )
        self.max_radii2d = torch.cat(
            (self.max_radii2d, torch.zeros(count, device=device, dtype=dtype))
        )
        supplied: list[dict[str, Any]]
        if metadata is None:
            supplied = [{} for _ in range(count)]
        elif isinstance(metadata, Mapping):
            if count != 1:
                raise ValueError("one metadata mapping requires one row")
            supplied = [copy.deepcopy(dict(metadata))]
        else:
            if len(metadata) != count:
                raise ValueError("metadata length must match rows")
            supplied = [copy.deepcopy(dict(row)) for row in metadata]
        for index, row in enumerate(supplied):
            row.setdefault("parent_row", None)
            row.update(
                {
                    "birth_kind": birth_kind,
                    "stable_id": int(stable[index].item()),
                    "parent_stable_id": (
                        None
                        if int(parent_stable_id[index].item()) < 0
                        else int(parent_stable_id[index].item())
                    ),
                    "generation": int(generation[index].item()),
                    "birth_frame": int(start_value[index].item()),
                    "initial_xyz": [float(v) for v in xyz[index].detach().cpu().tolist()],
                    "root_anchor_xyz": [
                        float(v)
                        for v in root_anchor_xyz[index].detach().cpu().tolist()
                    ],
                    "root_anchor_scale": float(
                        root_anchor_scale[index].detach().cpu().item()
                    ),
                }
            )
            self.metadata.append(row)
            self.observed_view_ids[int(stable[index].item())] = set()
        self.validate()
        return torch.arange(old_count, old_count + count, device=device, dtype=torch.long)

    def append_gradient_children(
        self,
        *,
        parent_rows: torch.Tensor,
        split: bool,
        timestamp: int,
        optimizer: torch.optim.Optimizer,
        trigger_scores: torch.Tensor,
        children_per_split: int = 2,
        random_seed: int = 0,
        replace_split_parent: bool = True,
    ) -> torch.Tensor:
        rows = parent_rows.detach().to(device=self._xyz.device, dtype=torch.long).flatten()
        if rows.numel() == 0:
            return rows
        if bool((rows < 0).any() or (rows >= self.num_gaussians).any()):
            raise IndexError("gradient children must reference NEW sidecar rows")
        if not bool(self.active_mask(float(timestamp))[rows].all()):
            raise ValueError("gradient children can only inherit from active NEW rows")
        scores = trigger_scores.detach().to(device=self._xyz.device, dtype=self._xyz.dtype)
        if scores.shape != rows.shape:
            raise ValueError("trigger_scores must align with parent rows")
        repeat = int(children_per_split) if split else 1
        if split:
            devices = [self._xyz.device.index or 0] if self._xyz.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(random_seed))
                child_xyz, child_scaling = fastgs_split_children(
                    self._xyz.detach()[rows],
                    self.get_scaling.detach()[rows],
                    self._rotation.detach()[rows],
                    self.scaling_inverse_activation,
                    children_per_source=repeat,
                )
        else:
            child_xyz = self._xyz.detach()[rows].clone()
            child_scaling = self._scaling.detach()[rows].clone()
        parent_ids = self.stable_id[rows].repeat(repeat)
        generations = self.generation[rows].repeat(repeat) + 1
        child_dc = self.new_dc.detach()[rows].repeat(repeat, 1, 1)
        child_opacity = self.get_opacity.detach()[rows].repeat(repeat, 1)
        child_rotation = self._rotation.detach()[rows].repeat(repeat, 1)
        child_root_anchor_xyz = self.root_anchor_xyz[rows].repeat(repeat, 1)
        child_root_anchor_scale = self.root_anchor_scale[rows].repeat(repeat)
        score_values = scores.repeat(repeat)
        parent_rows_at_birth = rows.repeat(repeat)
        metadata = [
            {
                "densification_trigger_score": float(score),
                "densification_kind": "split" if split else "clone",
                "parent_row": int(parent_row),
            }
            for score, parent_row in zip(
                score_values.detach().cpu().tolist(),
                parent_rows_at_birth.detach().cpu().tolist(),
            )
        ]
        child_rows = self._append(
            xyz=child_xyz,
            start=float(timestamp),
            scaling=child_scaling,
            opacity=child_opacity,
            dc=child_dc,
            rotation=child_rotation,
            parent_stable_id=parent_ids,
            generation=generations,
            birth_kind="gradient_densified",
            metadata=metadata,
            optimizer=optimizer,
            root_anchor_xyz=child_root_anchor_xyz,
            root_anchor_scale=child_root_anchor_scale,
        )
        if split and replace_split_parent:
            # Standard 3DGS split replaces each large source with its children.
            prune = torch.zeros(self.num_gaussians, device=self._xyz.device, dtype=torch.bool)
            prune[rows] = True
            child_ids = self.stable_id[child_rows].clone()
            self.prune_rows(prune, optimizer=optimizer, reason="split_parent_replaced")
            id_to_row = {int(value): i for i, value in enumerate(self.stable_id.tolist())}
            child_rows = torch.tensor(
                [id_to_row[int(value)] for value in child_ids.tolist()],
                device=self._xyz.device,
                dtype=torch.long,
            )
        return child_rows

    def prune_rows(
        self,
        prune_mask: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        reason: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(prune_mask, torch.Tensor) or prune_mask.dtype != torch.bool:
            raise TypeError("prune_mask must be boolean")
        prune_mask = prune_mask.to(device=self._xyz.device).flatten()
        if prune_mask.shape != (self.num_gaussians,):
            raise ValueError("prune_mask must align with NEW rows")
        if not bool(prune_mask.any()):
            return []
        keep = ~prune_mask
        removed = []
        for row in torch.nonzero(prune_mask, as_tuple=False).flatten().tolist():
            event = copy.deepcopy(self.metadata[row])
            event["prune_reason"] = reason
            removed.append(event)
            self.observed_view_ids.pop(int(self.stable_id[row].item()), None)
        values = {
            "xyz": self._xyz.detach()[keep],
            "dc": self.new_dc.detach()[keep],
            "opacity": self._opacity.detach()[keep],
            "scaling": self._scaling.detach()[keep],
            "rotation": self._rotation.detach()[keep],
        }
        self._install_parameter_values(values, optimizer=optimizer, selector=keep)
        for name in (
            "_features_rest",
            "start",
            "end",
            "stable_id",
            "parent_stable_id",
            "generation",
            "birth_frame",
            "initial_xyz",
            "root_anchor_xyz",
            "root_anchor_scale",
            "densification_count",
            "positive_support",
            "contradiction_support",
            "xyz_gradient_accum",
            "xyz_gradient_accum_abs",
            "gradient_denom",
            "max_radii2d",
        ):
            setattr(self, name, getattr(self, name)[keep])
        self.metadata = [row for row, selected in zip(self.metadata, keep.tolist()) if selected]
        self.validate()
        return removed

    def _parameter_map(self) -> dict[str, nn.Parameter]:
        return {
            "xyz": self._xyz,
            "dc": self.new_dc,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

    def _install_parameter_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        optimizer: torch.optim.Optimizer | None,
        selector: torch.Tensor | None,
    ) -> None:
        old = self._parameter_map()
        replacements = {
            name: nn.Parameter(values[name].detach().contiguous())
            for name in _PARAMETER_NAMES
        }
        self._xyz = replacements["xyz"]
        self.new_dc = replacements["dc"]
        self._opacity = replacements["opacity"]
        self._scaling = replacements["scaling"]
        self._rotation = replacements["rotation"]
        if optimizer is None:
            return
        for name in _PARAMETER_NAMES:
            old_parameter = old[name]
            new_parameter = replacements[name]
            group = next(
                (
                    group
                    for group in optimizer.param_groups
                    if group.get("name") == name
                    or any(parameter is old_parameter for parameter in group["params"])
                ),
                None,
            )
            if group is None:
                optimizer.add_param_group(
                    {"params": [new_parameter], "name": name, "lr": 0.0}
                )
            else:
                group["params"] = [new_parameter]
            previous = optimizer.state.pop(old_parameter, {})
            next_state: dict[str, Any] = {}
            old_count = int(old_parameter.shape[0])
            new_count = int(new_parameter.shape[0])
            for key, value in previous.items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    next_state[key] = copy.deepcopy(value)
                elif value.shape[0] != old_count:
                    next_state[key] = value.detach().clone()
                elif selector is not None:
                    next_state[key] = value.detach()[selector].clone()
                else:
                    extension = torch.zeros(
                        (new_count - old_count, *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    next_state[key] = torch.cat((value.detach(), extension), dim=0)
            optimizer.state[new_parameter] = next_state

    def reset_density_statistics(self) -> None:
        self.xyz_gradient_accum.zero_()
        self.xyz_gradient_accum_abs.zero_()
        self.gradient_denom.zero_()
        self.max_radii2d.zero_()

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "contract": "e4d_active_new_geometry",
            "sh_degree": self.max_sh_degree,
            "xyz": self._xyz.detach().clone(),
            "dc": self.new_dc.detach().clone(),
            "opacity": self._opacity.detach().clone(),
            "scaling": self._scaling.detach().clone(),
            "rotation": self._rotation.detach().clone(),
            "features_rest": self._features_rest.detach().clone(),
            "start": self.start.detach().clone(),
            "end": self.end.detach().clone(),
            "stable_id": self.stable_id.detach().clone(),
            "parent_stable_id": self.parent_stable_id.detach().clone(),
            "generation": self.generation.detach().clone(),
            "birth_frame": self.birth_frame.detach().clone(),
            "initial_xyz": self.initial_xyz.detach().clone(),
            "root_anchor_xyz": self.root_anchor_xyz.detach().clone(),
            "root_anchor_scale": self.root_anchor_scale.detach().clone(),
            "densification_count": self.densification_count.detach().clone(),
            "positive_support": self.positive_support.detach().clone(),
            "contradiction_support": self.contradiction_support.detach().clone(),
            "xyz_gradient_accum": self.xyz_gradient_accum.detach().clone(),
            "xyz_gradient_accum_abs": self.xyz_gradient_accum_abs.detach().clone(),
            "gradient_denom": self.gradient_denom.detach().clone(),
            "max_radii2d": self.max_radii2d.detach().clone(),
            "metadata": copy.deepcopy(self.metadata),
            "observed_view_ids": {
                int(key): sorted(values) for key, values in self.observed_view_ids.items()
            },
            "next_stable_id": int(self.next_stable_id),
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any]
    ) -> "ActiveNewGaussianModel":
        """Restore a complete E4d sidecar without touching a reference bank."""

        if checkpoint.get("contract") != "e4d_active_new_geometry":
            raise ValueError("checkpoint is not an E4d active NEW geometry state")
        xyz = checkpoint["xyz"]
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("checkpoint xyz must have shape [N,3]")
        model = cls(
            sh_degree=int(checkpoint["sh_degree"]),
            device=xyz.device,
            dtype=xyz.dtype,
        )
        model._xyz = nn.Parameter(xyz.detach().clone())
        model.new_dc = nn.Parameter(checkpoint["dc"].detach().clone())
        model._opacity = nn.Parameter(checkpoint["opacity"].detach().clone())
        model._scaling = nn.Parameter(checkpoint["scaling"].detach().clone())
        model._rotation = nn.Parameter(checkpoint["rotation"].detach().clone())
        for name in (
            "features_rest",
            "start",
            "end",
            "stable_id",
            "parent_stable_id",
            "generation",
            "birth_frame",
            "initial_xyz",
            "positive_support",
            "contradiction_support",
            "xyz_gradient_accum",
            "xyz_gradient_accum_abs",
            "gradient_denom",
            "max_radii2d",
        ):
            attribute = "_features_rest" if name == "features_rest" else name
            setattr(model, attribute, checkpoint[name].detach().clone())
        model.root_anchor_xyz = checkpoint.get(
            "root_anchor_xyz", checkpoint["initial_xyz"]
        ).detach().clone()
        model.root_anchor_scale = checkpoint.get(
            "root_anchor_scale",
            torch.exp(checkpoint["scaling"]).amax(dim=1),
        ).detach().clone()
        model.densification_count = checkpoint.get(
            "densification_count",
            torch.zeros(
                xyz.shape[0], device=xyz.device, dtype=torch.long
            ),
        ).detach().clone()
        model.metadata = copy.deepcopy(list(checkpoint["metadata"]))
        model.observed_view_ids = {
            int(key): {int(value) for value in values}
            for key, values in dict(checkpoint["observed_view_ids"]).items()
        }
        for stable_id in model.stable_id.detach().cpu().tolist():
            model.observed_view_ids.setdefault(int(stable_id), set())
        model.next_stable_id = int(checkpoint["next_stable_id"])
        model.validate()
        return model

    def validate(self) -> None:
        count = self.num_gaussians
        tensors = {
            "dc": self.new_dc,
            "opacity": self._opacity,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "features_rest": self._features_rest,
            "start": self.start,
            "end": self.end,
            "stable_id": self.stable_id,
            "parent_stable_id": self.parent_stable_id,
            "generation": self.generation,
            "birth_frame": self.birth_frame,
            "initial_xyz": self.initial_xyz,
            "root_anchor_xyz": self.root_anchor_xyz,
            "root_anchor_scale": self.root_anchor_scale,
            "densification_count": self.densification_count,
            "positive_support": self.positive_support,
            "contradiction_support": self.contradiction_support,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "xyz_gradient_accum_abs": self.xyz_gradient_accum_abs,
            "gradient_denom": self.gradient_denom,
            "max_radii2d": self.max_radii2d,
        }
        for name, tensor in tensors.items():
            if tensor.shape[0] != count:
                raise RuntimeError(f"{name} is not aligned with NEW topology")
        if len(self.metadata) != count:
            raise RuntimeError("metadata is not aligned with NEW topology")
        if count and torch.unique(self.stable_id).numel() != count:
            raise RuntimeError("stable IDs must be unique")
        valid_interval = self.start < self.end
        never_open = torch.isposinf(self.start) & torch.isposinf(self.end)
        if count and not bool(torch.all(valid_interval | never_open)):
            raise RuntimeError(
                "NEW lifespan rows must be valid intervals or NEVER_OPEN"
            )


class ActiveNewGeometryView:
    """Selected-row renderer adapter with independently detachable attributes.

    By default the selected rows are the currently active NEW rows.  An
    explicit ``row_mask`` also supports quarantined NEVER_OPEN geometry
    refinement without changing lifecycle state. ``detach_geometry=True`` is
    the learned-DC-only view; ``detach_opacity=True`` keeps pending opacity
    fixed while xyz, scale, and rotation remain trainable.
    """

    def __init__(
        self,
        model: ActiveNewGaussianModel,
        timestamp: float | torch.Tensor,
        *,
        row_mask: torch.Tensor | None = None,
        detach_geometry: bool = False,
        detach_dc: bool = False,
        detach_opacity: bool = False,
    ) -> None:
        if row_mask is None:
            attrs = model.get_active_render_attributes(timestamp)
        else:
            if row_mask.dtype != torch.bool or row_mask.ndim != 1:
                raise ValueError("row_mask must be boolean [N]")
            if row_mask.shape[0] != model.num_gaussians:
                raise ValueError("row_mask must align with the NEW seed bank")
            selected = row_mask.to(device=model._xyz.device)
            rows = torch.nonzero(selected, as_tuple=False).flatten()
            attrs = {
                "rows": rows,
                "xyz": model._xyz[selected],
                "dc": model.new_dc[selected],
                "features_rest": model._features_rest[selected],
                "opacity": model.get_opacity[selected],
                "scaling": model.get_scaling[selected],
                "rotation": model.get_rotation[selected],
            }

        def geometry(value: torch.Tensor) -> torch.Tensor:
            return value.detach() if detach_geometry else value

        def dc(value: torch.Tensor) -> torch.Tensor:
            return value.detach() if detach_dc else value

        self.global_rows = attrs["rows"]
        self.active_sh_degree = 0
        self.max_sh_degree = model.max_sh_degree
        self.scaling_activation = model.scaling_activation
        self.opacity_activation = model.opacity_activation
        self.rotation_activation = model.rotation_activation
        self.covariance_activation = model.covariance_activation
        self._xyz = geometry(attrs["xyz"])
        self._features_dc = dc(attrs["dc"])
        self._features_rest = attrs["features_rest"]
        self._opacity_value = (
            attrs["opacity"].detach()
            if detach_opacity
            else geometry(attrs["opacity"])
        )
        self._scaling_value = geometry(attrs["scaling"])
        self._rotation_value = geometry(attrs["rotation"])

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
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )


class FrozenBaseActiveNewView:
    """Frozen reference plus active NEW rows for training/evaluation probes.

    ``detach_new_geometry`` supports the E4a-compatible DC objective used by
    E4d: the current NEW DC remains differentiable, while xyz, opacity, scale,
    and rotation receive gradients only from the separate active-only geometry
    coverage render.
    """

    def __init__(
        self,
        base: Any,
        model: ActiveNewGaussianModel,
        *,
        timestamp: float | torch.Tensor,
        base_dc: torch.Tensor | None = None,
        detach_new: bool = False,
        detach_new_geometry: bool = False,
        detach_new_dc: bool = False,
    ) -> None:
        attrs = model.get_active_render_attributes(timestamp)
        self.new_global_rows = attrs["rows"]
        self.base_count = int(base.get_xyz.shape[0])
        self.active_sh_degree = 0
        self.max_sh_degree = max(base.max_sh_degree, model.max_sh_degree)
        self.scaling_activation = base.scaling_activation
        self.opacity_activation = base.opacity_activation
        self.rotation_activation = base.rotation_activation
        self.covariance_activation = base.covariance_activation

        def geometry(value: torch.Tensor) -> torch.Tensor:
            return value.detach() if detach_new or detach_new_geometry else value

        def dc(value: torch.Tensor) -> torch.Tensor:
            return value.detach() if detach_new or detach_new_dc else value

        base_dc_value = base._features_dc.detach() if base_dc is None else base_dc.detach()
        rest_channels = max(base._features_rest.shape[1], attrs["features_rest"].shape[1])

        def pad(value: torch.Tensor) -> torch.Tensor:
            if value.shape[1] == rest_channels:
                return value
            extra = torch.zeros(
                (value.shape[0], rest_channels - value.shape[1], 3),
                device=value.device,
                dtype=value.dtype,
            )
            return torch.cat((value, extra), dim=1)

        self._xyz = torch.cat((base.get_xyz.detach(), geometry(attrs["xyz"])))
        self._features_dc = torch.cat((base_dc_value, dc(attrs["dc"])))
        self._features_rest = torch.cat(
            (pad(base._features_rest.detach()), pad(attrs["features_rest"].detach()))
        )
        self._opacity_value = torch.cat(
            (base.get_opacity.detach(), geometry(attrs["opacity"]))
        )
        self._scaling_value = torch.cat(
            (base.get_scaling.detach(), geometry(attrs["scaling"]))
        )
        self._rotation_value = torch.cat(
            (base.get_rotation.detach(), geometry(attrs["rotation"]))
        )

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
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )
