"""Dynamic topology for one equal-status mutable ``R_change`` Gaussian bank.

Every row is governed by the same contract.  The binary detector decides only
whether the row is currently ACTIVE or INACTIVE.  ACTIVE rows may be optimized,
cloned, split, or pruned; INACTIVE rows are preserved exactly.  A newly created
row copies its source detector/controller/lifespan state at the topology event,
but owns independent tensors and Bayesian state afterwards.

This module deliberately has no immutable-prefix, root, or residual-bank
semantics.  Stable IDs and parent IDs are diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from utils.fastgs_topology import fastgs_split_children


_PARAMETER_BINDINGS = {
    "dc": ("change_dc", "_features_dc"),
    "xyz": ("xyz", "_xyz"),
    "features_rest": ("features_rest", "_features_rest"),
    "opacity": ("opacity", "_opacity"),
    "scaling": ("scaling", "_scaling"),
    "rotation": ("rotation", "_rotation"),
}

_LIFECYCLE_BUFFERS = (
    "state_start",
    "state_end",
    "state_valid",
    "state_status",
    "num_states",
    "current_state_index",
)

_FILTER_BUFFERS = ("p_active", "visible_observations", "last_timestamp")
_CONTROLLER_BUFFERS = ("open_support_count", "close_support_count")


@dataclass(frozen=True)
class DynamicDensityMasks:
    """Pre-mutation row masks over the topology at event entry."""

    active: torch.Tensor
    gradient_observed: torch.Tensor
    clone: torch.Tensor
    split: torch.Tensor


@dataclass(frozen=True)
class DynamicDensityResult:
    """Counts and masks for one active-only O-SCD density event."""

    initial_count: int
    final_count: int
    clone_source_count: int
    clone_child_count: int
    split_source_count: int
    split_child_count: int
    split_source_removed_count: int
    opacity_pruned_count: int
    size_pruned_count: int
    total_removed_count: int
    masks: DynamicDensityMasks


def _validate_positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


class DynamicGaussianTopologyManager:
    """Mutate parameters, optimizer, detector, and lifecycle in lockstep."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        tracker: Any,
        controller: Any,
        *,
        percent_dense: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.controller = controller
        self.percent_dense = _validate_positive_finite("percent_dense", percent_dense)
        n = int(model.current_state_index.shape[0])
        if n < 1:
            raise ValueError("dynamic Gaussian topology must start nonempty")
        device = model.current_state_index.device
        dtype = model.xyz.dtype
        self.stable_id = torch.arange(n, device=device, dtype=torch.long)
        self.parent_stable_id = torch.full((n,), -1, device=device, dtype=torch.long)
        self.generation = torch.zeros(n, device=device, dtype=torch.long)
        self.creation_timestamp = torch.full((n,), -1, device=device, dtype=torch.long)
        self.next_stable_id = n
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device, dtype=dtype)
        self.denom = torch.zeros((n, 1), device=device, dtype=dtype)
        self.max_radii2d = torch.zeros(n, device=device, dtype=dtype)
        self.lineage_events: list[dict[str, Any]] = []
        self._sync_base_density_buffers()
        self.validate()

    @property
    def count(self) -> int:
        return int(self.model.current_state_index.shape[0])

    def active_mask(self) -> torch.Tensor:
        return self.model.current_state_index >= 0

    def _sync_base_density_buffers(self) -> None:
        base = self.model.base
        base.xyz_gradient_accum = self.xyz_gradient_accum
        base.denom = self.denom
        base.max_radii2D = self.max_radii2d
        if hasattr(base, "xyz_gradient_accum_abs"):
            base.xyz_gradient_accum_abs = torch.zeros_like(self.xyz_gradient_accum)

    def _optimizer_group(self, old_parameter: nn.Parameter) -> dict[str, Any]:
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["params"][0] is old_parameter:
                return group
        raise RuntimeError("masked optimizer lost a mutable Gaussian parameter")

    def _replace_parameter(
        self,
        name: str,
        combined_value: torch.Tensor,
        keep: torch.Tensor,
        *,
        old_count: int,
    ) -> None:
        model_attribute, base_attribute = _PARAMETER_BINDINGS[name]
        old_parameter: nn.Parameter = getattr(self.model, model_attribute)
        group = self._optimizer_group(old_parameter)
        old_state = self.optimizer.state.pop(old_parameter)
        new_parameter = nn.Parameter(
            combined_value.detach()[keep], requires_grad=old_parameter.requires_grad
        )
        setattr(self.model, model_attribute, new_parameter)
        setattr(self.model.base, base_attribute, new_parameter)
        group["params"][0] = new_parameter

        combined_count = int(combined_value.shape[0])
        extension_count = combined_count - int(old_count)
        if extension_count < 0:
            raise RuntimeError("combined topology cannot be smaller before pruning")
        new_state: dict[str, Any] = {}
        for state_name, state_value in old_state.items():
            if (
                isinstance(state_value, torch.Tensor)
                and state_value.ndim >= 1
                and state_value.shape[0] == int(old_count)
            ):
                extension = torch.zeros(
                    (extension_count, *state_value.shape[1:]),
                    device=state_value.device,
                    dtype=state_value.dtype,
                )
                new_state[state_name] = torch.cat((state_value, extension), dim=0)[keep]
            else:
                new_state[state_name] = state_value
        self.optimizer.state[new_parameter] = new_state

    @staticmethod
    def _repeat_rows(value: torch.Tensor, rows: torch.Tensor, repeat: int) -> torch.Tensor:
        selected = value[rows]
        return selected.repeat((int(repeat),) + (1,) * (selected.ndim - 1))

    def _combined_child_rows(
        self,
        clone_rows: torch.Tensor,
        split_rows: torch.Tensor,
        *,
        split_children: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build clone/split parameters and source metadata in append order."""

        base = self.model.base
        clone_count = int(clone_rows.numel())
        split_source_count = int(split_rows.numel())
        split_count = split_source_count * int(split_children)
        split_xyz = base._xyz.new_empty((0, 3))
        split_scaling = base._scaling.new_empty((0, 3))
        if split_source_count:
            split_xyz, split_scaling = fastgs_split_children(
                base._xyz.detach()[split_rows],
                base.get_scaling.detach()[split_rows],
                base._rotation.detach()[split_rows],
                base.scaling_inverse_activation,
                children_per_source=int(split_children),
            )

        children: dict[str, torch.Tensor] = {}
        for name, (_model_attribute, base_attribute) in _PARAMETER_BINDINGS.items():
            value = getattr(base, base_attribute).detach()
            clone_value = value[clone_rows]
            split_value = self._repeat_rows(value, split_rows, int(split_children))
            if name == "xyz":
                split_value = split_xyz
            elif name == "scaling":
                split_value = split_scaling
            children[name] = torch.cat((clone_value, split_value), dim=0)

        source_rows = torch.cat((clone_rows, split_rows.repeat(int(split_children))))
        child_count = clone_count + split_count
        if int(source_rows.numel()) != child_count:
            raise RuntimeError("child source metadata lost topology alignment")
        child_ids = torch.arange(
            self.next_stable_id,
            self.next_stable_id + child_count,
            device=self.stable_id.device,
            dtype=torch.long,
        )
        parent_ids = self.stable_id[source_rows]
        generations = self.generation[source_rows] + 1
        return children, source_rows, child_ids, parent_ids, generations

    def add_gradient_stats(
        self,
        viewspace_points: torch.Tensor,
        radii: torch.Tensor,
        active_rows: torch.Tensor | None = None,
    ) -> int:
        """Accumulate the original O-SCD screen-space xyz gradient on ACTIVE rows."""

        if viewspace_points.grad is None:
            raise RuntimeError("view-space gradient is required for density control")
        if radii.shape != (self.count,):
            raise ValueError("radii must match the current Gaussian topology")
        active = self.active_mask() if active_rows is None else active_rows
        if active.dtype != torch.bool or active.shape != (self.count,):
            raise ValueError("active_rows must be boolean [N]")
        visible = active.to(device=radii.device) & (radii.detach() > 0)
        if not bool(visible.any()):
            return 0
        gradient = viewspace_points.grad.detach()
        if gradient.shape[0] != self.count or gradient.shape[1] < 2:
            raise ValueError("view-space gradients must have shape [N,>=2]")
        self.xyz_gradient_accum[visible] += torch.linalg.vector_norm(
            gradient[visible, :2], dim=-1, keepdim=True
        )
        self.denom[visible] += 1
        self.max_radii2d[visible] = torch.maximum(
            self.max_radii2d[visible], radii.detach()[visible]
        )
        return int(visible.sum().item())

    def reset_gradient_stats(self) -> None:
        n = self.count
        device = self.model.xyz.device
        dtype = self.model.xyz.dtype
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device, dtype=dtype)
        self.denom = torch.zeros((n, 1), device=device, dtype=dtype)
        self.max_radii2d = torch.zeros(n, device=device, dtype=dtype)
        self._sync_base_density_buffers()

    @torch.no_grad()
    def apply_active_oscd_density_control(
        self,
        *,
        timestamp: int,
        scene_extent: float,
        grad_threshold: float = 1e-3,
        min_opacity: float = 0.4,
        max_screen_size: float | None = None,
        split_children: int = 2,
    ) -> DynamicDensityResult:
        """Run active-only O-SCD clone/split plus explicit active-only pruning.

        The online O-SCD loop supplies the gradient clone/split schedule.  Its
        16-step loop does not prune; the opacity/size pruning below is the
        explicit extension requested by this ablation and uses the same
        criteria as ``GaussianModel.densify_and_prune``.
        """

        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TypeError("timestamp must be an integer")
        scene_extent = _validate_positive_finite("scene_extent", scene_extent)
        grad_threshold = _validate_positive_finite(
            "grad_threshold", grad_threshold, allow_zero=True
        )
        min_opacity = _validate_positive_finite(
            "min_opacity", min_opacity, allow_zero=True
        )
        if max_screen_size is not None:
            max_screen_size = _validate_positive_finite(
                "max_screen_size", max_screen_size
            )
        if isinstance(split_children, bool) or int(split_children) < 2:
            raise ValueError("split_children must be an integer >= 2")

        old_count = self.count
        active = self.active_mask()
        denominator = self.denom.clamp_min(1.0)
        gradients = torch.nan_to_num(self.xyz_gradient_accum / denominator)
        observed = self.denom[:, 0] > 0
        gradient_ok = (
            torch.linalg.vector_norm(gradients, dim=-1) >= float(grad_threshold)
        )
        small = self.model.base.get_scaling.max(dim=1).values <= (
            self.percent_dense * scene_extent
        )
        clone = active & observed & gradient_ok & small
        split = active & observed & gradient_ok & ~small
        clone_rows = torch.nonzero(clone, as_tuple=False).flatten()
        split_rows = torch.nonzero(split, as_tuple=False).flatten()

        children, source_rows, child_ids, parent_ids, generations = self._combined_child_rows(
            clone_rows,
            split_rows,
            split_children=int(split_children),
        )
        child_count = int(source_rows.numel())
        combined_count = old_count + child_count
        child_active = active[source_rows] if child_count else active.new_empty((0,))
        combined_active = torch.cat((active, child_active), dim=0)

        # Copy lifecycle/filter/controller state exactly at birth.  Parameter
        # and Bayesian tensors are independent storage after concatenation.
        combined_lifecycle = {
            name: torch.cat((getattr(self.model, name), getattr(self.model, name)[source_rows]), dim=0)
            for name in _LIFECYCLE_BUFFERS
        }
        combined_filter = {
            name: torch.cat((getattr(self.tracker, name), getattr(self.tracker, name)[source_rows]), dim=0)
            for name in _FILTER_BUFFERS
        }
        controller_names = tuple(
            name for name in _CONTROLLER_BUFFERS if hasattr(self.controller, name)
        )
        combined_controller = {
            name: torch.cat((getattr(self.controller, name), getattr(self.controller, name)[source_rows]), dim=0)
            for name in controller_names
        }

        combined_parameters = {
            name: torch.cat((getattr(self.model, model_attribute).detach(), children[name]), dim=0)
            for name, (model_attribute, _base_attribute) in _PARAMETER_BINDINGS.items()
        }
        combined_stable_id = torch.cat((self.stable_id, child_ids), dim=0)
        combined_parent_id = torch.cat((self.parent_stable_id, parent_ids), dim=0)
        combined_generation = torch.cat((self.generation, generations), dim=0)
        combined_creation = torch.cat(
            (
                self.creation_timestamp,
                torch.full(
                    (child_count,),
                    int(timestamp),
                    device=self.creation_timestamp.device,
                    dtype=torch.long,
                ),
            ),
            dim=0,
        )

        split_source_mask = torch.zeros(
            combined_count, device=active.device, dtype=torch.bool
        )
        split_source_mask[split_rows] = True
        opacity_prune = combined_active & (
            self.model.base.opacity_activation(combined_parameters["opacity"])
            < float(min_opacity)
        ).squeeze(-1)
        size_prune = torch.zeros_like(opacity_prune)
        if max_screen_size is not None:
            combined_radii = torch.cat(
                (
                    self.max_radii2d,
                    torch.zeros(child_count, device=active.device, dtype=self.max_radii2d.dtype),
                )
            )
            screen_large = combined_radii > float(max_screen_size)
            world_large = (
                self.model.base.scaling_activation(combined_parameters["scaling"])
                .max(dim=1)
                .values
                > 0.1 * scene_extent
            )
            size_prune = combined_active & (screen_large | world_large)
        remove = split_source_mask | opacity_prune | size_prune
        keep = ~remove
        if not bool(keep.any()):
            raise RuntimeError("active-only density control would remove every Gaussian")

        # Record child creation before deletion, including children immediately
        # rejected by the source opacity/size criteria.
        clone_child_count = int(clone_rows.numel())
        for position, child_id in enumerate(child_ids.detach().cpu().tolist()):
            kind = "clone" if position < clone_child_count else "split"
            self.lineage_events.append(
                {
                    "action": "CREATE",
                    "kind": kind,
                    "stable_id": int(child_id),
                    "parent_stable_id": int(parent_ids[position].item()),
                    "timestamp": int(timestamp),
                    "generation": int(generations[position].item()),
                }
            )
        for row in torch.nonzero(remove, as_tuple=False).flatten().tolist():
            reason = "split_source"
            if bool(opacity_prune[row]):
                reason = "low_opacity"
            elif bool(size_prune[row]):
                reason = "oversized"
            self.lineage_events.append(
                {
                    "action": "DELETE",
                    "reason": reason,
                    "stable_id": int(combined_stable_id[row].item()),
                    "timestamp": int(timestamp),
                }
            )

        for name, combined in combined_parameters.items():
            self._replace_parameter(name, combined, keep, old_count=old_count)
        for name, combined in combined_lifecycle.items():
            setattr(self.model, name, combined[keep])
        for name, combined in combined_filter.items():
            setattr(self.tracker, name, combined[keep])
        for name, combined in combined_controller.items():
            setattr(self.controller, name, combined[keep])
        self.stable_id = combined_stable_id[keep]
        self.parent_stable_id = combined_parent_id[keep]
        self.generation = combined_generation[keep]
        self.creation_timestamp = combined_creation[keep]
        self.next_stable_id += child_count
        self.reset_gradient_stats()
        self.validate()

        result = DynamicDensityResult(
            initial_count=old_count,
            final_count=self.count,
            clone_source_count=int(clone.sum().item()),
            clone_child_count=int(clone.sum().item()),
            split_source_count=int(split.sum().item()),
            split_child_count=int(split.sum().item()) * int(split_children),
            split_source_removed_count=int(split_source_mask.sum().item()),
            opacity_pruned_count=int(opacity_prune.sum().item()),
            size_pruned_count=int(size_prune.sum().item()),
            total_removed_count=int(remove.sum().item()),
            masks=DynamicDensityMasks(
                active=active.detach().clone(),
                gradient_observed=observed.detach().clone(),
                clone=clone.detach().clone(),
                split=split.detach().clone(),
            ),
        )
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_stable_id": int(self.next_stable_id),
            "stable_id": self.stable_id.clone(),
            "parent_stable_id": self.parent_stable_id.clone(),
            "generation": self.generation.clone(),
            "creation_timestamp": self.creation_timestamp.clone(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone(),
            "denom": self.denom.clone(),
            "max_radii2d": self.max_radii2d.clone(),
            "lineage_events": list(self.lineage_events),
        }

    def validate(self) -> bool:
        n = self.count
        base = self.model.base
        for name, (model_attribute, base_attribute) in _PARAMETER_BINDINGS.items():
            parameter = getattr(self.model, model_attribute)
            if not isinstance(parameter, nn.Parameter) or parameter.shape[0] != n:
                raise RuntimeError(f"mutable parameter {name} lost topology alignment")
            if getattr(base, base_attribute) is not parameter:
                raise RuntimeError(f"model/base parameter alias for {name} was broken")
            group = self._optimizer_group(parameter)
            if group["params"][0] is not parameter:
                raise RuntimeError(f"optimizer parameter alias for {name} was broken")
            for state_name, state_value in self.optimizer.state[parameter].items():
                if isinstance(state_value, torch.Tensor) and state_value.ndim >= 1:
                    if state_value.shape[0] != n:
                        raise RuntimeError(
                            f"optimizer state {name}.{state_name} lost topology alignment"
                        )
        for name in _LIFECYCLE_BUFFERS:
            if getattr(self.model, name).shape[0] != n:
                raise RuntimeError(f"lifecycle buffer {name} lost topology alignment")
        for name in _FILTER_BUFFERS:
            if getattr(self.tracker, name).shape != (n,):
                raise RuntimeError(f"filter buffer {name} lost topology alignment")
        for name in _CONTROLLER_BUFFERS:
            if hasattr(self.controller, name) and getattr(self.controller, name).shape != (n,):
                raise RuntimeError(f"controller buffer {name} lost topology alignment")
        for name in (
            "stable_id",
            "parent_stable_id",
            "generation",
            "creation_timestamp",
            "max_radii2d",
        ):
            if getattr(self, name).shape != (n,):
                raise RuntimeError(f"manager buffer {name} lost topology alignment")
        for name in ("xyz_gradient_accum", "denom"):
            if getattr(self, name).shape != (n, 1):
                raise RuntimeError(f"manager buffer {name} lost topology alignment")
        if torch.unique(self.stable_id).numel() != n:
            raise RuntimeError("stable Gaussian IDs must remain unique")
        if not self.model.validate_lifecycle():
            raise RuntimeError("lifespan lifecycle validation failed")
        return True
