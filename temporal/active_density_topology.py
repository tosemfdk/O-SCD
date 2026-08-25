"""Topology-safe density control for active temporal Gaussian episodes.

The Bayesian detector remains indexed by the immutable reference prefix.  This
module grows a separate render-anchor topology by appending residual children
after that prefix, while resizing the temporal ``[N, S, ...]`` sidecar and its
row/slot Adam state in lockstep.  Reference-prefix rows are never removed.

Every residual child records one immutable reference root and one temporal
episode slot.  It is therefore active only for that episode, closes with its
root, and cannot silently reappear when the root later opens a new slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from utils.fastgs_topology import fastgs_split_children


_PARAMETER_ATTRIBUTES = {
    "dc": "state_change_dc",
    "xyz": "state_xyz_delta",
    "opacity": "state_opacity_delta",
    "scaling": "state_scaling_delta",
    "rotation": "state_rotation_delta",
}

_BASE_ATTRIBUTES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)


@dataclass(frozen=True)
class TemporalDensityScore:
    """K-view alpha-T cue support for the current active topology."""

    view_indices: tuple[int, ...]
    positive_mass: torch.Tensor
    negative_mass: torch.Tensor
    total_mass: torch.Tensor
    importance_score: torch.Tensor
    visible_view_count: torch.Tensor
    support_view_count: torch.Tensor
    change_ratio: torch.Tensor


@dataclass(frozen=True)
class TemporalDensityResult:
    """One topology mutation and its pre-mutation decision masks."""

    initial_count: int
    final_count: int
    importance_count: int
    clone_count: int
    split_source_count: int
    split_child_count: int
    vcp_pruned_count: int
    split_residual_removed_count: int
    removed_count: int
    clone_mask: torch.Tensor
    split_mask: torch.Tensor
    prune_mask: torch.Tensor


def _as_bool_rows(mask: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError("row mask must be a one-dimensional boolean tensor")
    if mask.shape[0] != int(size):
        raise ValueError(f"row mask must have shape ({int(size)},)")
    return mask.to(device=device)


def aggregate_temporal_view_evidence(
    positive_masses: Sequence[torch.Tensor],
    negative_masses: Sequence[torch.Tensor],
    eligible_masks: Sequence[torch.Tensor],
    *,
    min_mass: float,
    support_threshold: float = 0.5,
    eps: float = 1e-8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Aggregate soft masses and discrete K-view support diagnostics.

    The FastGS-style importance retains the source ``sum / K`` denominator.
    ``eligible_masks`` prevents views earlier than a row's current lifespan (or
    residual creation) from contributing to that row.
    """

    if not positive_masses or not (
        len(positive_masses) == len(negative_masses) == len(eligible_masks)
    ):
        raise ValueError("positive, negative, and eligibility sequences must align")
    if not math.isfinite(float(min_mass)) or float(min_mass) < 0.0:
        raise ValueError("min_mass must be finite and nonnegative")
    if not 0.0 <= float(support_threshold) <= 1.0:
        raise ValueError("support_threshold must be in [0,1]")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    first = positive_masses[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 1 or not torch.is_floating_point(first):
        raise TypeError("evidence masses must be floating [N] tensors")
    positive_sum = torch.zeros_like(first)
    negative_sum = torch.zeros_like(first)
    visible_count = torch.zeros(first.shape, device=first.device, dtype=torch.long)
    support_count = torch.zeros_like(visible_count)
    for positive, negative, eligible in zip(
        positive_masses, negative_masses, eligible_masks
    ):
        if (
            positive.shape != first.shape
            or negative.shape != first.shape
            or positive.device != first.device
            or negative.device != first.device
            or positive.dtype != first.dtype
            or negative.dtype != first.dtype
        ):
            raise ValueError("all evidence tensors must share shape/device/dtype")
        eligible = _as_bool_rows(eligible, first.shape[0], first.device)
        if not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
            raise ValueError("evidence masses must be finite")
        if bool((positive < 0).any() or (negative < 0).any()):
            raise ValueError("evidence masses must be nonnegative")
        total = positive + negative
        observed = eligible & (total >= float(min_mass)) & (total > 0)
        ratio = positive / (total + float(eps))
        positive_sum = positive_sum + torch.where(eligible, positive, torch.zeros_like(positive))
        negative_sum = negative_sum + torch.where(eligible, negative, torch.zeros_like(negative))
        visible_count = visible_count + observed.long()
        support_count = support_count + (observed & (ratio >= float(support_threshold))).long()

    total_sum = positive_sum + negative_sum
    importance = positive_sum / float(len(positive_masses))
    ratio = positive_sum / (total_sum + float(eps))
    return (
        positive_sum,
        negative_sum,
        total_sum,
        importance,
        visible_count,
        support_count,
        ratio,
    )


class TemporalTopologyManager:
    """Resize a temporal geometry model and render anchor in exact lockstep."""

    def __init__(self, model: Any, optimizer: Any, immutable_count: int) -> None:
        self.model = model
        self.optimizer = optimizer
        self.immutable_count = int(immutable_count)
        n = int(model.state_valid.shape[0])
        if self.immutable_count < 1 or n != self.immutable_count:
            raise ValueError("manager must start with exactly the immutable prefix")
        device = model.state_valid.device
        dtype = model.state_change_dc.dtype
        self.stable_id = torch.arange(n, device=device, dtype=torch.long)
        self.root_index = torch.arange(n, device=device, dtype=torch.long)
        self.episode_slot = torch.full((n,), -1, device=device, dtype=torch.long)
        self.creation_timestamp = torch.full((n,), -1, device=device, dtype=torch.long)
        self.generation = torch.zeros(n, device=device, dtype=torch.long)
        self.is_residual = torch.zeros(n, device=device, dtype=torch.bool)
        self.parent_stable_id = torch.full((n,), -1, device=device, dtype=torch.long)
        self.next_stable_id = n
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device, dtype=dtype)
        self.xyz_gradient_accum_abs = torch.zeros((n, 1), device=device, dtype=dtype)
        self.denom = torch.zeros((n, 1), device=device, dtype=dtype)
        self.max_radii2d = torch.zeros(n, device=device, dtype=dtype)
        self.lineage_events: list[dict[str, Any]] = []
        self._closed_snapshots: list[dict[str, Any]] = []
        self.validate()

    @property
    def count(self) -> int:
        return int(self.model.state_valid.shape[0])

    def active_mask(self) -> torch.Tensor:
        return self.model.current_state_index >= 0

    def current_episode_start(self) -> torch.Tensor:
        indices = self.model.current_state_index
        rows = torch.arange(self.count, device=indices.device)
        starts = torch.full(
            (self.count,), float("inf"), device=indices.device, dtype=self.model.state_start.dtype
        )
        active = indices >= 0
        starts[active] = self.model.state_start[rows[active], indices[active]]
        return starts

    def _optimizer_group(self, old_parameter: nn.Parameter) -> dict[str, Any]:
        for group in self.optimizer.param_groups:
            if group["params"][0] is old_parameter:
                return group
        raise RuntimeError("temporal optimizer lost a model parameter")

    def _replace_parameter_rows(
        self,
        name: str,
        new_value: torch.Tensor,
        row_selector: torch.Tensor | None,
    ) -> None:
        attribute = _PARAMETER_ATTRIBUTES[name]
        old_parameter: nn.Parameter = getattr(self.model, attribute)
        group = self._optimizer_group(old_parameter)
        old_state = self.optimizer.state.pop(old_parameter)
        new_parameter = nn.Parameter(
            new_value.detach(), requires_grad=old_parameter.requires_grad
        )
        setattr(self.model, attribute, new_parameter)
        group["params"][0] = new_parameter
        old_n = int(old_parameter.shape[0])
        new_state: dict[str, Any] = {}
        for state_name, state_value in old_state.items():
            if not isinstance(state_value, torch.Tensor) or state_value.ndim == 0:
                new_state[state_name] = state_value
                continue
            if state_value.shape[0] != old_n:
                new_state[state_name] = state_value
                continue
            if row_selector is None:
                extension_shape = (new_parameter.shape[0] - old_n, *state_value.shape[1:])
                extension = torch.zeros(
                    extension_shape, device=state_value.device, dtype=state_value.dtype
                )
                new_state[state_name] = torch.cat((state_value, extension), dim=0)
            else:
                new_state[state_name] = state_value[row_selector]
        self.optimizer.state[new_parameter] = new_state

    @staticmethod
    def _replace_base_rows(base: Any, values: Mapping[str, torch.Tensor]) -> None:
        for attribute in _BASE_ATTRIBUTES:
            old = getattr(base, attribute)
            value = values[attribute]
            setattr(
                base,
                attribute,
                nn.Parameter(value.detach(), requires_grad=False),
            )
        base.max_radii2D = torch.zeros(
            values["_xyz"].shape[0],
            device=values["_xyz"].device,
            dtype=values["_xyz"].dtype,
        )

    def _append_rows(
        self,
        *,
        base_rows: Mapping[str, torch.Tensor],
        state_rows: Mapping[str, torch.Tensor],
        lifecycle_rows: Mapping[str, torch.Tensor],
        root_index: torch.Tensor,
        episode_slot: torch.Tensor,
        creation_timestamp: torch.Tensor,
        generation: torch.Tensor,
        parent_stable_id: torch.Tensor,
        kind: str,
    ) -> torch.Tensor:
        count = int(root_index.numel())
        if count == 0:
            return torch.empty(0, device=self.stable_id.device, dtype=torch.long)
        old_n = self.count
        device = self.stable_id.device
        metadata = {
            "episode_slot": episode_slot,
            "creation_timestamp": creation_timestamp,
            "generation": generation,
            "parent_stable_id": parent_stable_id,
        }
        for key, value in metadata.items():
            if value.shape != (count,) or value.device != device:
                raise ValueError(f"{key} must be a [{count}] tensor on the model device")
        base = self.model.base
        new_base = {
            attribute: torch.cat((getattr(base, attribute).detach(), base_rows[attribute].detach()), dim=0)
            for attribute in _BASE_ATTRIBUTES
        }
        self._replace_base_rows(base, new_base)
        for name, attribute in _PARAMETER_ATTRIBUTES.items():
            old = getattr(self.model, attribute)
            value = torch.cat((old.detach(), state_rows[name].detach()), dim=0)
            self._replace_parameter_rows(name, value, row_selector=None)
        for attribute in ("state_start", "state_end", "state_valid", "state_status"):
            setattr(
                self.model,
                attribute,
                torch.cat((getattr(self.model, attribute), lifecycle_rows[attribute]), dim=0),
            )
        self.model.num_states = torch.cat((self.model.num_states, lifecycle_rows["num_states"]), dim=0)
        self.model.current_state_index = torch.cat(
            (self.model.current_state_index, lifecycle_rows["current_state_index"]), dim=0
        )
        new_ids = torch.arange(
            self.next_stable_id,
            self.next_stable_id + count,
            device=device,
            dtype=torch.long,
        )
        self.next_stable_id += count
        self.stable_id = torch.cat((self.stable_id, new_ids))
        self.root_index = torch.cat((self.root_index, root_index.long()))
        self.episode_slot = torch.cat((self.episode_slot, episode_slot.long()))
        self.creation_timestamp = torch.cat(
            (self.creation_timestamp, creation_timestamp.long())
        )
        self.generation = torch.cat((self.generation, generation.long()))
        self.parent_stable_id = torch.cat(
            (self.parent_stable_id, parent_stable_id.long())
        )
        self.is_residual = torch.cat(
            (self.is_residual, torch.ones(count, device=device, dtype=torch.bool))
        )
        dtype = self.xyz_gradient_accum.dtype
        self.xyz_gradient_accum = torch.cat(
            (self.xyz_gradient_accum, torch.zeros((count, 1), device=device, dtype=dtype))
        )
        self.xyz_gradient_accum_abs = torch.cat(
            (self.xyz_gradient_accum_abs, torch.zeros((count, 1), device=device, dtype=dtype))
        )
        self.denom = torch.cat(
            (self.denom, torch.zeros((count, 1), device=device, dtype=dtype))
        )
        self.max_radii2d = torch.cat(
            (self.max_radii2d, torch.zeros(count, device=device, dtype=dtype))
        )
        for child_id, parent_id, root, slot, created, child_generation in zip(
            new_ids.detach().cpu().tolist(),
            parent_stable_id.detach().cpu().tolist(),
            root_index.detach().cpu().tolist(),
            episode_slot.detach().cpu().tolist(),
            creation_timestamp.detach().cpu().tolist(),
            generation.detach().cpu().tolist(),
        ):
            self.lineage_events.append(
                {
                    "action": "CREATE",
                    "kind": kind,
                    "stable_id": int(child_id),
                    "parent_stable_id": int(parent_id),
                    "root_index": int(root),
                    "episode_slot": int(slot),
                    "timestamp": int(created),
                    "generation": int(child_generation),
                }
            )
        rows = torch.arange(old_n, old_n + count, device=device, dtype=torch.long)
        self.validate()
        return rows

    def _capture_sources(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = rows.long().flatten()
        slots = self.model.current_state_index[rows]
        if bool((slots < 0).any()):
            raise RuntimeError("densification source rows must be OPEN")
        return {
            "rows": rows,
            "slots": slots,
            "stable_id": self.stable_id[rows],
            "root_index": self.root_index[rows],
            "generation": self.generation[rows],
            "dc": self.model.state_change_dc[rows, slots].detach(),
            "xyz": (self.model.base._xyz[rows] + self.model.state_xyz_delta[rows, slots]).detach(),
            "opacity_raw": (self.model.base._opacity[rows] + self.model.state_opacity_delta[rows, slots]).detach(),
            "scaling_raw": (self.model.base._scaling[rows] + self.model.state_scaling_delta[rows, slots]).detach(),
            "rotation_raw": (self.model.base._rotation[rows] + self.model.state_rotation_delta[rows, slots]).detach(),
            "features_rest": self.model.base._features_rest[rows].detach(),
        }

    def _child_payload(
        self,
        source: Mapping[str, torch.Tensor],
        *,
        timestamp: int,
        split: bool,
        split_children: int = 2,
    ) -> dict[str, Any]:
        source_count = int(source["rows"].numel())
        repeat = int(split_children) if split else 1
        if source_count == 0:
            return {"count": 0}
        xyz = source["xyz"]
        scaling_raw = source["scaling_raw"]
        rotation_raw = source["rotation_raw"]
        if split:
            xyz, scaling_raw = fastgs_split_children(
                xyz,
                self.model.base.scaling_activation(scaling_raw),
                rotation_raw,
                self.model.base.scaling_inverse_activation,
                children_per_source=repeat,
            )
        else:
            xyz = xyz.clone()
            scaling_raw = scaling_raw.clone()
        slots = source["slots"].repeat(repeat)
        count = source_count * repeat
        device = xyz.device
        dtype = xyz.dtype
        max_states = int(self.model.max_states)

        def zeros(*tail: int) -> torch.Tensor:
            return torch.zeros((count, max_states, *tail), device=device, dtype=dtype)

        dc_state = zeros(1, 3)
        xyz_state = zeros(3)
        opacity_state = zeros(1)
        scaling_state = zeros(3)
        rotation_state = zeros(4)
        child_dc = source["dc"].repeat(repeat, 1, 1)
        child_rows = torch.arange(count, device=device)
        dc_state[child_rows, slots] = child_dc
        state_start = torch.zeros((count, max_states), device=device, dtype=dtype)
        state_end = torch.full((count, max_states), float("inf"), device=device, dtype=dtype)
        state_valid = torch.zeros((count, max_states), device=device, dtype=torch.bool)
        state_status = torch.zeros((count, max_states), device=device, dtype=torch.int8)
        state_start[child_rows, slots] = float(timestamp)
        state_valid[child_rows, slots] = True
        state_status[child_rows, slots] = 1
        base_rows = {
            "_xyz": xyz,
            "_features_dc": torch.zeros_like(child_dc),
            "_features_rest": source["features_rest"].repeat(repeat, 1, 1),
            "_opacity": source["opacity_raw"].repeat(repeat, 1),
            "_scaling": scaling_raw,
            "_rotation": source["rotation_raw"].repeat(repeat, 1),
        }
        return {
            "count": count,
            "base_rows": base_rows,
            "state_rows": {
                "dc": dc_state,
                "xyz": xyz_state,
                "opacity": opacity_state,
                "scaling": scaling_state,
                "rotation": rotation_state,
            },
            "lifecycle_rows": {
                "state_start": state_start,
                "state_end": state_end,
                "state_valid": state_valid,
                "state_status": state_status,
                "num_states": torch.ones(count, device=device, dtype=torch.long),
                "current_state_index": slots,
            },
            "root_index": source["root_index"].repeat(repeat),
            "episode_slot": slots,
            "creation_timestamp": torch.full(
                (count,), int(timestamp), device=device, dtype=torch.long
            ),
            "generation": source["generation"].repeat(repeat) + 1,
            "parent_stable_id": source["stable_id"].repeat(repeat),
        }

    def _append_payload(self, payload: Mapping[str, Any], kind: str) -> torch.Tensor:
        if int(payload.get("count", 0)) == 0:
            return torch.empty(0, device=self.stable_id.device, dtype=torch.long)
        return self._append_rows(
            base_rows=payload["base_rows"],
            state_rows=payload["state_rows"],
            lifecycle_rows=payload["lifecycle_rows"],
            root_index=payload["root_index"],
            episode_slot=payload["episode_slot"],
            creation_timestamp=payload["creation_timestamp"],
            generation=payload["generation"],
            parent_stable_id=payload["parent_stable_id"],
            kind=kind,
        )

    def _prune_rows(
        self,
        prune_mask: torch.Tensor,
        *,
        timestamp: int,
        reason: str,
        reason_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> int:
        prune_mask = _as_bool_rows(prune_mask, self.count, self.stable_id.device)
        if bool(prune_mask[: self.immutable_count].any()):
            raise RuntimeError("immutable reference-prefix rows cannot be pruned")
        count = int(prune_mask.sum().item())
        if count == 0:
            return 0
        selected_rows = torch.nonzero(prune_mask, as_tuple=False).flatten()
        ids = self.stable_id[selected_rows].detach().cpu().tolist()
        roots = self.root_index[selected_rows].detach().cpu().tolist()
        slots = self.episode_slot[selected_rows].detach().cpu().tolist()
        normalized_reasons: dict[str, torch.Tensor] = {}
        if reason_masks is not None:
            normalized_reasons = {
                label: _as_bool_rows(mask, self.count, self.stable_id.device)
                for label, mask in reason_masks.items()
            }
        for row, stable_id, root, slot in zip(
            selected_rows.detach().cpu().tolist(), ids, roots, slots
        ):
            row_reason = reason
            for label, mask in normalized_reasons.items():
                if bool(mask[int(row)].item()):
                    row_reason = label
                    break
            self.lineage_events.append(
                {
                    "action": "DELETE",
                    "reason": row_reason,
                    "stable_id": int(stable_id),
                    "root_index": int(root),
                    "episode_slot": int(slot),
                    "timestamp": int(timestamp),
                }
            )
        keep = ~prune_mask
        old_n = self.count
        base = self.model.base
        self._replace_base_rows(
            base,
            {attribute: getattr(base, attribute).detach()[keep] for attribute in _BASE_ATTRIBUTES},
        )
        for name, attribute in _PARAMETER_ATTRIBUTES.items():
            value = getattr(self.model, attribute).detach()[keep]
            self._replace_parameter_rows(name, value, row_selector=keep)
        for attribute in ("state_start", "state_end", "state_valid", "state_status"):
            setattr(self.model, attribute, getattr(self.model, attribute)[keep])
        self.model.num_states = self.model.num_states[keep]
        self.model.current_state_index = self.model.current_state_index[keep]
        for attribute in (
            "stable_id",
            "root_index",
            "episode_slot",
            "creation_timestamp",
            "generation",
            "is_residual",
            "parent_stable_id",
            "xyz_gradient_accum",
            "xyz_gradient_accum_abs",
            "denom",
            "max_radii2d",
        ):
            value = getattr(self, attribute)
            if value.shape[0] != old_n:
                raise RuntimeError(f"manager tensor {attribute} lost topology alignment")
            setattr(self, attribute, value[keep])
        self.validate()
        return count

    def add_gradient_stats(
        self,
        viewspace_points: torch.Tensor,
        radii: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        if viewspace_points.grad is None:
            raise RuntimeError("view-space gradient is required for density statistics")
        active_mask = _as_bool_rows(active_mask, self.count, self.stable_id.device)
        if radii.shape != (self.count,):
            raise ValueError("radii must match current topology")
        visible = active_mask & (radii.detach() > 0)
        if not bool(visible.any()):
            return
        grad = viewspace_points.grad.detach()
        self.xyz_gradient_accum[visible] += torch.linalg.vector_norm(
            grad[visible, :2], dim=-1, keepdim=True
        )
        self.xyz_gradient_accum_abs[visible] += torch.linalg.vector_norm(
            grad[visible, 2:], dim=-1, keepdim=True
        )
        self.denom[visible] += 1
        self.max_radii2d[visible] = torch.maximum(
            self.max_radii2d[visible], radii.detach()[visible]
        )

    def reset_gradient_stats(self) -> None:
        n = self.count
        device = self.stable_id.device
        dtype = self.model.state_change_dc.dtype
        self.xyz_gradient_accum = torch.zeros((n, 1), device=device, dtype=dtype)
        self.xyz_gradient_accum_abs = torch.zeros((n, 1), device=device, dtype=dtype)
        self.denom = torch.zeros((n, 1), device=device, dtype=dtype)
        self.max_radii2d = torch.zeros(n, device=device, dtype=dtype)

    def _capture_closed_snapshot(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        if rows.numel() == 0:
            return
        snapshot: dict[str, Any] = {
            "stable_id": self.stable_id[rows].detach().cpu().clone(),
            "slots": slots.detach().cpu().clone(),
            "parameters": {},
            "optimizer": {},
        }
        for name, parameter in self.model.state_parameter_items():
            snapshot["parameters"][name] = parameter.detach()[rows, slots].cpu().clone()
            state = self.optimizer.state[parameter]
            snapshot["optimizer"][name] = {
                state_name: state_value.detach()[rows, slots].cpu().clone()
                for state_name, state_value in state.items()
                if isinstance(state_value, torch.Tensor)
                and state_value.ndim >= 2
                and tuple(state_value.shape[:2]) == tuple(parameter.shape[:2])
            }
        self._closed_snapshots.append(snapshot)

    @torch.no_grad()
    def close_descendants(
        self,
        root_rows: torch.Tensor,
        root_slots: torch.Tensor,
        *,
        timestamp: int,
    ) -> int:
        root_rows = root_rows.long().flatten().to(self.root_index.device)
        root_slots = root_slots.long().flatten().to(self.root_index.device)
        if root_rows.shape != root_slots.shape:
            raise ValueError("root rows and slots must align")
        selected = torch.zeros(self.count, device=self.root_index.device, dtype=torch.bool)
        for root, slot in zip(root_rows.tolist(), root_slots.tolist()):
            selected |= (
                self.is_residual
                & (self.root_index == int(root))
                & (self.episode_slot == int(slot))
                & (self.model.current_state_index >= 0)
            )
        rows = torch.nonzero(selected, as_tuple=False).flatten()
        if rows.numel() == 0:
            return 0
        slots = self.model.current_state_index[rows].clone()
        self._capture_closed_snapshot(rows, slots)
        self.model.close_rows(rows, timestamp)
        self.validate()
        return int(rows.numel())

    @torch.no_grad()
    def apply_density_control(
        self,
        score: TemporalDensityScore,
        *,
        timestamp: int,
        scene_extent: float,
        importance_threshold: float = 5.0,
        grad_threshold: float = 2e-4,
        grad_abs_threshold: float = 1.2e-3,
        dense_fraction: float = 1e-3,
        min_densify_views: int = 3,
        min_prune_views: int = 8,
        max_prune_support_views: int = 2,
        prune_grace_frames: int = 10,
    ) -> TemporalDensityResult:
        n = self.count
        for name, value in (
            ("importance_score", score.importance_score),
            ("visible_view_count", score.visible_view_count),
            ("support_view_count", score.support_view_count),
        ):
            if value.shape != (n,):
                raise ValueError(f"{name} must match current topology")
        if float(scene_extent) <= 0 or float(dense_fraction) <= 0:
            raise ValueError("scene_extent and dense_fraction must be positive")
        if min(min_densify_views, min_prune_views, prune_grace_frames) < 0:
            raise ValueError("view and grace thresholds must be nonnegative")
        active = self.active_mask()
        denom = self.denom.clamp_min(1.0)
        gradients = torch.nan_to_num(self.xyz_gradient_accum / denom)
        gradients_abs = torch.nan_to_num(self.xyz_gradient_accum_abs / denom)
        small = self.model.get_active_render_attributes(float(timestamp))["scaling"].max(dim=1).values <= (
            float(dense_fraction) * float(scene_extent)
        )
        sufficiently_seen = score.visible_view_count >= int(min_densify_views)
        important = (
            active
            & sufficiently_seen
            & (score.importance_score > float(importance_threshold))
        )
        clone = (
            important
            & small
            & (torch.linalg.vector_norm(gradients, dim=-1) >= float(grad_threshold))
        )
        split = (
            important
            & ~small
            & (torch.linalg.vector_norm(gradients_abs, dim=-1) >= float(grad_abs_threshold))
        )
        age = int(timestamp) - self.creation_timestamp
        prune = (
            active
            & self.is_residual
            & (age >= int(prune_grace_frames))
            & (score.visible_view_count >= int(min_prune_views))
            & (score.support_view_count <= int(max_prune_support_views))
        )
        clone &= ~prune
        split &= ~prune
        clone_rows = torch.nonzero(clone, as_tuple=False).flatten()
        split_rows = torch.nonzero(split, as_tuple=False).flatten()
        clone_source = self._capture_sources(clone_rows)
        split_source = self._capture_sources(split_rows)
        clone_payload = self._child_payload(clone_source, timestamp=timestamp, split=False)
        split_payload = self._child_payload(split_source, timestamp=timestamp, split=True)
        split_residual = split & self.is_residual
        remove = prune | split_residual
        vcp_pruned_count = int(prune.sum().item())
        split_residual_removed_count = int(split_residual.sum().item())
        removed_count = self._prune_rows(
            remove,
            timestamp=timestamp,
            reason="density_remove",
            reason_masks={
                "cue_vcp": prune,
                "fastgs_split_source": split_residual,
            },
        )
        clone_child_rows = self._append_payload(clone_payload, "clone")
        split_child_rows = self._append_payload(split_payload, "split")
        self.reset_gradient_stats()
        result = TemporalDensityResult(
            initial_count=n,
            final_count=self.count,
            importance_count=int(important.sum().item()),
            clone_count=int(clone_child_rows.numel()),
            split_source_count=int(split_rows.numel()),
            split_child_count=int(split_child_rows.numel()),
            vcp_pruned_count=vcp_pruned_count,
            split_residual_removed_count=split_residual_removed_count,
            removed_count=int(removed_count),
            clone_mask=clone.detach().clone(),
            split_mask=split.detach().clone(),
            prune_mask=prune.detach().clone(),
        )
        self.validate()
        return result

    @torch.no_grad()
    def verify_closed_residuals(self) -> dict[str, Any]:
        maximum = 0.0
        missing = 0
        checked = 0
        per_state: dict[str, float] = {}
        index_by_id = {
            int(stable_id): index
            for index, stable_id in enumerate(self.stable_id.detach().cpu().tolist())
        }
        parameters = dict(self.model.state_parameter_items())
        for snapshot in self._closed_snapshots:
            stable_ids = snapshot["stable_id"].tolist()
            slots_cpu = snapshot["slots"]
            positions = [index_by_id.get(int(value), -1) for value in stable_ids]
            if any(position < 0 for position in positions):
                missing += sum(position < 0 for position in positions)
                continue
            rows = torch.tensor(positions, device=self.stable_id.device, dtype=torch.long)
            slots = slots_cpu.to(device=self.stable_id.device)
            checked += len(positions)
            for name, expected in snapshot["parameters"].items():
                difference = float(
                    (parameters[name].detach()[rows, slots].cpu() - expected).abs().max().item()
                ) if expected.numel() else 0.0
                per_state[f"{name}.parameter"] = max(
                    per_state.get(f"{name}.parameter", 0.0), difference
                )
                maximum = max(maximum, difference)
            for name, values in snapshot["optimizer"].items():
                state = self.optimizer.state[parameters[name]]
                for state_name, expected in values.items():
                    difference = float(
                        (state[state_name].detach()[rows, slots].cpu() - expected).abs().max().item()
                    ) if expected.numel() else 0.0
                    key = f"{name}.{state_name}"
                    per_state[key] = max(per_state.get(key, 0.0), difference)
                    maximum = max(maximum, difference)
        return {
            "passed": maximum == 0.0 and missing == 0,
            "max_abs": float(maximum),
            "audited_pair_count": int(checked),
            "missing_stable_ids": int(missing),
            "per_parameter_and_optimizer_state_max_abs": per_state,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "immutable_count": self.immutable_count,
            "next_stable_id": self.next_stable_id,
            "stable_id": self.stable_id,
            "root_index": self.root_index,
            "episode_slot": self.episode_slot,
            "creation_timestamp": self.creation_timestamp,
            "generation": self.generation,
            "is_residual": self.is_residual,
            "parent_stable_id": self.parent_stable_id,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "xyz_gradient_accum_abs": self.xyz_gradient_accum_abs,
            "denom": self.denom,
            "max_radii2d": self.max_radii2d,
            "lineage_events": list(self.lineage_events),
        }

    def validate(self) -> bool:
        n = self.count
        base = self.model.base
        for attribute in _BASE_ATTRIBUTES:
            value = getattr(base, attribute)
            if value.shape[0] != n:
                raise RuntimeError(f"base {attribute} topology mismatch")
            if value.requires_grad:
                raise RuntimeError(f"base {attribute} must remain frozen")
        for _name, parameter in self.model.state_parameter_items():
            if tuple(parameter.shape[:2]) != (n, int(self.model.max_states)):
                raise RuntimeError("temporal parameter topology mismatch")
        for attribute in (
            "state_start",
            "state_end",
            "state_valid",
            "state_status",
        ):
            if tuple(getattr(self.model, attribute).shape) != (n, int(self.model.max_states)):
                raise RuntimeError(f"temporal buffer {attribute} topology mismatch")
        for attribute in ("num_states", "current_state_index"):
            if getattr(self.model, attribute).shape != (n,):
                raise RuntimeError(f"temporal buffer {attribute} topology mismatch")
        for attribute in (
            "stable_id",
            "root_index",
            "episode_slot",
            "creation_timestamp",
            "generation",
            "is_residual",
            "parent_stable_id",
            "max_radii2d",
        ):
            if getattr(self, attribute).shape != (n,):
                raise RuntimeError(f"manager {attribute} topology mismatch")
        for attribute in ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom"):
            if getattr(self, attribute).shape != (n, 1):
                raise RuntimeError(f"manager {attribute} topology mismatch")
        if not bool(torch.equal(self.stable_id[: self.immutable_count], torch.arange(self.immutable_count, device=self.stable_id.device))):
            raise RuntimeError("immutable prefix stable IDs changed")
        if bool(self.is_residual[: self.immutable_count].any()):
            raise RuntimeError("immutable prefix cannot be residual")
        if n > self.immutable_count:
            residual = self.is_residual
            if not bool(residual[self.immutable_count :].all()):
                raise RuntimeError("all appended rows must be residual")
            if bool((self.root_index[residual] < 0).any() or (self.root_index[residual] >= self.immutable_count).any()):
                raise RuntimeError("residual root index is invalid")
            if bool((self.episode_slot[residual] < 0).any() or (self.episode_slot[residual] >= self.model.max_states).any()):
                raise RuntimeError("residual episode slot is invalid")
            active_residual = residual & (self.model.current_state_index >= 0)
            if bool(active_residual.any()):
                root_slots = self.model.current_state_index[self.root_index[active_residual]]
                if not bool(torch.equal(root_slots, self.episode_slot[active_residual])):
                    raise RuntimeError("active residual outlived or changed its root episode")
        self.model.validate_lifecycle()
        return True


def compute_temporal_multiview_score(
    views: Sequence[Any],
    view_indices: Sequence[int],
    model: Any,
    pipe: Any,
    background: torch.Tensor,
    *,
    current_timestamp: int,
    cue_scale: float = 1.0,
    min_mass: float = 1e-6,
    support_threshold: float = 0.5,
) -> TemporalDensityScore:
    """Probe only the current OPEN temporal topology against causal soft cues."""

    from gaussian_renderer import render_change

    if not view_indices:
        raise ValueError("view_indices must be nonempty")
    if not math.isfinite(float(cue_scale)) or float(cue_scale) <= 0.0:
        raise ValueError("cue_scale must be finite and positive")
    indices = tuple(int(value) for value in view_indices)
    if min(indices) < 0 or max(indices) > int(current_timestamp):
        raise RuntimeError("density score attempted to access a future view")
    attributes = model.get_active_render_attributes(float(current_timestamp))
    n = int(model.state_valid.shape[0])
    device = model.state_change_dc.device
    dtype = model.state_change_dc.dtype
    rows = torch.arange(n, device=device)
    slots = model.current_state_index
    starts = torch.full((n,), float("inf"), device=device, dtype=model.state_start.dtype)
    active = slots >= 0
    starts[active] = model.state_start[rows[active], slots[active]]
    positive: list[torch.Tensor] = []
    negative: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    for index in indices:
        view = views[index]
        cue = getattr(view, "candidate_map", None)
        if not isinstance(cue, torch.Tensor):
            raise AttributeError("each sampled view must expose candidate_map")
        if cue.ndim == 3 and cue.shape[0] == 1:
            cue = cue[0]
        if cue.ndim != 2:
            raise ValueError("candidate_map must have shape [H,W] or [1,H,W]")
        cue = torch.clamp(cue.to(device=device, dtype=dtype) / float(cue_scale), 0.0, 1.0)
        probe = torch.zeros((n, 3), device=device, dtype=dtype, requires_grad=True)
        package = render_change(
            view,
            model.base,
            pipe,
            torch.zeros_like(background),
            override_color=probe,
            override_opacity=attributes["opacity"].detach(),
            override_xyz=attributes["xyz"].detach(),
            override_scaling=attributes["scaling"].detach(),
            override_rotation=attributes["rotation"].detach(),
            clamp_output=False,
        )
        rendered = package["render"]
        if tuple(rendered.shape[1:]) != tuple(cue.shape):
            raise ValueError("cue/render spatial shape mismatch")
        weights = torch.zeros_like(rendered)
        weights[0] = cue
        weights[1] = 1.0 - cue
        gradient = torch.autograd.grad(
            rendered,
            probe,
            grad_outputs=weights,
            retain_graph=False,
            create_graph=False,
        )[0].detach()
        if not bool(torch.isfinite(gradient).all()) or bool((gradient[:, :2] < -1e-6).any()):
            raise RuntimeError("temporal alpha-T probe returned invalid responsibility")
        positive.append(gradient[:, 0].clamp_min(0.0))
        negative.append(gradient[:, 1].clamp_min(0.0))
        eligible.append(active & (starts <= float(index)))
    aggregated = aggregate_temporal_view_evidence(
        positive,
        negative,
        eligible,
        min_mass=min_mass,
        support_threshold=support_threshold,
    )
    return TemporalDensityScore(
        view_indices=indices,
        positive_mass=aggregated[0],
        negative_mass=aggregated[1],
        total_mass=aggregated[2],
        importance_score=aggregated[3],
        visible_view_count=aggregated[4],
        support_view_count=aggregated[5],
        change_ratio=aggregated[6],
    )
