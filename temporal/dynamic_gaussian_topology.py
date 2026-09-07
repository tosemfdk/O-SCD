"""Dynamic topology for one equal-status mutable ``R_change`` Gaussian bank.

Every row is governed by the same contract.  The binary detector decides only
whether the row is currently ACTIVE or INACTIVE.  ACTIVE rows may be optimized,
cloned, split, or pruned; INACTIVE rows are preserved exactly.  A newly created
row copies its source detector/controller/lifespan state at the topology event,
but owns independent tensors and Bayesian state afterwards.

This module deliberately has no immutable-prefix, root, or residual-bank
parameter semantics.  Generation and direct-parent metadata additionally guard
the optional child-only black pruning ablation: initial rows are never eligible,
and at least one live representative is retained when a removed split source no
longer exists.
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
    gradient_split: torch.Tensor
    cue_mixture_split: torch.Tensor
    signed_score_candidate: torch.Tensor
    signed_score_clone: torch.Tensor
    signed_score_split: torch.Tensor
    signed_score_prune: torch.Tensor
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
    gradient_split_source_count: int
    cue_mixture_split_source_count: int
    cue_mixture_only_split_source_count: int
    signed_score_candidate_count: int
    signed_score_clone_source_count: int
    signed_score_split_source_count: int
    signed_score_prune_candidate_count: int
    signed_score_pruned_count: int
    split_source_removed_count: int
    opacity_pruned_count: int
    size_pruned_count: int
    black_child_prune_candidate_count: int
    black_child_pruned_count: int
    black_child_retained_for_support_count: int
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

    @staticmethod
    def _row_buffer_names(owner: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve explicitly declared row-aligned state for topology mutation."""

        names = tuple(getattr(owner, "topology_buffer_names", fallback))
        if len(names) != len(set(names)):
            raise RuntimeError("topology buffer names must be unique")
        for name in names:
            value = getattr(owner, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise RuntimeError(
                    f"topology buffer {type(owner).__name__}.{name} must be a 1D tensor"
                )
        return names

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

    def _stable_topk_rows(
        self,
        candidate: torch.Tensor,
        score: torch.Tensor,
        max_sources: int | None,
    ) -> torch.Tensor:
        """Return deterministic score-desc/stable-id-asc candidate rows."""

        rows = torch.nonzero(candidate, as_tuple=False).flatten()
        if max_sources is None:
            return rows
        if isinstance(max_sources, bool) or int(max_sources) < 0:
            raise ValueError("signed_density_max_sources must be a nonnegative integer")
        if int(rows.numel()) <= int(max_sources):
            return rows
        if int(max_sources) == 0:
            return rows[:0]
        # Deterministic ordering without materializing all candidates on CPU:
        # first impose stable-id ascending, then stable-sort by score descending.
        # Tied scores therefore preserve the prior stable-id order.
        stable_order = torch.argsort(self.stable_id[rows], stable=True)
        rows = rows[stable_order]
        score_order = torch.argsort(score[rows], descending=True, stable=True)
        return rows[score_order[: int(max_sources)]]

    def _current_episode_start(self, timestamp: int | float) -> torch.Tensor:
        """Return per-row active lifecycle start; inactive rows get timestamp."""

        rows = torch.arange(self.count, device=self.model.current_state_index.device)
        slots = self.model.current_state_index
        starts = torch.full(
            (self.count,),
            float(timestamp),
            device=self.model.state_start.device,
            dtype=self.model.state_start.dtype,
        )
        active = slots >= 0
        starts[active] = self.model.state_start[rows[active], slots[active]]
        return starts

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
        cue_mixture_score: torch.Tensor | None = None,
        cue_mixture_threshold: float = 0.5,
        signed_density_score: torch.Tensor | None = None,
        signed_density_threshold: float = 0.0,
        signed_density_max_sources: int | None = None,
        signed_density_prune_threshold: float | None = None,
        signed_density_prune_min_age_frames: int = 1,
        black_child_prune_threshold: float | None = None,
        black_child_prune_min_age_frames: int = 1,
    ) -> DynamicDensityResult:
        """Run active-only O-SCD density plus optional cue-mixture splitting.

        The online O-SCD loop supplies the gradient clone/split schedule.  Its
        16-step loop does not prune; the opacity/size pruning below is the
        explicit extension requested by this ablation and uses the same
        criteria as ``GaussianModel.densify_and_prune``.

        ``cue_mixture_score`` is an optional detached per-row value in ``[0,1]``
        computed from the current frame's pre-optimization raw cue evidence.
        It may only add large-Gaussian split sources; it never clones small
        rows, changes lifecycle state, or replaces the original gradient rule.

        ``signed_density_score`` is an optional detached per-row signed value.
        When supplied, positive rows above ``signed_density_threshold`` fully
        replace the gradient/cue-mixture topology selection: small rows clone,
        large rows split, and source selection is deterministic score-desc then
        stable-id-asc with an optional ``signed_density_max_sources`` budget.
        Negative rows are never densified.  They may prune only ACTIVE children
        with ``generation > 0`` when ``signed_density_prune_threshold`` is set;
        initial generation-zero rows remain protected.

        ``black_child_prune_threshold`` optionally hard-prunes only previously
        densified ACTIVE children whose learned intrinsic DC render value is
        below the threshold.  Initial rows and children born in this density
        event are protected.  If a direct parent is already gone and all of its
        surviving children are black candidates, the least-black child is kept
        so detector support for that split lineage is not deleted wholesale.
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
        cue_mixture_threshold = _validate_positive_finite(
            "cue_mixture_threshold", cue_mixture_threshold
        )
        if cue_mixture_threshold > 1.0:
            raise ValueError("cue_mixture_threshold must be at most one")
        signed_density_threshold = _validate_positive_finite(
            "signed_density_threshold",
            signed_density_threshold,
            allow_zero=True,
        )
        if signed_density_max_sources is not None and (
            isinstance(signed_density_max_sources, bool)
            or not isinstance(signed_density_max_sources, int)
            or signed_density_max_sources < 0
        ):
            raise ValueError("signed_density_max_sources must be a nonnegative integer")
        if signed_density_prune_threshold is not None:
            signed_density_prune_threshold = _validate_positive_finite(
                "signed_density_prune_threshold",
                signed_density_prune_threshold,
                allow_zero=True,
            )
        if (
            isinstance(signed_density_prune_min_age_frames, bool)
            or not isinstance(signed_density_prune_min_age_frames, int)
            or signed_density_prune_min_age_frames < 1
        ):
            raise ValueError("signed_density_prune_min_age_frames must be an integer >= 1")
        if black_child_prune_threshold is not None:
            black_child_prune_threshold = _validate_positive_finite(
                "black_child_prune_threshold",
                black_child_prune_threshold,
                allow_zero=True,
            )
            if black_child_prune_threshold > 1.0:
                raise ValueError("black_child_prune_threshold must be at most one")
        if (
            isinstance(black_child_prune_min_age_frames, bool)
            or not isinstance(black_child_prune_min_age_frames, int)
            or black_child_prune_min_age_frames < 1
        ):
            raise ValueError("black_child_prune_min_age_frames must be an integer >= 1")
        if cue_mixture_score is None:
            mixture_score = torch.zeros(
                self.count, device=self.model.xyz.device, dtype=self.model.xyz.dtype
            )
        else:
            if not isinstance(cue_mixture_score, torch.Tensor):
                raise TypeError("cue_mixture_score must be a tensor")
            mixture_score = cue_mixture_score.detach().to(
                device=self.model.xyz.device, dtype=self.model.xyz.dtype
            ).flatten()
            if mixture_score.shape != (self.count,):
                raise ValueError("cue_mixture_score must match the current topology")
            if not bool(torch.isfinite(mixture_score).all()) or bool(
                ((mixture_score < 0.0) | (mixture_score > 1.0)).any()
            ):
                raise ValueError("cue_mixture_score must be finite and in [0,1]")
        signed_score: torch.Tensor | None = None
        if signed_density_score is not None:
            if not isinstance(signed_density_score, torch.Tensor):
                raise TypeError("signed_density_score must be a tensor")
            signed_score = signed_density_score.detach().to(
                device=self.model.xyz.device, dtype=self.model.xyz.dtype
            ).flatten()
            if signed_score.shape != (self.count,):
                raise ValueError("signed_density_score must match the current topology")
            if not bool(torch.isfinite(signed_score).all()):
                raise ValueError("signed_density_score must be finite")

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
        signed_candidate = torch.zeros_like(active)
        signed_clone = torch.zeros_like(active)
        signed_split = torch.zeros_like(active)
        signed_prune_candidate_pre = torch.zeros_like(active)
        if signed_score is None:
            gradient_candidate = active & observed & gradient_ok
            clone = gradient_candidate & small
            gradient_split = gradient_candidate & ~small
            cue_mixture_split = (
                active
                & ~small
                & (mixture_score >= float(cue_mixture_threshold))
            )
            split = gradient_split | cue_mixture_split
        else:
            positive = active & (signed_score > float(signed_density_threshold))
            selected_positive_rows = self._stable_topk_rows(
                positive,
                signed_score,
                signed_density_max_sources,
            )
            signed_candidate[selected_positive_rows] = True
            signed_clone = signed_candidate & small
            signed_split = signed_candidate & ~small
            clone = signed_clone
            split = signed_split
            gradient_split = torch.zeros_like(active)
            cue_mixture_split = torch.zeros_like(active)
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
        filter_names = self._row_buffer_names(self.tracker, _FILTER_BUFFERS)
        combined_filter = {
            name: torch.cat((getattr(self.tracker, name), getattr(self.tracker, name)[source_rows]), dim=0)
            for name in filter_names
        }
        controller_names = self._row_buffer_names(
            self.controller,
            tuple(
                name for name in _CONTROLLER_BUFFERS if hasattr(self.controller, name)
            ),
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
        if signed_score is None:
            combined_signed_score = torch.zeros(
                combined_count, device=active.device, dtype=self.model.xyz.dtype
            )
        else:
            child_score = (
                signed_score[source_rows]
                if child_count
                else signed_score.new_empty((0,))
            )
            combined_signed_score = torch.cat(
                (signed_score, child_score),
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
        signed_score_prune = torch.zeros_like(opacity_prune)
        if signed_score is not None and signed_density_prune_threshold is not None:
            rows = torch.arange(
                combined_count,
                device=combined_lifecycle["current_state_index"].device,
                dtype=torch.long,
            )
            slots = combined_lifecycle["current_state_index"]
            current_starts = torch.full(
                (combined_count,),
                float(timestamp),
                device=combined_lifecycle["state_start"].device,
                dtype=combined_lifecycle["state_start"].dtype,
            )
            active_slots = slots >= 0
            current_starts[active_slots] = combined_lifecycle["state_start"][
                rows[active_slots], slots[active_slots]
            ]
            episode_old_enough = (
                float(timestamp) - current_starts
            ) >= float(signed_density_prune_min_age_frames)
            signed_score_prune = (
                combined_active
                & (combined_generation > 0)
                & episode_old_enough.to(device=combined_active.device)
                & (combined_signed_score <= -float(signed_density_prune_threshold))
            )
            signed_prune_candidate_pre = signed_score_prune[:old_count].detach().clone()
        base_remove = split_source_mask | opacity_prune | size_prune | signed_score_prune

        black_child_candidate = torch.zeros_like(base_remove)
        black_child_prune = torch.zeros_like(base_remove)
        black_child_retained = torch.zeros_like(base_remove)
        if black_child_prune_threshold is not None:
            intrinsic_dc = (
                combined_parameters["dc"] * 0.28209479177387814 + 0.5
            ).mean(dim=tuple(range(1, combined_parameters["dc"].ndim)))
            old_enough = (
                combined_creation >= 0
            ) & (
                int(timestamp) - combined_creation
                >= int(black_child_prune_min_age_frames)
            )
            black_child_candidate = (
                combined_active
                & (combined_generation > 0)
                & old_enough
                & (intrinsic_dc < float(black_child_prune_threshold))
                & ~base_remove
            )

            if bool(black_child_candidate.any()):
                supporting = ~base_remove & ~black_child_candidate
                id_capacity = int(self.next_stable_id) + child_count
                stable_support = torch.zeros(
                    id_capacity, device=active.device, dtype=torch.bool
                )
                stable_support[combined_stable_id[supporting]] = True
                sibling_support_count = torch.zeros(
                    id_capacity, device=active.device, dtype=torch.long
                )
                supporting_children = supporting & (combined_parent_id >= 0)
                sibling_support_count.scatter_add_(
                    0,
                    combined_parent_id[supporting_children],
                    torch.ones_like(
                        combined_parent_id[supporting_children], dtype=torch.long
                    ),
                )

                candidate_rows = torch.nonzero(
                    black_child_candidate, as_tuple=False
                ).flatten()
                candidate_parents = combined_parent_id[candidate_rows]
                has_support = (
                    stable_support[candidate_parents]
                    | (sibling_support_count[candidate_parents] > 0)
                )
                black_child_prune[candidate_rows[has_support]] = True

                unsupported_rows = candidate_rows[~has_support]
                if int(unsupported_rows.numel()):
                    unsupported_parents = combined_parent_id[unsupported_rows]
                    unique_parents, inverse = torch.unique(
                        unsupported_parents, sorted=False, return_inverse=True
                    )
                    best_dc = torch.full(
                        (int(unique_parents.numel()),),
                        -torch.inf,
                        device=intrinsic_dc.device,
                        dtype=intrinsic_dc.dtype,
                    )
                    best_dc.scatter_reduce_(
                        0,
                        inverse,
                        intrinsic_dc[unsupported_rows],
                        reduce="amax",
                        include_self=True,
                    )
                    is_best = intrinsic_dc[unsupported_rows] == best_dc[inverse]
                    max_stable_id = torch.iinfo(torch.long).max
                    best_stable_id = torch.full(
                        (int(unique_parents.numel()),),
                        max_stable_id,
                        device=combined_stable_id.device,
                        dtype=torch.long,
                    )
                    best_stable_id.scatter_reduce_(
                        0,
                        inverse[is_best],
                        combined_stable_id[unsupported_rows[is_best]],
                        reduce="amin",
                        include_self=True,
                    )
                    retain = (
                        combined_stable_id[unsupported_rows]
                        == best_stable_id[inverse]
                    )
                    black_child_retained[unsupported_rows[retain]] = True
                    black_child_prune[unsupported_rows[~retain]] = True

        remove = base_remove | black_child_prune
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
            elif bool(signed_score_prune[row]):
                reason = "negative_signed_score"
            elif bool(black_child_prune[row]):
                reason = "black_child"
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
            gradient_split_source_count=int(gradient_split.sum().item()),
            cue_mixture_split_source_count=int(cue_mixture_split.sum().item()),
            cue_mixture_only_split_source_count=int(
                (cue_mixture_split & ~gradient_split).sum().item()
            ),
            signed_score_candidate_count=int(signed_candidate.sum().item()),
            signed_score_clone_source_count=int(signed_clone.sum().item()),
            signed_score_split_source_count=int(signed_split.sum().item()),
            signed_score_prune_candidate_count=int(
                signed_prune_candidate_pre.sum().item()
            ),
            signed_score_pruned_count=int(signed_score_prune.sum().item()),
            split_source_removed_count=int(split_source_mask.sum().item()),
            opacity_pruned_count=int(opacity_prune.sum().item()),
            size_pruned_count=int(size_prune.sum().item()),
            black_child_prune_candidate_count=int(
                black_child_candidate.sum().item()
            ),
            black_child_pruned_count=int(black_child_prune.sum().item()),
            black_child_retained_for_support_count=int(
                black_child_retained.sum().item()
            ),
            total_removed_count=int(remove.sum().item()),
            masks=DynamicDensityMasks(
                active=active.detach().clone(),
                gradient_observed=observed.detach().clone(),
                gradient_split=gradient_split.detach().clone(),
                cue_mixture_split=cue_mixture_split.detach().clone(),
                signed_score_candidate=signed_candidate.detach().clone(),
                signed_score_clone=signed_clone.detach().clone(),
                signed_score_split=signed_split.detach().clone(),
                signed_score_prune=signed_prune_candidate_pre.detach().clone(),
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
        for name in self._row_buffer_names(self.tracker, _FILTER_BUFFERS):
            if getattr(self.tracker, name).shape != (n,):
                raise RuntimeError(f"filter buffer {name} lost topology alignment")
        for name in self._row_buffer_names(
            self.controller,
            tuple(
                name for name in _CONTROLLER_BUFFERS if hasattr(self.controller, name)
            ),
        ):
            if getattr(self.controller, name).shape != (n,):
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
