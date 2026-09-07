"""Panel-10 NEW/non-NEW partitioned rendering and training helpers.

The helpers in this module are intentionally standalone so the viewer can opt
into them behind a ``--training-partition panel10_new`` switch without changing
legacy update code.  They implement the user's current contract:

* the Panel-10 NEW map supervises DA3 seed rows;
* the complement of that map inside the learned soft cue supervises base rows;
* every branch still renders all NEVER_OPEN rows as frozen black occluders;
* CLOSED and future rows are excluded;
* rows that were OPEN in a sampled historical view but are not OPEN at the
  current timestamp render detached, so replay cannot resurrect optimizer state.

Losses used by :func:`train_partition_update` are exactly two O-SCD SSF terms:

```text
L_new  = SSF(A · Q_new,                 render(seed OPEN + all NEVER_OPEN))
L_base = SSF(A · (Q_change - Q_new),    render(base OPEN + all NEVER_OPEN))
L      = L_new + L_base
```

where ``A`` is ``replay.args.representation_cue_amplitude`` (defaulting to 2),
``Q_new`` is ``item.new_target`` and ``Q_change`` is ``item.cue_target``.  The
same SSF objective drives seed DC and seed geometry; there is no white-coverage
or projected-BCE side objective in this path.

The opt-in ``panel10_render_mode=joint_channels`` keeps these two losses but
uses one shared geometry/opacity scene. R carries NEW radiance, G carries BASE
radiance, B is zero. Mutual occlusion and its seed-geometry gradients are an
intentional model ablation, not a claim of split-render mathematical equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from temporal.fusion import compute_ssf_loss
from utils.sh_utils import RGB2SH, SH2RGB

Branch = Literal["base", "new", "joint"]

# Tests may monkeypatch this symbol to avoid importing the CUDA renderer.
render_change = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PartitionMasks:
    """Global row masks used to build and optimize a partition render."""

    base_open: torch.Tensor
    base_never_open: torch.Tensor
    base_trainable_open: torch.Tensor
    seed_open: torch.Tensor
    seed_never_open: torch.Tensor
    seed_trainable_open: torch.Tensor


class Panel10PartitionView:
    """Concatenated adapter accepted by ``gaussian_renderer.render_change``.

    ``base_rows`` and ``seed_rows`` map each rendered row back to its global
    source row.  Non-source entries are ``-1``.  ``trainable_render_rows`` marks
    rows that are OPEN at both the sampled timestamp and the current timestamp;
    all other rendered rows are detached occluders or historical-only replay
    rows whose values and optimizer moments must be preserved.
    """

    def __init__(
        self,
        *,
        base: Any,
        seed_model: Any | None,
        base_rows: torch.Tensor,
        seed_rows: torch.Tensor,
        xyz: torch.Tensor,
        dc: torch.Tensor,
        features_rest: torch.Tensor,
        opacity: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        trainable_render_rows: torch.Tensor,
        trainable_base_rows: torch.Tensor,
        trainable_seed_rows: torch.Tensor,
    ) -> None:
        self.base = base
        self.seed_model = seed_model
        self.base_rows = base_rows
        self.seed_rows = seed_rows
        self.trainable_render_rows = trainable_render_rows
        self.trainable_base_rows = trainable_base_rows
        self.trainable_seed_rows = trainable_seed_rows
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.scaling_activation = base.scaling_activation
        self.opacity_activation = base.opacity_activation
        self.rotation_activation = base.rotation_activation
        self.covariance_activation = base.covariance_activation
        self._xyz = xyz
        self._features_dc = dc
        self._features_rest = features_rest
        self._opacity_value = opacity
        self._opacity = opacity
        self._scaling_value = scaling
        self._rotation_value = rotation

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        # Publicly this adapter is still a degree-0 SH model, so callers that
        # inspect ``get_features`` should see only DC.  Internally we keep a
        # one-coefficient dummy ``_features_rest`` buffer because the local
        # FastGS CUDA backward only enters computeColorFromSH (and therefore
        # writes dL/ddc) when the SH-rest pointer is non-null.
        if self.max_sh_degree == 0:
            return self._features_dc
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


def _seed_count(model: Any) -> int:
    return int(model.num_gaussians)


def _probe_count(probe: Any) -> int:
    return int(probe.num_seeds)


def _mask_from(lifecycle: Any, name: str, timestamp: int) -> torch.Tensor:
    mask = getattr(lifecycle, name)(timestamp)
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
    ):
        raise ValueError(f"{name} must return a boolean [N] tensor")
    return mask


def _materialized_mask(lifecycle: Any, timestamp: int) -> torch.Tensor:
    return _mask_from(lifecycle, "materialized_mask", timestamp)


def _rowwise_detach(value: torch.Tensor, trainable_rows: torch.Tensor) -> torch.Tensor:
    if value.shape[0] == 0:
        return value
    trainable_rows = trainable_rows.to(device=value.device)
    view_shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    return torch.where(trainable_rows.reshape(view_shape), value, value.detach())


def _zero_rest(rows: int, *, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros((rows, 1, 3), device=like.device, dtype=like.dtype)


def _selected_or_empty(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return value[rows] if rows.numel() else value[:0]


def _seed_masks(
    replay: Any, timestamp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_model = replay.seed_model
    if _seed_count(seed_model) == 0:
        device = replay.base.get_xyz.device
        empty = torch.empty(0, device=device, dtype=torch.bool)
        return empty, empty, empty
    active = _mask_from(replay.seed_lifecycle, "active_mask", timestamp)
    never = _mask_from(replay.seed_lifecycle, "never_open_mask", timestamp)
    materialized = _materialized_mask(replay.seed_lifecycle, timestamp)
    expected = _seed_count(seed_model)
    if active.shape[0] != expected or never.shape[0] != expected:
        raise RuntimeError("DA3 lifecycle history is not aligned with seed rows")
    if bool((active & never).any()):
        raise RuntimeError("DA3 historical OPEN/NEVER_OPEN masks overlap")
    if bool(((active | never) & ~materialized).any()):
        raise RuntimeError("future DA3 rows entered a partition render")
    return active, never, materialized


def _base_masks(
    replay: Any, timestamp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = _mask_from(replay.lifecycle, "active_mask", timestamp)
    never = _mask_from(replay.lifecycle, "never_open_mask", timestamp)
    materialized = _materialized_mask(replay.lifecycle, timestamp)
    if bool((active & never).any()):
        raise RuntimeError("base OPEN/NEVER_OPEN masks overlap")
    if bool(((active | never) & ~materialized).any()):
        raise RuntimeError("future base rows entered a partition render")
    return active, never, materialized


def _partition_masks(
    replay: Any, timestamp: int, branch: Branch, train: bool
) -> PartitionMasks:
    current_timestamp = int(replay.current_index)
    if timestamp < 0 or timestamp > current_timestamp:
        raise RuntimeError(
            f"replay timestamp {timestamp} is outside the causal range [0,{current_timestamp}]"
        )
    base_active, base_never, _ = _base_masks(replay, timestamp)
    seed_active, seed_never, _ = _seed_masks(replay, timestamp)
    base_current = _mask_from(replay.lifecycle, "active_mask", current_timestamp)
    seed_current = (
        _mask_from(replay.seed_lifecycle, "active_mask", current_timestamp)
        if seed_active.numel()
        else seed_active
    )
    if base_current.shape != base_active.shape:
        raise RuntimeError("base current lifecycle is not aligned with sampled rows")
    if seed_current.shape != seed_active.shape:
        raise RuntimeError("seed current lifecycle is not aligned with sampled rows")

    base_open = (
        base_active if branch in {"base", "joint"} else torch.zeros_like(base_active)
    )
    seed_open = (
        seed_active if branch in {"new", "joint"} else torch.zeros_like(seed_active)
    )
    return PartitionMasks(
        base_open=base_open,
        base_never_open=base_never,
        base_trainable_open=(
            (base_open & base_current) if train else torch.zeros_like(base_open)
        ),
        seed_open=seed_open,
        seed_never_open=seed_never,
        seed_trainable_open=(
            (seed_open & seed_current) if train else torch.zeros_like(seed_open)
        ),
    )


def build_partition_view(
    replay: Any,
    timestamp: int,
    branch: Branch,
    train: bool = True,
) -> Panel10PartitionView:
    """Build a differentiable render adapter for a Panel-10 partition.

    Branch semantics:

    ```text
    base : base OPEN learned DC + all base/seed NEVER_OPEN black occluders
    new  : seed OPEN learned DC/geometry + all base/seed NEVER_OPEN black occluders
    joint: base OPEN + seed OPEN + all NEVER_OPEN black occluders
    ```

    In every branch, CLOSED and future rows are omitted.  OPEN rows are
    differentiable only when they are OPEN at both ``timestamp`` and
    ``replay.current_index``; historical-only rows render detached.
    """

    if branch not in {"base", "new", "joint"}:
        raise ValueError("branch must be 'base', 'new', or 'joint'")
    timestamp = int(timestamp)
    masks = _partition_masks(replay, timestamp, branch, train)
    base = replay.base
    seed_model = replay.seed_model
    base_xyz_full = base.get_xyz
    base_scaling_full = base.get_scaling
    base_rotation_full = base.get_rotation
    base_opacity_full = base.get_opacity
    base_dc_full = replay.change_dc
    base_black = RGB2SH(base_dc_full.new_zeros(()))
    base_dc_values = torch.where(
        masks.base_open[:, None, None], base_dc_full, base_black
    )
    base_selected = masks.base_open | masks.base_never_open
    base_idx = torch.nonzero(base_selected, as_tuple=False).flatten()
    base_train_local = masks.base_trainable_open[base_idx]

    parts_xyz: list[torch.Tensor] = []
    parts_dc: list[torch.Tensor] = []
    parts_rest: list[torch.Tensor] = []
    parts_opacity: list[torch.Tensor] = []
    parts_scaling: list[torch.Tensor] = []
    parts_rotation: list[torch.Tensor] = []
    base_rows_parts: list[torch.Tensor] = []
    seed_rows_parts: list[torch.Tensor] = []
    train_parts: list[torch.Tensor] = []

    if base_idx.numel():
        base_xyz = _selected_or_empty(base_xyz_full, base_idx).detach()
        base_scaling = _selected_or_empty(base_scaling_full, base_idx).detach()
        base_rotation = _selected_or_empty(base_rotation_full, base_idx).detach()
        base_opacity = _selected_or_empty(base_opacity_full, base_idx).detach()
        base_dc = _rowwise_detach(
            _selected_or_empty(base_dc_values, base_idx), base_train_local
        )
        parts_xyz.append(base_xyz)
        parts_dc.append(base_dc)
        parts_rest.append(_zero_rest(base_idx.numel(), like=base_dc_full))
        parts_opacity.append(base_opacity)
        parts_scaling.append(base_scaling)
        parts_rotation.append(base_rotation)
        base_rows_parts.append(base_idx)
        seed_rows_parts.append(torch.full_like(base_idx, -1))
        train_parts.append(base_train_local)

    if _seed_count(seed_model) > 0:
        seed_selected = masks.seed_open | masks.seed_never_open
        seed_idx = torch.nonzero(seed_selected, as_tuple=False).flatten()
        if seed_idx.numel():
            seed_train_local = masks.seed_trainable_open[seed_idx]
            probe = replay.seed_detector_probe
            if _probe_count(probe) != _seed_count(seed_model):
                raise RuntimeError("DA3 detector probe is not aligned with seed rows")
            seed_dc_full = seed_model.new_dc
            seed_black = RGB2SH(seed_dc_full.new_zeros(()))
            seed_xyz_values = torch.where(
                masks.seed_open[:, None], seed_model.get_xyz, probe.get_xyz.detach()
            )
            seed_dc_values = torch.where(
                masks.seed_open[:, None, None], seed_dc_full, seed_black
            )
            seed_opacity_values = torch.where(
                masks.seed_open[:, None],
                seed_model.get_opacity,
                probe.get_opacity.detach(),
            )
            seed_scaling_values = torch.where(
                masks.seed_open[:, None],
                seed_model.get_scaling,
                probe.get_scaling.detach(),
            )
            seed_rotation_values = torch.where(
                masks.seed_open[:, None],
                seed_model.get_rotation,
                probe.get_rotation.detach(),
            )
            parts_xyz.append(
                _rowwise_detach(seed_xyz_values[seed_idx], seed_train_local)
            )
            parts_dc.append(
                _rowwise_detach(seed_dc_values[seed_idx], seed_train_local)
            )
            parts_rest.append(_zero_rest(seed_idx.numel(), like=seed_dc_full))
            parts_opacity.append(
                _rowwise_detach(seed_opacity_values[seed_idx], seed_train_local)
            )
            parts_scaling.append(
                _rowwise_detach(seed_scaling_values[seed_idx], seed_train_local)
            )
            parts_rotation.append(
                _rowwise_detach(seed_rotation_values[seed_idx], seed_train_local)
            )
            base_rows_parts.append(torch.full_like(seed_idx, -1))
            seed_rows_parts.append(seed_idx)
            train_parts.append(seed_train_local)

    if parts_xyz:
        xyz = torch.cat(parts_xyz, dim=0)
        dc = torch.cat(parts_dc, dim=0)
        rest = torch.cat(parts_rest, dim=0)
        opacity = torch.cat(parts_opacity, dim=0)
        scaling = torch.cat(parts_scaling, dim=0)
        rotation = torch.cat(parts_rotation, dim=0)
        base_rows = torch.cat(base_rows_parts, dim=0)
        seed_rows = torch.cat(seed_rows_parts, dim=0)
        train_rows = torch.cat(train_parts, dim=0).to(device=xyz.device)
    else:
        like = base_dc_full
        xyz = torch.empty((0, 3), device=like.device, dtype=like.dtype)
        dc = torch.empty((0, 1, 3), device=like.device, dtype=like.dtype)
        rest = torch.empty((0, 1, 3), device=like.device, dtype=like.dtype)
        opacity = torch.empty((0, 1), device=like.device, dtype=like.dtype)
        scaling = torch.empty((0, 3), device=like.device, dtype=like.dtype)
        rotation = torch.empty((0, 4), device=like.device, dtype=like.dtype)
        base_rows = torch.empty((0,), device=like.device, dtype=torch.long)
        seed_rows = torch.empty((0,), device=like.device, dtype=torch.long)
        train_rows = torch.empty((0,), device=like.device, dtype=torch.bool)

    return Panel10PartitionView(
        base=base,
        seed_model=seed_model,
        base_rows=base_rows,
        seed_rows=seed_rows,
        xyz=xyz,
        dc=dc,
        features_rest=rest,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        trainable_render_rows=train_rows,
        trainable_base_rows=masks.base_trainable_open,
        trainable_seed_rows=masks.seed_trainable_open,
    )


def _renderer():
    global render_change
    if render_change is None:
        from gaussian_renderer import render_change as imported_render_change

        render_change = imported_render_change
    return render_change


def _render(
    replay: Any, item: Any, view: Panel10PartitionView, *, override_color=None
) -> dict[str, torch.Tensor]:
    options = {} if override_color is None else {"override_color": override_color}
    return _renderer()(  # type: ignore[misc]
        item.view,
        view,
        replay.pipe,
        replay.evidence_background,
        clamp_output=False,
        **options,
    )


def joint_partition_colors(view: Panel10PartitionView) -> torch.Tensor:
    """Pack scalar NEW/BASE radiance while preserving the DC-to-color gradient.

    Match degree-zero CUDA's per-color lower clamp BEFORE taking the mean.
    Compositing is linear in color at fixed alpha-T; only the two groups' T
    changes relative to separately rendered scenes. Black NEVER_OPEN stays zero.
    """
    radiance = SH2RGB(view._features_dc[:, 0, :]).clamp_min(0).mean(dim=1)
    return torch.stack((radiance * (view.seed_rows >= 0),
                        radiance * (view.base_rows >= 0),
                        torch.zeros_like(radiance)), dim=1)


def _visible_global_rows(
    view: Panel10PartitionView,
    package: dict[str, torch.Tensor],
    source: Literal["base", "seed"],
) -> torch.Tensor:
    radii = package.get("radii")
    if not isinstance(radii, torch.Tensor):
        raise RuntimeError("render package is missing radii")
    if radii.ndim != 1 or radii.shape[0] != view.trainable_render_rows.shape[0]:
        raise RuntimeError("render radii do not align with partition rows")
    rows = view.base_rows if source == "base" else view.seed_rows
    visible = (radii.detach() > 0) & view.trainable_render_rows & (rows >= 0)
    return rows[visible]


def _accumulate_seed_screen_gradients(
    replay: Any,
    view: Panel10PartitionView,
    package: dict[str, torch.Tensor],
) -> int:
    seed_model = replay.seed_model
    if _seed_count(seed_model) == 0:
        return 0
    radii = package.get("radii")
    points = package.get("viewspace_points")
    if not isinstance(radii, torch.Tensor):
        raise RuntimeError("render package is missing radii")
    visible = (radii.detach() > 0) & view.trainable_render_rows & (view.seed_rows >= 0)
    rows = view.seed_rows[visible]
    replay.seed_last_visible_rows = torch.zeros(
        _seed_count(seed_model), device=radii.device, dtype=torch.bool
    )
    if rows.numel() == 0:
        return 0
    replay.seed_last_visible_rows[rows] = True
    if not isinstance(points, torch.Tensor) or points.grad is None:
        raise RuntimeError("view-space points did not receive a seed geometry gradient")
    grad = points.grad[visible]
    if grad.ndim != 2 or grad.shape[1] < 2:
        raise RuntimeError("view-space point gradients must have shape [N,D>=2]")
    signed = torch.linalg.vector_norm(grad[:, :2], dim=-1, keepdim=True)
    absolute_source = grad[:, 2:] if grad.shape[1] > 2 else grad[:, :2]
    absolute = torch.linalg.vector_norm(absolute_source, dim=-1, keepdim=True)
    seed_model.xyz_gradient_accum[rows] += signed.detach()
    seed_model.xyz_gradient_accum_abs[rows] += absolute.detach()
    seed_model.gradient_denom[rows] += 1.0
    seed_model.max_radii2d[rows] = torch.maximum(
        seed_model.max_radii2d[rows], radii.detach()[visible]
    )
    return int(rows.numel())


def _zero_grad(optimizer: Any) -> None:
    optimizer.zero_grad(set_to_none=True)



def _validate_partition_targets(
    cue_target: torch.Tensor, new_target: torch.Tensor, *, amplitude: float
) -> None:
    if not isinstance(cue_target, torch.Tensor) or not isinstance(
        new_target, torch.Tensor
    ):
        raise TypeError("cue_target and new_target must be tensors")
    if cue_target.ndim != 3 or cue_target.shape[0] != 1:
        raise ValueError("cue_target must have shape [1,H,W]")
    if new_target.shape != cue_target.shape:
        raise ValueError("new_target must have the same [1,H,W] shape as cue_target")
    if cue_target.device != new_target.device:
        raise ValueError("cue_target and new_target must share a device")
    if not torch.is_floating_point(cue_target) or not torch.is_floating_point(
        new_target
    ):
        raise TypeError("cue_target and new_target must be floating-point tensors")
    if not torch.isfinite(cue_target).all() or not torch.isfinite(new_target).all():
        raise ValueError("cue_target and new_target must be finite")
    if not torch.isfinite(torch.as_tensor(amplitude)) or float(amplitude) <= 0.0:
        raise ValueError("representation_cue_amplitude must be finite and positive")
    invalid_range = (
        (new_target < 0.0)
        | (cue_target < 0.0)
        | (new_target > cue_target)
        | (cue_target > 1.0)
    )
    if bool(invalid_range.any()):
        raise ValueError("targets must satisfy 0 <= new_target <= cue_target <= 1")


def train_partition_update(
    replay: Any,
    item: Any,
    *,
    current_timestamp: int,
    dc_item: Any | None = None,
) -> dict[str, float | int]:
    """Run one Panel-10 partitioned SSF update.

    If ``dc_item`` is supplied, it replaces ``item`` for both seed and base
    partition losses.  This keeps DC and geometry on the same sampled view in
    the new path and avoids the legacy split where geometry could use a
    different replay item.
    """

    current_timestamp = int(current_timestamp)
    sampled_item = item if dc_item is None else dc_item
    sampled_timestamp = int(sampled_item.timestamp)
    if current_timestamp < 0:
        raise RuntimeError("current timestamp must be nonnegative")
    if sampled_timestamp < 0 or sampled_timestamp > current_timestamp:
        raise RuntimeError(
            f"replay timestamp {sampled_timestamp} is outside the causal range [0,{current_timestamp}]"
        )
    if int(replay.current_index) != current_timestamp:
        raise RuntimeError("current_timestamp must match replay.current_index")

    _zero_grad(replay.base_optimizer)
    _zero_grad(replay.seed_optimizer)

    amplitude = float(
        getattr(
            getattr(replay, "args", object()),
            "representation_cue_amplitude",
            2.0,
        )
    )
    _validate_partition_targets(
        sampled_item.cue_target, sampled_item.new_target, amplitude=amplitude
    )
    new_target = amplitude * sampled_item.new_target
    base_target = amplitude * (sampled_item.cue_target - sampled_item.new_target)

    render_mode = getattr(replay.args, "panel10_render_mode", "split")
    if render_mode == "joint_channels":
        if bool((replay.evidence_background != 0).any()):
            raise ValueError("joint partition channels require a black background")
        new_view = base_view = build_partition_view(replay, sampled_timestamp, "joint", train=True)
        new_package = base_package = _render(
            replay, sampled_item, new_view, override_color=joint_partition_colors(new_view)
        )
        rgb = new_package["render"]
        new_loss, _ = compute_ssf_loss(new_target, rgb[0:1].expand(3, -1, -1))
        base_loss, _ = compute_ssf_loss(base_target, rgb[1:2].expand(3, -1, -1))
        loss = new_loss + base_loss
        if bool(new_view.trainable_render_rows.any()) and loss.requires_grad:
            loss.backward()
        visible_seed_rows = _visible_global_rows(new_view, new_package, "seed")
        seed_visible_count = _accumulate_seed_screen_gradients(replay, new_view, new_package)
    elif render_mode == "split":
        new_view = build_partition_view(replay, sampled_timestamp, "new", train=True)
        new_package = _render(replay, sampled_item, new_view)
        new_loss, _ = compute_ssf_loss(new_target, new_package["render"])
        if bool(new_view.trainable_render_rows.any()) and new_loss.requires_grad:
            new_loss.backward(retain_graph=False)
        visible_seed_rows = _visible_global_rows(new_view, new_package, "seed")
        seed_visible_count = _accumulate_seed_screen_gradients(replay, new_view, new_package)

        base_view = build_partition_view(replay, sampled_timestamp, "base", train=True)
        base_package = _render(replay, sampled_item, base_view)
        base_loss, _ = compute_ssf_loss(base_target, base_package["render"])
        if bool(base_view.trainable_render_rows.any()) and base_loss.requires_grad:
            base_loss.backward(retain_graph=False)
    else:
        raise ValueError("panel10_render_mode must be split or joint_channels")
    visible_base_rows = _visible_global_rows(base_view, base_package, "base")

    seed_count = _seed_count(replay.seed_model)
    if seed_count == 0:
        device = replay.base.get_xyz.device
        replay.seed_last_visible_rows = torch.zeros(0, device=device, dtype=torch.bool)

    base_mask = torch.zeros_like(base_view.trainable_base_rows)
    if visible_base_rows.numel():
        base_mask[visible_base_rows] = True
        replay.base_optimizer.step(base_mask)

    seed_mask = torch.zeros_like(new_view.trainable_seed_rows)
    if visible_seed_rows.numel():
        seed_mask[visible_seed_rows] = True
        replay.seed_optimizer.step(
            {
                "xyz": seed_mask,
                "dc": seed_mask,
                "opacity": seed_mask,
                "scaling": seed_mask,
                "rotation": seed_mask,
            }
        )
        replay._constrain_typed_seeds(seed_mask)

    if replay.seed_geometry_update_counts.shape != seed_mask.shape:
        raise RuntimeError("DA3 geometry update counts are misaligned")
    replay.seed_geometry_update_counts[seed_mask] += 1

    return {
        "loss": float((new_loss + base_loss).detach().item()),
        "base_loss": float(base_loss.detach().item()),
        "new_loss": float(new_loss.detach().item()),
        "trainable_open_rows": int(
            _mask_from(replay.lifecycle, "active_mask", current_timestamp)
            .sum()
            .item()
        ),
        "active_seed_rows": (
            int(
                _mask_from(
                    replay.seed_lifecycle,
                    "active_mask",
                    current_timestamp,
                )
                .sum()
                .item()
            )
            if seed_count
            else 0
        ),
        "visible_seed_rows": int(seed_visible_count),
        "visible_pending_seed_rows": 0,
        "lifespan_render_violations": 0,
    }
