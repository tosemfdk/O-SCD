"""Panel-10 NEW seed topology helpers for the evolving-scene viewer.

This module is intentionally viewer-light: it operates on a replay-like object
that owns the active NEW sidecar, the immutable detector probe, lifecycle
history, and BF tracker.  It does not import the viewer module at import time.

Rows are an append-only archive.  Density split/prune retire rows by closing
lifespans and masking them from future topology decisions; rows are not GPU-
compacted so historical renders and optimizer state remain auditable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _device_from_replay(replay: Any) -> torch.device:
    if hasattr(replay, "device"):
        return torch.device(replay.device)
    if hasattr(replay.seed_model, "_xyz"):
        return replay.seed_model._xyz.device
    return torch.device("cpu")


def _dtype_from_replay(replay: Any) -> torch.dtype:
    if hasattr(replay.seed_model, "_xyz"):
        return replay.seed_model._xyz.dtype
    return torch.float32


def _as_metadata_list(metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None, count: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(count)]
    if isinstance(metadata, Mapping):
        if count != 1:
            raise ValueError("single metadata mapping requires one appended seed")
        return [copy.deepcopy(dict(metadata))]
    if len(metadata) != count:
        raise ValueError("metadata length must match seed rows")
    return [copy.deepcopy(dict(row)) for row in metadata]



def _ensure_archive_vectors(replay: Any) -> None:
    """Install append-only seed bookkeeping vectors if an older replay lacks them."""

    device = _device_from_replay(replay)
    count = int(replay.seed_model.num_gaussians)
    if not hasattr(replay, "accepted_da3_source_rows"):
        replay.accepted_da3_source_rows = list(range(count))
    if not hasattr(replay, "accepted_da3_birth_global"):
        replay.accepted_da3_birth_global = [-1] * count
    if not hasattr(replay, "seed_retired"):
        replay.seed_retired = torch.zeros(count, device=device, dtype=torch.bool)
    if not hasattr(replay, "seed_geometry_update_counts"):
        replay.seed_geometry_update_counts = torch.zeros(count, device=device, dtype=torch.long)
    if not hasattr(replay, "seed_last_visible_rows"):
        replay.seed_last_visible_rows = torch.zeros(count, device=device, dtype=torch.bool)

    for name, dtype, fill in (
        ("seed_retired", torch.bool, False),
        ("seed_geometry_update_counts", torch.long, 0),
        ("seed_last_visible_rows", torch.bool, False),
    ):
        value = getattr(replay, name).to(device=device, dtype=dtype).flatten()
        if value.numel() < count:
            pad = torch.full((count - value.numel(),), fill, device=device, dtype=dtype)
            value = torch.cat((value, pad))
        elif value.numel() > count:
            raise RuntimeError(f"{name} has more rows than the seed model archive")
        setattr(replay, name, value)
    if len(replay.accepted_da3_source_rows) != count or len(replay.accepted_da3_birth_global) != count:
        raise RuntimeError("accepted DA3 seed mappings are not aligned with the archive")


def _tracker_capacity(tracker: Any) -> int:
    for name in ("num_rows", "num_gaussians"):
        if hasattr(tracker, name):
            return int(getattr(tracker, name))
    if hasattr(tracker, "stable_a"):
        return int(tracker.stable_a.numel())
    raise TypeError("tracker must expose a row capacity")


def _base_row_offset(replay: Any) -> int:
    """Return the immutable base prefix length in the unified detector tracker."""

    return int(replay.count)


def _joint_tracker(replay: Any) -> Any:
    tracker = getattr(replay, "tracker", None)
    if tracker is None:
        raise RuntimeError("replay.tracker is required for typed NEW seeds")
    return tracker


def _ensure_tracker_capacity(replay: Any, archive_rows: int) -> None:
    """Reserve append-only BF storage without imposing a seed birth budget."""
    tracker = _joint_tracker(replay)
    base_offset = _base_row_offset(replay)
    capacity = _tracker_capacity(tracker)
    required = base_offset + int(archive_rows)
    if capacity < required:
        # Grow only the suffix in chunks, avoiding a copy of the large base
        # prefix for every small birth. Spare storage is not an archive cap.
        suffix_capacity = ((int(archive_rows) + 4095) // 4096) * 4096
        tracker.ensure_capacity(base_offset + suffix_capacity)


def _tracker_copy_rows(tracker: Any, parent_rows: torch.Tensor, child_rows: torch.Tensor, *, timestamp: int) -> None:
    names = tuple(getattr(tracker, "topology_buffer_names", ()))
    if not names or child_rows.numel() == 0:
        return
    repeat = int(child_rows.numel() // max(int(parent_rows.numel()), 1))
    expanded_parents = parent_rows.repeat(repeat) if repeat > 1 else parent_rows
    if expanded_parents.numel() != child_rows.numel():
        # Fall back to the exact order used by ActiveNewGaussianModel for split children.
        expanded_parents = parent_rows.repeat_interleave(repeat)
    for name in names:
        value = getattr(tracker, name)
        if value.shape[0] <= int(child_rows.max().item()):
            raise RuntimeError(f"tracker.{name} is not preallocated for child rows")
        value[child_rows] = value[expanded_parents].clone()
    if hasattr(tracker, "last_timestamp"):
        tracker.last_timestamp[child_rows] = int(timestamp)



def _close_retire_rows(replay: Any, rows: torch.Tensor, timestamp: int) -> None:
    if rows.numel() == 0:
        return
    active = replay.seed_lifecycle.active_mask(int(timestamp))[rows]
    rows = rows[active]
    if rows.numel() == 0:
        return
    replay.seed_lifecycle.close_rows(rows, int(timestamp))
    replay.seed_model.close_rows(rows, int(timestamp))
    replay.seed_retired[rows] = True


def append_typed_seeds(
    replay: Any,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    timestamp: int,
    metadata: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> int:
    """Append Panel-10 NEW DA3 seeds as NEVER_OPEN generation-0 rows.

    The accepted source-row mapping is internal row identity, not the upstream
    DA3 proposal ID.  The unified BF tracker grows as needed; seed rows use
    ``base_offset + internal_seed_row``.  Appending only materializes topology;
    detector evidence is not consumed here, so the caller may append first and
    evaluate the same current view exactly once afterwards.
    """

    _ensure_archive_vectors(replay)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    requested = int(xyz.shape[0])
    if requested == 0:
        return 0
    if scaling.shape[0] != requested:
        raise ValueError("scaling must align with xyz")
    max_rows = int(_arg(replay.args, "da3_max_rows", 20000))
    available = requested if max_rows == 0 else max(0, max_rows - int(replay.seed_model.num_gaussians))
    accepted = min(requested, available)
    if accepted <= 0:
        return 0
    _ensure_tracker_capacity(replay, int(replay.seed_model.num_gaussians) + accepted)

    device = _device_from_replay(replay)
    dtype = _dtype_from_replay(replay)
    xyz = xyz[:accepted].detach().to(device=device, dtype=dtype).contiguous()
    scaling = scaling[:accepted].detach().to(device=device, dtype=dtype).contiguous()
    rows_metadata = _as_metadata_list(metadata, requested)[:accepted]
    first_row = int(replay.seed_model.num_gaussians)
    for offset, row in enumerate(rows_metadata):
        row.setdefault("source", "panel10_da3_new_seed")
        row.setdefault("source_row", first_row + offset)
        row["panel10_type"] = "new"
        row["internal_seed_row"] = first_row + offset
        row["root_seed_row"] = first_row + offset
        row["retirement_semantics"] = "append_only_close_no_compaction"

    rows = replay.seed_model.append_xfeat_anchors(
        xyz=xyz,
        start=float(timestamp),
        scaling=scaling,
        opacity=float(_arg(replay.args, "da3_initial_opacity", 0.1)),
        metadata=rows_metadata,
        optimizer=getattr(replay, "seed_optimizer", None),
        start_active=False,
    )
    expected_rows = torch.arange(first_row, first_row + accepted, device=device, dtype=torch.long)
    if not torch.equal(rows, expected_rows):
        raise RuntimeError("typed seed sidecar did not append in row-identity order")

    replay.seed_detector_probe.append(
        xyz=xyz,
        start=float(timestamp),
        scaling=scaling,
        opacity=float(_arg(replay.args, "da3_initial_opacity", 0.1)),
        metadata=rows_metadata,
    )
    replay.seed_detector_probe.seed_dc.requires_grad_(False)
    lifecycle_rows = replay.seed_lifecycle.append_rows(accepted, materialized_timestamp=int(timestamp))
    if not torch.equal(lifecycle_rows, expected_rows):
        raise RuntimeError("typed seed lifecycle did not append in row-identity order")

    replay.accepted_da3_source_rows.extend(range(first_row, first_row + accepted))
    replay.accepted_da3_birth_global.extend([int(timestamp)] * accepted)
    replay.seed_retired = torch.cat((replay.seed_retired, torch.zeros(accepted, device=device, dtype=torch.bool)))
    replay.seed_geometry_update_counts = torch.cat(
        (replay.seed_geometry_update_counts, torch.zeros(accepted, device=device, dtype=torch.long))
    )
    replay.seed_last_visible_rows = torch.cat(
        (replay.seed_last_visible_rows, torch.zeros(accepted, device=device, dtype=torch.bool))
    )
    _ensure_archive_vectors(replay)
    return accepted


def _active_visible_density_candidates(replay: Any, timestamp: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_archive_vectors(replay)
    model = replay.seed_model
    device = model._xyz.device
    active = replay.seed_lifecycle.active_mask(int(timestamp)).to(device=device)
    visible = replay.seed_last_visible_rows.to(device=device, dtype=torch.bool)
    observed = model.gradient_denom.squeeze(1) > 0
    not_retired = ~replay.seed_retired.to(device=device, dtype=torch.bool)
    base = active & visible & observed & not_retired
    denom = model.gradient_denom.clamp_min(1.0)
    signed = (model.xyz_gradient_accum / denom).squeeze(1)
    absolute = (model.xyz_gradient_accum_abs / denom).squeeze(1)
    return base, signed, absolute


def _root_child_counts(model: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row, meta in enumerate(model.metadata):
        if int(model.generation[row].item()) <= 0:
            continue
        root = int(meta.get("root_seed_row", meta.get("parent_row", row)))
        counts[root] = counts.get(root, 0) + 1
    return counts


def _eligible_by_lineage(replay: Any, rows: torch.Tensor) -> torch.Tensor:
    model = replay.seed_model
    max_generation = int(_arg(replay.args, "da3_max_generation", 2))
    keep = []
    for row in rows.detach().cpu().tolist():
        generation_ok = int(model.generation[row].item()) < max_generation
        one_shot_ok = int(model.densification_count[row].item()) < 1
        keep.append(generation_ok and one_shot_ok)
    return torch.as_tensor(keep, device=rows.device, dtype=torch.bool)


def _root_for_row(model: Any, row: int) -> int:
    return int(model.metadata[int(row)].get("root_seed_row", int(row)))


def _select_with_root_capacity(
    replay: Any,
    rows: torch.Tensor,
    scores: torch.Tensor,
    *,
    per_source_children: int,
    global_child_limit: int,
    root_counts: dict[int, int],
) -> torch.Tensor:
    """Select rows by score without exceeding cumulative per-root child caps."""

    if global_child_limit <= 0 or rows.numel() == 0:
        return rows[:0]
    model = replay.seed_model
    max_root_children = int(_arg(replay.args, "da3_max_root_children", 32))
    order = torch.argsort(scores[rows], descending=True, stable=True)
    selected: list[int] = []
    remaining_global = int(global_child_limit)
    for row in rows[order].detach().cpu().tolist():
        root = _root_for_row(model, int(row))
        remaining_root = max_root_children - int(root_counts.get(root, 0))
        if remaining_root < per_source_children or remaining_global < per_source_children:
            continue
        selected.append(int(row))
        root_counts[root] = int(root_counts.get(root, 0)) + int(per_source_children)
        remaining_global -= int(per_source_children)
    if not selected:
        return rows[:0]
    return torch.as_tensor(selected, device=rows.device, dtype=torch.long)


def _small_by_root_ratio(replay: Any) -> torch.Tensor:
    model = replay.seed_model
    ratio = float(_arg(replay.args, "da3_split_scale_ratio", 1.0))
    current_scale = model.get_scaling.detach().amax(dim=1)
    threshold = model.root_anchor_scale.detach() * ratio
    return current_scale <= threshold


def _extend_probe_from_active_children(replay: Any, child_rows: torch.Tensor) -> None:
    if child_rows.numel() == 0:
        return
    model = replay.seed_model
    probe = replay.seed_detector_probe
    probe.append(
        xyz=model._xyz.detach()[child_rows],
        start=model.start.detach()[child_rows],
        scaling=model._scaling.detach()[child_rows],
        rotation=model._rotation.detach()[child_rows],
        opacity=model.get_opacity.detach()[child_rows],
        dc=model.new_dc.detach()[child_rows],
        metadata=[copy.deepcopy(model.metadata[int(row)]) for row in child_rows.detach().cpu().tolist()],
    )
    probe.seed_dc.requires_grad_(False)


def _append_child_bookkeeping(
    replay: Any,
    *,
    parent_rows: torch.Tensor,
    child_rows: torch.Tensor,
    timestamp: int,
) -> None:
    if child_rows.numel() == 0:
        return
    device = _device_from_replay(replay)
    child_count = int(child_rows.numel())
    lifecycle_rows = replay.seed_lifecycle.append_rows(child_count, materialized_timestamp=int(timestamp))
    if not torch.equal(lifecycle_rows, child_rows.to(device=lifecycle_rows.device)):
        raise RuntimeError("child lifecycle rows are not aligned with model rows")
    replay.seed_lifecycle.open_rows(child_rows, int(timestamp))
    _extend_probe_from_active_children(replay, child_rows)
    replay.accepted_da3_source_rows.extend(int(row) for row in child_rows.detach().cpu().tolist())
    replay.accepted_da3_birth_global.extend([int(timestamp)] * child_count)
    replay.seed_retired = torch.cat((replay.seed_retired, torch.zeros(child_count, device=device, dtype=torch.bool)))
    replay.seed_geometry_update_counts = torch.cat(
        (replay.seed_geometry_update_counts, torch.zeros(child_count, device=device, dtype=torch.long))
    )
    replay.seed_last_visible_rows = torch.cat(
        (replay.seed_last_visible_rows, torch.zeros(child_count, device=device, dtype=torch.bool))
    )
    # ActiveNewGaussianModel stores parent_row but not root row; make root lineage explicit.
    repeat = child_count // max(int(parent_rows.numel()), 1)
    expanded = parent_rows.repeat(repeat) if repeat > 1 else parent_rows
    if expanded.numel() != child_rows.numel():
        expanded = parent_rows.repeat_interleave(repeat)
    for child, parent in zip(child_rows.detach().cpu().tolist(), expanded.detach().cpu().tolist()):
        parent_meta = replay.seed_model.metadata[int(parent)]
        root = int(parent_meta.get("root_seed_row", parent))
        replay.seed_model.metadata[int(child)]["root_seed_row"] = root
        replay.seed_model.metadata[int(child)]["retirement_semantics"] = "append_only_close_no_compaction"
        replay.seed_detector_probe.metadata[int(child)]["root_seed_row"] = root
    base_offset = _base_row_offset(replay)
    _tracker_copy_rows(
        _joint_tracker(replay),
        parent_rows + base_offset,
        child_rows + base_offset,
        timestamp=int(timestamp),
    )
    _ensure_archive_vectors(replay)


@dataclass(frozen=True)
class TypedDensityResult:
    clone_count: int
    split_source_count: int
    children: int
    count_before: int
    count_after: int
    events: list[dict[str, Any]]


def apply_typed_density(replay: Any, timestamp: int, random_seed: int = 0) -> dict[str, Any]:
    """Clone/split currently OPEN visible Panel-10 NEW seed rows.

    This is append-only topology retirement: split sources are CLOSED and marked
    ``seed_retired`` but not physically deleted, preserving past renders and Adam
    moments for audit.  Density statistics are reset after the call.
    """

    _ensure_archive_vectors(replay)
    model = replay.seed_model
    count_before = int(model.num_gaussians)
    max_rows = int(_arg(replay.args, "da3_max_rows", 20000))
    max_children = int(_arg(replay.args, "da3_density_max_children", 128))
    available = max_children if max_rows == 0 else max(0, min(max_children, max_rows - count_before))
    events: list[dict[str, Any]] = []
    if available <= 0:
        model.reset_density_statistics()
        return TypedDensityResult(0, 0, 0, count_before, count_before, events).__dict__

    base, signed, absolute = _active_visible_density_candidates(replay, int(timestamp))
    rows = torch.nonzero(base, as_tuple=False).flatten()
    if rows.numel():
        rows = rows[_eligible_by_lineage(replay, rows)]
    small = _small_by_root_ratio(replay)
    clone_threshold = float(_arg(replay.args, "da3_density_grad_threshold", 2.0e-4))
    split_threshold = float(_arg(replay.args, "da3_density_abs_grad_threshold", 1.2e-3))
    open_ts = replay.seed_lifecycle.open_timestamp.to(device=model._xyz.device)
    clone_rows = rows[small[rows] & (signed[rows] >= clone_threshold)] if rows.numel() else rows
    split_rows = rows[(~small[rows]) & (absolute[rows] >= split_threshold) & (open_ts[rows] < int(timestamp))] if rows.numel() else rows

    split_children = 2
    root_counts = _root_child_counts(model)
    split_sources = _select_with_root_capacity(
        replay,
        split_rows,
        absolute,
        per_source_children=split_children,
        global_child_limit=available,
        root_counts=root_counts,
    )
    available -= int(split_sources.numel()) * split_children
    clone_sources = _select_with_root_capacity(
        replay,
        clone_rows,
        signed,
        per_source_children=1,
        global_child_limit=available,
        root_counts=root_counts,
    )

    # Reserve before mutating model/lifecycle so allocation failure cannot
    # leave children without matching BF rows.
    _ensure_tracker_capacity(
        replay, count_before + int(split_sources.numel()) * split_children + int(clone_sources.numel())
    )

    child_total = 0
    clone_count = 0
    split_count = 0
    if split_sources.numel():
        scores = absolute[split_sources].detach().clone()
        child_rows = model.append_gradient_children(
            parent_rows=split_sources,
            split=True,
            timestamp=int(timestamp),
            optimizer=replay.seed_optimizer,
            trigger_scores=scores,
            children_per_split=split_children,
            random_seed=int(random_seed),
            replace_split_parent=False,
        )
        model.densification_count[split_sources] += 1
        _append_child_bookkeeping(replay, parent_rows=split_sources, child_rows=child_rows, timestamp=int(timestamp))
        _close_retire_rows(replay, split_sources, int(timestamp))
        child_total += int(child_rows.numel())
        split_count += int(split_sources.numel())
        for row, score in zip(split_sources.detach().cpu().tolist(), scores.detach().cpu().tolist()):
            events.append({"timestamp": int(timestamp), "action": "split_retire_parent", "row": int(row), "children": split_children, "trigger_score": float(score)})

    if clone_sources.numel():
        # Split retirement does not compact rows, so clone row indices remain stable.
        scores = signed[clone_sources].detach().clone()
        child_rows = model.append_gradient_children(
            parent_rows=clone_sources,
            split=False,
            timestamp=int(timestamp),
            optimizer=replay.seed_optimizer,
            trigger_scores=scores,
            random_seed=int(random_seed) + 1,
            replace_split_parent=False,
        )
        model.densification_count[clone_sources] += 1
        _append_child_bookkeeping(replay, parent_rows=clone_sources, child_rows=child_rows, timestamp=int(timestamp))
        child_total += int(child_rows.numel())
        clone_count += int(clone_sources.numel())
        for row, score in zip(clone_sources.detach().cpu().tolist(), scores.detach().cpu().tolist()):
            events.append({"timestamp": int(timestamp), "action": "clone", "row": int(row), "children": 1, "trigger_score": float(score)})

    model.reset_density_statistics()
    return TypedDensityResult(clone_count, split_count, child_total, count_before, int(model.num_gaussians), events).__dict__


def prune_typed_seeds(replay: Any, timestamp: int) -> dict[str, Any]:
    """Retire weak learned-opacity NEW rows without physical compaction."""

    _ensure_archive_vectors(replay)
    model = replay.seed_model
    active = replay.seed_lifecycle.active_mask(int(timestamp)).to(device=model._xyz.device)
    visible = replay.seed_last_visible_rows.to(device=model._xyz.device, dtype=torch.bool)
    not_retired = ~replay.seed_retired.to(device=model._xyz.device, dtype=torch.bool)
    open_ts = replay.seed_lifecycle.open_timestamp.to(device=model._xyz.device)
    age_ok = (int(timestamp) - open_ts) >= int(_arg(replay.args, "da3_prune_grace_frames", 3))
    updates_ok = replay.seed_geometry_update_counts.to(device=model._xyz.device) >= int(
        _arg(replay.args, "da3_prune_min_updates", 4)
    )
    opacity_ok = model.get_opacity.detach().squeeze(1) <= float(_arg(replay.args, "da3_prune_opacity", 0.02))
    fresh_ok = open_ts < int(timestamp)
    rows = torch.nonzero(active & visible & not_retired & age_ok & updates_ok & opacity_ok & fresh_ok, as_tuple=False).flatten()
    _close_retire_rows(replay, rows, int(timestamp))
    events = [
        {"timestamp": int(timestamp), "action": "prune_retire", "row": int(row), "reason": "low_opacity_append_only"}
        for row in rows.detach().cpu().tolist()
    ]
    return {"pruned": int(rows.numel()), "retired": int(rows.numel()), "events": events, "count_after": int(model.num_gaussians)}
