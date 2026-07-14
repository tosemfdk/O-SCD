# Target identity resolution and episode freeze (docs/target_gaussian_nbv.md §9).
# The persistent-ID buffer itself lives on GaussianModel; this module is the
# adapter layer that target_nbv code talks to.

from __future__ import annotations

from contextlib import contextmanager

import torch

from target_nbv.types import TargetHandle


class UnknownTargetError(KeyError):
    pass


def resolve_target_by_persistent_id(model, persistent_id: int) -> int:
    """Return the current row index for a persistent Gaussian ID."""
    if model._persistent_id.numel() == 0:
        raise UnknownTargetError("model has no persistent IDs (old checkpoint?)")
    rows = (model._persistent_id == int(persistent_id)).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        raise UnknownTargetError(f"persistent id {persistent_id} not present (pruned?)")
    if rows.numel() > 1:
        raise RuntimeError(f"persistent id {persistent_id} is duplicated — ID invariant broken")
    return int(rows.item())


def resolve_target_by_current_index(model, index: int) -> int:
    """Return the persistent ID currently living at a row index."""
    n = model._persistent_id.numel()
    if not (0 <= index < n):
        raise UnknownTargetError(f"row index {index} out of range [0, {n})")
    return int(model._persistent_id[index].item())


def make_handle(model, persistent_id: int, frozen: bool = True) -> TargetHandle:
    row = resolve_target_by_persistent_id(model, persistent_id)
    return TargetHandle(persistent_id=int(persistent_id), current_indices=[row],
                        frozen_during_episode=frozen)


def is_target_frozen(model, persistent_id: int) -> bool:
    return int(persistent_id) in model._protected_pids


def begin_target_episode(model, persistent_id: int) -> int:
    resolve_target_by_persistent_id(model, persistent_id)  # existence check
    model.protect_ids([persistent_id])
    return int(persistent_id)


def end_target_episode(model, persistent_id: int) -> None:
    model.unprotect_ids([persistent_id])


@contextmanager
def target_episode(model, persistent_id: int):
    """Freeze the target for the duration of a selection episode."""
    begin_target_episode(model, persistent_id)
    try:
        yield make_handle(model, persistent_id)
    finally:
        end_target_episode(model, persistent_id)
