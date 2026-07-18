# Deterministic batch-static greedy selection (spec §11-12).
from __future__ import annotations

from typing import Mapping

import torch

from view_selection.criteria import score_candidate
from view_selection.types import SelectionResult, SelectionStep


def greedy_static_select(infos: Mapping[int, torch.Tensor], budget: int,
                         criterion: str, lambda_value: float,
                         model_revision: int = 0) -> SelectionResult:
    """infos: frame_id -> diagonal b (N,) float32 (fixed, reference scoring
    state). Deterministic: ties break to the smaller frame ID; the result is
    invariant to candidate iteration order."""
    ids = sorted(infos)
    if not ids:
        raise ValueError("no candidates")
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if budget > len(ids):
        raise ValueError(f"budget {budget} > candidates {len(ids)}")
    n = infos[ids[0]].shape[0]
    device = infos[ids[0]].device
    for i in ids:
        b = infos[i]
        if b.shape != (n,):
            raise ValueError(f"frame {i}: diagonal shape {tuple(b.shape)}")
        if not torch.isfinite(b).all():
            raise FloatingPointError(f"frame {i}: non-finite information")
        if (b < 0).any():
            raise ValueError(f"frame {i}: negative information")

    prior = torch.full((n,), float(lambda_value), dtype=torch.float32,
                       device=device)
    remaining = list(ids)
    selected: list[int] = []
    steps: list[SelectionStep] = []
    for step in range(budget):
        scores = {fid: score_candidate(criterion, prior, infos[fid])
                  for fid in remaining}
        best = min((-s, fid) for fid, s in scores.items())[1]
        selected.append(best)
        prior.add_(infos[best])
        remaining.remove(best)
        steps.append(SelectionStep(
            step=step, selected_frame_id=best, selected_score=scores[best],
            candidate_scores=dict(sorted(scores.items())),
            lambda_value=float(lambda_value), gaussian_count=n,
            model_revision=model_revision))
    return SelectionResult(greedy_order=selected,
                           replay_order=sorted(selected), steps=steps)
