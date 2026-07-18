# Selection criteria over diagonal change information (spec §6.1).
# Element computation float32, final sums accumulated in float64.
from __future__ import annotations

import torch

CRITERIA = ("dopt", "trace_reduction", "fisher_ratio", "candidate_only")


def score_candidate(criterion: str, prior: torch.Tensor,
                    candidate_info: torch.Tensor) -> float:
    """prior h (>0, includes lambda) and candidate b, both (N,) float32."""
    if prior.shape != candidate_info.shape:
        raise ValueError(f"shape mismatch {prior.shape} vs {candidate_info.shape}")
    b = candidate_info
    h = prior
    if criterion == "dopt":
        val = torch.log1p(b / h).double().sum()
    elif criterion == "trace_reduction":
        val = (1.0 / h - 1.0 / (h + b)).double().sum()
    elif criterion == "fisher_ratio":
        val = (b / h).double().sum()
    elif criterion == "candidate_only":
        val = b.double().sum()
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    out = float(val)
    if out != out:  # NaN
        raise FloatingPointError(f"{criterion} score is NaN")
    return out
