# Target parameter adapter (docs/target_gaussian_nbv.md §3, stage 6).
# theta_t = [mu_xyz, log_s_xyz] (6,). _scaling is stored in log-space, so both
# groups are identity mappings onto the model tensors. Writes go through .data
# so optimizer state stays aligned; perturbations must always be wrapped in
# perturbed() (try/finally restore).

from __future__ import annotations

from contextlib import contextmanager

import torch

from target_nbv.types import TargetParameterSpec

_GROUP_TENSORS = {"mean": "_xyz", "log_scale": "_scaling"}


def get_theta(model, row: int, spec: TargetParameterSpec) -> torch.Tensor:
    """Packed float64 CPU copy of the target's parameter block."""
    parts = [getattr(model, _GROUP_TENSORS[name])[row].detach().double().cpu()
             for name in spec.parameter_names]
    return torch.cat(parts)


def set_theta(model, row: int, spec: TargetParameterSpec, theta: torch.Tensor) -> None:
    sl = spec.slices()
    for name in spec.parameter_names:
        tensor = getattr(model, _GROUP_TENSORS[name])
        with torch.no_grad():
            tensor.data[row] = theta[sl[name]].to(tensor.device, tensor.dtype)


def fd_epsilons(model, row: int, spec: TargetParameterSpec,
                mean_epsilon_rel: float, log_scale_epsilon: float) -> torch.Tensor:
    """Per-parameter central-difference steps, float64 (D,)."""
    r_world = float(model.get_scaling[row].detach().max())
    eps = torch.empty(spec.dimension, dtype=torch.float64)
    sl = spec.slices()
    for name in spec.parameter_names:
        if name == "mean":
            eps[sl[name]] = mean_epsilon_rel * r_world
        elif name == "log_scale":
            eps[sl[name]] = log_scale_epsilon
    return eps


@contextmanager
def perturbed(model, row: int, spec: TargetParameterSpec, theta: torch.Tensor,
              verify_restoration: bool = False):
    """Temporarily install `theta` on the target row; restore exactly on exit."""
    theta0 = get_theta(model, row, spec)
    try:
        set_theta(model, row, spec, theta)
        yield
    finally:
        set_theta(model, row, spec, theta0)
        if verify_restoration:
            after = get_theta(model, row, spec)
            if not torch.equal(after, theta0):
                raise RuntimeError("model restoration after perturbation failed")
