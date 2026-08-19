"""Adam-style optimizers for row/slot state tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import torch
from torch.optim import Optimizer


ALLOWED_NAMES = ("dc", "xyz", "opacity", "scaling", "rotation")


class MaskedRowSlotAdam(Optimizer):
    """Adam optimizer that updates only selected Gaussian row-slot pairs.

    Parameters are indexed by ``[N, S, ...]`` and only the entries where
    ``active_mask[row, slot]`` is true receive gradient updates. Parameters and
    optimizer moments for inactive pairs are preserved exactly.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
        *,
        thaw_names: Iterable[str] = ALLOWED_NAMES,
        lrs: Mapping[str, float] | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        named_parameters = dict(named_parameters)
        allowed = set(ALLOWED_NAMES)
        requested_names = tuple(thaw_names)
        if not requested_names:
            raise ValueError("at least one thawed parameter must be enabled")
        if len(requested_names) != len(set(requested_names)):
            raise ValueError("thaw_names must not contain duplicates")
        unknown = sorted(set(requested_names) - allowed)
        if unknown:
            raise ValueError(f"unknown thaw parameter names: {unknown}")
        active_names = requested_names
        if lrs is None:
            lrs = {}
        for name in lrs:
            if name not in allowed:
                raise ValueError(f"unknown parameter name for lr: {name}")

        selected: list[dict[str, Any]] = []
        pair_shape: tuple[int, int] | None = None
        for name in ALLOWED_NAMES:
            if name in active_names and name not in named_parameters:
                raise KeyError(f"missing expected parameter: {name}")
            if name not in active_names:
                continue
            parameter = named_parameters[name]
            lr = float(lrs.get(name, 1e-3))
            if not math.isfinite(lr) or lr < 0:
                raise ValueError(
                    f"learning rate for {name} must be finite and nonnegative"
                )
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"{name} must be a torch.nn.Parameter")
            if parameter.ndim < 2 or not torch.is_floating_point(parameter):
                raise ValueError(f"{name} must be a floating [N,S,...] parameter")
            current_shape = tuple(parameter.shape[:2])
            if pair_shape is None:
                pair_shape = current_shape
            elif current_shape != pair_shape:
                raise ValueError("all temporal parameters must share [N,S]")
            selected.append({"params": [parameter], "lr": lr, "name": name})

        if not (
            isinstance(betas, tuple)
            and len(betas) == 2
            and all(
                math.isfinite(float(beta)) and 0 <= beta < 1 for beta in betas
            )
        ):
            raise ValueError("betas must be a pair in [0,1)")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")

        defaults = {
            "lr": 1e-3,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(selected, defaults)

        for group in self.param_groups:
            parameter = group["params"][0]
            n, s_states = parameter.shape[:2]
            self.state[parameter]["step"] = torch.zeros((n, s_states), device=parameter.device, dtype=torch.long)
            self.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
            self.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
            if amsgrad:
                self.state[parameter]["max_exp_avg_sq"] = torch.zeros_like(parameter)

    @staticmethod
    def _validate_pair_mask(mask: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2 or not mask.dtype == torch.bool:
            raise ValueError("pair mask must be a boolean [N, S] tensor")
        n, s = parameter.shape[:2]
        if tuple(mask.shape) != (n, s):
            raise ValueError(
                f"pair mask shape {tuple(mask.shape)} does not match parameter shape {tuple(parameter.shape[:2])}"
            )
        if mask.device != parameter.device:
            mask = mask.to(device=parameter.device)
        return mask

    @torch.no_grad()
    def reset_state_pairs(self, active_mask: torch.Tensor) -> None:
        """Zero optimizer state buffers for exactly selected row-slot pairs."""
        for group in self.param_groups:
            parameter = group["params"][0]
            state = self.state[parameter]
            mask = self._validate_pair_mask(active_mask, parameter)
            flat_mask = mask.reshape(-1)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = state["step"]

            exp_avg_view = exp_avg.reshape(exp_avg.shape[0] * exp_avg.shape[1], -1)
            exp_avg_sq_view = exp_avg_sq.reshape(exp_avg_sq.shape[0] * exp_avg_sq.shape[1], -1)
            exp_avg_view[flat_mask] = 0
            exp_avg_sq_view[flat_mask] = 0
            step.reshape(-1)[flat_mask] = 0
            if "max_exp_avg_sq" in state:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                max_view = max_exp_avg_sq.reshape(max_exp_avg_sq.shape[0] * max_exp_avg_sq.shape[1], -1)
                max_view[flat_mask] = 0


    @torch.no_grad()
    def step(self, active_mask: torch.Tensor, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a boolean tensor")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            amsgrad = bool(group["amsgrad"])
            parameter = group["params"][0]
            grad = parameter.grad
            if grad is None:
                continue

            mask = self._validate_pair_mask(active_mask, parameter)
            flat_mask = mask.reshape(-1)
            if not bool(flat_mask.any()):
                continue

            state = self.state[parameter]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step_state = state["step"]

            n, s = parameter.shape[:2]
            grad_pair = grad.reshape(n * s, -1)
            parameter_pair = parameter.reshape(n * s, -1)
            exp_avg_pair = exp_avg.reshape(n * s, -1)
            exp_avg_sq_pair = exp_avg_sq.reshape(n * s, -1)
            step_pair = step_state.reshape(n * s)
            active_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()

            if group["amsgrad"] and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(exp_avg)
            if amsgrad:
                max_exp_avg_sq_pair = state["max_exp_avg_sq"].reshape(n * s, -1)

            g = grad_pair.index_select(0, active_indices)
            if weight_decay != 0:
                g = g + weight_decay * parameter_pair.index_select(0, active_indices)

            step_selected = step_pair.index_select(0, active_indices).add_(1)
            step_pair.index_copy_(0, active_indices, step_selected)
            step_f = step_selected.to(dtype=parameter.dtype).unsqueeze(1)

            beta1_t = torch.pow(torch.as_tensor(beta1, device=parameter.device, dtype=parameter.dtype), step_f)
            beta2_t = torch.pow(torch.as_tensor(beta2, device=parameter.device, dtype=parameter.dtype), step_f)
            bias_correction1 = 1.0 - beta1_t
            bias_correction2 = 1.0 - beta2_t

            exp_active = exp_avg_pair.index_select(0, active_indices).mul_(beta1).add_(
                g, alpha=1 - beta1
            )
            exp_sq_active = exp_avg_sq_pair.index_select(0, active_indices).mul_(beta2).addcmul_(
                g, g, value=1 - beta2
            )

            exp_avg_pair.index_copy_(0, active_indices, exp_active)
            exp_avg_sq_pair.index_copy_(0, active_indices, exp_sq_active)

            denom_sq = exp_sq_active / bias_correction2
            if amsgrad:
                max_active = max_exp_avg_sq_pair.index_select(0, active_indices)
                torch.maximum(max_active, exp_sq_active, out=max_active)
                max_exp_avg_sq_pair.index_copy_(0, active_indices, max_active)
                denom_sq = max_active / bias_correction2

            denom = denom_sq.sqrt().add_(group["eps"])
            step_term = (exp_active / bias_correction1) / denom
            parameter_pair.index_add_(0, active_indices, -lr * step_term)

        return loss
