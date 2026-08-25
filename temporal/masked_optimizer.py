"""Adam-style optimizers for row/slot state tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import torch
from torch.optim import Optimizer


ALLOWED_NAMES = ("dc", "xyz", "opacity", "scaling", "rotation")
PERSISTENT_ALLOWED_NAMES = (
    "dc",
    "xyz",
    "features_rest",
    "opacity",
    "scaling",
    "rotation",
)


def active_visible_pair_mask(
    active_pair_mask: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Restrict current lifespan pairs to rows visible in one render.

    ``render_change_temporal`` follows the Gaussian renderer convention that a
    row is visible when its rasterized radius is positive.  The returned mask
    can be passed directly to :class:`MaskedRowSlotAdam`, ensuring that an OPEN
    pair outside the current camera view preserves both its parameter value and
    its optimizer-owned moments exactly.
    """
    if active_pair_mask.ndim != 2 or active_pair_mask.dtype != torch.bool:
        raise ValueError("active_pair_mask must be a boolean [N, S] tensor")
    if radii.ndim != 1 or radii.shape[0] != active_pair_mask.shape[0]:
        raise ValueError("radii must have shape [N] matching active_pair_mask")
    if radii.device != active_pair_mask.device:
        radii = radii.to(device=active_pair_mask.device)
    return active_pair_mask & (radii.detach() > 0).unsqueeze(1)


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


class MaskedRowAdam(Optimizer):
    """Adam that updates only explicitly selected Gaussian rows.

    This optimizer is the direct-parameter counterpart of
    :class:`MaskedRowSlotAdam`.  Its tensors have shape ``[N, ...]`` rather
    than ``[N, S, ...]`` because the parameters persist across every lifespan.
    Inactive and current-view-invisible rows preserve both values and Adam
    moments exactly.  Reopening a row intentionally resumes those moments.
    """

    def __init__(
        self,
        named_parameters: Mapping[
            str, torch.nn.Parameter
        ] | Iterable[tuple[str, torch.nn.Parameter]],
        *,
        thaw_names: Iterable[str] = PERSISTENT_ALLOWED_NAMES,
        lrs: Mapping[str, float] | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        named_parameters = dict(named_parameters)
        requested = tuple(thaw_names)
        allowed = set(PERSISTENT_ALLOWED_NAMES)
        if not requested:
            raise ValueError("at least one persistent parameter must be enabled")
        if len(requested) != len(set(requested)):
            raise ValueError("thaw_names must not contain duplicates")
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"unknown persistent parameter names: {unknown}")
        lrs = {} if lrs is None else dict(lrs)
        unknown_lrs = sorted(set(lrs) - allowed)
        if unknown_lrs:
            raise ValueError(f"unknown persistent learning-rate names: {unknown_lrs}")

        groups: list[dict[str, Any]] = []
        row_count: int | None = None
        for name in PERSISTENT_ALLOWED_NAMES:
            if name not in requested:
                continue
            if name not in named_parameters:
                raise KeyError(f"missing expected persistent parameter: {name}")
            parameter = named_parameters[name]
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(f"{name} must be an nn.Parameter")
            if parameter.ndim < 1 or not torch.is_floating_point(parameter):
                raise ValueError(f"{name} must be a floating [N,...] parameter")
            if row_count is None:
                row_count = int(parameter.shape[0])
            elif int(parameter.shape[0]) != row_count:
                raise ValueError("all persistent parameters must share row count N")
            lr = float(lrs.get(name, 1e-3))
            if not math.isfinite(lr) or lr < 0:
                raise ValueError(f"learning rate for {name} must be finite and nonnegative")
            groups.append({"params": [parameter], "lr": lr, "name": name})

        if not (
            isinstance(betas, tuple)
            and len(betas) == 2
            and all(math.isfinite(float(v)) and 0 <= float(v) < 1 for v in betas)
        ):
            raise ValueError("betas must be a pair in [0,1)")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        super().__init__(
            groups,
            {
                "lr": 1e-3,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
            },
        )
        for group in self.param_groups:
            parameter = group["params"][0]
            n = int(parameter.shape[0])
            self.state[parameter]["step"] = torch.zeros(
                n, device=parameter.device, dtype=torch.long
            )
            self.state[parameter]["exp_avg"] = torch.zeros_like(parameter)
            self.state[parameter]["exp_avg_sq"] = torch.zeros_like(parameter)
            if amsgrad:
                self.state[parameter]["max_exp_avg_sq"] = torch.zeros_like(parameter)

    @staticmethod
    def _validate_row_mask(mask: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError("row mask must be boolean [N]")
        if mask.shape[0] != parameter.shape[0]:
            raise ValueError("row mask must match persistent parameter row count")
        return mask.to(device=parameter.device)

    @torch.no_grad()
    def step(self, active_rows: torch.Tensor, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if active_rows.dtype != torch.bool or active_rows.ndim != 1:
            raise ValueError("active_rows must be a boolean [N] tensor")

        for group in self.param_groups:
            parameter = group["params"][0]
            gradient = parameter.grad
            if gradient is None:
                continue
            mask = self._validate_row_mask(active_rows, parameter)
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            state = self.state[parameter]
            beta1, beta2 = group["betas"]
            parameter_rows = parameter.reshape(parameter.shape[0], -1)
            gradient_rows = gradient.reshape(gradient.shape[0], -1)
            exp_avg_rows = state["exp_avg"].reshape(parameter.shape[0], -1)
            exp_avg_sq_rows = state["exp_avg_sq"].reshape(parameter.shape[0], -1)
            step_rows = state["step"]

            selected_gradient = gradient_rows.index_select(0, indices)
            if float(group["weight_decay"]) != 0.0:
                selected_gradient = selected_gradient + float(
                    group["weight_decay"]
                ) * parameter_rows.index_select(0, indices)
            selected_step = step_rows.index_select(0, indices).add_(1)
            step_rows.index_copy_(0, indices, selected_step)
            step_float = selected_step.to(dtype=parameter.dtype).unsqueeze(1)

            selected_avg = exp_avg_rows.index_select(0, indices).mul_(beta1).add_(
                selected_gradient, alpha=1 - beta1
            )
            selected_avg_sq = exp_avg_sq_rows.index_select(0, indices).mul_(
                beta2
            ).addcmul_(selected_gradient, selected_gradient, value=1 - beta2)
            exp_avg_rows.index_copy_(0, indices, selected_avg)
            exp_avg_sq_rows.index_copy_(0, indices, selected_avg_sq)

            beta1_power = torch.pow(
                torch.as_tensor(beta1, device=parameter.device, dtype=parameter.dtype),
                step_float,
            )
            beta2_power = torch.pow(
                torch.as_tensor(beta2, device=parameter.device, dtype=parameter.dtype),
                step_float,
            )
            variance = selected_avg_sq / (1.0 - beta2_power)
            if bool(group["amsgrad"]):
                max_rows = state["max_exp_avg_sq"].reshape(parameter.shape[0], -1)
                selected_max = max_rows.index_select(0, indices)
                torch.maximum(selected_max, selected_avg_sq, out=selected_max)
                max_rows.index_copy_(0, indices, selected_max)
                variance = selected_max / (1.0 - beta2_power)
            update = (selected_avg / (1.0 - beta1_power)) / (
                variance.sqrt().add_(float(group["eps"]))
            )
            parameter_rows.index_add_(0, indices, -float(group["lr"]) * update)
        return loss
