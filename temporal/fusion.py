"""Loss helpers shared by baseline and temporal change rendering."""

import torch


def compute_growth_replay_regularization(
    local_support: torch.Tensor,
    change_probability: torch.Tensor,
    previous_growth_map: torch.Tensor,
) -> torch.Tensor:
    """Penalize current mask mass where the prior update grew without support.

    All three tensors live in the previous camera's pixel coordinates.  The
    growth map is treated as fixed evidence, while gradients flow through the
    current rendering of that previous view.
    """
    if local_support.ndim != 3 or local_support.shape[0] != 1:
        raise ValueError("local_support must have shape [1, H, W]")
    if change_probability.shape != local_support.shape:
        raise ValueError(
            "change_probability and local_support must share shape"
        )
    if previous_growth_map.shape != local_support.shape:
        raise ValueError(
            "previous_growth_map and local_support must share shape"
        )
    if not (
        local_support.device
        == change_probability.device
        == previous_growth_map.device
    ):
        raise ValueError("growth replay tensors must share a device")
    if not all(
        torch.is_floating_point(value)
        for value in (local_support, change_probability, previous_growth_map)
    ):
        raise TypeError("growth replay tensors must be floating-point")

    return (
        previous_growth_map.detach()
        * (1.0 - local_support)
        * change_probability
    ).mean()


def compute_ssf_loss(
    candidate_map: torch.Tensor,
    rendered_change_rgb: torch.Tensor,
    *,
    regularizer_offset: float = 1.0,
    regularization_mode: str = "global",
    local_support: torch.Tensor | None = None,
    regularization_weight: float = 1.0,
    previous_growth: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the O-SCD fusion loss with selectable sparsity regularization.

    The default arguments preserve the original O-SCD objective exactly.
    ``local_support`` is an explicitly normalized map in ``[0, 1]`` and is
    required by modes with a local term. In modes with a previous-update
    term, ``previous_growth`` is a detached scalar in ``[0, 1]`` that weights
    the original view-global regularizer.
    """
    if candidate_map.ndim != 3 or candidate_map.shape[0] != 1:
        raise ValueError("candidate_map must have shape [1, H, W]")
    if rendered_change_rgb.ndim != 3 or rendered_change_rgb.shape[0] != 3:
        raise ValueError("rendered_change_rgb must have shape [3, H, W]")
    if candidate_map.shape[1:] != rendered_change_rgb.shape[1:]:
        raise ValueError("candidate_map and rendered_change_rgb must share H and W")
    if candidate_map.device != rendered_change_rgb.device:
        raise ValueError("candidate_map and rendered_change_rgb must share a device")
    if not torch.is_floating_point(candidate_map) or not torch.is_floating_point(
        rendered_change_rgb
    ):
        raise TypeError("SSF inputs must be floating-point tensors")
    if regularizer_offset <= 0.0:
        raise ValueError("regularizer_offset must be positive")
    if regularization_weight < 0.0:
        raise ValueError("regularization_weight must be nonnegative")
    valid_modes = {
        "global",
        "local",
        "previous_update_global",
        "local_plus_previous_update_global",
    }
    if regularization_mode not in valid_modes:
        raise ValueError(
            "regularization_mode must be 'global', 'local', "
            "'previous_update_global', or "
            "'local_plus_previous_update_global'"
        )
    uses_local = regularization_mode in {
        "local",
        "local_plus_previous_update_global",
    }
    uses_previous_growth = regularization_mode in {
        "previous_update_global",
        "local_plus_previous_update_global",
    }
    if uses_local:
        if not isinstance(local_support, torch.Tensor):
            raise TypeError("local_support must be a tensor in local mode")
        if local_support.shape != candidate_map.shape:
            raise ValueError("local_support and candidate_map must share shape")
        if local_support.device != candidate_map.device:
            raise ValueError("local_support and candidate_map must share a device")
        if not torch.is_floating_point(local_support):
            raise TypeError("local_support must be floating-point")
    elif local_support is not None:
        raise ValueError("local_support is only valid in local mode")
    if uses_previous_growth:
        if previous_growth is None:
            raise TypeError(
                "previous_growth is required in previous_update_global mode"
            )
        previous_growth_tensor = torch.as_tensor(
            previous_growth,
            dtype=candidate_map.dtype,
            device=candidate_map.device,
        )
        if previous_growth_tensor.numel() != 1:
            raise ValueError("previous_growth must be scalar")
        previous_growth_tensor = previous_growth_tensor.reshape(())
        if not bool(torch.isfinite(previous_growth_tensor).item()):
            raise ValueError("previous_growth must be finite")
        previous_growth_value = float(previous_growth_tensor.item())
        if not 0.0 <= previous_growth_value <= 1.0:
            raise ValueError("previous_growth must be in [0, 1]")
        previous_growth_tensor = previous_growth_tensor.detach()
    elif previous_growth is not None:
        raise ValueError(
            "previous_growth is only valid in a previous-update mode"
        )

    change_probability = torch.sigmoid(
        rendered_change_rgb.mean(dim=0, keepdim=True)
    )
    detection = (candidate_map * (1.0 - change_probability)).mean()
    if regularization_mode == "global":
        regularization = torch.log(
            change_probability.mean() ** 2 + regularizer_offset
        )
    elif regularization_mode == "local":
        regularization = ((1.0 - local_support) * change_probability).mean()
    elif regularization_mode == "previous_update_global":
        regularization = previous_growth_tensor * torch.log(
            change_probability.mean() ** 2 + regularizer_offset
        )
    else:
        local_regularization = (
            (1.0 - local_support) * change_probability
        ).mean()
        previous_global_regularization = previous_growth_tensor * torch.log(
            change_probability.mean() ** 2 + regularizer_offset
        )
        regularization = local_regularization + previous_global_regularization
    loss = detection + regularization_weight * regularization
    return loss, {
        "loss": loss,
        "detection": detection,
        "regularization": regularization,
        "change_probability": change_probability,
    }
