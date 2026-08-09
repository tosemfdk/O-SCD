"""Loss helpers shared by baseline and temporal change rendering."""

import torch


def compute_ssf_loss(
    candidate_map: torch.Tensor,
    rendered_change_rgb: torch.Tensor,
    *,
    regularizer_offset: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the original O-SCD self-supervised fusion loss."""
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

    change_probability = torch.sigmoid(
        rendered_change_rgb.mean(dim=0, keepdim=True)
    )
    detection = (candidate_map * (1.0 - change_probability)).mean()
    regularization = torch.log(
        change_probability.mean() ** 2 + regularizer_offset
    )
    loss = detection + regularization
    return loss, {
        "loss": loss,
        "detection": detection,
        "regularization": regularization,
        "change_probability": change_probability,
    }
