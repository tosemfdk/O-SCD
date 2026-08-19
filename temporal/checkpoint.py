"""Checkpoint loading shared by temporal experiments and interactive viewers."""

from pathlib import Path

import torch

from scene import GaussianModel

from .change_model import CLOSED, OPEN, TemporalChangeModel
from .geometry_change_model import TemporalGeometryChangeModel
from .shared_geometry_change_model import TemporalSharedGeometryChangeModel


def _infer_lifecycle_buffers(migrated: dict[str, torch.Tensor]) -> None:
    if not {"state_valid", "state_end"}.issubset(migrated):
        return
    valid = migrated["state_valid"].bool()
    end = migrated["state_end"]
    status = torch.zeros(valid.shape, dtype=torch.int8, device=valid.device)
    finite = torch.isfinite(end)
    positive_inf = torch.isposinf(end)
    if bool((valid & ~(finite | positive_inf)).any()):
        raise ValueError(
            "valid legacy state_end values must be finite or positive infinity"
        )
    status[valid & finite] = CLOSED
    status[valid & positive_inf] = OPEN
    slot_ids = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    migrated["num_states"] = valid.sum(dim=1, dtype=torch.long)
    open_slot = torch.where(
        status == OPEN, slot_ids, torch.full_like(slot_ids, -1)
    ).amax(dim=1)
    migrated["current_state_index"] = open_slot.long()
    migrated["state_status"] = status


def migrate_temporal_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add deterministic defaults required by newer temporal schemas."""
    migrated = dict(state_dict)
    _infer_lifecycle_buffers(migrated)
    if "shared_xyz_delta" in migrated and "geometry_frozen" not in migrated:
        gaussian_count = int(migrated["shared_xyz_delta"].shape[0])
        migrated["geometry_frozen"] = torch.zeros(
            gaussian_count,
            dtype=torch.bool,
            device=migrated["shared_xyz_delta"].device,
        )
    return migrated


def load_temporal_model(
    checkpoint_path: str | Path,
) -> (
    TemporalChangeModel
    | TemporalGeometryChangeModel
    | TemporalSharedGeometryChangeModel
):
    """Reconstruct a temporal sidecar and its fixed base on CUDA."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    base_ply = Path(checkpoint["base_ply"])
    if not base_ply.exists():
        raise FileNotFoundError(base_ply)
    state_dict = migrate_temporal_state_dict(checkpoint["state_dict"])
    max_states = int(state_dict["state_change_dc"].shape[1])
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    if "shared_xyz_delta" in state_dict:
        model_class = TemporalSharedGeometryChangeModel
    elif "state_xyz_delta" in state_dict:
        model_class = TemporalGeometryChangeModel
    else:
        model_class = TemporalChangeModel
    model = model_class.from_gaussians(
        base,
        max_states=max_states,
        initial_time=0.0,
    )
    cuda_state = {
        name: value.cuda(non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in state_dict.items()
    }
    model.load_state_dict(cuda_state, strict=True)
    model.validate_lifecycle()
    model.eval()
    return model
