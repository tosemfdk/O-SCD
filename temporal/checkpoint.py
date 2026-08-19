"""Checkpoint loading shared by temporal experiments and interactive viewers."""

from pathlib import Path

import torch

from scene import GaussianModel

from .change_model import TemporalChangeModel
from .geometry_change_model import TemporalGeometryChangeModel
from .shared_geometry_change_model import TemporalSharedGeometryChangeModel


def migrate_temporal_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add deterministic defaults required by newer temporal schemas."""
    migrated = dict(state_dict)
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
    model.eval()
    return model
