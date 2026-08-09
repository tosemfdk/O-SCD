"""Materialize a DC-only temporal R_change state for the static Viser viewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from plyfile import PlyData, PlyElement


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float_or_none(value: torch.Tensor) -> float | None:
    scalar = float(value.item())
    return scalar if torch.isfinite(value) else None


def _resolve_base_ply(checkpoint_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (checkpoint_path.parent / path).resolve()


def _load_dc_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("temporal checkpoint must be a dictionary")
    contract = str(checkpoint.get("contract", ""))
    if "dc_only" not in contract:
        raise ValueError(
            "viewer export currently supports DC-only temporal checkpoints; "
            f"got contract={contract!r}"
        )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("temporal checkpoint is missing state_dict")
    required = ("state_change_dc", "state_start", "state_end", "state_valid")
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(f"temporal checkpoint is missing tensors: {missing}")
    base_ply_value = checkpoint.get("base_ply")
    if not base_ply_value:
        raise ValueError("temporal checkpoint is missing base_ply")
    base_ply = _resolve_base_ply(checkpoint_path, base_ply_value)
    if not base_ply.is_file():
        raise FileNotFoundError(f"base PLY not found: {base_ply}")
    return checkpoint, state_dict, base_ply


def export_dc_state_for_viewer(
    checkpoint: str | Path,
    output_ply: str | Path,
    *,
    state_id: int = 0,
) -> dict[str, Any]:
    """Export one trained lifespan slot as an ordinary change PLY.

    The temporal renderer gives inactive rows exactly zero opacity. The static
    viewer cannot carry the lifespan sidecar, so those rows are removed. Active
    rows retain the reference geometry, scale, rotation, and raw opacity while
    their DC values are replaced by the selected learned temporal slot.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_ply).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0:
        raise ValueError("state_id must be a non-negative integer")

    payload, state_dict, base_ply = _load_dc_checkpoint(checkpoint_path)
    dc = state_dict["state_change_dc"]
    valid = state_dict["state_valid"]
    starts = state_dict["state_start"]
    ends = state_dict["state_end"]
    if dc.ndim != 4 or tuple(dc.shape[2:]) != (1, 3):
        raise ValueError(f"state_change_dc must have shape [N,S,1,3], got {tuple(dc.shape)}")
    if valid.shape != dc.shape[:2] or starts.shape != valid.shape or ends.shape != valid.shape:
        raise ValueError("temporal state tensor shapes are inconsistent")
    if state_id >= dc.shape[1]:
        raise ValueError(f"state_id {state_id} is outside [0, {dc.shape[1] - 1}]")
    if not torch.isfinite(dc[:, state_id]).all():
        raise ValueError(f"state {state_id} contains non-finite DC values")

    active = valid[:, state_id].to(dtype=torch.bool).contiguous()
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError(f"state {state_id} has no active Gaussians to inspect")

    source = PlyData.read(str(base_ply))
    if not source.elements or source.elements[0].name != "vertex":
        raise ValueError("base PLY must contain a leading vertex element")
    vertices = source.elements[0]
    if len(vertices.data) != dc.shape[0]:
        raise ValueError(
            "checkpoint/base PLY Gaussian count mismatch: "
            f"{dc.shape[0]} != {len(vertices.data)}"
        )
    names = set(vertices.data.dtype.names or ())
    dc_fields = ("f_dc_0", "f_dc_1", "f_dc_2")
    missing_fields = [name for name in dc_fields if name not in names]
    if missing_fields:
        raise ValueError(f"base PLY is missing DC fields: {missing_fields}")

    selected = vertices.data[active.numpy()].copy()
    selected_dc = dc[active, state_id, 0].detach().to(torch.float32).numpy()
    for channel, field in enumerate(dc_fields):
        selected[field] = selected_dc[:, channel]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    exported_vertex = PlyElement.describe(
        selected,
        vertices.name,
        comments=list(vertices.comments),
    )
    PlyData(
        [exported_vertex],
        text=source.text,
        byte_order=source.byte_order,
        comments=list(source.comments),
        obj_info=list(source.obj_info),
    ).write(str(temp_path))
    temp_path.replace(output_path)

    active_starts = starts[active, state_id]
    active_ends = ends[active, state_id]
    manifest = {
        "schema_version": 1,
        "kind": "dc_only_temporal_state_viewer_export",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_contract": str(payload.get("contract", "")),
        "base_ply": str(base_ply),
        "base_ply_sha256": _sha256(base_ply),
        "output_ply": str(output_path),
        "output_ply_sha256": _sha256(output_path),
        "state_id": state_id,
        "state_start_min": _finite_float_or_none(active_starts.min()),
        "state_start_max": _finite_float_or_none(active_starts.max()),
        "state_end_min": _finite_float_or_none(active_ends.min()),
        "state_end_max": _finite_float_or_none(active_ends.max()),
        "source_gaussian_count": int(dc.shape[0]),
        "active_gaussian_count": active_count,
        "inactive_gaussian_count": int(dc.shape[0]) - active_count,
        "inactive_policy": "pruned_because_temporal_renderer_assigns_exact_zero_opacity",
        "preserved_attributes": ["xyz", "raw_opacity", "raw_scaling", "raw_rotation", "features_rest"],
        "replaced_attributes": ["features_dc"],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def ensure_dc_state_viewer_export(
    checkpoint: str | Path,
    *,
    state_id: int = 0,
    output_ply: str | Path | None = None,
) -> dict[str, Any]:
    """Reuse a matching viewer export or materialize it when absent/stale."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if output_ply is None:
        output_path = checkpoint_path.parent / "viewer" / f"state_{state_id:03d}_change.ply"
    else:
        output_path = Path(output_ply).expanduser().resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                int(manifest.get("state_id", -1)) == state_id
                and Path(manifest.get("checkpoint", "")).resolve() == checkpoint_path
                and manifest.get("checkpoint_sha256") == _sha256(checkpoint_path)
                and manifest.get("output_ply_sha256") == _sha256(output_path)
            ):
                manifest["manifest"] = str(manifest_path)
                manifest["reused"] = True
                return manifest
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    manifest = export_dc_state_for_viewer(
        checkpoint_path,
        output_path,
        state_id=state_id,
    )
    manifest["reused"] = False
    return manifest


__all__ = ["ensure_dc_state_viewer_export", "export_dc_state_for_viewer"]
