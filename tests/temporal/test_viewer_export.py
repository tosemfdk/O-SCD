import json
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from temporal.viewer_export import ensure_dc_state_viewer_export, export_dc_state_for_viewer


def _write_base_ply(path: Path, count: int = 4) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    data = np.zeros(count, dtype=dtype)
    data["x"] = np.arange(count, dtype=np.float32)
    data["opacity"] = np.linspace(-2.0, 2.0, count, dtype=np.float32)
    data["scale_0"] = 1.0
    data["scale_1"] = 2.0
    data["scale_2"] = 3.0
    data["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def _write_checkpoint(path: Path, base_ply: Path, *, active=(True, False, True, False)) -> None:
    n = len(active)
    dc = torch.zeros(n, 2, 1, 3)
    dc[:, 0, 0] = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    dc[:, 1, 0] = 100.0
    valid = torch.tensor(active, dtype=torch.bool)[:, None].repeat(1, 2)
    starts = torch.tensor([[0.0, 5.0]]).repeat(n, 1)
    ends = torch.tensor([[5.0, float("inf")]]).repeat(n, 1)
    torch.save(
        {
            "contract": "fixed_topology_oracle_boundary_dc_only_oracle_stream_no_gt",
            "base_ply": str(base_ply),
            "state_dict": {
                "state_change_dc": dc,
                "state_valid": valid,
                "state_start": starts,
                "state_end": ends,
            },
        },
        path,
    )


def test_export_dc_state_prunes_inactive_rows_and_replaces_only_dc(tmp_path: Path):
    base_ply = tmp_path / "base.ply"
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "viewer" / "state0.ply"
    _write_base_ply(base_ply)
    _write_checkpoint(checkpoint, base_ply)

    result = export_dc_state_for_viewer(checkpoint, output, state_id=0)

    exported = PlyData.read(output)["vertex"].data
    assert result["active_gaussian_count"] == 2
    assert result["inactive_gaussian_count"] == 2
    assert exported["x"].tolist() == [0.0, 2.0]
    assert exported["f_dc_0"].tolist() == [0.0, 6.0]
    assert exported["f_dc_1"].tolist() == [1.0, 7.0]
    assert exported["f_dc_2"].tolist() == [2.0, 8.0]
    assert exported["opacity"].tolist() == pytest.approx([-2.0, 2.0 / 3.0])
    assert exported["scale_2"].tolist() == [3.0, 3.0]
    manifest = json.loads(output.with_suffix(".ply.json").read_text(encoding="utf-8"))
    assert manifest["inactive_policy"].startswith("pruned")
    assert manifest["state_end_min"] == 5.0


def test_ensure_dc_state_export_reuses_matching_artifact(tmp_path: Path):
    base_ply = tmp_path / "base.ply"
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "state0.ply"
    _write_base_ply(base_ply)
    _write_checkpoint(checkpoint, base_ply)

    first = ensure_dc_state_viewer_export(checkpoint, state_id=0, output_ply=output)
    second = ensure_dc_state_viewer_export(checkpoint, state_id=0, output_ply=output)

    assert first["reused"] is False
    assert second["reused"] is True
    assert first["output_ply_sha256"] == second["output_ply_sha256"]


def test_export_rejects_state_without_active_gaussians(tmp_path: Path):
    base_ply = tmp_path / "base.ply"
    checkpoint = tmp_path / "checkpoint.pt"
    _write_base_ply(base_ply)
    _write_checkpoint(checkpoint, base_ply, active=(False, False, False, False))

    with pytest.raises(ValueError, match="no active Gaussians"):
        export_dc_state_for_viewer(checkpoint, tmp_path / "state0.ply", state_id=0)
