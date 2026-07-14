# Shared IO for the target-NBV CLI tools (stage 12): camera JSON <-> MiniCam,
# checkpoint loading, pose matrix export. Camera JSON entries are
#   {"view_id": str, "position": [x,y,z], "wxyz": [w,x,y,z] | "look_at": [x,y,z],
#    optional "fovy" (RADIANS), "width", "height"}
# with wxyz the OpenGL c2w quaternion (real-first), same as CandidateCamera —
# so a best_view.json from tools/select_target_nbv.py is directly reusable as
# a camera entry.

from __future__ import annotations

import json

import numpy as np

from target_nbv.candidates import (build_mini_cam, look_at_wxyz,
                                   quaternion_to_rotation_matrix)


def load_observed_cameras(path: str, default_fovy: float,
                          default_width: int, default_height: int) -> list[tuple[str, object]]:
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty JSON list of camera entries")
    out = []
    for i, d in enumerate(entries):
        view_id, cam = camera_from_json(d, default_fovy, default_width,
                                        default_height, fallback_id=f"obs{i}")
        out.append((view_id, cam))
    return out


def camera_from_json(d: dict, default_fovy: float, default_width: int,
                     default_height: int, fallback_id: str = "cam"):
    position = np.asarray(d["position"], dtype=np.float64)
    if "wxyz" in d:
        wxyz = np.asarray(d["wxyz"], dtype=np.float64)
    elif "look_at" in d:
        wxyz = look_at_wxyz(position, np.asarray(d["look_at"], dtype=np.float64))
    else:
        raise ValueError(f"camera entry needs 'wxyz' or 'look_at': {d}")
    fovy = float(d.get("fovy", default_fovy))
    width = int(d.get("width", default_width))
    height = int(d.get("height", default_height))
    cam = build_mini_cam(wxyz, position, fovy, width, height)
    return str(d.get("view_id", fallback_id)), cam


def default_pipe():
    """PipelineParams with repo defaults, without a real CLI parse."""
    from argparse import ArgumentParser
    from arguments import PipelineParams
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    return pp.extract(parser.parse_args([]))


def load_gaussian_model(ply_path: str, sh_degree: int):
    """RGB GaussianModel from a point_cloud.ply (+ .pid.pt sidecar if present)."""
    from scene import GaussianModel
    model = GaussianModel(sh_degree)
    model.load_ply(ply_path)
    return model


def pose_matrices(wxyz: np.ndarray, position: np.ndarray) -> dict:
    """COLMAP-convention c2w/w2c 4x4 for a stored OpenGL c2w quaternion."""
    R_c2w = quaternion_to_rotation_matrix(wxyz) @ np.diag([1.0, -1.0, -1.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = R_c2w.T
    w2c[:3, 3] = -R_c2w.T @ np.asarray(position, dtype=np.float64)
    return {"c2w": c2w.tolist(), "w2c": w2c.tolist()}
