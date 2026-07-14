# Target-centered candidate camera generation (docs/target_gaussian_nbv.md, stage 4).
#
# Coordinate recipe copied from viewer.py:create_mini_cam (the repo's normative
# camera-synthesis path): OpenGL c2w rotation -> COLMAP flip diag(1,-1,-1) ->
# transposed w2c / full-proj matrices -> MiniCam. Kept dependency-free of the
# viewer (viser import is heavy/optional).

from __future__ import annotations

import math

import numpy as np
import torch

from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, fov2focal

from target_nbv.config import TargetNBVConfig
from target_nbv.types import CandidateCamera


# --- rotation helpers (wxyz, real part first) --------------------------------

def quaternion_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def look_at_wxyz(position: np.ndarray, target: np.ndarray,
                 up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """OpenGL c2w quaternion for a camera at `position` looking at `target`.
    OpenGL camera frame: +x right, +y up, -z forward."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    fn = np.linalg.norm(forward)
    if fn < 1e-12:
        raise ValueError("camera position coincides with target")
    forward = forward / fn
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(forward, up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 0.0, 1.0])  # pole singularity fallback
        if abs(float(np.dot(forward, up))) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    R_c2w_opengl = np.stack([right, cam_up, -forward], axis=1)
    return rotation_matrix_to_quaternion(R_c2w_opengl)


# --- MiniCam synthesis (verbatim create_mini_cam math, viewer.py:125-165) ----

def build_mini_cam(wxyz: np.ndarray, position: np.ndarray, fovy: float,
                   width: int, height: int, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    R_c2w_opengl = quaternion_to_rotation_matrix(wxyz).astype(np.float32)
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_c2w_colmap = R_c2w_opengl @ flip
    R_w2c = R_c2w_colmap.T
    t = -R_w2c @ np.asarray(position, dtype=np.float32)

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3] = t

    aspect = width / height
    FoVy = fovy
    FoVx = 2.0 * math.atan(math.tan(FoVy / 2.0) * aspect)

    proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).cuda()
    world_view = torch.tensor(W2C, dtype=torch.float32).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width=width, height=height, fovy=FoVy, fovx=FoVx,
                   znear=znear, zfar=zfar,
                   world_view_transform=world_view, full_proj_transform=full_proj)


def project_point(cam: MiniCam, xyz_world: np.ndarray):
    """Project a world point through the camera. Returns (ndc_xy, depth_cam)."""
    p = torch.tensor([*np.asarray(xyz_world, dtype=np.float32), 1.0], device="cuda")
    clip = p @ cam.full_proj_transform  # row-vector convention
    ndc = (clip[:3] / clip[3]).detach().cpu().numpy()
    depth = float((p @ cam.world_view_transform)[2])  # camera-space z (COLMAP forward)
    return ndc[:2], depth


def fibonacci_sphere(count: int) -> np.ndarray:
    """Deterministic unit directions, shape (count, 3)."""
    i = np.arange(count, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / count
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


def movement_cost(position: np.ndarray, wxyz: np.ndarray, current_cam,
                  translation_scale: float, rotation_weight: float) -> float:
    if current_cam is None:
        return 0.0
    cur_center = current_cam.camera_center.detach().cpu().numpy().astype(np.float64)
    trans = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - cur_center))
    # relative rotation angle between candidate c2w (COLMAP) and current cam c2w
    R_cand = quaternion_to_rotation_matrix(wxyz) @ np.diag([1.0, -1.0, -1.0])
    wvt = current_cam.world_view_transform.detach().cpu().numpy().astype(np.float64)
    R_cur_c2w = wvt[:3, :3]  # wvt is w2c transposed -> its top-left 3x3 IS R_w2c^T = R_c2w
    cos = (np.trace(R_cand.T @ R_cur_c2w) - 1.0) / 2.0
    angle = math.acos(min(1.0, max(-1.0, cos)))
    return trans / translation_scale + rotation_weight * angle / math.pi


def generate_candidates(model, target_row: int, cfg: TargetNBVConfig,
                        width: int, height: int, fovy: float,
                        current_cam=None) -> list[CandidateCamera]:
    """Generate target-centered candidate cameras. Deterministic given cfg.seed."""
    mu = model._xyz[target_row].detach().cpu().numpy().astype(np.float64)
    r_world = float(model.get_scaling[target_row].detach().max())

    f_px = fov2focal(fovy, height)  # fovy pairs with image height
    d_nominal = f_px * r_world / cfg.candidates.desired_projected_radius_px
    znear, zfar = 0.01, 100.0
    d_nominal = min(max(d_nominal, cfg.candidates.min_distance, 2 * znear),
                    cfg.candidates.max_distance, 0.5 * zfar)

    dirs = fibonacci_sphere(cfg.candidates.count)
    if cfg.candidates.jitter > 0:
        rng = np.random.default_rng(cfg.seed)
        dirs = dirs + cfg.candidates.jitter * rng.standard_normal(dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    out, cand_id = [], 0
    for shell_index, shell in enumerate(cfg.candidates.radius_shells):
        d = d_nominal * shell
        if not (cfg.candidates.min_distance <= d <= cfg.candidates.max_distance):
            continue
        r_px = f_px * r_world / d
        if not (cfg.candidates.min_projected_radius_px <= r_px
                <= cfg.candidates.max_projected_radius_px):
            continue
        for direction in dirs:
            position = mu + d * direction
            wxyz = look_at_wxyz(position, mu)
            cam = CandidateCamera(
                cand_id=cand_id, position=position, wxyz=wxyz,
                fovx=2.0 * math.atan(math.tan(fovy / 2.0) * (width / height)),
                fovy=fovy, width=width, height=height, shell_index=shell_index,
                meta={"distance": d, "projected_radius_px": r_px},
            )
            cam.minicam = build_mini_cam(wxyz, position, fovy, width, height)

            ndc, depth = project_point(cam.minicam, mu)
            if depth <= znear:
                cam.meta["rejected"] = "behind_camera"
            elif abs(ndc[0]) > 0.9 or abs(ndc[1]) > 0.9:
                cam.meta["rejected"] = "outside_image"
            if "rejected" in cam.meta:
                cand_id += 1
                continue

            cam.movement_cost = movement_cost(
                position, wxyz, current_cam,
                cfg.scoring.translation_scale, cfg.scoring.rotation_weight)
            out.append(cam)
            cand_id += 1
    return out
