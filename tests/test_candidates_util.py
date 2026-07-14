# Shared camera helper for GPU tests. Builds a MiniCam with the repo's
# transposed w2c convention (scene/cameras.py, utils/graphics_utils.py).
import math

import torch

from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


def cam_from_w2c(R_w2c, t, width=128, height=128, fov_deg=60.0):
    """R_w2c: (3,3) torch float32 cpu; t = -R_w2c @ cam_pos."""
    W2C = torch.eye(4, dtype=torch.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3, 3] = t
    fov = math.radians(fov_deg)
    world_view = W2C.transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(width, height, fov, fov, 0.01, 100.0, world_view, full_proj)


def simple_cam(distance=3.0, width=128, height=128, fov_deg=60.0):
    """Camera on -z axis looking at the origin along +z (COLMAP forward)."""
    R = torch.eye(3, dtype=torch.float32)
    pos = torch.tensor([0.0, 0.0, -distance])
    t = -R @ pos
    return cam_from_w2c(R, t, width, height, fov_deg)
