import os
import sys
from argparse import ArgumentParser

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_opt_args():
    from arguments import OptimizationParams
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    args = parser.parse_args([])
    return op.extract(args)


def make_toy_model(n: int = 100, seed: int = 0, spatial_lr_scale: float = 1.0,
                   scale_range=(0.02, 0.2), setup_optimizer: bool = True):
    """Programmatic GaussianModel on cuda: grid-ish points, isotropic-ish scales.

    _scaling is set in LOG space (storage convention), rotations identity wxyz,
    opacity logit of 0.9, DC color mid-grey, no higher SH (sh_degree 0).
    """
    import torch
    from torch import nn
    from scene import GaussianModel
    from utils.general_utils import inverse_sigmoid

    g = torch.Generator(device="cpu").manual_seed(seed)
    xyz = (torch.rand((n, 3), generator=g) - 0.5) * 2.0
    scales = torch.exp(torch.rand((n, 3), generator=g)
                       * (torch.log(torch.tensor(scale_range[1])) - torch.log(torch.tensor(scale_range[0])))
                       + torch.log(torch.tensor(scale_range[0])))
    rots = torch.zeros((n, 4)); rots[:, 0] = 1.0
    opac = inverse_sigmoid(0.9 * torch.ones((n, 1)))
    f_dc = 0.5 * torch.ones((n, 1, 3))
    f_rest = torch.zeros((n, 0, 3))

    model = GaussianModel(0, 0)
    model.spatial_lr_scale = spatial_lr_scale
    model._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
    model._scaling = nn.Parameter(torch.log(scales).cuda().requires_grad_(True))
    model._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
    model._opacity = nn.Parameter(opac.cuda().requires_grad_(True))
    model._features_dc = nn.Parameter(f_dc.cuda().requires_grad_(True))
    model._features_rest = nn.Parameter(f_rest.cuda().requires_grad_(True))
    model.max_radii2D = torch.zeros(n, device="cuda")
    model.active_sh_degree = 0
    model._init_persistent_ids(n)
    if setup_optimizer:
        model.training_setup_change(make_opt_args())
    return model


def make_scene(points, scales, opacities=None, colors=None, setup_optimizer: bool = False):
    """GaussianModel with exact placement. points/scales: (N,3) lists or arrays;
    scales are LINEAR sigmas (stored as log internally)."""
    import torch
    from torch import nn
    from scene import GaussianModel
    from utils.general_utils import inverse_sigmoid

    pts = torch.tensor(points, dtype=torch.float32)
    n = pts.shape[0]
    scl = torch.tensor(scales, dtype=torch.float32)
    opac = torch.full((n, 1), 0.9) if opacities is None else torch.tensor(opacities, dtype=torch.float32).reshape(n, 1)
    col = 0.5 * torch.ones((n, 1, 3)) if colors is None else torch.tensor(colors, dtype=torch.float32).reshape(n, 1, 3)
    rots = torch.zeros((n, 4)); rots[:, 0] = 1.0

    model = GaussianModel(0, 0)
    model.spatial_lr_scale = 1.0
    model._xyz = nn.Parameter(pts.cuda().requires_grad_(True))
    model._scaling = nn.Parameter(torch.log(scl).cuda().requires_grad_(True))
    model._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
    model._opacity = nn.Parameter(inverse_sigmoid(opac).cuda().requires_grad_(True))
    model._features_dc = nn.Parameter(col.cuda().requires_grad_(True))
    model._features_rest = nn.Parameter(torch.zeros((n, 0, 3)).cuda().requires_grad_(True))
    model.max_radii2D = torch.zeros(n, device="cuda")
    model.active_sh_degree = 0
    model._init_persistent_ids(n)
    if setup_optimizer:
        model.training_setup_change(make_opt_args())
    return model


def make_pipe():
    from arguments import PipelineParams
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    return pp.extract(parser.parse_args([]))


@pytest.fixture
def toy_model():
    return make_toy_model()


@pytest.fixture
def pipe():
    return make_pipe()


@pytest.fixture
def background():
    import torch
    return torch.zeros(3, device="cuda")
