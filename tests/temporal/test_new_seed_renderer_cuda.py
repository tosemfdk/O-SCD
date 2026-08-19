import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for NEW seed renderer tests", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")
pytest.importorskip("simple_knn._C")

from gaussian_renderer import render_change
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from temporal.new_seed_gaussians import ActiveNewSeedView, NewSeedGaussianModel, ConcatenatedChangeView
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import RGB2SH


def _camera(image_size=64):
    device = torch.device("cuda")
    fov = math.radians(60.0)
    world_view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).t().to(device)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(image_size, image_size, fov, fov, 0.01, 100.0, world_view, full_projection)


def _base_model():
    device = torch.device("cuda")
    base = GaussianModel(sh_degree=0, active_sh_degree=0)
    base._xyz = nn.Parameter(torch.tensor([[-0.15, 0.0, 2.0], [0.15, 0.0, 2.2]], device=device))
    base._features_dc = nn.Parameter(RGB2SH(torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0]], device=device)).view(2, 1, 3))
    base._features_rest = nn.Parameter(torch.zeros((2, 0, 3), device=device))
    base._opacity = nn.Parameter(inverse_sigmoid(torch.full((2, 1), 0.7, device=device)))
    base._scaling = nn.Parameter(torch.full((2, 3), math.log(0.08), device=device))
    rotation = torch.zeros((2, 4), device=device)
    rotation[:, 0] = 1.0
    base._rotation = nn.Parameter(rotation)
    return base


def test_zero_seed_render_matches_base_render_and_base_hashes():
    base = _base_model()
    seeds = NewSeedGaussianModel(device="cuda")
    camera = _camera()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    base_snapshot = {name: getattr(base, name).detach().clone() for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")}

    base_render = render_change(camera, base, pipe, bg, override_dc=base._features_dc.detach())["render"]
    view = ConcatenatedChangeView(base, seeds, timestamp=0.0)
    view_render = render_change(camera, view, pipe, bg)["render"]

    assert torch.allclose(view_render, base_render, atol=1e-6)
    for name, before in base_snapshot.items():
        assert torch.equal(getattr(base, name).detach(), before), name


def test_seed_dc_is_connected_to_adapter_autograd_and_render_keeps_base_frozen():
    base = _base_model()
    seeds = NewSeedGaussianModel(device="cuda")
    seeds.append(
        xyz=torch.tensor([[0.0, 0.0, 1.8]], device="cuda"),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.35), device="cuda"),
        opacity=torch.full((1, 1), 0.9, device="cuda"),
        dc=RGB2SH(torch.tensor([[0.0, 0.0, 0.8]], device="cuda")).view(1, 1, 3),
    )
    camera = _camera()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    view = ConcatenatedChangeView(base, seeds, timestamp=0.0)
    rendered = render_change(camera, view, pipe, bg, clamp_output=False)["render"]
    assert rendered.is_cuda
    assert torch.isfinite(rendered).all()

    # The local FastGS rasterizer build used in this repo does not expose a
    # useful color/DC gradient in this synthetic smoke path, so assert the
    # adapter-side autograd contract directly: seed rows are connected, base
    # rows are detached, and geometry buffers remain non-trainable.
    view._features_dc.sum().backward()
    assert seeds.seed_dc.grad is not None
    assert seeds.seed_dc.grad.abs().sum() > 0
    assert all(buffer.grad is None for _, buffer in seeds.named_buffers())
    assert base._features_dc.grad is None
    assert base._xyz.grad is None


def test_dynamic_append_preserves_cuda_adam_state():
    seeds = NewSeedGaussianModel(device="cuda")
    seeds.append(xyz=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), start=0.0, dc=torch.ones(1, 1, 3, device="cuda"))
    opt = torch.optim.Adam(seeds.optimizer_parameter_groups(lr=0.01))
    seeds.seed_dc.sum().backward()
    opt.step()
    old_param = seeds.seed_dc
    old_exp_avg = opt.state[old_param]["exp_avg"].clone()

    seeds.append(xyz=torch.tensor([[1.0, 0.0, 1.0]], device="cuda"), start=1.0, optimizer=opt)

    assert seeds.seed_dc.is_cuda
    assert seeds.seed_dc is opt.param_groups[0]["params"][0]
    assert torch.equal(opt.state[seeds.seed_dc]["exp_avg"][:1], old_exp_avg)
    assert torch.equal(opt.state[seeds.seed_dc]["exp_avg"][1:], torch.zeros_like(opt.state[seeds.seed_dc]["exp_avg"][1:]))


def test_active_seed_only_view_renders_without_reference_rows():
    seeds = NewSeedGaussianModel(device="cuda")
    seeds.append(
        xyz=torch.tensor([[0.0, 0.0, 1.8]], device="cuda"),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.20), device="cuda"),
        opacity=0.3,
        dc=RGB2SH(torch.tensor([[0.0, 0.0, 0.8]], device="cuda")).view(1, 1, 3),
    )
    camera = _camera()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")

    active = seeds.active_view(0.0)
    assert isinstance(active, ActiveNewSeedView)
    assert active.get_xyz.shape == (1, 3)
    rendered = render_change(camera, active, pipe, bg, clamp_output=False)["render"]
    assert rendered.shape == (3, 64, 64)
    assert torch.isfinite(rendered).all()
