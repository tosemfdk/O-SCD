import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

import torch
from torch import nn


class DummyBase(SimpleNamespace):
    def opacity_activation(self, x):
        return torch.sigmoid(x)

    def scaling_activation(self, x):
        return torch.exp(x)

    def rotation_activation(self, x):
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def make_base(n=4, *, dtype=torch.float32, rest_channels=2):
    rotation = torch.zeros(n, 4, dtype=dtype)
    rotation[:, 0] = 1.0
    return DummyBase(
        _xyz=nn.Parameter(torch.arange(n * 3, dtype=dtype).reshape(n, 3) / 10.0),
        _features_dc=nn.Parameter(torch.arange(n * 3, dtype=dtype).reshape(n, 1, 3)),
        _features_rest=nn.Parameter(torch.arange(n * rest_channels * 3, dtype=dtype).reshape(n, rest_channels, 3)),
        _opacity=nn.Parameter(torch.linspace(-4.0, 1.0, n, dtype=dtype).reshape(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3, dtype=dtype)),
        _rotation=nn.Parameter(rotation),
    )


def make_mcmc_gaussians(*, raw_opacity, scaling=None, xyz=None):
    raw = torch.as_tensor(raw_opacity, dtype=torch.float32).reshape(-1, 1)
    n = raw.shape[0]
    if xyz is None:
        xyz = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10.0
    else:
        xyz = torch.as_tensor(xyz, dtype=torch.float32).reshape(n, 3)
    if scaling is None:
        scaling = torch.zeros(n, 3, dtype=torch.float32)
    else:
        scaling = torch.as_tensor(scaling, dtype=torch.float32).reshape(n, 3)
    rotation = torch.zeros(n, 4, dtype=torch.float32)
    rotation[:, 0] = 1.0
    return SimpleNamespace(
        xyz=nn.Parameter(xyz.clone()),
        change_dc=nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 1, 3)),
        change_opacity_raw=nn.Parameter(raw.clone()),
        scaling_raw=nn.Parameter(scaling.clone()),
        rotation_raw=nn.Parameter(rotation),
    )


def pytest_collection_modifyitems(items):
    import pytest

    for item in items:
        item.add_marker(pytest.mark.mcmc)
