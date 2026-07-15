# Gate G: target-conditioned change Fisher (GPU).
import math

import pytest
import torch

from tests.conftest import make_scene, make_pipe
from tests.test_candidates_util import simple_cam
from target_nbv.change.fisher import (ChangeFisherUnsupported,
                                      change_fisher_gain,
                                      change_view_information)

pytestmark = pytest.mark.gpu


def change_scene():
    """Change-model stand-in: DC features are logits, zero-initialized
    (= undecided), active_sh_degree 0 — same shape load_ply_change produces."""
    return make_scene(points=[[0.0, 0.0, 0.0], [0.6, 0.4, 0.0]],
                      scales=[[0.05] * 3, [0.05] * 3],
                      colors=[[0.0] * 3, [0.0] * 3])


def test_visible_target_has_information():
    model = change_scene()
    dh = change_view_information(model, simple_cam(distance=2.0), 0, make_pipe())
    assert dh > 1e-3


def test_invisible_target_has_no_information():
    model = make_scene(points=[[0.0, 0.0, 0.0], [0.0, 0.0, -5.0]],  # behind cam
                       scales=[[0.05] * 3, [0.05] * 3],
                       colors=[[0.0] * 3, [0.0] * 3])
    dh_vis = change_view_information(model, simple_cam(distance=2.0), 0, make_pipe())
    dh_inv = change_view_information(model, simple_cam(distance=2.0), 1, make_pipe())
    assert dh_inv < 1e-6 * max(dh_vis, 1.0)


def test_model_restored_after_fd():
    model = change_scene()
    before = model._features_dc.detach().clone()
    change_view_information(model, simple_cam(distance=2.0), 0, make_pipe())
    assert torch.equal(model._features_dc.detach(), before)


def test_gain_formula():
    assert change_fisher_gain(1.0, 0.0) == 0.0
    assert change_fisher_gain(1.0, math.e ** 2 - 1.0) == pytest.approx(1.0)
    # repeated observation of the same view has diminishing gain
    dh = 3.0
    g1 = change_fisher_gain(1.0, dh)
    g2 = change_fisher_gain(1.0 + dh, dh)
    assert 0 < g2 < g1
    with pytest.raises(ValueError, match="h_prior"):
        change_fisher_gain(0.0, 1.0)
    with pytest.raises(ValueError, match="delta_h"):
        change_fisher_gain(1.0, -1.0)


def test_unsupported_model_explicit_error():
    model = change_scene()
    model.active_sh_degree = 3  # looks like an RGB model, not a change model
    with pytest.raises(ChangeFisherUnsupported, match="active_sh_degree"):
        change_view_information(model, simple_cam(distance=2.0), 0, make_pipe())
