# Direction-aware D-opt frame scorer: the property that motivated it (GPU).
import pytest
import torch

from tests.conftest import make_scene, make_pipe
from tests.test_scorers_exact import cam_at
from target_nbv.change.counts import responsibilities
from target_nbv.change.dopt_scorer import DoptFrameState

pytestmark = pytest.mark.gpu


def scene_and_state(damping=1.0):
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05] * 3])
    return model, DoptFrameState(1, damping=damping), make_pipe()


def test_orthogonal_baseline_beats_same_direction_revisit():
    model, state, pipe = scene_and_state()
    w = torch.ones(1, device="cuda")
    cam_z = cam_at([0.0, 0.0, -1.5])

    # before any commit, both directions are roughly equally informative
    s_z0 = state.score_frame(model, cam_z, w, pipe)
    assert s_z0 > 0

    state.commit(model, cam_z, responsibilities(model, cam_z, pipe))

    s_z = state.score_frame(model, cam_at([0.0, 0.0, -1.4]), w, pipe)  # revisit
    s_x = state.score_frame(model, cam_at([1.5, 0.0, 0.0]), w, pipe)   # orthogonal
    assert s_x > 2.0 * s_z, f"orthogonal {s_x:.3f} vs revisit {s_z:.3f}"


def test_marginal_gain_decreases_on_repeat_commit():
    model, state, pipe = scene_and_state()
    w = torch.ones(1, device="cuda")
    cam = cam_at([1.5, 0.0, 0.0])
    tau = responsibilities(model, cam, pipe)
    s1 = state.score_frame(model, cam, w, pipe)
    state.commit(model, cam, tau)
    s2 = state.score_frame(model, cam, w, pipe)
    assert 0 <= s2 < s1


def test_invisible_target_scores_zero():
    model, state, pipe = scene_and_state()
    w = torch.ones(1, device="cuda")
    away = cam_at([1.5, 0.0, 0.0], target=(3.0, 0.0, 0.0))  # looks away
    assert state.score_frame(model, away, w, pipe) == 0.0


def test_weight_gates_contribution():
    model, state, pipe = scene_and_state()
    cam = cam_at([1.5, 0.0, 0.0])
    s_on = state.score_frame(model, cam, torch.ones(1, device="cuda"), pipe)
    s_off = state.score_frame(model, cam, torch.zeros(1, device="cuda"), pipe)
    assert s_on > 0 and s_off == 0.0


def test_append_and_validation():
    _, state, _ = scene_and_state()
    state.append(2)
    assert len(state) == 3
    with pytest.raises(ValueError, match="damping"):
        DoptFrameState(1, damping=0.0)
    with pytest.raises(ValueError, match="tau size"):
        state.commit(None, None, torch.zeros(1, dtype=torch.float64, device="cuda"))