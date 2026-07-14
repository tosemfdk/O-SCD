# Gate E: exact target-conditioned P-optimal scorer.
# CPU tests exercise the information-matrix algebra via score_delta;
# GPU tests run the full FD-Jacobian scoring path on toy scenes.
import math

import numpy as np
import pytest
import torch

from target_nbv.candidates import build_mini_cam, look_at_wxyz
from target_nbv.config import TargetNBVConfig
from target_nbv.info_builder import TargetInformationBuilder, view_information
from target_nbv.scorers.exact import TargetPOptimalScorer, commit_observed_view
from target_nbv.types import (CandidateCamera, CandidateScore,
                              TargetInformationState, TargetParameterSpec)

SPEC = TargetParameterSpec()
CFG = TargetNBVConfig().validate()
FOVY = math.radians(60)


def make_state(H_data: np.ndarray) -> TargetInformationState:
    return TargetInformationState(target_pid=1, spec=SPEC, H_data=H_data,
                                  absolute_damping=1e-6, relative_damping=1e-6)


def _rand_delta(n=40, d=6, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return view_information(scale * torch.randn((n, d), generator=g, dtype=torch.float64))


# --- CPU: score_delta algebra -------------------------------------------------

def test_gains_match_numpy_reference():
    state = make_state(np.diag([4.0, 3.0, 2.0, 1.0, 1.0, 1.0]))
    H_prior = state.H_prior()
    delta = _rand_delta(seed=1)
    scorer = TargetPOptimalScorer(CFG)
    gains, reason, _ = scorer.score_delta(H_prior, scorer.prior_quantities(H_prior),
                                          delta, movement_cost=0.0)
    assert reason is None
    H_post = H_prior + delta
    assert np.isclose(gains["d_gain"],
                      0.5 * (np.linalg.slogdet(H_post)[1] - np.linalg.slogdet(H_prior)[1]))
    assert np.isclose(gains["trace_gain"],
                      np.trace(np.linalg.inv(H_prior)) - np.trace(np.linalg.inv(H_post)),
                      rtol=1e-8)
    assert np.isclose(gains["e_gain"],
                      1.0 / np.linalg.eigvalsh(H_prior)[0] - 1.0 / np.linalg.eigvalsh(H_post)[0],
                      rtol=1e-8)


def test_gains_nonnegative_for_psd_increment():
    state = make_state(np.eye(6))
    H_prior = state.H_prior()
    scorer = TargetPOptimalScorer(CFG)
    pq = scorer.prior_quantities(H_prior)
    for seed in range(5):
        gains, reason, warnings = scorer.score_delta(H_prior, pq, _rand_delta(seed=seed), 0.0)
        assert reason is None and warnings == []
        assert gains["d_gain"] >= 0 and gains["trace_gain"] >= 0 and gains["e_gain"] >= 0


def test_marginal_gain_decreases_after_commit():
    delta = _rand_delta(seed=7)
    state1 = make_state(np.eye(6))
    H1 = state1.H_prior()
    scorer = TargetPOptimalScorer(CFG)
    g1, _, _ = scorer.score_delta(H1, scorer.prior_quantities(H1), delta, 0.0)

    state2 = make_state(np.eye(6) + delta)  # same view committed once
    H2 = state2.H_prior()
    g2, _, _ = scorer.score_delta(H2, scorer.prior_quantities(H2), delta, 0.0)
    assert g2["d_gain"] < g1["d_gain"]
    assert g2["score"] < g1["score"]


def test_non_finite_delta_invalid():
    state = make_state(np.eye(6))
    H = state.H_prior()
    scorer = TargetPOptimalScorer(CFG)
    bad = np.full((6, 6), np.nan)
    gains, reason, _ = scorer.score_delta(H, scorer.prior_quantities(H), bad, 0.0)
    assert gains is None and reason == "non_finite_delta_H"


def test_e_gain_prefers_weakest_axis():
    # axis 0 is weakest; equal-magnitude info on axis 0 beats axis 1 in E-mode
    state = make_state(np.diag([0.1, 10.0, 10.0, 10.0, 10.0, 10.0]))
    H = state.H_prior()
    cfg = TargetNBVConfig()
    cfg.scoring.d_weight = 0.0
    cfg.scoring.e_weight = 1.0
    scorer = TargetPOptimalScorer(cfg.validate())
    pq = scorer.prior_quantities(H)

    def axis_delta(k):
        d = np.zeros((6, 6)); d[k, k] = 1.0
        return d

    g_weak, _, _ = scorer.score_delta(H, pq, axis_delta(0), 0.0)
    g_strong, _, _ = scorer.score_delta(H, pq, axis_delta(1), 0.0)
    assert g_weak["score"] > g_strong["score"]


def test_movement_cost_lowers_score():
    state = make_state(np.eye(6))
    H = state.H_prior()
    scorer = TargetPOptimalScorer(CFG)
    pq = scorer.prior_quantities(H)
    delta = _rand_delta(seed=2)
    g0, _, _ = scorer.score_delta(H, pq, delta, movement_cost=0.0)
    g1, _, _ = scorer.score_delta(H, pq, delta, movement_cost=5.0)
    assert g1["score"] == pytest.approx(
        g0["score"] - CFG.scoring.movement_weight * 5.0)


# --- GPU: full FD-Jacobian scoring path ---------------------------------------

def cam_at(position, target=(0.0, 0.0, 0.0), width=128, height=128):
    position = np.asarray(position, dtype=np.float64)
    wxyz = look_at_wxyz(position, np.asarray(target, dtype=np.float64))
    return build_mini_cam(wxyz, position, FOVY, width, height)


def candidate_at(cand_id, position, movement_cost=0.0, target=(0.0, 0.0, 0.0)):
    position = np.asarray(position, dtype=np.float64)
    wxyz = look_at_wxyz(position, np.asarray(target, dtype=np.float64))
    cand = CandidateCamera(cand_id=cand_id, position=position, wxyz=wxyz,
                           fovx=FOVY, fovy=FOVY, width=128, height=128,
                           movement_cost=movement_cost)
    cand.minicam = build_mini_cam(wxyz, position, FOVY, 128, 128)
    return cand


def cscore(cand):
    return CandidateScore(candidate=cand, movement_cost=cand.movement_cost)


def _builder_with_views(model, positions, cfg=CFG):
    from tests.conftest import make_pipe
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")
    b = TargetInformationBuilder(1, SPEC, cfg)
    for i, p in enumerate(positions):
        assert commit_observed_view(b, f"v{i}", model, cam_at(p), 0, pipe, bg, cfg)
    return b, pipe, bg


@pytest.mark.gpu
def test_end_to_end_scoring_and_model_restoration():
    from tests.conftest import make_scene
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    snap = {n: getattr(model, n).detach().clone()
            for n in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc")}
    b, pipe, bg = _builder_with_views(model, [[0.0, 0.0, -1.5]])

    cands = [cscore(candidate_at(0, [1.5, 0.0, 0.0])),
             cscore(candidate_at(1, [0.0, 0.0, -1.5]))]  # same as observed view
    scorer = TargetPOptimalScorer(CFG)
    out = scorer.score_candidates(b.state, cands, model, 0, pipe, bg)

    assert all(s.valid for s in out)
    assert all(np.isfinite(s.exact_score) for s in out)
    # orthogonal baseline must beat re-observing the already-seen pose
    assert out[0].candidate.cand_id == 0
    for name, before in snap.items():
        assert torch.equal(getattr(model, name).detach(), before), f"{name} mutated"
    # scoring must not mutate the prior state
    assert b.state.observed_view_ids == ["v0"]


@pytest.mark.gpu
def test_repeated_view_marginal_gain_decreases():
    from tests.conftest import make_scene
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    b, pipe, bg = _builder_with_views(model, [[0.0, 0.0, -1.5]])
    scorer = TargetPOptimalScorer(CFG)
    pose = [1.5, 0.0, 0.0]

    g1 = scorer.score_candidates(b.state, [cscore(candidate_at(0, pose))],
                                 model, 0, pipe, bg)[0].d_gain
    assert commit_observed_view(b, "v1", model, cam_at(pose), 0, pipe, bg, CFG)
    g2 = scorer.score_candidates(b.state, [cscore(candidate_at(0, pose))],
                                 model, 0, pipe, bg)[0].d_gain
    assert 0 <= g2 < g1


@pytest.mark.gpu
def test_occluded_candidate_gains_less():
    from tests.conftest import make_scene
    # opaque occluder between the -z camera and the target; +x side is clear
    model = make_scene(points=[[0.0, 0.0, 0.0], [0.0, 0.0, -0.5]],
                       scales=[[0.05, 0.05, 0.05], [0.2, 0.2, 0.02]],
                       opacities=[[0.99], [0.99]])
    b, pipe, bg = _builder_with_views(model, [[1.5, 0.0, 0.0]])
    scorer = TargetPOptimalScorer(CFG)
    out = scorer.score_candidates(
        b.state,
        [cscore(candidate_at(0, [0.0, 0.0, -1.5])),   # blocked by occluder
         cscore(candidate_at(1, [0.0, 1.5, 0.0]))],   # clear top view
        model, 0, pipe, bg)
    gains = {s.candidate.cand_id: s.d_gain for s in out if s.valid}
    assert gains[1] > gains[0]


@pytest.mark.gpu
def test_target_behind_candidate_invalid():
    from tests.conftest import make_scene
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    b, pipe, bg = _builder_with_views(model, [[0.0, 0.0, -1.5]])
    # camera at +x looking AWAY from the target
    away = candidate_at(0, [1.5, 0.0, 0.0], target=(3.0, 0.0, 0.0))
    out = TargetPOptimalScorer(CFG).score_candidates(
        b.state, [cscore(away)], model, 0, pipe, bg)
    assert not out[0].valid
    assert out[0].invalid_reason.startswith("jacobian:")


@pytest.mark.gpu
def test_ranking_stable_under_epsilon_halving():
    from tests.conftest import make_scene
    model = make_scene(points=[[0.0, 0.0, 0.0]], scales=[[0.05, 0.05, 0.05]])
    b, pipe, bg = _builder_with_views(model, [[0.0, 0.0, -1.5]])
    cands = lambda: [cscore(candidate_at(0, [1.5, 0.0, 0.0])),
                     cscore(candidate_at(1, [0.0, 0.0, -1.2]))]

    cfg2 = TargetNBVConfig()
    cfg2.jacobian.mean_epsilon_rel = CFG.jacobian.mean_epsilon_rel / 2
    cfg2.jacobian.log_scale_epsilon = CFG.jacobian.log_scale_epsilon / 2
    top1 = TargetPOptimalScorer(CFG).score_candidates(
        b.state, cands(), model, 0, pipe, bg)[0].candidate.cand_id
    top1_half = TargetPOptimalScorer(cfg2.validate()).score_candidates(
        b.state, cands(), model, 0, pipe, bg)[0].candidate.cand_id
    assert top1 == top1_half
