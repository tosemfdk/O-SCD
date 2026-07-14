# Gate E: geometry FIM proxy scorer (CPU — synthetic candidates/visibilities).
import math
import random

import numpy as np
import pytest

from target_nbv.config import TargetNBVConfig
from target_nbv.scorers.proxy import (GeometryProxyScorer,
                                      normalize_responsibilities, proxy_top_k)
from target_nbv.types import (CandidateCamera, TargetInformationState,
                              TargetParameterSpec, TargetVisibility)

MU = np.zeros(3)


def make_state(mean_diag=(10.0, 10.0, 0.1)) -> TargetInformationState:
    H = np.zeros((6, 6))
    H[:3, :3] = np.diag(mean_diag)
    H[3:, 3:] = np.eye(3)
    return TargetInformationState(
        target_pid=1, spec=TargetParameterSpec(), H_data=H,
        absolute_damping=1e-6, relative_damping=1e-6)


def make_candidate(cand_id: int, direction, distance: float = 1.0,
                   movement_cost: float = 0.0) -> CandidateCamera:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return CandidateCamera(
        cand_id=cand_id, position=MU + distance * direction,
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        fovx=math.radians(60), fovy=math.radians(60), width=128, height=128,
        movement_cost=movement_cost)


def visible(rho: float = 1.0) -> TargetVisibility:
    return TargetVisibility(valid=True, responsibility_sum=rho,
                            visible_pixel_count=100, occlusion_ratio=0.0)


def rank(cands, viss, state=None, cfg=None):
    cfg = cfg or TargetNBVConfig().validate()
    return GeometryProxyScorer(cfg).rank(state or make_state(), cands, viss, MU)


def test_baseline_perpendicular_to_uncertain_axis_wins():
    # z information is weakest; an x-side view adds (I - r r^T) info in the
    # y/z plane, a head-on z view only in x/y -> x-side must rank first.
    cands = [make_candidate(0, [0, 0, 1]), make_candidate(1, [1, 0, 0])]
    scores = rank(cands, [visible(), visible()])
    assert scores[0].candidate.cand_id == 1
    assert scores[0].proxy_score > scores[1].proxy_score


def test_score_decreases_with_distance():
    cands = [make_candidate(0, [1, 0, 0], distance=1.0),
             make_candidate(1, [1, 0, 0], distance=2.0),
             make_candidate(2, [1, 0, 0], distance=4.0)]
    scores = rank(cands, [visible(1.0)] * 3)
    by_id = {s.candidate.cand_id: s.proxy_score for s in scores}
    assert by_id[0] > by_id[1] > by_id[2]


def test_zero_responsibility_scores_near_zero():
    scores = rank([make_candidate(0, [1, 0, 0])], [visible(rho=0.0)])
    assert scores[0].valid
    assert abs(scores[0].proxy_score) < 1e-12


def test_fully_occluded_candidate_invalid_and_last():
    occ = TargetVisibility(valid=False, invalid_reason="fully_occluded")
    scores = rank([make_candidate(0, [1, 0, 0]), make_candidate(1, [0, 1, 0])],
                  [occ, visible()])
    assert scores[-1].candidate.cand_id == 0
    assert not scores[-1].valid and scores[-1].invalid_reason == "fully_occluded"
    assert [s.candidate.cand_id for s in proxy_top_k(scores, 5)] == [1]


def test_movement_penalty_prefers_near_candidate():
    # identical geometry, one pose is expensive to reach
    cands = [make_candidate(0, [1, 0, 0], movement_cost=10.0),
             make_candidate(1, [-1, 0, 0], movement_cost=0.0)]
    scores = rank(cands, [visible(), visible()])
    assert scores[0].candidate.cand_id == 1


def test_ranking_invariant_to_input_order():
    cands = [make_candidate(i, [math.cos(a), 0.3, math.sin(a)], distance=1 + 0.1 * i)
             for i, a in enumerate(np.linspace(0, math.pi, 8))]
    viss = [visible(rho=0.5 + 0.1 * i) for i in range(8)]
    ref = [s.candidate.cand_id for s in rank(list(cands), list(viss))]
    idx = list(range(8))
    random.Random(3).shuffle(idx)
    shuffled = [s.candidate.cand_id
                for s in rank([cands[i] for i in idx], [viss[i] for i in idx])]
    assert ref == shuffled


def test_percentile90_normalization():
    rhos = np.array([0.0, 1.0, 2.0, 100.0])
    out = normalize_responsibilities(rhos, "percentile90")
    p90 = np.percentile(rhos, 90.0)
    assert np.allclose(out, np.clip(rhos / p90, 0, 1))
    assert out[3] == 1.0  # outlier clamped
    assert np.allclose(normalize_responsibilities(np.zeros(3), "percentile90"),
                       np.zeros(3))
    with pytest.raises(ValueError, match="normalization"):
        normalize_responsibilities(rhos, "minmax")


def test_render_free_mode_allows_none_visibility():
    cfg = TargetNBVConfig()
    cfg.proxy.uses_visibility = False
    scores = rank([make_candidate(0, [1, 0, 0])], [None], cfg=cfg.validate())
    assert scores[0].valid and scores[0].proxy_score > 0

    with pytest.raises(ValueError, match="visibility required"):
        rank([make_candidate(0, [1, 0, 0])], [None])


def test_directional_gain_diagnostic():
    # e_max is z (weakest axis): x-side view has large directional gain,
    # z head-on view ~0 (its ray is parallel to e_max).
    scores = rank([make_candidate(0, [1, 0, 0]), make_candidate(1, [0, 0, 1])],
                  [visible(), visible()])
    g = {s.candidate.cand_id: s.candidate.meta["proxy_directional_gain"]
         for s in scores}
    assert g[0] > 0.5 and g[1] < 1e-9
