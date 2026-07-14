# Gate F: end-to-end selector (GPU).
import numpy as np
import pytest
import torch

from tests.conftest import make_pipe, make_scene
from tests.test_scorers_exact import cam_at
from target_nbv.config import TargetNBVConfig
from target_nbv.selector import (NoValidCandidateError, build_information_state,
                                 select_next_view)
from target_nbv.target_registry import UnknownTargetError
from target_nbv.types import TargetParameterSpec

pytestmark = pytest.mark.gpu

SPEC = TargetParameterSpec()


def small_cfg(**overrides) -> TargetNBVConfig:
    cfg = TargetNBVConfig()
    cfg.candidates.count = 16
    cfg.proxy.top_k = 6
    cfg.jacobian.exact_top_k = 3
    for k, v in overrides.items():
        obj = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, v)
    return cfg.validate()


def scene_and_views():
    model = make_scene(
        points=[[0.0, 0.0, 0.0], [0.6, 0.5, 0.5], [-0.5, -0.6, 0.4]],
        scales=[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    # two nearby views: narrow baseline, so orthogonal candidates are informative
    observed = [("v0", cam_at([0.0, 0.0, -1.5])),
                ("v1", cam_at([0.0, 0.3, -1.4]))]
    return model, observed


def run(model, observed, cfg, **kw):
    return select_next_view(model, 0, observed, cfg, make_pipe(),
                            torch.zeros(3, device="cuda"), **kw)


def test_end_to_end_geometry_exact():
    model, observed = scene_and_views()
    snap = model._xyz.detach().clone()
    result = run(model, observed, small_cfg())

    assert result.best is not None and result.best.valid
    assert result.best.delta_H is not None
    assert result.H_before.shape == (6, 6)
    assert np.allclose(result.predicted_H_after,
                       result.H_before + result.best.delta_H)
    assert {"information", "candidates", "visibility", "proxy", "exact",
            "total"} <= set(result.runtime)
    meta = result.config_snapshot["_selection_meta"]
    assert meta["target_pid"] == 0
    assert meta["observed_views_used"] == ["v0", "v1"]
    assert torch.equal(model._xyz.detach(), snap)
    # episode freeze released after the call
    assert not model._protected_pids


def test_deterministic():
    model, observed = scene_and_views()
    cfg = small_cfg()
    r1 = run(model, observed, cfg)
    r2 = run(model, observed, cfg)
    assert r1.best.candidate.cand_id == r2.best.candidate.cand_id
    assert ([s.candidate.cand_id for s in r1.scores]
            == [s.candidate.cand_id for s in r2.scores])
    assert r1.best.exact_score == pytest.approx(r2.best.exact_score, abs=1e-12)


def test_dry_run_leaves_prebuilt_state_untouched():
    model, observed = scene_and_views()
    cfg = small_cfg()
    builder, skipped = build_information_state(
        model, 0, 0, SPEC, observed, make_pipe(),
        torch.zeros(3, device="cuda"), cfg)
    assert not skipped
    version_before = builder.state.version
    views_before = list(builder.state.observed_view_ids)

    run(model, observed, cfg, information_builder=builder)
    assert builder.state.version == version_before
    assert builder.state.observed_view_ids == views_before


def test_invalid_target_id():
    model, observed = scene_and_views()
    with pytest.raises(UnknownTargetError, match="999999"):
        select_next_view(model, 999999, observed, small_cfg(), make_pipe(),
                         torch.zeros(3, device="cuda"))


def test_all_candidates_invalid_is_explicit():
    model, observed = scene_and_views()
    cfg = small_cfg(**{"visibility.min_responsibility": 1e9})
    with pytest.raises(NoValidCandidateError, match="fully_occluded"):
        run(model, observed, cfg)


def test_fewer_valid_candidates_than_top_k():
    model, observed = scene_and_views()
    # single far shell -> few candidates survive the projected-radius gate
    cfg = small_cfg(**{"candidates.count": 6, "proxy.top_k": 6,
                       "jacobian.exact_top_k": 6,
                       "candidates.radius_shells": [1.0]})
    result = run(model, observed, cfg)
    assert result.best is not None and result.best.valid


def test_geometry_proxy_mode():
    model, observed = scene_and_views()
    cfg = small_cfg(mode="geometry_proxy")
    result = run(model, observed, cfg)
    assert result.best.valid and result.best.proxy_score > 0
    assert result.best.delta_H is None
    assert result.predicted_H_after is None
    assert "exact" not in result.runtime


def test_unsupported_mode_explicit():
    model, observed = scene_and_views()
    cfg = small_cfg(mode="geometry_schur", **{"neighbors.enabled": True})
    with pytest.raises(NotImplementedError, match="geometry_schur"):
        run(model, observed, cfg)


def test_state_target_mismatch_rejected():
    model, observed = scene_and_views()
    cfg = small_cfg()
    builder, _ = build_information_state(
        model, 1, 1, SPEC, observed, make_pipe(),
        torch.zeros(3, device="cuda"), cfg)  # built for target 1
    with pytest.raises(ValueError, match="information state is for target 1"):
        run(model, observed, cfg, information_builder=builder)
