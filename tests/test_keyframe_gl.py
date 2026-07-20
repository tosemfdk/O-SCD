# Part 2 cycle 2 — global-local keyframe selection tests (spec §12).
#
# Coverage map (spec test number -> here):
#   1  first-round candidate set = all frames, no frame-0 preinsert
#   2  no-GT: static source scan + FrameAccessGuard leakage during scoring
#   4  topology independence (R_global N != R_local N, no index mixing)
#   5  mask decomposition identities / shape / range / finiteness
#   6  first-round equivalence: kf_glu_dir ranking == kf_g_dir ranking
#   7  clean-rebuild seed depends on the SET only (order-free); the full
#      fusion-level check runs once against a manual replay during the pilot
#      (rasterizer atomics make bitwise equality machine-level, not unit-level)
#   8  direction-block soft-weight exactness (single-pixel weights make the
#      Hutchinson block estimator EXACT — no Monte-Carlo tolerance needed)
#   9  candidate order invariance (rank fusion + repeated full runs)
#   3, 10 are driver-level (global-context regression vs all25_repeats.csv;
#      final replay == manual subset_oscd) — see experiments/run_keyframe_gl_eval.py.
import glob
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import make_pipe, make_scene  # noqa: E402
from tests.test_change_information_exact import (  # noqa: E402
    CAMS, look_at_mini_cam, make_change_toy)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- CPU tests

def test_no_gt_static_scan():
    """Spec test 2a: the selection package must never reference GT artifacts
    or the evaluator. String-level scan keeps this honest against future edits."""
    forbidden = ("gt_mask", "utils.evaluate", "utils/evaluate", "ground_truth",
                 "Mean IoU")
    for path in glob.glob(os.path.join(REPO, "view_selection", "*.py")):
        src = open(path, encoding="utf-8", errors="replace").read()
        for token in forbidden:
            assert token not in src, f"{os.path.basename(path)} references {token!r}"


def test_mask_decomposition_identities():
    """Spec test 5: unresolved = relu(g-l), overlap = g*l, local_only =
    relu(l-g); all masks finite, in [0,1], and consistently gated by V."""
    from view_selection.global_local_keyframe import component_masks

    g = torch.Generator().manual_seed(0)
    m_g = torch.rand(24, 32, generator=g)
    m_l = torch.rand(24, 32, generator=g)
    V = (torch.rand(24, 32, generator=g) > 0.3).float()
    masks = component_masks(V, m_g, m_l)

    assert set(masks) == {"global", "local", "unresolved", "overlap",
                          "local_only"}
    for name, m in masks.items():
        assert m.shape == (24, 32), name
        assert torch.isfinite(m).all(), name
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0, name
        assert torch.equal(m * V, m), f"{name} leaks outside visibility"
    assert torch.allclose(masks["unresolved"], V * torch.relu(m_g - m_l))
    assert torch.allclose(masks["overlap"], V * m_g * m_l)
    assert torch.allclose(masks["local_only"], V * torch.relu(m_l - m_g))
    # a pixel is never both unresolved and local-only
    assert float((masks["unresolved"] * masks["local_only"]).abs().max()) == 0.0
    # difference identity: unresolved - local_only == V * (m_g - m_l)
    assert torch.allclose(masks["unresolved"] - masks["local_only"],
                          V * (m_g - m_l), atol=1e-6)


def test_percentile_rank_range_ties_and_order_invariance():
    """Spec §6 + test 9 (rank level): ranks live in [0,1], ties are averaged,
    and insertion order of the score dict never changes the result."""
    from view_selection.global_local_keyframe import percentile_rank

    scores = {3: 0.5, 7: 0.5, 1: -2.0, 12: 9.0, 5: 4.0}
    r = percentile_rank(scores)
    assert r[1] == 0.0 and r[12] == 1.0
    assert r[3] == r[7] == pytest.approx(0.375)  # ranks 1,2 tie -> 1.5/4
    shuffled = {k: scores[k] for k in [12, 1, 5, 7, 3]}
    assert percentile_rank(shuffled) == r
    assert percentile_rank({4: 1.23}) == {4: 1.0}


def test_local_rebuild_seed_set_dependence():
    """Spec test 7 (contract level): the rebuild RNG seed is a function of the
    SET S, not of greedy order; different sets / train seeds separate."""
    from view_selection.global_local_keyframe import local_rebuild_seed

    assert (local_rebuild_seed(0, [4, 1, 19])
            == local_rebuild_seed(0, [19, 4, 1])
            == local_rebuild_seed(0, (1, 4, 19)))
    assert local_rebuild_seed(0, [1, 4, 19]) != local_rebuild_seed(0, [1, 4, 20])
    assert local_rebuild_seed(0, [1, 4, 19]) != local_rebuild_seed(1, [1, 4, 19])
    assert 0 <= local_rebuild_seed(0, [0]) < 2 ** 32


def test_global_context_cache_key_and_lookup(tmp_path):
    """Spec §10: key carries every reproducibility field; lookup only returns
    complete, key-matching cache dirs."""
    from view_selection.global_context import (find_cached_context,
                                               global_context_dir,
                                               global_context_key,
                                               write_metadata)

    names = [f"im_{i}" for i in range(25)]
    key = global_context_key("Garden", "Instance_1", "abcd1234",
                             names, global_seed=1, resolution=4,
                             alpha_threshold=0.5, repo_root=REPO)
    for field in ("scene", "instance", "reference_checkpoint_hash",
                  "all25_frame_manifest_hash", "fusion_iterations",
                  "global_seed", "resolution", "alpha_threshold",
                  "code_commit", "rasterizer_commit"):
        assert field in key, field
    assert key["fusion_iterations"] == 16

    # frame manifest hash must react to the frame list
    key2 = global_context_key("Garden", "Instance_1", "abcd1234",
                              names[:24], 1, 4, 0.5, REPO)
    assert key["all25_frame_manifest_hash"] != key2["all25_frame_manifest_hash"]

    root = str(tmp_path)
    d = global_context_dir(root, "Garden", "Instance_1", key)
    assert find_cached_context(root, "Garden", "Instance_1", key) is None
    os.makedirs(d, exist_ok=True)
    for n in ("r_global.ply", "all25_rendered_soft_masks.pt", "alpha_masks.pt"):
        open(os.path.join(d, n), "wb").close()
    write_metadata(d, key, {"note": "test"})
    assert find_cached_context(root, "Garden", "Instance_1", key) == d
    # key mismatch (different seed) -> miss, never a silent wrong hit
    key_other = global_context_key("Garden", "Instance_1", "abcd1234",
                                   names, 2, 4, 0.5, REPO)
    assert find_cached_context(root, "Garden", "Instance_1", key_other) is None


# ---------------------------------------------------------------- GPU tests

def _gl_toy(n_views=8, num_probes=2):
    """Toy fixture: reference model (visibility), R_global with real change
    mass, cameras, config."""
    from view_selection.types import InformationConfig

    model_ref = make_change_toy(c_value=0.5, seed=3)  # opaque scene for alpha
    r_global = make_change_toy(c_value=0.0, seed=3)
    with torch.no_grad():  # concentrated "change" on a few Gaussians
        r_global._features_dc.data[2] = 2.5
        r_global._features_dc.data[5] = 1.5
    cams = [look_at_mini_cam(p) for p in CAMS[:n_views]]
    cfg = InformationConfig(num_probes=num_probes)
    return model_ref, r_global, cams, cfg


def _stub_rebuild(seed_base=11):
    """Deterministic set-dependent stand-in for the clean R_local rebuild."""
    def rebuild(S_sorted):
        model = make_change_toy(c_value=0.0, seed=3)
        with torch.no_grad():
            for s in S_sorted:  # local suspicion depends on the set only
                model._features_dc.data[(s + seed_base) % 7] = 1.0 + 0.1 * s
        return model
    return rebuild


@pytest.mark.gpu
def test_first_round_all_candidates_no_forced_seed():
    """Spec tests 1 + 7 fields: round 1 scores every frame, frame 0 is not
    preinserted, and the manifest carries the free-seed bookkeeping."""
    from view_selection.global_local_keyframe import select_keyframes_gl

    model_ref, r_global, cams, cfg = _gl_toy()
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")
    manifest = select_keyframes_gl(
        "kf_g_dir", cams, model_ref, r_global, 3, cfg, pipe, bg,
        _stub_rebuild(), "toy", train_seed=0)

    r1 = manifest["rounds"][0]
    assert r1["candidates"] == list(range(len(cams)))
    assert r1["components"] == ["global"]
    assert manifest["first_frame_forced"] is False
    assert manifest["claim_scope"] == "offline_pool_keyframe_selection"
    assert manifest["gt_used_for_selection"] is False
    assert manifest["global_context_uses_all_25"] is True
    # the pick is the round-1 argmax (ties -> smaller id), NOT frame 0 by fiat
    best = min((-v, int(i)) for i, v in r1["total_score"].items())[1]
    assert manifest["greedy_order"][0] == best == manifest["first_frame_id"]
    assert manifest["frame0_global_rank"] is not None
    assert len(manifest["greedy_order"]) == 3
    assert manifest["replay_order"] == sorted(manifest["greedy_order"])
    assert json.dumps(manifest)  # fully serializable


@pytest.mark.gpu
def test_first_round_equivalence_glu_equals_g():
    """Spec test 6: with S empty the kf_glu_dir scoring must reduce to
    kf_g_dir exactly (same probes, same single global component)."""
    from view_selection.global_local_keyframe import select_keyframes_gl

    model_ref, r_global, cams, cfg = _gl_toy()
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")
    kwargs = dict(budget=1, config=cfg, pipe=pipe, background=bg,
                  rebuild_local_fn=_stub_rebuild(), scene_id="toy",
                  train_seed=0)
    m_g = select_keyframes_gl("kf_g_dir", cams, model_ref, r_global, **kwargs)
    m_glu = select_keyframes_gl("kf_glu_dir", cams, model_ref, r_global,
                                **kwargs)
    assert m_g["rounds"][0]["total_score"] == m_glu["rounds"][0]["total_score"]
    assert m_g["greedy_order"][0] == m_glu["greedy_order"][0]


@pytest.mark.gpu
def test_topology_independence_and_determinism():
    """Spec tests 4 + 9: R_local may have a different Gaussian count than
    R_global every round; reruns with identical inputs select identically."""
    from view_selection.global_local_keyframe import select_keyframes_gl

    model_ref, r_global, cams, cfg = _gl_toy()
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")

    def rebuild_bigger(S_sorted):
        n = 7 + 3 * len(S_sorted)  # topology changes with |S|
        g = torch.Generator().manual_seed(99)
        pts = ((torch.rand((n, 3), generator=g) - 0.5) * 1.6).tolist()
        scales = (0.08 + 0.10 * torch.rand((n, 3), generator=g)).tolist()
        cols = [[0.0] * 3] * n
        model = make_scene(pts, scales, colors=cols)
        with torch.no_grad():
            model._features_dc.data[S_sorted[0] % n] = 1.2
        return model

    runs = [select_keyframes_gl("kf_glu_dir", cams, model_ref, r_global, 3,
                                cfg, pipe, bg, rebuild_bigger, "toy", 0)
            for _ in range(2)]
    assert runs[0]["greedy_order"] == runs[1]["greedy_order"]
    for rnd in runs[0]["rounds"][1:]:
        assert rnd["gaussian_count_local"] != rnd["gaussian_count_global"]
        assert rnd["components"] == ["global", "local", "unresolved"]


@pytest.mark.gpu
def test_scoring_never_touches_candidate_content():
    """Spec test 2b: planting exploding sentinels as image content on every
    view proves scoring is pose+model only (the guard would raise anyway;
    this pins the property even outside the guard's own bookkeeping)."""
    from view_selection.global_local_keyframe import select_keyframes_gl
    from view_selection.types import LeakageError

    class _Boom:
        def __getattr__(self, *_): raise LeakageError("image content read")
        def __getitem__(self, *_): raise LeakageError("image content read")

    model_ref, r_global, cams, cfg = _gl_toy()
    for cam in cams:
        cam.original_image = _Boom()
        cam.candidate_map = _Boom()
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")
    manifest = select_keyframes_gl("kf_gu_dir", cams, model_ref, r_global, 2,
                                   cfg, pipe, bg, _stub_rebuild(), "toy", 0)
    assert len(manifest["greedy_order"]) == 2


@pytest.mark.gpu
def test_block_information_exact_on_single_pixel_soft_weights():
    """Spec test 8 (new coverage): with a weight map supported on ONE pixel,
    g = sqrt(w)*xi*J_p with xi = ±1, so every probe returns exactly
    w * J_p^T J_p — the Hutchinson block estimator has zero variance and must
    match the autograd Jacobian outer product. Also: soft-weight scaling is
    linear, zero/degenerate weights hit the radii guard and yield zero."""
    from view_selection.information import (_render_raw,
                                            hutchinson_block_information)
    from view_selection.types import InformationConfig

    model = make_change_toy(c_value=0.3)
    pipe, bg = make_pipe(), torch.zeros(3, device="cuda")
    cam = look_at_mini_cam(CAMS[0])
    cfg = InformationConfig(num_probes=3)

    # pick a covered pixel with a NONZERO position Jacobian (a Gaussian's
    # dead center has ~zero spatial derivative, so search by response)
    y0, _ = _render_raw(model, cam, pipe, bg)
    order = y0.detach().flatten().argsort(descending=True)
    r = c = None
    for flat in order[:64].tolist():
        rr, cc = divmod(flat, y0.shape[1])
        y_probe, _ = _render_raw(model, cam, pipe, bg)
        (j,) = torch.autograd.grad(y_probe[rr, cc], model._xyz)
        if float(j.abs().max()) > 1e-6:
            r, c, jac = rr, cc, j
            break
    assert r is not None, "no pixel with a nonzero xyz Jacobian found"
    w = torch.zeros_like(y0.detach())
    w[r, c] = 0.7

    blocks = hutchinson_block_information(model, cam, w, cfg, ("t",), pipe, bg,
                                          frame_id=0)
    exact = 0.7 * jac.unsqueeze(2) * jac.unsqueeze(1)
    assert torch.allclose(blocks, exact, atol=1e-6, rtol=1e-4)
    assert float(blocks.abs().sum()) > 0  # the pixel actually sees Gaussians

    w4 = w * 4.0
    blocks4 = hutchinson_block_information(model, cam, w4, cfg, ("t",), pipe,
                                           bg, frame_id=0)
    assert torch.allclose(blocks4, 4.0 * blocks, atol=1e-5, rtol=1e-4)

    zero = hutchinson_block_information(model, cam, torch.zeros_like(w), cfg,
                                        ("t",), pipe, bg, frame_id=0)
    assert float(zero.abs().sum()) == 0.0


@pytest.mark.gpu
def test_frozen_change_model_roundtrip(tmp_path):
    """R_global reload must preserve the change scalar c (load_ply_change
    deliberately zeroes it; load_frozen_change_model restores it)."""
    from view_selection.global_context import load_frozen_change_model

    model = make_change_toy(n_extra_rest=False, c_value=0.0)
    with torch.no_grad():
        model._features_dc.data = torch.randn_like(model._features_dc.data)
    path = str(tmp_path / "r_global.ply")
    model.save_ply_change(path)

    loaded = load_frozen_change_model(path, sh_degree=0)
    assert loaded.get_xyz.shape == model.get_xyz.shape
    assert torch.allclose(loaded._features_dc.detach().cpu(),
                          model._features_dc.detach().cpu(), atol=1e-6)
    assert loaded._xyz.requires_grad  # blocks need the position Jacobian
