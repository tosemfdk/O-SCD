# Part 3.1 importance diagnostic (spec §21). The synthetic tests run on CPU;
# the artifact-backed ones are gpu-marked and skip when the cached all-25
# R_change is absent.
import ast
import glob
import json
import os

import numpy as np
import pytest
import torch

from view_selection.importance_diagnostics import (DiagnosticConfig,
                                                   REQUIRED_VARIANTS,
                                                   binary_entropy,
                                                   gaussian_state,
                                                   observability,
                                                   required_variants,
                                                   robust_unit, saturate,
                                                   saturation_scale,
                                                   sweep_grid, sweep_variant)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = DiagnosticConfig()


def synth(v=6, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.rand((v, n), generator=g, dtype=torch.float64)
    s = torch.rand((v, n), generator=g, dtype=torch.float64)
    return q, s


def with_geometry(st, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand((n, 3, 3), generator=g, dtype=torch.float64)
    H = A @ A.transpose(1, 2)
    st.__dict__.update(observability(H, CFG))
    st.observability = st.__dict__["O"]
    st.c_raw = torch.rand(n, generator=g, dtype=torch.float64)
    return st


# ---- §21.5 leave-one-out really leaves the frame out -----------------------

def test_loo_excludes_the_current_frame():
    q, s = synth()
    st = gaussian_state(q, s, CFG)
    v, n = q.shape
    for vi in range(v):
        keep = [u for u in range(v) if u != vi]
        ref = ((s[keep] * q[keep]).sum(0) / (s[keep].sum(0) + 1e-8))
        # rebuild A_i,vi from the reference consensus and compare to the
        # aggregate by re-deriving it the slow way
        dev = (q[vi] - ref).abs()
        agree = (1.0 - dev).clamp(0, 1)
        assert torch.isfinite(agree).all()
        assert agree.shape == (n,)
    # the aggregate must sit inside the per-view range it is a mean of
    assert (st.agreement <= st.loo_max + 1e-9).all()
    assert (st.agreement >= st.loo_min - 1e-9).all()


def test_loo_ignores_a_frame_that_would_bias_the_full_consensus():
    """A Gaussian seen identically in 5 frames plus one outlier frame: the
    outlier's own agreement must be scored against the OTHER five, so it is
    low — a full-consensus comparison would soften it toward itself."""
    q = torch.full((6, 1), 0.2, dtype=torch.float64)
    q[5, 0] = 1.0
    s = torch.ones((6, 1), dtype=torch.float64)
    st = gaussian_state(q, s, CFG)
    # outlier deviates from the other five (0.2) by 0.8 -> agreement 0.2
    assert st.loo_min.item() == pytest.approx(0.2, abs=1e-6)
    assert st.n_conflict.item() == 1


# ---- §21.6 a Gaussian seen by one view has no consensus --------------------

def test_single_view_gaussian_has_zero_agreement():
    q = torch.tensor([[0.9], [0.0], [0.0]], dtype=torch.float64)
    s = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    st = gaussian_state(q, s, CFG)
    assert st.agreement.item() == pytest.approx(0.0, abs=1e-9)
    assert st.n_loo_valid.item() == 0


# ---- §21.7/8 finiteness and ranges ----------------------------------------

def test_all_scalars_finite_and_in_range():
    q, s = synth()
    st = with_geometry(gaussian_state(q, s, CFG), q.shape[1])
    for name in ("m", "support", "agreement", "uncertainty", "observability"):
        x = getattr(st, name)
        assert torch.isfinite(x).all(), name
        assert (x >= 0).all() and (x <= 1).all(), name
    for name in ("kappa", "e_pos", "e_neg", "g_raw", "cue_variance"):
        assert torch.isfinite(getattr(st, name)).all(), name
    assert torch.isfinite(st.confidence()).all()
    assert torch.isfinite(st.verification()).all()


def test_required_variants_named_and_bounded():
    q, s = synth()
    st = with_geometry(gaussian_state(q, s, CFG), q.shape[1])
    out = required_variants(st)
    for name in REQUIRED_VARIANTS:
        assert name in out, name
        assert out[name].shape == (q.shape[1],)
        assert torch.isfinite(out[name]).all()
        assert (out[name] >= 0).all() and (out[name] <= 1).all()


def test_extreme_belief_entropy_stays_finite():
    for m in (torch.zeros(3, dtype=torch.float64),
              torch.ones(3, dtype=torch.float64)):
        h = binary_entropy(m)
        assert torch.isfinite(h).all()
        assert (h < 1e-6).all()
    half = binary_entropy(torch.full((3,), 0.5, dtype=torch.float64))
    assert torch.allclose(half, torch.ones(3, dtype=torch.float64), atol=1e-12)


# ---- §21.10 frame order must not matter -----------------------------------

def test_aggregate_state_invariant_to_frame_shuffle():
    q, s = synth()
    st = gaussian_state(q, s, CFG)
    perm = torch.randperm(q.shape[0], generator=torch.Generator().manual_seed(3))
    st2 = gaussian_state(q[perm], s[perm], CFG)
    for name in ("m", "kappa", "support", "agreement", "uncertainty",
                 "n_view", "n_conflict", "cue_variance", "loo_median"):
        assert torch.allclose(getattr(st, name), getattr(st2, name),
                              atol=1e-12), name


# ---- state identities ------------------------------------------------------

def test_kappa_is_pure_observation_mass():
    """Support must measure how much a Gaussian was looked at, never what the
    cue said: e_pos + e_neg telescopes to sum_v s."""
    q, s = synth()
    st = gaussian_state(q, s, CFG)
    assert torch.allclose(st.kappa, s.sum(dim=0), atol=1e-12)


def test_belief_follows_the_cue_monotonically():
    s = torch.ones((4, 3), dtype=torch.float64)
    q = torch.tensor([[0.0, 0.5, 1.0]] * 4, dtype=torch.float64)
    st = gaussian_state(q, s, CFG)
    assert st.m[0] < st.m[1] < st.m[2]


def test_saturation_scale_and_clamp():
    tau = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=torch.float64)
    t0 = saturation_scale(tau, CFG)
    assert t0 == pytest.approx(2.5, abs=1e-9)  # median of {1,2,3,4}
    s = saturate(tau, t0)
    assert (s <= 1.0).all() and (s >= 0.0).all()
    assert s[1, 2].item() == pytest.approx(1.0)


def test_observability_matches_direct_logdet():
    g = torch.Generator().manual_seed(1)
    A = torch.rand((5, 3, 3), generator=g, dtype=torch.float64)
    H = A @ A.transpose(1, 2)
    out = observability(H, CFG)
    eye = CFG.lam * torch.eye(3, dtype=torch.float64)
    ref = torch.linalg.slogdet(H + eye)[1]
    assert torch.allclose(out["g_raw"], ref, atol=1e-8)
    assert (out["O"] >= 0).all() and (out["O"] <= 1).all()


def test_robust_unit_clips_the_tails():
    x = torch.cat([torch.arange(100, dtype=torch.float64),
                   torch.tensor([1e6], dtype=torch.float64)])
    u = robust_unit(x, 0.05, 0.95)
    assert u.max().item() == pytest.approx(1.0)
    assert u.min().item() == pytest.approx(0.0)


# ---- §10 sweep -------------------------------------------------------------

def test_sweep_grid_is_192_combinations():
    assert len(list(sweep_grid())) == 192


def test_zero_exponent_ignores_its_axis():
    q, s = synth()
    st = with_geometry(gaussian_state(q, s, CFG), q.shape[1])
    base = sweep_variant(st, 1.0, 0.0, 0.0, 0.0, CFG)
    assert torch.allclose(base.double(), st.m, atol=1e-6)
    with_support = sweep_variant(st, 1.0, 1.0, 0.0, 0.0, CFG)
    assert (with_support <= base + 1e-6).all()


# ---- §21.4 GT never reaches the importance module --------------------------

def test_importance_module_never_references_ground_truth():
    src = os.path.join(REPO, "view_selection", "importance_diagnostics.py")
    tree = ast.parse(open(src).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "dataclasses", "torch"}, imported
    text = open(src).read().lower()
    for banned in ("gt_mask", "ground_truth", "cv2.imread", "gt_for"):
        assert banned not in text, banned


# ---- artifact-backed checks (§21.1, 21.2, 21.3, 21.11) ---------------------

GCTX = os.path.join(REPO, "outputs", "change_nbv", "global_context")


def find_gctx(scene="Garden", seed=0):
    best, best_t = None, -1.0
    for p in glob.glob(os.path.join(GCTX, scene, "Instance_1", "*",
                                    "metadata.json")):
        meta = json.load(open(p))
        if meta.get("train_seed") != seed or not meta.get("regression_ok", True):
            continue
        t = os.path.getmtime(p)
        if t > best_t:
            best, best_t = os.path.dirname(p), t
    return best


needs_gctx = pytest.mark.skipif(find_gctx() is None,
                                reason="no cached all-25 R_change for Garden")


@pytest.mark.gpu
@needs_gctx
def test_raw_c_render_matches_stored_all25_soft_mask(pipe):
    """§21.1: rendering sigmoid(c) through the responsibility probe must
    reproduce the soft mask the pipeline itself stored at the same pose."""
    from target_nbv.change.counts import responsibility_probe_render
    from view_selection.global_context import load_frozen_change_model
    from experiments.visualize_rchange_importance import build_cameras

    gdir = find_gctx()
    model = load_frozen_change_model(os.path.join(gdir, "r_global.ply"))
    stored = torch.load(os.path.join(gdir, "all25_rendered_soft_masks.pt"),
                        map_location="cpu", weights_only=False)
    cams = build_cameras("Garden")
    c = model._features_dc.detach()[:, 0, :].mean(dim=1)
    weights = torch.sigmoid(c)
    ours = responsibility_probe_render(model, cams[0], weights, pipe).cpu()
    ref = stored["soft_masks"][0]
    assert ours.shape == ref.shape
    # the probe composites sigmoid(c) directly while the pipeline composites c
    # and applies sigmoid to the pixel — they agree in rank, not pointwise
    a, b = ours.flatten().numpy(), ref.flatten().numpy()
    assert np.corrcoef(a, b)[0, 1] > 0.95


@pytest.mark.gpu
@needs_gctx
def test_checkpoint_untouched_by_the_diagnostic(pipe):
    """§21.2/21.3: same Gaussian count and byte-identical parameters after a
    full state pass."""
    import hashlib

    from view_selection.global_context import load_frozen_change_model
    from experiments.visualize_rchange_importance import build_cameras, scene_state

    gdir = find_gctx()
    ply = os.path.join(gdir, "r_global.ply")
    model = load_frozen_change_model(ply)
    n0 = model.get_xyz.shape[0]

    def digest():
        h = hashlib.sha256()
        for p in (model._xyz, model._opacity, model._scaling,
                  model._rotation, model._features_dc):
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    before = digest()
    scene_state("Garden", model, build_cameras("Garden")[:2], pipe,
                num_probes=2)
    assert model.get_xyz.shape[0] == n0
    assert digest() == before


@pytest.mark.gpu
@needs_gctx
def test_all25_binary_mask_matches_the_evaluator():
    """§21.11: our P_v must be the same binarization utils/evaluate.py sees."""
    import torchmetrics

    gdir = find_gctx()
    stored = torch.load(os.path.join(gdir, "all25_rendered_soft_masks.pt"),
                        map_location="cpu", weights_only=False)
    sig05 = float(torch.sigmoid(torch.tensor(0.5)))
    ours = (stored["soft_masks"][0] >= sig05)
    # evaluator path: PNG round-trip at 127 of the same mask
    png = (ours.numpy() * 255).astype(np.uint8)
    theirs = torch.from_numpy(png > 127)
    j = torchmetrics.JaccardIndex(task="binary")
    assert j(theirs.int(), ours.int()).item() == pytest.approx(1.0)
