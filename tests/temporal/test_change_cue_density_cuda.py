import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cuda
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for change-cue density tests", allow_module_level=True)

pytest.importorskip("diff_gaussian_rasterization_fastgs")

from experiments.temporal_lifespan_smoke import build_synthetic_temporal_scene  # noqa: E402
from temporal.change_cue_density import (  # noqa: E402
    apply_fastgs_change_densification,
    compute_soft_multiview_change_score,
    topology_integrity,
)
from temporal.change_evidence import alpha_t_evidence_vjp  # noqa: E402


def _optimization_args():
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        lowfeature_lr=0.0025,
        highfeature_lr=0.005,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
    )


def test_k1_soft_score_matches_current_view_alpha_t_vjp():
    temporal, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    bank = temporal.base
    cue = torch.linspace(0.0, 1.0, 32, device="cuda").repeat(32, 1)[None]
    camera.candidate_map = cue
    direct = alpha_t_evidence_vjp(camera, bank, pipe, background, cue)
    score = compute_soft_multiview_change_score(
        [camera], [0], bank, pipe, background
    )
    assert score.view_indices == (0,)
    # FastGS backward uses atomic accumulation, so two separate VJPs are
    # mathematically equivalent but not guaranteed bitwise identical.
    assert torch.allclose(score.positive_mass, direct.e_plus, rtol=1e-5, atol=1e-5)
    assert torch.allclose(score.negative_mass, direct.e_minus, rtol=1e-5, atol=1e-5)
    assert torch.allclose(score.importance_score, direct.e_plus, rtol=1e-5, atol=1e-5)


def test_fastgs_clone_split_preserve_all_topology_and_optimizer_shapes():
    temporal, _camera, _pipe, _background = build_synthetic_temporal_scene(image_size=24)
    bank = temporal.base
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        getattr(bank, name).requires_grad_(True)
    bank.spatial_lr_scale = 1.0
    bank.training_setup_update(_optimization_args())

    # Initialize Adam moments for every optimizer-owned tensor before topology
    # mutation, including the separate FastGS SH optimizer.
    main_loss = sum(parameter.sum() for group in bank.optimizer.param_groups for parameter in group["params"])
    main_loss.backward()
    bank.optimizer.step()
    bank.optimizer.zero_grad(set_to_none=True)
    sh_loss = bank._features_rest.sum()
    sh_loss.backward()
    bank.shoptimizer.step()
    bank.shoptimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        bank.xyz_gradient_accum.fill_(2.0)
        bank.xyz_gradient_accum_abs.fill_(2.0)
        bank.denom.fill_(1.0)
        # Row 0 is small (clone); row 2 is large (split).
        bank._scaling[2].fill_(math.log(2.0))
    importance = torch.tensor([6.0, 0.0, 6.0, 0.0, 0.0], device="cuda")
    result = apply_fastgs_change_densification(
        bank,
        torch.ones(5, device="cuda"),
        importance,
        scene_extent=10.0,
        importance_threshold=5.0,
        grad_threshold=1.0,
        grad_abs_threshold=1.0,
        dense_fraction=0.02,
    )
    assert result.clone_count == 1
    assert result.split_count == 1
    assert result.initial_count == 5
    assert result.final_count == 7
    assert topology_integrity(bank)["passed"]
