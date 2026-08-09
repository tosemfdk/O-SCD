import math

import pytest
import torch

from temporal.mcmc_dynamics import (
    eq9_relocation_terms,
    synthetic_relocation_invariance_audit,
)


def upstream_scale_with_coeff_opacity(
    old_opacity: torch.Tensor, coeff_opacity: torch.Tensor, group_total: int
):
    coeff = torch.zeros((), device=old_opacity.device, dtype=old_opacity.dtype)
    for i in range(1, group_total + 1):
        for k in range(i):
            coeff = coeff + (
                math.comb(i - 1, k)
                * ((-1.0) ** k)
                / math.sqrt(k + 1.0)
                * torch.pow(coeff_opacity, k + 1)
            )
    return old_opacity.pow(2) / coeff.pow(2)


def upstream_eq9_formula(old_opacity: torch.Tensor, group_total: int):
    theoretical_new = 1.0 - torch.pow(1.0 - old_opacity, 1.0 / group_total)
    return theoretical_new, upstream_scale_with_coeff_opacity(
        old_opacity, theoretical_new, group_total
    )


@pytest.mark.parametrize("old_opacity", [0.005, 0.1, 0.5, 0.95])
@pytest.mark.parametrize("group_total", [2, 4, 8])
def test_eq9_scale_matches_upstream_formula_before_opacity_writeback_clamp(
    old_opacity, group_total
):
    old = torch.tensor(old_opacity, dtype=torch.float64)

    written, actual_scale, theoretical = eq9_relocation_terms(
        old, group_total, return_theoretical=True
    )
    expected_theoretical, expected_scale = upstream_eq9_formula(old, group_total)

    assert torch.allclose(theoretical, expected_theoretical, rtol=1e-12, atol=1e-12)
    assert torch.allclose(actual_scale, expected_scale, rtol=1e-12, atol=1e-12)
    assert written >= torch.tensor(0.005, dtype=torch.float64)


def test_eq9_low_opacity_scale_uses_unclamped_theoretical_opacity():
    old = torch.tensor(0.005, dtype=torch.float64)
    group_total = 8

    written, actual_scale, theoretical = eq9_relocation_terms(
        old, group_total, return_theoretical=True
    )
    _expected_theoretical, expected_scale = upstream_eq9_formula(old, group_total)
    clamped_scale = upstream_scale_with_coeff_opacity(old, written, group_total)

    assert theoretical < written
    assert written == torch.tensor(0.005, dtype=torch.float64)
    assert torch.allclose(actual_scale, expected_scale, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual_scale, clamped_scale, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("old_opacity", [0.1, 0.5, 0.95])
@pytest.mark.parametrize("group_total", [2, 4, 8])
@pytest.mark.parametrize("principal_scales", [(1.0, 1.0), (0.75, 2.25)])
def test_synthetic_relocation_invariance_audit_beats_naive_clone(
    old_opacity, group_total, principal_scales
):
    stats = synthetic_relocation_invariance_audit(
        old_opacity=old_opacity,
        group_total=group_total,
        principal_scales=principal_scales,
        grid_resolution=129,
    )

    assert stats["finite"] is True
    assert stats["mcmc_mean_abs_error"] < stats["naive_mean_abs_error"]
    assert stats["mean_error_improvement"] > 1.0
    assert stats["covariance_scale"] > 0.0
    assert stats["written_new_opacity"] == pytest.approx(
        stats["theoretical_new_opacity"]
    )
    assert set(stats) >= {
        "old_opacity",
        "group_total",
        "principal_scales",
        "mcmc_mean_abs_error",
        "naive_mean_abs_error",
        "mean_error_improvement",
        "mcmc_mean_le_threshold",
    }


@pytest.mark.parametrize("group_total", [2, 4, 8])
def test_synthetic_relocation_invariance_threshold_cases_match_upstream_behavior(
    group_total,
):
    # With the deterministic grid used by the audit, upstream Eq. 9 gets below
    # the intended 1e-4 mean alpha-response error for low-opacity isotropic
    # relocation.  Higher opacities are still much better than naive cloning but
    # do not achieve pointwise alpha-response invariance at this threshold.
    low = synthetic_relocation_invariance_audit(
        old_opacity=0.1,
        group_total=group_total,
        principal_scales=(1.0, 1.0),
        grid_resolution=129,
    )
    high = synthetic_relocation_invariance_audit(
        old_opacity=0.95,
        group_total=group_total,
        principal_scales=(1.0, 1.0),
        grid_resolution=129,
    )

    assert low["mcmc_mean_le_threshold"] is True
    assert low["mcmc_mean_abs_error"] <= 1e-4
    assert high["mcmc_mean_le_threshold"] is False
    assert high["mcmc_mean_abs_error"] > 1e-4
    assert high["mcmc_mean_abs_error"] < high["naive_mean_abs_error"]
