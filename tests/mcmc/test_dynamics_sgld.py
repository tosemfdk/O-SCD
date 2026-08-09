import torch

from temporal.mcmc_dynamics import apply_sgld_xyz_noise_, sgld_xyz_noise
from .conftest import make_mcmc_gaussians


def test_sgld_zero_noise_scale_returns_exact_zero_noise_and_finite_stats():
    gaussians = make_mcmc_gaussians(raw_opacity=[-8.0, -4.6, 0.0, 4.0])

    noise, stats = sgld_xyz_noise(gaussians, xyz_lr=0.1, noise_scale=0.0, seed=3)

    assert torch.equal(noise, torch.zeros_like(gaussians.xyz))
    assert stats.noise_max_abs == 0.0
    assert stats.noise_rms == 0.0
    assert torch.isfinite(torch.tensor([stats.gate_min, stats.gate_mean, stats.gate_max])).all()


def test_sgld_noise_decreases_as_change_opacity_increases_for_same_seed():
    low_opacity = make_mcmc_gaussians(raw_opacity=[-9.0])
    high_opacity = make_mcmc_gaussians(raw_opacity=[9.0])

    low_noise, low_stats = sgld_xyz_noise(low_opacity, xyz_lr=0.1, noise_scale=1.0, seed=7)
    high_noise, high_stats = sgld_xyz_noise(high_opacity, xyz_lr=0.1, noise_scale=1.0, seed=7)

    assert low_stats.gate_mean > high_stats.gate_mean
    assert low_noise.norm() > high_noise.norm()


def test_sgld_noise_respects_anisotropic_covariance_axes():
    gaussians = make_mcmc_gaussians(raw_opacity=[-9.0] * 512, scaling=[2.0, 0.0, -2.0] * 512)

    noise, _stats = sgld_xyz_noise(gaussians, xyz_lr=1.0, noise_scale=1.0, seed=11)
    axis_rms = torch.sqrt((noise * noise).mean(dim=0))

    assert axis_rms[0] > axis_rms[1] > axis_rms[2]


def test_apply_sgld_is_seed_reproducible_and_preserves_parameter_identity():
    first = make_mcmc_gaussians(raw_opacity=[-8.0, -7.0, -6.0])
    second = make_mcmc_gaussians(raw_opacity=[-8.0, -7.0, -6.0])
    first_id = id(first.xyz)

    apply_sgld_xyz_noise_(first, xyz_lr=0.05, noise_scale=1.0, seed=13)
    apply_sgld_xyz_noise_(second, xyz_lr=0.05, noise_scale=1.0, seed=13)

    assert id(first.xyz) == first_id
    assert torch.equal(first.xyz, second.xyz)
