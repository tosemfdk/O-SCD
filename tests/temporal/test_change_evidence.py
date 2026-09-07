import pytest
import torch

from temporal.change_evidence import (
    cue_to_change_probability,
    evidence_counts,
    evidence_probe_scaling,
)

def test_cue_modes_binary_and_soft_fractional_extension():
    candidate = torch.tensor([[0.2, 0.6, 2.0]])
    assert torch.equal(cue_to_change_probability(candidate, mode="binary", threshold=0.5), torch.tensor([[0.0, 1.0, 1.0]]))
    assert torch.equal(cue_to_change_probability(candidate, mode="soft", scale=2.0), torch.tensor([[0.1, 0.3, 1.0]]))

def test_raw_and_capped_counts_and_unobserved_rows():
    plus = torch.tensor([2.0, 0.01])
    minus = torch.tensor([1.0, 0.01])
    da, db, mass, obs = evidence_counts(plus, minus, mode="raw", min_evidence_mass=0.1)
    assert torch.equal(obs, torch.tensor([True, False]))
    assert torch.equal(da, torch.tensor([2.0, 0.0]))
    assert torch.equal(db, torch.tensor([1.0, 0.0]))
    da, db, mass, obs = evidence_counts(plus, minus, mode="capped", mass_saturation=6.0, min_evidence_mass=0.1)
    assert torch.allclose(da[0], torch.tensor(1/3))
    assert torch.allclose(db[0], torch.tensor(1/6))
    assert da[1].item() == 0.0 and db[1].item() == 0.0

def test_capped_binary_counts_threshold_post_aggregation_only():
    plus = torch.tensor([2.0, 1.0, 0.0, 0.02, 0.30])
    minus = torch.tensor([1.0, 1.0, 0.0, 0.01, 0.20])

    da, db, mass, obs = evidence_counts(
        plus,
        minus,
        mode="capped_binary",
        mass_saturation=3.0,
        min_evidence_mass=0.1,
    )

    assert torch.equal(mass, plus + minus)
    assert torch.equal(obs, torch.tensor([True, True, False, False, True]))
    assert torch.allclose(da, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0 / 6.0]))
    assert torch.allclose(db, torch.tensor([0.0, 2.0 / 3.0, 0.0, 0.0, 0.0]))
    assert torch.equal(da * db, torch.zeros_like(da))
    assert torch.allclose(
        da + db,
        torch.where(obs, torch.clamp(mass / 3.0, 0.0, 1.0), torch.zeros_like(mass)),
    )

def test_capped_binary_zero_mass_stays_zero_when_observed_threshold_allows_it():
    plus = torch.tensor([0.0])
    minus = torch.tensor([0.0])

    da, db, mass, obs = evidence_counts(
        plus, minus, mode="capped_binary", mass_saturation=2.0, min_evidence_mass=0.0
    )

    assert obs.item() is True
    assert mass.item() == 0.0
    assert da.item() == 0.0
    assert db.item() == 0.0

def test_capped_binary_uses_gaussian_majority_not_pixel_thresholding():
    # Equal alpha-T responsibilities over three pixels with soft cue [0.99, 0.49, 0.49].
    # Correct Gaussian-level threshold after aggregation is active because 1.97 > 1.03.
    soft_plus = torch.tensor([0.99 + 0.49 + 0.49])
    soft_minus = torch.tensor([0.01 + 0.51 + 0.51])
    da, db, mass, obs = evidence_counts(
        soft_plus, soft_minus, mode="capped_binary", mass_saturation=1.0
    )
    assert obs.item() is True
    assert mass.item() == pytest.approx(3.0)
    assert da.item() == pytest.approx(1.0)
    assert db.item() == pytest.approx(0.0)

    # Incorrect pre-aggregation pixel thresholding would invert the same frame.
    pixel_thresholded_plus = torch.tensor([1.0])
    pixel_thresholded_minus = torch.tensor([2.0])
    wrong_da, wrong_db, _, _ = evidence_counts(
        pixel_thresholded_plus,
        pixel_thresholded_minus,
        mode="capped_binary",
        mass_saturation=1.0,
    )
    assert wrong_da.item() == pytest.approx(0.0)
    assert wrong_db.item() == pytest.approx(1.0)

def test_invalid_cues_and_evidence_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        cue_to_change_probability(
            torch.tensor([[float("nan")]]), mode="binary"
        )
    with pytest.raises(ValueError, match="same shape"):
        evidence_counts(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        evidence_counts(torch.tensor([-1.0]), torch.tensor([1.0]))

def test_isotropic_min_probe_scaling_shortens_only_long_axes_and_detaches():
    scaling = torch.tensor(
        [[1.0, 4.0, 2.0], [0.5, 0.25, 3.0]], requires_grad=True
    )

    native = evidence_probe_scaling(scaling, mode="native")
    isotropic = evidence_probe_scaling(scaling, mode="isotropic_min")

    assert not native.requires_grad
    assert not isotropic.requires_grad
    assert torch.equal(native, scaling.detach())
    assert torch.equal(
        isotropic,
        torch.tensor([[1.0, 1.0, 1.0], [0.25, 0.25, 0.25]]),
    )
    assert bool((isotropic <= scaling.detach()).all())

def test_probe_scaling_rejects_unknown_mode():
    with pytest.raises(ValueError, match=r"native\|isotropic_min"):
        evidence_probe_scaling(torch.ones(2, 3), mode="unknown")
