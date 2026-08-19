import pytest
import torch

from temporal.change_evidence import cue_to_change_probability, evidence_counts


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


def test_invalid_cues_and_evidence_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        cue_to_change_probability(
            torch.tensor([[float("nan")]]), mode="binary"
        )
    with pytest.raises(ValueError, match="same shape"):
        evidence_counts(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="nonnegative"):
        evidence_counts(torch.tensor([-1.0]), torch.tensor([1.0]))
