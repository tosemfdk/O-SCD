from __future__ import annotations

import pytest
import torch

from temporal.signed_lifespan_density import compute_signed_lifespan_score


def test_signed_lifespan_score_aligns_plus_as_new_and_weights_by_age():
    result = compute_signed_lifespan_score(
        plus_mass=torch.tensor([0.8, 0.8]),
        minus_mass=torch.tensor([0.2, 0.2]),
        p_plus_is_new=torch.tensor([1.0, 0.0]),
        active=torch.tensor([True, True]),
        episode_start=torch.tensor([0.0, 2.0]),
        timestamp=3,
    )

    assert torch.allclose(result.sign_support, torch.tensor([0.6, -0.6]))
    assert torch.allclose(result.age, torch.tensor([4.0, 2.0]))
    assert torch.allclose(result.age_weight, torch.tensor([0.25, 0.5]))
    assert torch.allclose(result.score, torch.tensor([0.15, -0.3]))


def test_signed_lifespan_score_handles_remove_support_and_inactive_rows():
    result = compute_signed_lifespan_score(
        plus_mass=torch.tensor([0.1, 0.9, 0.9]),
        minus_mass=torch.tensor([0.7, 0.1, 0.1]),
        p_plus_is_new=1.0,
        active=torch.tensor([True, False, True]),
        episode_start=torch.tensor([5.0, 99.0, 4.0]),
        timestamp=5,
    )

    assert result.score[0].item() < 0.0
    assert result.score[1].item() == pytest.approx(0.0)
    assert result.sign_support[1].item() == pytest.approx(0.0)
    assert result.score[2].item() == pytest.approx(0.4)


def test_signed_lifespan_score_rejects_nonpositive_active_age():
    with pytest.raises(ValueError, match="active episode age"):
        compute_signed_lifespan_score(
            plus_mass=torch.tensor([1.0]),
            minus_mass=torch.tensor([0.0]),
            p_plus_is_new=1.0,
            active=torch.tensor([True]),
            episode_start=torch.tensor([3.0]),
            timestamp=1,
        )


def test_signed_lifespan_score_rejects_invalid_posterior_and_mass():
    with pytest.raises(ValueError, match="p_plus_is_new"):
        compute_signed_lifespan_score(
            plus_mass=torch.tensor([1.0]),
            minus_mass=torch.tensor([0.0]),
            p_plus_is_new=1.1,
            active=torch.tensor([True]),
            episode_start=torch.tensor([0.0]),
            timestamp=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        compute_signed_lifespan_score(
            plus_mass=torch.tensor([-1.0]),
            minus_mass=torch.tensor([0.0]),
            p_plus_is_new=1.0,
            active=torch.tensor([True]),
            episode_start=torch.tensor([0.0]),
            timestamp=0,
        )
