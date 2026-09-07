import pytest
import torch

from temporal.change_cue_fusion import (
    OscdPixelTerms,
    fuse_power_product,
    normalized_oscd_pixel_cue,
    normalized_oscd_pixel_cue_from_terms,
    power_product_from_cached_sum,
    semantic_from_cached_sum,
    sigmoid_soft_binarize,
    smoothstep_soft_binarize,
)


def test_power_product_recovers_semantic_term_from_sum() -> None:
    pixel = torch.tensor([[[0.0, 0.01], [0.25, 1.0]]])
    semantic = torch.tensor([[[0.8, 0.7], [0.6, 0.5]]])

    result = power_product_from_cached_sum(pixel + semantic, pixel, exponent=0.3)

    torch.testing.assert_close(result.pixel, pixel)
    torch.testing.assert_close(result.semantic, semantic)
    torch.testing.assert_close(result.fused, 2.0 * pixel.pow(0.3) * semantic)


def test_power_product_clamps_small_reconstruction_noise() -> None:
    pixel = torch.tensor([[[0.4, 0.8]]])
    cached_sum = torch.tensor([[[0.399999, 1.800001]]])

    result = power_product_from_cached_sum(cached_sum, pixel)

    assert result.semantic.tolist() == [[[0.0, 1.0]]]


def test_power_product_rejects_nonpositive_exponent() -> None:
    cue = torch.zeros((1, 2, 2))

    with pytest.raises(ValueError, match="positive"):
        power_product_from_cached_sum(cue, cue, exponent=0.0)


def test_normalized_pixel_cue_has_expected_shape_and_range() -> None:
    reference = torch.zeros((3, 16, 16))
    online = reference.clone()
    online[:, 4:12, 4:12] = 1.0

    cue = normalized_oscd_pixel_cue(reference, online)

    assert cue.shape == (1, 16, 16)
    assert float(cue.min()) == 0.0
    assert float(cue.max()) == pytest.approx(1.0)


def test_l1_only_power_is_applied_before_weighting_and_normalization() -> None:
    terms = OscdPixelTerms(
        l1=torch.tensor([[0.0, 0.01], [0.1, 1.0]]),
        structural=torch.tensor([[0.4, 0.3], [0.2, 0.1]]),
    )

    actual = normalized_oscd_pixel_cue_from_terms(terms, l1_exponent=0.3)
    raw = 0.8 * terms.l1.pow(0.3) + 0.2 * terms.structural
    expected = ((raw - raw.min()) / (raw.max() - raw.min() + 1e-8)).unsqueeze(0)

    torch.testing.assert_close(actual, expected)


def test_l1_only_power_uses_original_pixel_only_to_recover_semantic() -> None:
    original_pixel = torch.tensor([[[0.0, 0.2], [0.5, 1.0]]])
    powered_l1_pixel = torch.tensor([[[0.0, 0.5], [0.8, 1.0]]])
    semantic = torch.tensor([[[0.7, 0.6], [0.5, 0.4]]])

    recovered = semantic_from_cached_sum(original_pixel + semantic, original_pixel)
    fused = fuse_power_product(powered_l1_pixel, recovered, exponent=1.0)

    torch.testing.assert_close(recovered, semantic)
    torch.testing.assert_close(fused, 2.0 * powered_l1_pixel * semantic)


def test_smoothstep_soft_binarize_has_exact_plateaus_and_soft_center() -> None:
    cue = torch.tensor([0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 1.0])

    sharpened = smoothstep_soft_binarize(cue, low=0.15, high=0.35)

    torch.testing.assert_close(
        sharpened,
        torch.tensor([0.0, 0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0]),
    )


@pytest.mark.parametrize("low,high", [(0.4, 0.4), (0.6, 0.4), (-0.1, 0.4)])
def test_smoothstep_soft_binarize_rejects_invalid_band(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="band"):
        smoothstep_soft_binarize(torch.tensor([0.5]), low=low, high=high)


def test_sigmoid_soft_binarize_uses_width_as_transition_half_width() -> None:
    cue = torch.tensor([0.15, 0.25, 0.35])

    sharpened = sigmoid_soft_binarize(cue, tau=0.25, width=0.10)

    torch.testing.assert_close(sharpened, torch.tensor([0.05, 0.5, 0.95]))
