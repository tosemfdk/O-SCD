import torch

from temporal.learnable_cue_boundary import (
    HistogramBoundaryMLP,
    histogram_stage2_loss,
)


def test_zero_initialized_head_reproduces_heuristic_boundary() -> None:
    model = HistogramBoundaryMLP(bins=4, initial_tau=0.25, initial_width=0.10)
    histograms = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 1.0, 0.0, 0.0]])

    tau, width = model(histograms)

    torch.testing.assert_close(tau, torch.full((2,), 0.25))
    torch.testing.assert_close(width, torch.full((2,), 0.10))


def test_boundary_predictions_remain_inside_configured_intervals() -> None:
    model = HistogramBoundaryMLP(
        bins=4,
        tau_bounds=(0.10, 0.60),
        width_bounds=(0.03, 0.20),
    )
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([100.0, -100.0]))

    tau, width = model(torch.ones((1, 4)))

    assert 0.10 < float(tau.item()) < 0.60
    assert 0.03 < float(width.item()) < 0.20


def test_histogram_stage2_loss_prefers_teacher_aligned_prediction() -> None:
    counts = torch.tensor([[10.0, 10.0]])
    teacher_sums = torch.tensor([[0.0, 10.0]])
    aligned = torch.tensor([[0.05, 0.95]])
    inverted = torch.tensor([[0.95, 0.05]])

    good = histogram_stage2_loss(aligned, counts, teacher_sums)
    bad = histogram_stage2_loss(inverted, counts, teacher_sums)

    assert float(good.loss) < float(bad.loss)
    assert float(good.soft_iou) > float(bad.soft_iou)


def test_stage2_loss_backpropagates_to_tau_and_width() -> None:
    model = HistogramBoundaryMLP(bins=4)
    histograms = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    counts = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    teacher_sums = torch.tensor([[0.0, 1.0, 4.0, 5.0]])
    centers = torch.tensor([[0.125, 0.375, 0.625, 0.875]])
    tau, width = model(histograms)
    prediction = model.remap(centers, tau[:, None], width[:, None])

    loss = histogram_stage2_loss(prediction, counts, teacher_sums).loss
    loss.backward()

    assert model.head.bias.grad is not None
    assert bool(torch.isfinite(model.head.bias.grad).all())
    assert bool((model.head.bias.grad.abs() > 0).all())
