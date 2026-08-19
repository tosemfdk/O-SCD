import numpy as np
import pytest
import torch

from temporal import (
    CausalPC1,
    FollowEvidence,
    NewSignGate,
    SignMappingConfig,
    SignPosterior,
    component_balanced_follow,
    evidence_skip_reason,
    pooled_follow,
    probability_beta_less,
)


def _batch_pc1(delta: torch.Tensor) -> torch.Tensor:
    x64 = delta.double()
    mean = x64.mean(dim=0)
    covariance = x64.T @ x64 / x64.shape[0] - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    _, eigenvectors = torch.linalg.eigh(covariance)
    pc = eigenvectors[:, -1].float()
    anchor = int(torch.argmax(pc.abs()).item())
    if float(pc[anchor]) < 0:
        pc = -pc
    return pc


def test_causal_pc1_matches_exact_prefix_batch_pca_and_robust_epsilon():
    frame1 = torch.tensor(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, -0.5, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.1],
        ],
        dtype=torch.float32,
    )
    frame2 = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [3.0, 1.5, 0.2],
            [4.0, 2.0, -0.2],
            [5.0, 2.5, 0.1],
        ],
        dtype=torch.float32,
    )
    pca = CausalPC1(channels=3, stable_bank_size=32, seed=7, device="cpu")

    first = pca.update(frame1, torch.ones(4, dtype=torch.bool), eps_sigma=2.5)
    assert torch.allclose(first.pc, _batch_pc1(frame1), atol=1e-6)
    assert first.prefix_tokens == 4
    assert first.stable_bank_tokens == 4

    second = pca.update(frame2, torch.tensor([True, False, True, False]), eps_sigma=2.5)
    expected = _batch_pc1(torch.cat([frame1, frame2], dim=0))
    if torch.dot(second.pc, expected) < 0:
        expected = -expected
    assert torch.allclose(second.pc, expected, atol=1e-6)
    assert second.prefix_tokens == 8
    assert second.stable_tokens_seen == 6

    stable_bank = torch.cat([frame1, frame2[[0, 2]]], dim=0).half().float()
    stable_scores = stable_bank @ second.pc
    center = torch.median(stable_scores)
    sigma = 1.4826 * torch.median(torch.abs(stable_scores - center)).clamp_min(1e-8)
    assert second.stable_center == pytest.approx(float(center.item()))
    assert second.stable_sigma_mad == pytest.approx(float(sigma.item()))
    assert second.epsilon_negative == pytest.approx(float((center - 2.5 * sigma).item()))
    assert second.epsilon_positive == pytest.approx(float((center + 2.5 * sigma).item()))


def test_causal_pc1_sign_aligns_to_previous_axis_direction():
    delta = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    pca = CausalPC1(channels=2, stable_bank_size=16, seed=0, device="cpu")
    first = pca.update(delta, torch.ones(5, dtype=torch.bool))

    pca.prev_pc = -first.pc.clone()
    second = pca.update(delta, torch.ones(5, dtype=torch.bool))

    assert second.signed_axis_cosine_before_alignment < 0.0
    assert second.axis_stability == pytest.approx(1.0)
    assert torch.dot(second.pc, pca.prev_pc) == pytest.approx(1.0)
    assert torch.dot(second.pc, first.pc) == pytest.approx(-1.0)


def test_strong_signed_masks_use_pc_thresholds_and_active_cue():
    delta = torch.tensor([[-2.0], [0.1], [0.2], [3.0]], dtype=torch.float32)
    pca = CausalPC1(channels=1, stable_bank_size=8, seed=0, device="cpu")
    update = pca.update(delta, torch.ones(4, dtype=torch.bool), eps_sigma=0.0)

    plus, minus, score = pca.strong_signed_masks(
        delta,
        torch.tensor([[True, False], [False, True]]),
        update,
        output_shape=(2, 2),
    )

    assert score.shape == (2, 2)
    assert plus.tolist() == [[False, False], [False, True]]
    assert minus.tolist() == [[True, False], [False, False]]


def test_global_and_component_balanced_follow_match_prototype_formulas():
    config = SignMappingConfig(
        memory_weight_floor=0.0,
        component_threshold=0.1,
        min_memory_cells=1,
        min_component_cells=1,
        component_area_cap=2,
    )
    memory = np.zeros((4, 5), dtype=np.float64)
    memory[0, 0:4] = 1.0  # component A: area 4, follow 3/4
    memory[3, 3:5] = 1.0  # component B: area 2, follow 0/2
    same = np.zeros_like(memory, dtype=bool)
    opposite = np.zeros_like(memory, dtype=bool)
    same[0, 0:3] = True
    opposite[0, 3] = True
    opposite[3, 3:5] = True

    global_ev = pooled_follow(memory, same, opposite, config)
    balanced_ev = component_balanced_follow(memory, same, opposite, config)

    assert global_ev.follow == pytest.approx(3.0 / 6.0)
    assert global_ev.conflict == pytest.approx(3.0 / 6.0)
    # Component A and B are equally weighted after area cap: avg(3/4, 0).
    assert balanced_ev.follow == pytest.approx(0.375)
    assert balanced_ev.conflict == pytest.approx(0.625)
    assert [row["balance_weight"] for row in balanced_ev.components] == [2.0, 2.0]


def test_invalid_view_skip_reasons_follow_the_causal_update_order():
    ok = FollowEvidence(0.8, 0.1, 0.5, 5.0, 1.0, 5)
    bad = FollowEvidence(None, None, None, 0.0, 0.0, 0, reason="zero_sign_contrast")
    config = SignMappingConfig(axis_stability_min=0.9, min_camera_translation=0.1, min_sign_cells=4)

    assert evidence_skip_reason(1, 1.0, 1.0, 10, 10, ok, ok, config) == "bootstrap_frame"
    assert evidence_skip_reason(2, 0.5, 1.0, 10, 10, ok, ok, config) == "unstable_pc1_axis"
    assert evidence_skip_reason(2, 1.0, 0.01, 10, 10, ok, ok, config) == "insufficient_camera_translation"
    assert evidence_skip_reason(2, 1.0, 1.0, 3, 10, ok, ok, config) == "insufficient_positive_cue"
    assert evidence_skip_reason(2, 1.0, 1.0, 10, 3, ok, ok, config) == "insufficient_negative_cue"
    assert evidence_skip_reason(2, 1.0, 1.0, 10, 10, bad, ok, config) == "plus_zero_sign_contrast"
    assert evidence_skip_reason(2, 1.0, 1.0, 10, 10, ok, ok, config) is None


def test_fractional_beta_update_uses_probability_q_plus_less_than_q_minus():
    assert probability_beta_less(1.0, 1.0, 1.0, 1.0) == pytest.approx(0.5)
    assert probability_beta_less(2.0, 1.0, 1.0, 2.0) == pytest.approx(1.0 / 6.0, abs=1e-6)

    posterior = SignPosterior("global", confirm_probability=0.6, confirm_views=2)
    first = posterior.update(0.0, 1.0, "f001")
    assert first.alpha_plus == pytest.approx(1.0)
    assert first.beta_plus == pytest.approx(2.0)
    assert first.alpha_minus == pytest.approx(2.0)
    assert first.beta_minus == pytest.approx(1.0)
    assert first.p_plus_is_add == pytest.approx(5.0 / 6.0, abs=1e-6)
    assert first.pending_side == "+=ADD,-=REMOVE"
    assert first.confirmed_mapping is None

    second = posterior.update(0.0, 1.0, "f002")
    assert second.confirmed_mapping == "+=ADD,-=REMOVE"
    assert second.confirmed_at_frame == "f002"
    assert second.valid_updates == 2


def test_new_sign_gate_requires_global_and_balanced_consensus_and_reports_flip():
    gate = NewSignGate(threshold=0.8)

    paused = gate.update(0.85, 0.50, "f001")
    assert paused.new_sign is None
    assert paused.status == "paused"
    assert gate.snapshot()["current_new_sign"] is None

    opened = gate.update(0.81, 0.92, "f002")
    assert opened.new_sign == "+"
    assert opened.confidence == pytest.approx(0.81)
    assert opened.opened_this_update is True
    assert opened.flipped_this_update is False
    assert gate.snapshot()["first_open_frame"] == "f002"

    neutral = gate.update(0.3, 0.4, "f003")
    assert neutral.new_sign is None
    assert neutral.status == "paused"
    # Pause does not erase the last active sign; track managers decide what to do.
    assert gate.snapshot()["current_new_sign"] == "+"

    flipped = gate.update(0.19, 0.05, "f004")
    assert flipped.new_sign == "-"
    assert flipped.confidence == pytest.approx(0.81)
    assert flipped.flipped_this_update is True
    assert gate.snapshot()["flip_count"] == 1
