import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[2] / "temporal" / "bernoulli_bocd.py"
spec = importlib.util.spec_from_file_location("bernoulli_bocd_under_test", MODULE_PATH)
bocd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bocd
spec.loader.exec_module(bocd)

BernoulliBOCDConfig = bocd.BernoulliBOCDConfig
BetaBernoulliBOCD = bocd.BetaBernoulliBOCD
AdamsMacKayBetaBernoulliBOCD = bocd.AdamsMacKayBetaBernoulliBOCD
MAPResetBernoulliFilter = bocd.MAPResetBernoulliFilter
make_bocd_filter = bocd.make_bocd_filter


def _state_clone(filter_):
    return {k: v.clone() for k, v in filter_.state_dict().items()}


def _scalar_beta_log_predictive(s, f, a, b):
    import math

    return (
        math.lgamma(a + s)
        + math.lgamma(b + f)
        - math.lgamma(a + b + s + f)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )


def _scalar_logsumexp(values):
    import math

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def _scalar_adams_mackay_steps(observations, *, prior_a, prior_b, hazard, max_run_length):
    import math

    size = max_run_length + 1
    logp = [-math.inf] * size
    logp[0] = 0.0
    a = [prior_a] * size
    b = [prior_b] * size
    evidence = [0.0] * size
    visible = [0] * size
    run_start = [0] * size
    total_visible = 0
    states = []
    for timestamp, s_count, f_count in observations:
        old_logp = logp[:]
        old_a = a[:]
        old_b = b[:]
        old_evidence = evidence[:]
        old_visible = visible[:]
        old_start = run_start[:]
        if total_visible == 0:
            old_start = [timestamp] * size

        pred = [
            _scalar_beta_log_predictive(s_count, f_count, old_a[r], old_b[r])
            for r in range(size)
        ]
        new_logp = [-math.inf] * size
        new_logp[0] = _scalar_logsumexp(
            old_logp[r] + pred[r] + math.log(hazard) for r in range(size)
        )
        for r in range(size - 1):
            new_logp[r + 1] = old_logp[r] + pred[r] + math.log1p(-hazard)
        norm = _scalar_logsumexp(new_logp)
        logp = [v - norm for v in new_logp]

        a = [prior_a] * size
        b = [prior_b] * size
        evidence = [0.0] * size
        visible = [0] * size
        run_start = [timestamp + 1] + [0] * max_run_length
        for r in range(size - 1):
            a[r + 1] = old_a[r] + s_count
            b[r + 1] = old_b[r] + f_count
            evidence[r + 1] = old_evidence[r] + s_count + f_count
            visible[r + 1] = old_visible[r] + 1
            run_start[r + 1] = old_start[r]
        total_visible += 1
        states.append((logp[:], a[:], b[:], evidence[:], visible[:], run_start[:]))
    return states


def test_fractional_counts_use_integrated_predictive_and_stay_normalized():
    cfg = BernoulliBOCDConfig(prior_a=2.0, prior_b=3.0, hazard=0.25, max_run_length=4)
    filt = BetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    result = filt.update(torch.tensor([0.25], dtype=torch.float64), torch.tensor([0.75], dtype=torch.float64))
    expected = bocd.beta_binomial_log_predictive(
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([0.75], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
    )[0]
    manual = torch.lgamma(torch.tensor(2.25, dtype=torch.float64)) + torch.lgamma(torch.tensor(3.75, dtype=torch.float64)) - torch.lgamma(torch.tensor(6.0, dtype=torch.float64)) - (torch.lgamma(torch.tensor(2.0, dtype=torch.float64)) + torch.lgamma(torch.tensor(3.0, dtype=torch.float64)) - torch.lgamma(torch.tensor(5.0, dtype=torch.float64)))
    assert torch.allclose(expected, manual)
    assert torch.allclose(result.run_length_posterior[0].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(filt.a[0, :2], torch.tensor([2.25, 2.25], dtype=torch.float64))
    assert torch.allclose(filt.b[0, :2], torch.tensor([3.75, 3.75], dtype=torch.float64))


def test_exact_changepoint_uses_prior_predictive_not_old_run_predictive():
    cfg = BernoulliBOCDConfig(prior_a=1.0, prior_b=1.0, hazard=0.3, max_run_length=6)
    filt = BetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    for _ in range(5):
        filt.update(torch.tensor([5.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))
    old_log = filt.log_run_probs.clone()
    old_a = filt.a.clone()
    old_b = filt.b.clone()
    filt.update(torch.tensor([0.0], dtype=torch.float64), torch.tensor([5.0], dtype=torch.float64))
    prior_pred = bocd.beta_binomial_log_predictive(torch.tensor([0.0], dtype=torch.float64), torch.tensor([5.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))[0]
    growth_pred = bocd.beta_binomial_log_predictive(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[5.0]], dtype=torch.float64), old_a[:, :-1], old_b[:, :-1])[0]
    unnorm = torch.full_like(old_log[0], -torch.inf)
    unnorm[0] = torch.logsumexp(old_log[0] + torch.log(torch.tensor(0.3, dtype=torch.float64)), dim=0) + prior_pred
    unnorm[1:] = old_log[0, :-1] + torch.log(torch.tensor(0.7, dtype=torch.float64)) + growth_pred
    expected_cp = (unnorm - torch.logsumexp(unnorm, dim=0)).exp()[0]
    assert torch.allclose(filt.p_run_zero[0], expected_cp, atol=1e-12, rtol=1e-12)


def test_no_observation_preserves_every_state_tensor_exactly():
    cfg = BernoulliBOCDConfig(hazard=0.2, max_run_length=3, min_evidence_mass=0.5)
    filt = BetaBernoulliBOCD(3, cfg)
    filt.update(torch.tensor([1.0, 0.0, 0.5]), torch.tensor([0.0, 1.0, 0.5]))
    before = _state_clone(filt)
    result = filt.update(torch.tensor([100.0, 100.0]), torch.tensor([100.0, 100.0]), total_mass=torch.tensor([0.0, 0.49]), row_indices=torch.tensor([0, 2]), timestamp=7)
    for key, value in before.items():
        assert torch.equal(value, filt.state_dict()[key]), key
    assert result.indices.tolist() == [0, 2]
    assert result.observed.tolist() == [False, False]


def test_chunked_updates_match_single_batch_updates():
    cfg = BernoulliBOCDConfig(hazard=0.1, max_run_length=8)
    full = BetaBernoulliBOCD(5, cfg, dtype=torch.float64)
    chunked = BetaBernoulliBOCD(5, cfg, dtype=torch.float64)
    frames = [torch.tensor([0.2, 1.0, 0.0, 0.7, 0.3], dtype=torch.float64), torch.tensor([0.1, 0.9, 0.4, 0.6, 0.8], dtype=torch.float64), torch.tensor([0.8, 0.2, 0.3, 0.4, 0.9], dtype=torch.float64)]
    for t, ep in enumerate(frames):
        em = 1.0 - ep
        full.update(ep, em, timestamp=t)
        chunked.update(ep[:2], em[:2], row_indices=torch.tensor([0, 1]), timestamp=t)
        chunked.update(ep[2:], em[2:], row_indices=torch.tensor([2, 3, 4]), timestamp=t)
    for key, value in full.state_dict().items():
        other = chunked.state_dict()[key]
        assert torch.allclose(value, other, atol=1e-12, rtol=1e-12) if value.is_floating_point() else torch.equal(value, other)


def test_exact_preserves_low_hazard_changepoint_lineage_until_it_becomes_map():
    cfg = BernoulliBOCDConfig(prior_a=1.0, prior_b=1.0, hazard=0.01, max_run_length=80)
    filt = BetaBernoulliBOCD(1, cfg, dtype=torch.float64)

    one = torch.tensor([1.0], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)
    for timestamp in range(40):
        filt.update(one, zero, timestamp=timestamp)

    first_failure = filt.update(zero, one, timestamp=40)
    assert 0.0 < first_failure.changepoint_probability.item() < 0.5
    assert first_failure.map_run_length.item() != 0
    assert first_failure.run_length_posterior[0, 0].item() > 0.0
    assert filt.run_start[0, 0].item() == 40

    for timestamp, expected_run_length in ((41, 1), (42, 2), (43, 3)):
        result = filt.update(zero, one, timestamp=timestamp)
        posterior = result.run_length_posterior[0]
        assert result.map_run_length.item() == expected_run_length
        assert result.estimated_run_start.item() == 40
        assert posterior[expected_run_length].item() == posterior.max().item()
        assert posterior[expected_run_length].item() > 0.5


def test_exact_unobserved_gap_preserves_low_hazard_lineage_without_advancing():
    cfg = BernoulliBOCDConfig(
        prior_a=1.0,
        prior_b=1.0,
        hazard=0.01,
        max_run_length=80,
        min_evidence_mass=0.5,
    )
    filt = BetaBernoulliBOCD(1, cfg, dtype=torch.float64)

    one = torch.tensor([1.0], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)
    for timestamp in range(40):
        filt.update(one, zero, timestamp=timestamp)

    first_failure = filt.update(zero, one, timestamp=40)
    assert 0.0 < first_failure.changepoint_probability.item() < 0.5
    assert filt.run_start[0, 0].item() == 40
    before_gap = _state_clone(filt)

    gap = filt.update(
        torch.tensor([99.0], dtype=torch.float64),
        torch.tensor([99.0], dtype=torch.float64),
        total_mass=torch.tensor([0.0], dtype=torch.float64),
        timestamp=41,
    )
    for key, value in before_gap.items():
        assert torch.equal(value, filt.state_dict()[key]), key
    assert gap.observed.tolist() == [False]
    assert gap.run_length_posterior[0, 0].item() == first_failure.run_length_posterior[0, 0].item()

    resumed = filt.update(zero, one, timestamp=42)
    assert resumed.map_run_length.item() == 1
    assert resumed.estimated_run_start.item() == 40
    assert resumed.run_length_posterior[0, 1].item() > 0.5

    for timestamp, expected_run_length in ((43, 2), (44, 3)):
        result = filt.update(zero, one, timestamp=timestamp)
        assert result.map_run_length.item() == expected_run_length
        assert result.estimated_run_start.item() == 40



def test_adams_mackay_matches_scalar_algorithm_one_recurrence_over_fractional_sequence():
    cfg = BernoulliBOCDConfig(prior_a=1.3, prior_b=2.1, hazard=0.07, max_run_length=5)
    filt = AdamsMacKayBetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    observations = [
        (3, 0.25, 0.75),
        (4, 1.20, 0.10),
        (5, 0.00, 0.80),
        (6, 0.55, 0.45),
    ]

    expected_states = _scalar_adams_mackay_steps(
        observations,
        prior_a=1.3,
        prior_b=2.1,
        hazard=0.07,
        max_run_length=5,
    )

    for (timestamp, e_plus, e_minus), expected in zip(observations, expected_states):
        filt.update(
            torch.tensor([e_plus], dtype=torch.float64),
            torch.tensor([e_minus], dtype=torch.float64),
            timestamp=timestamp,
        )
        logp, a, b, evidence, visible, run_start = expected
        assert torch.allclose(
            filt.log_run_probs[0], torch.tensor(logp, dtype=torch.float64), atol=1e-12, rtol=1e-12
        )
        assert torch.allclose(filt.a[0], torch.tensor(a, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        assert torch.allclose(filt.b[0], torch.tensor(b, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        assert torch.allclose(
            filt.total_run_evidence[0], torch.tensor(evidence, dtype=torch.float64), atol=1e-12, rtol=1e-12
        )
        assert filt.run_visible_observations[0].tolist() == visible
        assert filt.run_start[0].tolist() == run_start


def test_adams_mackay_constant_hazard_keeps_p_run_zero_at_h_without_truncation():
    cfg = BernoulliBOCDConfig(prior_a=1.0, prior_b=1.0, hazard=0.17, max_run_length=12)
    filt = AdamsMacKayBetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    observations = [
        (1.0, 0.0),
        (0.5, 0.5),
        (0.0, 1.0),
        (2.0, 0.0),
        (0.0, 2.0),
    ]

    for timestamp, (plus, minus) in enumerate(observations):
        result = filt.update(
            torch.tensor([plus], dtype=torch.float64),
            torch.tensor([minus], dtype=torch.float64),
            timestamp=timestamp,
        )
        assert torch.allclose(
            result.changepoint_probability[0],
            torch.tensor(0.17, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(result.run_length_posterior[0].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_adams_mackay_cp_branch_uses_previous_predictive_but_resets_stats_and_start():
    cfg = BernoulliBOCDConfig(prior_a=1.0, prior_b=1.0, hazard=0.3, max_run_length=6)
    filt = AdamsMacKayBetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    filt.update(torch.tensor([3.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), timestamp=10)
    old_log = filt.log_run_probs.clone()
    old_a = filt.a.clone()
    old_b = filt.b.clone()

    filt.update(torch.tensor([0.0], dtype=torch.float64), torch.tensor([3.0], dtype=torch.float64), timestamp=11)

    pred = bocd.beta_binomial_log_predictive(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[3.0]], dtype=torch.float64),
        old_a,
        old_b,
    )[0]
    unnorm = torch.full_like(old_log[0], -torch.inf)
    unnorm[0] = torch.logsumexp(old_log[0] + pred + torch.log(torch.tensor(0.3, dtype=torch.float64)), dim=0)
    unnorm[1:] = old_log[0, :-1] + pred[:-1] + torch.log(torch.tensor(0.7, dtype=torch.float64))
    expected_cp = (unnorm - torch.logsumexp(unnorm, dim=0)).exp()[0]

    assert torch.allclose(filt.p_run_zero[0], expected_cp, atol=1e-12, rtol=1e-12)
    assert torch.allclose(filt.a[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(filt.b[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(filt.total_run_evidence[0, 0], torch.tensor(0.0, dtype=torch.float64))
    assert filt.run_visible_observations[0, 0].item() == 0
    assert filt.run_start[0, 0].item() == 12
    assert filt.run_start[0, 1].item() == 11
    assert filt.run_start[0, 2].item() == 10


def test_adams_mackay_low_hazard_reset_branch_can_grow_after_contradiction():
    cfg = BernoulliBOCDConfig(prior_a=1.0, prior_b=1.0, hazard=0.01, max_run_length=100)
    filt = AdamsMacKayBetaBernoulliBOCD(1, cfg, dtype=torch.float64)

    one = torch.tensor([1.0], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)
    stable = None
    for timestamp in range(40):
        stable = filt.update(one, zero, timestamp=timestamp)

    assert stable is not None
    assert filt.run_start[0, 0].item() == 40
    assert stable.run_length_posterior[0, 0].item() == pytest.approx(0.01)

    first_failure = filt.update(zero, one, timestamp=40)
    assert first_failure.changepoint_probability[0].item() == pytest.approx(0.01)
    assert filt.run_start[0, 1].item() == 40
    assert first_failure.run_length_posterior[0, 1].item() == pytest.approx(
        0.14974600001749427
    )
    assert first_failure.map_run_length.item() != 1

    for timestamp, expected_run_length in ((41, 2), (42, 3), (43, 4)):
        result = filt.update(zero, one, timestamp=timestamp)
        assert filt.run_start[0, expected_run_length].item() == 40
        assert result.map_run_length.item() == expected_run_length
        assert result.estimated_run_start.item() == 40
        assert result.run_length_posterior[0, expected_run_length].item() > 0.5


def test_adams_mackay_no_observation_preserves_every_state_tensor_exactly():
    cfg = BernoulliBOCDConfig(hazard=0.2, max_run_length=3, min_evidence_mass=0.5)
    filt = AdamsMacKayBetaBernoulliBOCD(3, cfg)
    filt.update(torch.tensor([1.0, 0.0, 0.5]), torch.tensor([0.0, 1.0, 0.5]))
    before = _state_clone(filt)
    result = filt.update(
        torch.tensor([100.0, 100.0]),
        torch.tensor([100.0, 100.0]),
        total_mass=torch.tensor([0.0, 0.49]),
        row_indices=torch.tensor([0, 2]),
        timestamp=7,
    )
    for key, value in before.items():
        assert torch.equal(value, filt.state_dict()[key]), key
    assert result.indices.tolist() == [0, 2]
    assert result.observed.tolist() == [False, False]


def test_map_reset_filter_is_explicit_alternative_and_resets_on_surprise():
    cfg = BernoulliBOCDConfig(hazard=0.2, max_run_length=6)
    filt = MAPResetBernoulliFilter(1, cfg, dtype=torch.float64)
    for _ in range(4):
        filt.update(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64))
    assert filt.map_run_length.item() >= 1
    filt.update(torch.tensor([0.0], dtype=torch.float64), torch.tensor([20.0], dtype=torch.float64))
    assert filt.map_run_length.item() == 0
    assert 0.5 <= filt.last_changepoint_probability.item() <= 1.0


def test_map_reset_exposes_predictive_reset_probability_and_uses_threshold():
    permissive = MAPResetBernoulliFilter(
        1,
        BernoulliBOCDConfig(
            hazard=0.2,
            max_run_length=4,
            changepoint_probability=0.0,
        ),
        dtype=torch.float64,
    )
    strict = MAPResetBernoulliFilter(
        1,
        BernoulliBOCDConfig(
            hazard=0.2,
            max_run_length=4,
            changepoint_probability=1.0,
        ),
        dtype=torch.float64,
    )
    permissive.update(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), timestamp=1)
    strict.update(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), timestamp=1)

    permissive_result = permissive.update(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        timestamp=2,
    )
    strict_result = strict.update(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        timestamp=2,
    )

    assert 0.0 < permissive_result.changepoint_probability.item() < 1.0
    assert torch.equal(
        permissive_result.changepoint_probability,
        strict_result.changepoint_probability,
    )
    assert permissive_result.map_run_length.item() == 0
    assert strict_result.map_run_length.item() > 0
    assert permissive_result.visible_observations.item() == 1
    assert strict_result.visible_observations.item() == 2


def test_legacy_chunk_signature_and_factory_modes():
    cfg = BernoulliBOCDConfig(max_run_length=2)
    filt = make_bocd_filter("exact", 2, cfg)
    out = filt.update([0], torch.tensor([1.0]), torch.tensor([0.0]), timestamp=3)
    assert out.visible_observations.tolist() == [1]
    adams_mackay = make_bocd_filter("adams_mackay", 2, cfg)
    assert isinstance(adams_mackay, AdamsMacKayBetaBernoulliBOCD)
    assert adams_mackay.algorithm == "adams_mackay"
    assert isinstance(make_bocd_filter("map_reset", 2, cfg), MAPResetBernoulliFilter)


def test_exact_run_start_uses_timestamps_across_unobserved_gaps():
    cfg = BernoulliBOCDConfig(hazard=0.1, max_run_length=8, min_evidence_mass=0.5)
    filt = BetaBernoulliBOCD(1, cfg, dtype=torch.float64)
    first = filt.update(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), timestamp=10)
    before_gap = _state_clone(filt)
    gap = filt.update(
        torch.tensor([9.0], dtype=torch.float64),
        torch.tensor([9.0], dtype=torch.float64),
        total_mass=torch.tensor([0.0], dtype=torch.float64),
        timestamp=20,
    )
    for key, value in before_gap.items():
        assert torch.equal(value, filt.state_dict()[key]), key
    second = filt.update(torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), timestamp=30)

    assert gap.observed.tolist() == [False]
    assert first.estimated_run_start.item() == 10
    assert second.estimated_run_start.item() == 10
    assert second.estimated_run_start.item() != second.visible_observations.item() - second.map_run_length.item()


def test_map_reset_state_storage_is_linear_and_run_start_uses_timestamps():
    cfg = BernoulliBOCDConfig(hazard=0.2, max_run_length=16, min_evidence_mass=0.5)
    filt = MAPResetBernoulliFilter(7, cfg, dtype=torch.float64)
    for name, value in filt.state_dict().items():
        assert value.ndim <= 1, name
        assert value.shape == (7,), name

    first = filt.update(torch.tensor([0, 3]), torch.tensor([1.0, 1.0], dtype=torch.float64), torch.tensor([0.0, 0.0], dtype=torch.float64), timestamp=100)
    before_gap = _state_clone(filt)
    filt.update(
        torch.tensor([0, 3]),
        torch.tensor([9.0, 9.0], dtype=torch.float64),
        torch.tensor([9.0, 9.0], dtype=torch.float64),
        total_mass=torch.tensor([0.0, 0.0], dtype=torch.float64),
        timestamp=150,
    )
    for key, value in before_gap.items():
        assert torch.equal(value, filt.state_dict()[key]), key
    second = filt.update(torch.tensor([0, 3]), torch.tensor([1.0, 1.0], dtype=torch.float64), torch.tensor([0.0, 0.0], dtype=torch.float64), timestamp=200)

    assert first.estimated_run_start.tolist() == [100, 100]
    assert second.estimated_run_start.tolist() == [100, 100]
