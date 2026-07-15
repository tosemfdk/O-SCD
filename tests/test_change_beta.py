# Gate G: per-Gaussian Beta change state (CPU).
import numpy as np
import pytest
import torch
from scipy.stats import beta as scipy_beta

from target_nbv.change.beta_state import BetaChangeState, beta_entropy


def make_state(n=4, a=1.0, b=1.0, scale=1.0):
    return BetaChangeState(n, a, b, pseudo_count_scale=scale, device="cpu")


def test_entropy_matches_scipy():
    a = torch.tensor([1.0, 2.5, 0.7, 10.0], dtype=torch.float64)
    b = torch.tensor([1.0, 0.3, 4.2, 10.0], dtype=torch.float64)
    ours = beta_entropy(a, b).numpy()
    ref = [scipy_beta.entropy(float(ai), float(bi)) for ai, bi in zip(a, b)]
    assert np.allclose(ours, ref, atol=1e-12)


def test_eig_zero_at_zero_tau_positive_at_unit():
    st = make_state()
    assert torch.allclose(st.expected_eig(0.0), torch.zeros(4, dtype=torch.float64),
                          atol=1e-12)
    assert (st.expected_eig(1.0) > 0).all()
    assert (st.unit_eig() > 0).all()


def test_eig_increases_with_tau():
    st = make_state()
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]
    vals = [float(st.expected_eig(t)[0]) for t in taus]
    assert all(v2 > v1 for v1, v2 in zip(vals, vals[1:]))


def test_marginal_eig_decreases_with_repeated_updates():
    # At the uniform prior EIG is second-order small, so the FIRST update can
    # raise the unit EIG; from then on repeated evidence must saturate it.
    st = make_state(n=1)
    e1 = torch.tensor([0.7], dtype=torch.float64)
    e0 = torch.tensor([0.3], dtype=torch.float64)
    eigs = []
    for _ in range(6):
        eigs.append(float(st.unit_eig()[0]))
        st.update(e1, e0)
    assert all(later < earlier for earlier, later in zip(eigs[1:], eigs[2:]))
    assert eigs[-1] < eigs[1]


def test_update_moves_correct_parameter():
    st = make_state(n=2, scale=2.0)
    st.update(torch.tensor([1.0, 0.0], dtype=torch.float64),
              torch.tensor([0.0, 1.0], dtype=torch.float64))
    # pseudo_count_scale applied exactly once
    assert st.a.tolist() == [3.0, 1.0]
    assert st.b.tolist() == [1.0, 3.0]
    assert st.posterior_mean()[0] > 0.5 > st.posterior_mean()[1]


def test_append_gets_prior():
    st = make_state(n=1, a=2.0, b=3.0)
    st.update(torch.tensor([1.0], dtype=torch.float64),
              torch.tensor([0.0], dtype=torch.float64))
    st.append(2)
    assert len(st) == 3
    assert st.a[1:].tolist() == [2.0, 2.0] and st.b[1:].tolist() == [3.0, 3.0]


def test_save_load_round_trip(tmp_path):
    st = make_state(n=3, a=1.5, b=0.5, scale=0.25)
    st.update(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64),
              torch.tensor([0.5, 0.0, 3.0], dtype=torch.float64))
    p = str(tmp_path / "beta.pt")
    st.save(p)
    st2 = BetaChangeState.load(p, device="cpu")
    assert torch.equal(st.a, st2.a) and torch.equal(st.b, st2.b)
    assert st2.pseudo_count_scale == 0.25 and st2.n_updates == 1


def test_validation_errors():
    with pytest.raises(ValueError, match="priors"):
        make_state(a=0.0)
    with pytest.raises(ValueError, match="pseudo_count_scale"):
        make_state(scale=-1.0)
    st = make_state(n=2)
    with pytest.raises(ValueError, match="shape"):
        st.update(torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="negative"):
        st.update(torch.tensor([-1.0, 0.0], dtype=torch.float64),
                  torch.zeros(2, dtype=torch.float64))
