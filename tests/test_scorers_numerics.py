# Gate E: shared numerical policy (CPU).
import numpy as np

from target_nbv.scorers.numerics import (GAIN_TOL, clamp_gain,
                                         logdet_via_slogdet,
                                         max_eig_of_inverse, robust_cholesky,
                                         symmetrize, trace_of_inverse)


def _spd(d=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    return A @ A.T + d * np.eye(d)


def test_symmetrize():
    rng = np.random.default_rng(1)
    H = rng.standard_normal((6, 6))
    S = symmetrize(H)
    assert np.allclose(S, S.T)
    assert np.allclose(S, 0.5 * (H + H.T))


def test_robust_cholesky_spd_no_jitter():
    c, jitter = robust_cholesky(_spd(), 1e-8, 1e-2)
    assert c is not None and jitter == 0.0


def test_robust_cholesky_escalates_jitter():
    H = np.diag([1.0, 1.0, -1e-7])  # slightly indefinite
    c, jitter = robust_cholesky(H, 1e-8, 1e-2)
    assert c is not None and 0.0 < jitter <= 1e-2


def test_robust_cholesky_gives_up():
    H = -np.eye(3)  # hopeless within max_jitter
    c, jitter = robust_cholesky(H, 1e-8, 1e-2)
    assert c is None and jitter is None


def test_logdet_matches_numpy():
    H = _spd(seed=2)
    ld, jitter = logdet_via_slogdet(H, 1e-8, 1e-2)
    assert jitter == 0.0
    assert np.isclose(ld, np.linalg.slogdet(H)[1])


def test_logdet_failure_returns_none():
    ld, jitter = logdet_via_slogdet(-np.eye(3), 1e-8, 1e-2)
    assert ld is None and jitter is None


def test_trace_of_inverse():
    H = _spd(seed=3)
    tr = trace_of_inverse(H, 1e-8, 1e-2)
    assert np.isclose(tr, np.trace(np.linalg.inv(H)), rtol=1e-10)


def test_max_eig_of_inverse():
    H = np.diag([0.5, 2.0, 8.0])
    assert np.isclose(max_eig_of_inverse(H), 2.0)


def test_clamp_gain():
    warnings: list[str] = []
    assert clamp_gain(-GAIN_TOL / 10, "g", warnings) == 0.0     # noise -> 0
    assert clamp_gain(0.3, "g", warnings) == 0.3                # positive kept
    assert warnings == []
    big_neg = clamp_gain(-1e-3, "g", warnings)                  # bug -> kept + warned
    assert big_neg == -1e-3 and len(warnings) == 1 and "g" in warnings[0]
