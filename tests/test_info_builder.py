# Gate D: information builder algebra (CPU).
import numpy as np
import pytest
import torch

from target_nbv.config import TargetNBVConfig
from target_nbv.info_builder import TargetInformationBuilder, view_information
from target_nbv.types import TargetParameterSpec

CFG = TargetNBVConfig().validate()
SPEC = TargetParameterSpec()


def _rand_J(n=30, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, d), generator=g, dtype=torch.float64)


def test_view_information_symmetric_psd():
    H = view_information(_rand_J())
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() >= -1e-10


def test_incremental_equals_recompute():
    b = TargetInformationBuilder(1, SPEC, CFG)
    for i in range(5):
        b.add_view(f"v{i}", _rand_J(seed=i))
    assert np.allclose(b.state.H_data, b.recompute_from_cache(), atol=1e-12)


def test_duplicate_view_not_double_counted():
    b = TargetInformationBuilder(1, SPEC, CFG)
    J = _rand_J()
    assert b.add_view("v0", J) is True
    H_after_first = b.state.H_data.copy()
    assert b.add_view("v0", J) is False
    assert np.array_equal(b.state.H_data, H_after_first)


def test_damping_applied_exactly_once_and_readonly():
    b = TargetInformationBuilder(1, SPEC, CFG)
    b.add_view("v0", _rand_J())
    lam = b.state.damping()
    H1 = b.H_prior()
    H2 = b.H_prior()  # calling twice must not stack damping
    assert np.allclose(H1, H2)
    assert np.allclose(H1 - b.state.H_data, lam * np.eye(6), atol=1e-12)
    assert not H1.flags.writeable
    # PSD after damping
    assert np.linalg.eigvalsh(H1).min() > 0


def test_weights_are_applied():
    J = _rand_J(n=12)
    w = torch.full((12,), 4.0, dtype=torch.float64)
    assert np.allclose(view_information(J, w), 4.0 * view_information(J), atol=1e-12)


def test_save_load_round_trip(tmp_path):
    b = TargetInformationBuilder(7, SPEC, CFG, model_version="mv1")
    for i in range(3):
        b.add_view(f"v{i}", _rand_J(seed=10 + i))
    p = str(tmp_path / "state.pt")
    b.save(p)

    b2 = TargetInformationBuilder.load(p, CFG, expected_model_version="mv1")
    assert np.allclose(b2.state.H_data, b.state.H_data)
    assert b2.state.observed_view_ids == b.state.observed_view_ids
    assert b2.state.version == b.state.version

    with pytest.raises(ValueError, match="model version"):
        TargetInformationBuilder.load(p, CFG, expected_model_version="mv2")


def test_diagnostics():
    b = TargetInformationBuilder(1, SPEC, CFG)
    b.add_view("v0", _rand_J(n=100))
    d = b.diagnostics()
    assert d["num_views"] == 1
    assert d["condition_number"] >= 1.0
    assert min(d["eigenvalues"]) > 0
