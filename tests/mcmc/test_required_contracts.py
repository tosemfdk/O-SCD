"""Named regression gates from the oracle-boundary MCMC experiment spec."""

from dataclasses import replace

import pytest
import torch

from experiments import train_oracle_boundary_mcmc_rchange as runner
from temporal.mcmc_dynamics import inverse_sigmoid, relocate_dead_gaussians_
from temporal.mcmc_state import FixedCapacityChangeState
from .conftest import make_base


def test_fixed_capacity_count():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=8), capacity=8)
    parameter_ids = {name: id(value) for name, value in state.current_parameter_items()}
    with torch.no_grad():
        state.current_raw_change_opacity.copy_(
            torch.tensor([[-10.0], [-9.0], [1.0], [1.2], [1.4], [1.6], [1.8], [2.0]])
        )
    relocate_dead_gaussians_(state, opacity_threshold=0.005, seed=3)
    assert state.capacity == 8
    assert all(value.shape[0] == 8 for _name, value in state.current_parameter_items())
    assert {name: id(value) for name, value in state.current_parameter_items()} == parameter_ids


def test_no_densification_or_pruning_called():
    base = make_base(n=4)
    called = []
    for name in ("densify_and_clone", "densify_and_split", "densify_and_prune", "prune_points", "reset_opacity"):
        setattr(base, name, lambda *args, _name=name, **kwargs: called.append(_name))
    state = FixedCapacityChangeState.from_gaussians(base, capacity=4)
    with torch.no_grad():
        state.current_raw_change_opacity.copy_(torch.tensor([[-10.0], [1.0], [1.0], [1.0]]))
    relocate_dead_gaussians_(state, opacity_threshold=0.005, seed=0)
    assert called == []


def test_relocation_raw_parameter_roundtrip():
    probabilities = torch.tensor([0.005, 0.1, 0.5, 0.95], dtype=torch.float64)
    raw = inverse_sigmoid(probabilities)
    assert torch.allclose(torch.sigmoid(raw), probabilities, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(raw).all()


def test_no_gt_access_during_training():
    clean = runner.FrameRecord(0, 0, "frame.png", "/rgb/frame.png", "")
    runner.assert_no_gt_training_records([clean])
    contaminated = replace(clean, mask_path="/dataset/gt_mask/frame.png")
    with pytest.raises(RuntimeError, match="forbidden"):
        runner.assert_no_gt_training_records([contaminated])


def test_manifest_input_hash_match(tmp_path):
    expected = {"base": "abc", "frames": [{"name": "0.png", "sha256": "def"}]}
    path = tmp_path / "input_hashes.json"
    path.write_text(__import__("json").dumps(expected), encoding="utf-8")
    assert runner.validate_expected_input_hashes(expected, str(path)) == {"checked": True, "matched": True}
    with pytest.raises(RuntimeError, match="Input hash validation failed"):
        runner.validate_expected_input_hashes({"base": "changed"}, str(path))


def test_no_nan_after_relocation():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=6), capacity=6)
    with torch.no_grad():
        state.current_raw_change_opacity.copy_(torch.tensor([[-12.0], [-11.0], [0.0], [0.5], [1.0], [2.0]]))
        state.current_scaling.copy_(torch.tensor([[0.0, -1.0, 1.0]]).expand(6, -1))
    relocate_dead_gaussians_(state, opacity_threshold=0.005, seed=5)
    for _name, value in state.current_parameter_items():
        assert torch.isfinite(value).all()
