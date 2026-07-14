# CPU-only tests for target_nbv types and config (Gate A).
import numpy as np
import pytest

from target_nbv.types import (
    TargetHandle, TargetParameterSpec, CandidateCamera, TargetInformationState,
)
from target_nbv.config import TargetNBVConfig, CandidateConfig, ProxyConfig, InformationConfig, JacobianConfig


def test_default_config_valid():
    cfg = TargetNBVConfig().validate()
    assert cfg.mode == "geometry_exact"
    assert cfg.parameter_groups == ["mean", "log_scale"]


def test_quaternion_group_forbidden():
    with pytest.raises(ValueError, match="quaternion"):
        TargetNBVConfig(parameter_groups=["mean", "rotation"]).validate()
    with pytest.raises(ValueError, match="quaternion"):
        TargetParameterSpec(parameter_names=["mean", "quaternion"])


def test_unknown_group_and_mode_rejected():
    with pytest.raises(ValueError, match="unknown parameter group"):
        TargetNBVConfig(parameter_groups=["mean", "color"]).validate()
    with pytest.raises(ValueError, match="unsupported mode"):
        TargetNBVConfig(mode="global_dopt").validate()


def test_candidate_count_vs_topk():
    with pytest.raises(ValueError, match="candidate count"):
        TargetNBVConfig(candidates=CandidateConfig(count=8), proxy=ProxyConfig(top_k=12)).validate()
    with pytest.raises(ValueError, match="proxy top_k"):
        TargetNBVConfig(proxy=ProxyConfig(top_k=4), jacobian=JacobianConfig(exact_top_k=8)).validate()


def test_damping_positive_required():
    with pytest.raises(ValueError, match="absolute_damping"):
        TargetNBVConfig(information=InformationConfig(absolute_damping=0.0)).validate()


def test_radius_shells_sorted_positive():
    with pytest.raises(ValueError, match="radius_shells"):
        TargetNBVConfig(candidates=CandidateConfig(radius_shells=[1.0, 0.75])).validate()
    with pytest.raises(ValueError, match="radius_shells"):
        TargetNBVConfig(candidates=CandidateConfig(radius_shells=[])).validate()


def test_schur_mode_requires_neighbors():
    with pytest.raises(ValueError, match="geometry_schur"):
        TargetNBVConfig(mode="geometry_schur").validate()


def test_config_json_round_trip():
    cfg = TargetNBVConfig(seed=123)
    cfg2 = TargetNBVConfig.from_json(cfg.to_json())
    assert cfg2.to_dict() == cfg.to_dict()
    assert isinstance(cfg2.candidates, CandidateConfig)


def test_parameter_spec_slices():
    spec = TargetParameterSpec()
    assert spec.dimension == 6
    sl = spec.slices()
    assert sl["mean"] == slice(0, 3) and sl["log_scale"] == slice(3, 6)
    with pytest.raises(ValueError, match="duplicate"):
        TargetParameterSpec(parameter_names=["mean", "mean"])


def test_target_handle_single_requires_one_index():
    TargetHandle(persistent_id=5, current_indices=[3])
    with pytest.raises(ValueError, match="single"):
        TargetHandle(persistent_id=5, current_indices=[3, 4])
    with pytest.raises(ValueError, match="mode"):
        TargetHandle(persistent_id=5, current_indices=[3], mode="pair")


def test_candidate_camera_json_round_trip():
    cam = CandidateCamera(
        cand_id=7, position=np.array([1.0, 2.0, 3.0]), wxyz=np.array([1.0, 0, 0, 0]),
        fovx=1.2, fovy=1.0, width=128, height=128, shell_index=1, movement_cost=0.3,
        meta={"shell": 1}, minicam=object(),
    )
    d = cam.to_json_dict()
    assert "minicam" not in d
    cam2 = CandidateCamera.from_json_dict(d)
    assert cam2.cand_id == 7 and np.allclose(cam2.position, cam.position)
    assert cam2.minicam is None


def test_information_state_damping_and_readonly():
    spec = TargetParameterSpec()
    H_data = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0]).astype(np.float64)
    st = TargetInformationState(
        target_pid=1, spec=spec, H_data=H_data,
        absolute_damping=1e-6, relative_damping=0.1,
    )
    # relative damping = 0.1 * mean(diag)=0.25 dominates absolute
    assert st.damping() == pytest.approx(0.25)
    H = st.H_prior()
    assert H[0, 0] == pytest.approx(4.25)
    assert not H.flags.writeable
    with pytest.raises(ValueError):
        H[0, 0] = 99.0
