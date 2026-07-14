# Gate F: CLI round trip on a saved toy checkpoint (GPU).
import json
import os

import pytest
import torch

from tests.conftest import make_scene

pytestmark = pytest.mark.gpu


@pytest.fixture
def toy_checkpoint(tmp_path):
    model = make_scene(
        points=[[0.0, 0.0, 0.0], [0.6, 0.5, 0.5], [-0.5, -0.6, 0.4]],
        scales=[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    ply = str(tmp_path / "point_cloud.ply")
    model.save_ply(ply)
    assert os.path.exists(ply + ".pid.pt")  # persistent-ID sidecar

    cams = [{"view_id": "v0", "position": [0.0, 0.0, -1.5], "look_at": [0.0, 0.0, 0.0]},
            {"view_id": "v1", "position": [0.0, 0.3, -1.4], "look_at": [0.0, 0.0, 0.0]}]
    cams_path = str(tmp_path / "cams.json")
    with open(cams_path, "w") as f:
        json.dump(cams, f)
    return ply, cams_path, tmp_path


def _select(ply, cams_path, out_dir, extra=()):
    from tools.select_target_nbv import main
    return main(["--ply", ply, "--sh-degree", "0",
                 "--target-gaussian-id", "0",
                 "--observed-cameras", cams_path,
                 "--mode", "geometry_exact",
                 "--candidate-count", "16", "--proxy-top-k", "6",
                 "--exact-top-k", "3",
                 "--output-dir", str(out_dir), *extra])


def test_cli_select_output_schema(toy_checkpoint):
    ply, cams_path, tmp_path = toy_checkpoint
    out = _select(ply, cams_path, tmp_path / "run")

    for name in ("best_view.json", "candidates.json", "config_resolved.json",
                 "runtime.json", "information_before.pt",
                 "predicted_information_after.pt", "information_state.pt"):
        assert os.path.exists(os.path.join(out, name)), name
    assert os.path.exists(os.path.join(out, "debug", "best_view_rgb.png"))

    with open(os.path.join(out, "best_view.json")) as f:
        best = json.load(f)
    for key in ("position", "wxyz", "c2w", "w2c", "exact_score", "d_gain",
                "movement_cost", "target_persistent_id", "valid"):
        assert key in best, key
    assert best["valid"] is True and best["target_persistent_id"] == 0

    with open(os.path.join(out, "candidates.json")) as f:
        cands = json.load(f)
    assert len(cands) >= 1
    assert all("invalid_reason" in c and "proxy_score" in c for c in cands)

    H = torch.load(os.path.join(out, "information_before.pt"), weights_only=False)
    assert H.shape == (6, 6)


def test_cli_deterministic(toy_checkpoint):
    ply, cams_path, tmp_path = toy_checkpoint
    out1 = _select(ply, cams_path, tmp_path / "run1")
    out2 = _select(ply, cams_path, tmp_path / "run2")
    with open(os.path.join(out1, "best_view.json")) as f:
        b1 = json.load(f)
    with open(os.path.join(out2, "best_view.json")) as f:
        b2 = json.load(f)
    assert b1["cand_id"] == b2["cand_id"]
    assert b1["exact_score"] == pytest.approx(b2["exact_score"], abs=1e-12)


def test_cli_commit_updates_state_once(toy_checkpoint):
    from tools.update_target_information import main as update_main
    from target_nbv.config import TargetNBVConfig
    from target_nbv.info_builder import TargetInformationBuilder

    ply, cams_path, tmp_path = toy_checkpoint
    out = _select(ply, cams_path, tmp_path / "run")
    state_path = os.path.join(out, "information_state.pt")
    cfg = TargetNBVConfig().validate()

    views_before = list(TargetInformationBuilder.load(state_path, cfg)
                        .state.observed_view_ids)
    assert views_before == ["v0", "v1"]  # selection itself committed nothing

    args = ["--ply", ply, "--sh-degree", "0", "--state", state_path,
            "--config", os.path.join(out, "config_resolved.json"),
            "--camera", os.path.join(out, "best_view.json"),
            "--view-id", "sel_0"]
    assert update_main(args) is True
    after = TargetInformationBuilder.load(state_path, cfg).state
    assert after.observed_view_ids == ["v0", "v1", "sel_0"]

    assert update_main(args) is False  # duplicate commit is a no-op
    after2 = TargetInformationBuilder.load(state_path, cfg).state
    assert after2.observed_view_ids == ["v0", "v1", "sel_0"]
