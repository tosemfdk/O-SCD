import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from experiments import train_oracle_boundary_mcmc_rchange as runner


def _write(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_no_gt_frame_record_builder_never_requires_mask_files(tmp_path):
    image_dir = tmp_path / "inference_scene" / "images"
    _write(image_dir / "000.png")
    _write(image_dir / "001.png")
    _write(image_dir / "002.png")

    records, all_names = runner.build_no_gt_frame_records(tmp_path, (2,), prefix_frames=2)

    assert all_names == ["000.png", "001.png", "002.png"]
    assert [record.name for record in records] == ["000.png", "001.png"]
    assert [record.mask_path for record in records] == ["", ""]
    assert [record.segment_id for record in records] == [0, 0]


def test_input_hashes_change_when_selected_image_content_changes(tmp_path):
    source = tmp_path / "source"
    cue = tmp_path / "cue"
    _write(source / "inference_scene" / "images" / "000.png", b"before")
    _write(source / "inference_scene" / "images" / "001.png", b"other")
    _write(source / "manifest.json", json.dumps({"counts": {}}).encode())
    _write(source / runner.BASE_PLY_REL, b"ply")
    _write(tmp_path / "fixed.json", b"fixed")
    _write(cue / "metadata.json", b"meta")
    _write(cue / "cues" / "000.pt", b"cue")
    _write(cue / "cues" / "001.pt", b"cue1")
    args = Namespace(source_path=source, cue_cache_root=cue, fixed_cameras_json=tmp_path / "fixed.json")
    records, all_names = runner.build_no_gt_frame_records(source, (1,), prefix_frames=1)

    before = runner.input_hashes(args, records, all_names)
    _write(source / "inference_scene" / "images" / "000.png", b"after")
    after = runner.input_hashes(args, records, all_names)

    assert before["selected_frame_hashes"][0]["image_sha256"] != after["selected_frame_hashes"][0]["image_sha256"]
    assert before["all_frame_names_sha256"] == after["all_frame_names_sha256"]


def test_optional_mcmc_status_reports_current_module_names_available_when_exposed():
    if not hasattr(runner, "optional_mcmc_api_status"):
        pytest.skip("runner no longer exposes optional_mcmc_api_status helper")

    status = runner.optional_mcmc_api_status()

    assert status["available"] is True
    assert "temporal.mcmc_state" in status["loaded_symbols"]
    assert "temporal.mcmc_energy" in status["loaded_symbols"]
    assert "temporal.mcmc_dynamics" in status["loaded_symbols"]


def test_artifact_writer_records_relocation_audit_and_jsonl(tmp_path):
    event = {
        "global_step": 1,
        "protocol_step": 1,
        "state_id": 0,
        "frame": "000.png",
        "timestamp": 0.0,
        "attempted": 2,
        "applied": 1,
        "deferred": 1,
        "pre_energy": 3.0,
        "post_energy": 2.5,
        "delta_energy": -0.5,
        "pre_rendered_count": 4,
        "post_rendered_count": 4,
        "reason": "applied",
        "relocation": {"applied_count": 1},
    }
    writers = runner.ArtifactWriters(tmp_path)
    try:
        writers.write_relocation(event)
    finally:
        writers.close()

    assert (tmp_path / "relocation_audit.csv").read_text().splitlines()[-1].endswith("2,1,1,3.0,2.5,-0.5,4,4,applied")
    assert json.loads((tmp_path / "relocation_events.jsonl").read_text())["relocation"] == {"applied_count": 1}


def test_stream_schedule_is_causal_prefix_replay_only():
    views = [SimpleNamespace(segment_id=0, image_name=f"{idx:03d}.png") for idx in range(3)]

    schedule = runner.stream_schedule(views, updates_per_frame=2)

    visible_by_step = {
        1: {"000.png"},
        2: {"000.png", "001.png"},
        3: {"000.png", "001.png", "002.png"},
    }
    for protocol_step, _state, _local, view in schedule:
        assert view.image_name in visible_by_step[protocol_step]


def test_energy_forward_produces_no_nan_for_finite_cpu_tensors():
    from temporal.mcmc_energy import MCMCEnergy

    energy = MCMCEnergy(opacity_weight=0.0, scale_weight=0.0)
    candidate = torch.zeros(1, 2, 2)
    rendered = torch.zeros(3, 2, 2)

    total, parts = energy(candidate, rendered)

    assert torch.isfinite(total)
    assert all(torch.isfinite(part).all() for part in parts.values())
