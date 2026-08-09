import csv
import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import train_oracle_boundary_mcmc_rchange as runner
from temporal.mcmc_state import FixedCapacityChangeState, OracleBoundaryStateManager, StateArchive
from .conftest import make_base


@dataclass
class DummyPose:
    ok: bool = True


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_activate_all_slots_does_not_promote_cue_support_or_absent_opacity():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=3), capacity=3)
    before_support = state.cue_support_mask.clone()
    before_opacity = state.get_active_render_attributes()["opacity"].detach().clone()

    runner.activate_all_slots(state)

    assert torch.equal(state.cue_support_mask, before_support)
    assert torch.allclose(before_opacity, torch.full_like(before_opacity, 0.001), atol=1e-7)
    assert torch.allclose(state.get_active_render_attributes()["opacity"], before_opacity, atol=1e-7)


@pytest.mark.parametrize(
    "argv,expected_protocol,expected_updates",
    [
        (["prog", "--protocol", "oracle_stream", "--prefix-frames", "2"], "oracle_stream", 16),
        (["prog", "--protocol", "matched_exact"], "matched_exact", 120),
    ],
)
def test_parse_args_uses_protocol_aware_update_defaults(monkeypatch, argv, expected_protocol, expected_updates):
    monkeypatch.setattr(sys, "argv", argv)

    args = runner.parse_args()

    assert args.protocol == expected_protocol
    assert args.updates_per_frame == expected_updates


def test_conservative_mode_is_additive_and_keeps_a0_to_a4_mode_order(monkeypatch):
    assert runner.MODES[:6] == (
        "dc_only",
        "dc_opacity",
        "geo_adam",
        "geo_sgld",
        "geo_mcmc",
        "geo_mcmc_anchor",
    )
    assert runner.MODES[-1] == "geo_conservative_reloc"
    assert "geo_conservative_reloc" not in runner.SGLD_MODES

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--mode",
            "geo_conservative_reloc",
            "--conservative-min-observations",
            "4",
            "--conservative-min-support-views",
            "3",
            "--conservative-max-relocations",
            "12",
            "--conservative-burnin-steps",
            "64",
        ],
    )
    args = runner.parse_args()
    assert args.mode == "geo_conservative_reloc"
    assert args.conservative_min_observations == 4
    assert args.conservative_min_support_views == 3
    assert args.conservative_max_relocations == 12
    assert args.conservative_burnin_steps == 64
    assert args.conservative_max_relative_energy_increase == 0.0
    assert args.conservative_min_coverage_gain == pytest.approx(1e-6)


def test_conservative_stream_schedule_is_causal_and_replays_only_same_state_prefix():
    views = [
        SimpleNamespace(segment_id=0, image_name="000"),
        SimpleNamespace(segment_id=0, image_name="001"),
        SimpleNamespace(segment_id=1, image_name="002"),
        SimpleNamespace(segment_id=1, image_name="003"),
    ]

    schedule = runner.conservative_stream_schedule(views, 4, replay_buffer_size=2)

    for protocol_step, state, _local_step, training_view in schedule:
        arrival_index = protocol_step - 1
        assert int(training_view.segment_id) == state == int(views[arrival_index].segment_id)
        assert views.index(training_view) <= arrival_index


def test_conservative_burnin_views_are_deterministic_and_disjoint_when_buffer_is_full():
    views = [SimpleNamespace(image_name=str(index)) for index in range(8)]

    train, validation = runner.conservative_burnin_view_split(views)

    assert [view.image_name for view in train] == ["0", "2", "4", "6"]
    assert [view.image_name for view in validation] == ["1", "3", "5", "7"]
    assert not set(map(id, train)) & set(map(id, validation))


def test_negative_render_mass_uses_only_cue_negative_pixels(monkeypatch):
    view = SimpleNamespace()
    cue = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    render = torch.tensor([[[0.2, 0.9], [0.4, 0.8]]]).expand(3, -1, -1)
    monkeypatch.setattr(runner, "training_target", lambda _view: cue)

    mass = runner.aggregate_negative_render_mass([render], [view], cue_threshold=0.5)

    assert mass == pytest.approx(0.3)


def test_archive_replay_reuses_exact_recorded_view_names(monkeypatch):
    views = [SimpleNamespace(image_name="first"), SimpleNamespace(image_name="last")]
    manager = SimpleNamespace(archive=SimpleNamespace(records=[SimpleNamespace(state_id=0)]))
    audit = {
        "archive_index": 0,
        "render_checksum": "last",
        "parameter_checksum": "params",
        "view_names": ["last"],
    }

    def fake_archive_render_audit(_manager, _idx, selected, _base, _pipe, _background):
        return {
            "render_checksum": ",".join(view.image_name for view in selected),
            "parameter_checksum": "params",
        }

    monkeypatch.setattr(runner, "archive_render_audit", fake_archive_render_audit)
    result = runner.audit_archive_replay(
        [audit], manager, object(), object(), object(), {0: views}
    )

    assert result[0]["render_drift_zero"] is True
    assert result[0]["parameter_drift_zero"] is True


def test_pytest_ini_registers_mcmc_marker():
    text = Path("pytest.ini").read_text(encoding="utf-8")

    assert "mcmc:" in text
    assert "integration:" in text
    assert "cuda:" in text


def test_artifact_writer_creates_required_filenames_and_headers(tmp_path):
    writers = runner.ArtifactWriters(tmp_path)
    try:
        pass
    finally:
        writers.close()

    expected = {
        "training_energy.csv": ["global_step", "protocol_step", "state_id", "frame", "timestamp", "mode", "ssf", "energy"],
        "gaussian_counts.csv": ["global_step", "protocol_step", "state_id", "frame", "support_count", "rendered_count", "capacity"],
        "relocation_audit.csv": ["global_step", "protocol_step", "state_id", "frame", "attempted", "applied", "deferred", "pre_energy", "post_energy", "delta_energy", "attempted_post_energy", "attempted_relative_delta_total_energy", "reason"],
        "relocation_events.jsonl": None,
    }
    for filename, required_fields in expected.items():
        path = tmp_path / filename
        assert path.exists(), filename
        if required_fields is not None:
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            for field in required_fields:
                assert field in header


def test_synthetic_relocation_audit_records_render_improvement(tmp_path):
    event = {
        "global_step": 4,
        "protocol_step": 2,
        "state_id": 1,
        "frame": "001.png",
        "timestamp": 1.0,
        "attempted": 3,
        "applied": 2,
        "deferred": 1,
        "pre_energy": 5.0,
        "post_energy": 4.25,
        "delta_energy": -0.75,
        "pre_rendered_count": 7,
        "post_rendered_count": 8,
        "reason": "applied",
        "relocation": {
            "assignment_count": 3,
            "applied_count": 2,
            "deferred_count": 1,
            "events": [{"target_index": 9}],
            "applied_source_indices": [0, 1],
        },
    }
    writers = runner.ArtifactWriters(tmp_path)
    try:
        writers.write_relocation(event)
    finally:
        writers.close()

    payload = json.loads((tmp_path / "relocation_events.jsonl").read_text(encoding="utf-8"))
    row = next(csv.DictReader((tmp_path / "relocation_audit.csv").open(newline="", encoding="utf-8")))
    assert payload["post_energy"] < payload["pre_energy"]
    assert payload["post_rendered_count"] >= payload["pre_rendered_count"]
    assert float(row["delta_energy"]) < 0.0
    assert int(row["applied"]) == 2


def test_input_hash_mismatch_is_detectable_from_recomputed_hashes(tmp_path):
    source = tmp_path / "source"
    cue = tmp_path / "cue"
    fixed = tmp_path / "fixed.json"
    _write(source / "inference_scene" / "images" / "000.png", b"before")
    _write(source / "inference_scene" / "images" / "001.png", b"other")
    _write(source / "manifest.json", b"{}")
    _write(source / runner.BASE_PLY_REL, b"ply")
    _write(fixed, b"fixed")
    _write(cue / "metadata.json", b"meta")
    _write(cue / "cues" / "000.pt", b"cue")
    args = Namespace(source_path=source, cue_cache_root=cue, fixed_cameras_json=fixed)
    records, all_names = runner.build_no_gt_frame_records(source, (1,), prefix_frames=1)
    expected = runner.input_hashes(args, records, all_names)

    _write(source / "inference_scene" / "images" / "000.png", b"after")
    actual = runner.input_hashes(args, records, all_names)

    mismatches = [
        (before["name"], before["image_sha256"], after["image_sha256"])
        for before, after in zip(expected["selected_frame_hashes"], actual["selected_frame_hashes"])
        if before["image_sha256"] != after["image_sha256"]
    ]
    assert mismatches == [("000.png", expected["selected_frame_hashes"][0]["image_sha256"], actual["selected_frame_hashes"][0]["image_sha256"])]


def test_save_outputs_writes_required_artifacts_and_summary_no_gt_flags(tmp_path):
    base_ply = tmp_path / "source" / runner.BASE_PLY_REL
    fixed = tmp_path / "fixed.json"
    cue_root = tmp_path / "cue"
    _write(base_ply, b"ply")
    _write(fixed, b"fixed")
    _write(cue_root / "metadata.json", b"meta")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Existing writer artifacts should be checksummed into summary if present.
    writers = runner.ArtifactWriters(output_dir)
    writers.close()
    state = FixedCapacityChangeState.from_gaussians(make_base(n=2), capacity=2)
    manager = OracleBoundaryStateManager(state, boundaries=(1.0,), archive=StateArchive())
    args = Namespace(
        output_dir=output_dir,
        source_path=tmp_path / "source",
        mode="geo_mcmc",
        protocol="oracle_stream",
        init_policy="base_zero",
        boundaries=(1,),
        started_at=0.0,
        resolution=4.0,
        fixed_cameras_json=fixed,
        cue_cache_root=cue_root,
    )
    record = runner.FrameRecord(global_index=0, segment_id=0, name="000.png", image_path="/tmp/000.png", mask_path="")

    ckpt, summary_path, manifest_path = runner.save_outputs(
        state,
        manager,
        args,
        manifest={"counts": {"frames": 1}},
        records=[record],
        poses={"000.png": DummyPose()},
        intrinsics=np.eye(3),
        cue_metadata={"candidate_map_definition": "cached cue"},
        inputs={"selected_frame_hashes": []},
        support={"prefix_support_updates": True},
        loss_summary={"pre_train": {}, "post_train": {}},
        train_log=[],
        training_audit={"topology_ops": {"densification": False, "pruning": False, "relocation": True}},
        schedule_audit={"causal_prefix_only": True},
        archives=[],
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert ckpt.exists()
    assert summary["gt_used_for_training"] is False
    assert summary["gt_mask_pixels_loaded"] == 0
    assert summary["oracle_supervision"] is False
    assert summary["fixed_gaussian_topology"] is True
    assert summary.get("target_switch_known") is True
    assert summary.get("topology_mutation") is False
    assert summary.get("fixed_capacity_relocation") is True
    assert summary["bocd"] is False
    assert summary["support"]["prefix_support_updates"] is True
    assert {"training_energy.csv", "gaussian_counts.csv", "relocation_events.jsonl", "relocation_audit.csv"}.issubset(summary["artifact_sha256"])
    assert manifest["artifacts"]["checkpoint"] == str(ckpt)


def test_archive_replay_from_saved_state_dict_is_immutable_after_source_mutation():
    archive = StateArchive()
    tensors = {"xyz": torch.ones(2, 3), "raw_change_opacity": torch.zeros(2, 1)}
    archive.append(state_id=0, start_time=0.0, end_time=1.0, tensors=tensors)
    state_dict = archive.state_dict()

    tensors["xyz"].fill_(9.0)
    replay = StateArchive.from_state_dict(state_dict)
    replay_tensors = replay.tensors(0)
    replay_tensors["xyz"].fill_(7.0)

    assert torch.equal(replay.tensors(0)["xyz"], torch.ones(2, 3))
    replay.verify_checksums()
