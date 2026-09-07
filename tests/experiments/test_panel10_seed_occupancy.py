from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import experiments.view_bayesian_detector_steps as viewer
from experiments.panel10_seed_topology import append_typed_seeds
from temporal.active_new_gaussians import ActiveNewGaussianModel
from temporal.lifespan_gate_beta import LifespanGateBetaFilter
from temporal.masked_optimizer import MaskedRowAdam
from temporal.new_seed_gaussians import NewSeedGaussianModel


def _optimizer(model: ActiveNewGaussianModel) -> MaskedRowAdam:
    return MaskedRowAdam(
        {
            "xyz": model._xyz,
            "dc": model.new_dc,
            "opacity": model._opacity,
            "scaling": model._scaling,
            "rotation": model._rotation,
        },
        thaw_names=("xyz", "dc", "opacity", "scaling", "rotation"),
        lrs={name: 1e-3 for name in ("xyz", "dc", "opacity", "scaling", "rotation")},
        eps=1e-15,
    )


def _replay(*, coverage="3d_plus_2d", birth_max=1024, max_rows=100):
    replay = object.__new__(viewer.BayesianDetectorReplay)
    replay.args = Namespace(
        da3_max_rows=max_rows,
        da3_birth_max_per_frame=birth_max,
        da3_birth_stride=1,
        da3_birth_coverage=coverage,
        da3_coverage_2d_sigma=2.0,
        da3_coverage_min_updates=4,
        da3_coverage_sigma=2.0,
        da3_initial_opacity=0.2,
        max_states=4,
        cue_scale=1.0,
    )
    replay.device = torch.device("cpu")
    replay.count = 2
    replay.records = [SimpleNamespace(name="frame0"), SimpleNamespace(name="frame1"), SimpleNamespace(name="frame2"), SimpleNamespace(name="frame3"), SimpleNamespace(name="frame4"), SimpleNamespace(name="frame5")]
    replay.current_index = 2
    replay.current_view = SimpleNamespace(candidate_map=torch.ones((1, 64, 64)))
    replay._typed_new_mask_cache = {2: torch.ones((64, 64), dtype=torch.bool)}
    replay._typed_depth_cache = {2: (torch.ones((16, 16)) * 2.0, torch.ones((16, 16)) * 3.0, _K(native=True), torch.eye(4))}
    replay.seed_model = ActiveNewGaussianModel(device=replay.device)
    replay.seed_detector_probe = NewSeedGaussianModel(device=replay.device)
    replay.seed_detector_probe.seed_dc.requires_grad_(False)
    replay.seed_lifecycle = viewer.DetectorReplayLifespan(0, max_states=4, device=replay.device)
    replay.seed_optimizer = _optimizer(replay.seed_model)
    replay.tracker = LifespanGateBetaFilter(replay.count + max_rows, device=replay.device)
    replay.seed_tracker = None
    replay.accepted_da3_source_rows = []
    replay.accepted_da3_birth_global = []
    replay.seed_retired = torch.empty(0, dtype=torch.bool)
    replay.seed_geometry_update_counts = torch.empty(0, dtype=torch.long)
    replay.seed_last_visible_rows = torch.empty(0, dtype=torch.bool)
    replay.rendered_gs_online_da3_depth_difference_rgb = lambda: None
    replay._normalized_cue_target = lambda view: torch.ones((64, 64))
    replay._seed_lifecycle_masks = lambda timestamp: (
        replay.seed_lifecycle.active_mask(timestamp),
        replay.seed_lifecycle.never_open_mask(timestamp),
    )
    return replay


def _K(*, native=False):
    # Native 16x16 intrinsics scale to full 64x64: f=40, cx=32, cy=32.
    if native:
        return torch.tensor([[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]])
    return torch.tensor([[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]])


def _xyz_for_pixel(x, y, z=2.0):
    return torch.tensor([[(float(x) - 32.0) * z / 40.0, (float(y) - 32.0) * z / 40.0, z]])


def _batch(pixels, *, z=2.0, scale=0.20):
    xyz = torch.cat([_xyz_for_pixel(x, y, z=z) for x, y in pixels], dim=0)
    return SimpleNamespace(
        xyz=xyz,
        log_scaling=torch.full((len(pixels), 3), float(torch.log(torch.tensor(scale)))),
        pixels_xy=torch.tensor(pixels, dtype=torch.long),
        confidence=torch.linspace(1.0, 0.1, steps=len(pixels)),
    )


def _set_batch(monkeypatch, batch):
    monkeypatch.setattr(viewer, "panel10_da3_seed_proposals", lambda *args, **kwargs: batch)


def _append_existing(replay, pixel, *, timestamp=0, scale=0.20):
    return append_typed_seeds(
        replay,
        _xyz_for_pixel(*pixel),
        torch.full((1, 3), float(torch.log(torch.tensor(scale)))),
        timestamp,
        [{"pixel_xy": list(pixel)}],
    )


def test_same_frame_q_priority_reservation_and_count_identity(monkeypatch):
    replay = _replay(birth_max=4)
    _set_batch(monkeypatch, _batch([(32, 32), (33, 32), (50, 50), (51, 50)]))

    result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)

    assert result["accepted"] == 2
    assert result["coverage_2d_same_frame_rejected"] == 2
    assert result["coverage_2d_existing_rejected"] == 0
    assert result["budget_rejected"] == 0
    assert result["proposed"] == result["accepted"] + result["coverage_rejected"] + result["budget_rejected"]
    assert result["coverage_rejected"] == result["coverage_3d_rejected"] + result["coverage_2d_existing_rejected"] + result["coverage_2d_same_frame_rejected"]
    assert [m["pixel_xy"] for m in replay.seed_model.metadata] == [[32, 32], [50, 50]]


def test_pending_birth_probe_geometry_blocks_despite_learned_mutation_and_black_dc(monkeypatch):
    replay = _replay()
    _append_existing(replay, (32, 32), timestamp=1)
    replay.seed_model._xyz.data[:] = _xyz_for_pixel(5, 5)
    replay.seed_model.new_dc.data.zero_()
    _set_batch(monkeypatch, _batch([(32, 32)]))

    result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)

    assert result["accepted"] == 0
    assert result["coverage_2d_existing_rejected"] == 1
    assert result["coverage_3d_rejected"] == 0


def test_closed_retired_and_future_born_rows_are_excluded(monkeypatch):
    replay = _replay()
    _append_existing(replay, (32, 32), timestamp=0)
    replay.seed_lifecycle.open_rows(torch.tensor([0]), 0)
    replay.seed_model.open_rows(torch.tensor([0]), 0)
    replay.seed_lifecycle.close_rows(torch.tensor([0]), 1)
    replay.seed_model.close_rows(torch.tensor([0]), 1)
    _append_existing(replay, (40, 40), timestamp=0)
    replay.seed_retired[1] = True
    _append_existing(replay, (48, 48), timestamp=5)
    _set_batch(monkeypatch, _batch([(32, 32), (40, 40), (48, 48)]))

    result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)

    assert result["accepted"] == 3
    assert result["coverage_2d_existing_rejected"] == 0


def test_no_opacity_or_dc_dependency_for_existing_occupancy(monkeypatch):
    replay = _replay()
    _append_existing(replay, (32, 32), timestamp=0)
    replay.seed_lifecycle.open_rows(torch.tensor([0]), 1)
    replay.seed_model.open_rows(torch.tensor([0]), 1)
    replay.seed_model.new_dc.data.fill_(0.0)
    replay.seed_model._opacity.data.fill_(-100.0)
    _set_batch(monkeypatch, _batch([(32, 32)]))

    result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)

    assert result["accepted"] == 0
    assert result["coverage_2d_existing_rejected"] == 1


def test_3d_only_preserves_legacy_selection_when_2d_would_block(monkeypatch):
    batch = _batch([(32, 32), (33, 32), (34, 32)])
    legacy = _replay(coverage="3d_only")
    _set_batch(monkeypatch, batch)
    legacy_result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(legacy, 2)

    two_d = _replay(coverage="3d_plus_2d")
    _set_batch(monkeypatch, batch)
    two_d_result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(two_d, 2)

    assert legacy_result["accepted"] == 3
    assert legacy_result["coverage_2d_same_frame_rejected"] == 0
    assert two_d_result["accepted"] == 1
    assert two_d_result["coverage_2d_same_frame_rejected"] == 2


def test_budget_remaining_classified_as_budget_not_coverage(monkeypatch):
    replay = _replay(birth_max=1)
    _set_batch(monkeypatch, _batch([(32, 32), (33, 32), (34, 32)]))

    result = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)

    assert result["accepted"] == 1
    assert result["budget_rejected"] == 2
    assert result["coverage_2d_same_frame_rejected"] == 0


@pytest.mark.parametrize("existing_depth,candidate_depth", [(2.0, 8.0), (8.0, 2.0)])
def test_distinct_depth_overlap_is_intentionally_blocked_only_by_2d(
    monkeypatch, existing_depth, candidate_depth,
):
    """Document the depth-blind policy, including an occluded background seed."""
    results = {}
    for mode in ("3d_only", "3d_plus_2d"):
        replay = _replay(coverage=mode)
        append_typed_seeds(
            replay, _xyz_for_pixel(32, 32, z=existing_depth),
            torch.full((1, 3), float(torch.log(torch.tensor(0.2)))), 0,
        )
        replay.seed_lifecycle.open_rows(torch.tensor([0]), 1)
        replay.seed_model.open_rows(torch.tensor([0]), 1)
        replay.seed_geometry_update_counts[0] = 4
        _set_batch(monkeypatch, _batch([(32, 32)], z=candidate_depth))
        results[mode] = viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2)
        assert results[mode]["coverage_3d_rejected"] == 0
    assert results["3d_only"]["accepted"] == 1
    assert results["3d_plus_2d"]["accepted"] == 0
    assert results["3d_plus_2d"]["coverage_2d_existing_rejected"] == 1
