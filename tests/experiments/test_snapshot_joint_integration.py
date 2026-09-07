from __future__ import annotations

import torch

import experiments.view_bayesian_detector_steps as viewer
from experiments import panel10_split_training as split
from tests.experiments import test_panel10_seed_occupancy as occ
from tests.experiments.test_panel10_split_training import (
    CaptureOptimizer,
    item,
    make_replay,
    typed_fake_render,
)


def _cache_backed_lifespan(*, cache_states: bool) -> viewer.DetectorReplayLifespan:
    life = viewer.DetectorReplayLifespan(
        3,
        max_states=4,
        device=torch.device("cpu"),
        cache_states=cache_states,
    )
    life.open_rows([0, 1], 0)
    life.seal_snapshot(0)
    life.close_rows([1], 1)
    life.seal_snapshot(1)
    return life


def _joint_replay(*, cache_states: bool):
    replay = make_replay(current=1)
    replay.args.panel10_render_mode = "joint_channels"
    replay.lifecycle = _cache_backed_lifespan(cache_states=cache_states)
    replay.seed_lifecycle = _cache_backed_lifespan(cache_states=cache_states)
    return replay


def _tensor_state(parameter: torch.Tensor) -> torch.Tensor:
    grad = parameter.grad
    if grad is None:
        return torch.empty(0, dtype=parameter.dtype, device=parameter.device)
    return grad.detach().clone()


def _optimizer_state(optimizer: CaptureOptimizer) -> dict[str, torch.Tensor]:
    return {
        "step_count": optimizer.step_count.detach().clone(),
        **{name: value.detach().clone() for name, value in optimizer.params.items()},
        **{f"{name}.grad": _tensor_state(value) for name, value in optimizer.params.items()},
    }


def test_snapshot_cache_preserves_joint_channel_training_across_current_and_replay_timestamps(monkeypatch):
    monkeypatch.setattr(split, "render_change", typed_fake_render)
    uncached = _joint_replay(cache_states=False)
    cached = _joint_replay(cache_states=True)

    for timestamp in [0, 1, 0, 1]:
        results = [
            split.train_partition_update(replay, item(timestamp=timestamp), current_timestamp=1)
            for replay in (uncached, cached)
        ]
        assert results[0] == results[1]
        for optimizer_name in ("base_optimizer", "seed_optimizer"):
            left = _optimizer_state(getattr(uncached, optimizer_name))
            right = _optimizer_state(getattr(cached, optimizer_name))
            assert left.keys() == right.keys()
            for key in left:
                assert torch.equal(left[key], right[key]), key

    cached_hits = cached.lifecycle.state_snapshots.statistics()["hits"]
    assert cached_hits > 0


def _occupancy_replay(*, cache_states: bool):
    replay = occ._replay(coverage="3d_plus_2d", birth_max=10, max_rows=100)
    replay.seed_lifecycle = viewer.DetectorReplayLifespan(
        0,
        max_states=4,
        device=torch.device("cpu"),
        cache_states=cache_states,
    )
    replay._seed_lifecycle_masks = lambda timestamp: (
        replay.seed_lifecycle.active_mask(timestamp),
        replay.seed_lifecycle.never_open_mask(timestamp),
    )
    return replay


def _prepare_mixed_seed_archive(replay) -> None:
    # Pending NEVER_OPEN row: must reserve its detector-probe footprint.
    occ._append_existing(replay, (32, 32), timestamp=1)
    # Move the learned pending row away: pending 2D occupancy must use the
    # frozen detector probe, while the 3D learned-support test should not win.
    replay.seed_model._xyz.data[0] = torch.tensor([100.0, 100.0, 2.0])
    replay.seed_lifecycle.seal_snapshot(1)

    # OPEN row: learned geometry participates in existing 2D occupancy.
    occ._append_existing(replay, (50, 50), timestamp=0)
    replay.seed_lifecycle.open_rows(torch.tensor([1]), 1)
    replay.seed_model.open_rows(torch.tensor([1]), 1)
    # Keep learned-support updates below the 3D min so this OPEN row is
    # rejected by 2D occupancy, not by the older 3D radius filter.
    replay.seed_geometry_update_counts[1] = 0
    replay.seed_lifecycle.seal_snapshot(1)

    # CLOSED, retired, and future rows must not block root birth.
    occ._append_existing(replay, (10, 10), timestamp=0)
    replay.seed_lifecycle.open_rows(torch.tensor([2]), 0)
    replay.seed_model.open_rows(torch.tensor([2]), 0)
    replay.seed_lifecycle.seal_snapshot(0)
    replay.seed_lifecycle.close_rows(torch.tensor([2]), 1)
    replay.seed_model.close_rows(torch.tensor([2]), 1)
    replay.seed_lifecycle.seal_snapshot(1)

    occ._append_existing(replay, (30, 10), timestamp=0)
    replay.seed_retired[3] = True
    replay.seed_lifecycle.seal_snapshot(1)

    occ._append_existing(replay, (50, 10), timestamp=5)
    replay.seed_lifecycle.seal_snapshot(2)


def _seed_pixels(replay) -> list[list[int]]:
    return [metadata["pixel_xy"] for metadata in replay.seed_model.metadata]


def test_snapshot_cache_preserves_da3_3d2d_root_birth_selection_with_mixed_seed_states(monkeypatch):
    batch = occ._batch([(32, 32), (50, 50), (10, 10), (30, 10), (50, 10), (20, 50), (21, 50)])
    replays = [_occupancy_replay(cache_states=False), _occupancy_replay(cache_states=True)]
    for replay in replays:
        _prepare_mixed_seed_archive(replay)
        occ._set_batch(monkeypatch, batch)

    results = [viewer.BayesianDetectorReplay._append_panel10_da3_seeds(replay, 2) for replay in replays]

    assert results[0] == results[1]
    assert results[0]["accepted"] == 4
    assert results[0]["coverage_2d_existing_rejected"] == 2
    assert results[0]["coverage_2d_same_frame_rejected"] == 1
    assert results[0]["proposed"] == results[0]["accepted"] + results[0]["coverage_rejected"] + results[0]["budget_rejected"]
    assert _seed_pixels(replays[0]) == _seed_pixels(replays[1])
    assert _seed_pixels(replays[0])[-4:] == [[10, 10], [30, 10], [50, 10], [20, 50]]
    assert replays[1].seed_lifecycle.state_snapshots.statistics()["hits"] > 0
