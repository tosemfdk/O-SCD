import numpy as np
import pytest
import torch

from temporal.lifespan_state_snapshots import LifespanStateSnapshots, PackedLifespanState
from experiments.view_bayesian_detector_steps import DetectorReplayLifespan


@pytest.mark.parametrize('size', [0, 1, 2, 3, 4, 5, 63, 64, 65, 10003])
@pytest.mark.parametrize('pattern', ['constant', 'alternating', 'random'])
def test_codec_round_trip_and_payload_bound(size, pattern):
    states = np.full(size, 3, np.uint8)
    if pattern == 'alternating':
        states = (np.arange(size) % 4).astype(np.uint8)
    elif pattern == 'random':
        states = np.random.default_rng(16).integers(0, 4, size=size, dtype=np.uint8)
    snapshot = PackedLifespanState.encode(states)
    assert np.array_equal(snapshot.decode(), states)
    assert snapshot.nbytes <= (size + 3) // 4
    assert not snapshot.values.flags.writeable
    if pattern == 'constant' and size >= 64:
        assert snapshot.encoding == 'rle'
    if pattern == 'alternating' and size:
        assert snapshot.encoding == '2bit'


@pytest.mark.parametrize('states', [np.array([4], np.uint8), np.zeros((1, 2), np.uint8), np.zeros(4, np.int64)])
def test_codec_rejects_invalid_inputs(states):
    with pytest.raises(ValueError):
        PackedLifespanState.encode(states)


def test_bounded_cache_snapshot_invalidation_and_mask_ownership():
    life = DetectorReplayLifespan(100, max_states=4, device=torch.device('cpu'), cache_states=True)
    life.state_snapshots.decoded_capacity = 2
    life.open_rows([0], 0)
    for t in range(6):
        life.seal_snapshot(t)
    assert len(life.state_snapshots.frames) == 6
    assert len(life.state_snapshots.decoded) == 2
    original = life.state_snapshots.frames[0]
    life.active_mask(0).zero_()  # Public callers must not mutate cached masks.
    life.never_open_mask(0).zero_()
    life.materialized_mask(0).zero_()
    assert life.active_mask(0)[0] and life.never_open_mask(0)[1]
    life.close_rows([0], 4)  # Invalidate t>=4, preserving older snapshots.
    assert set(life.state_snapshots.frames) == {0, 1, 2, 3}
    assert life.state_snapshots.frames[0] is original
    assert life.closed_mask(5)[0]
    assert life.active_mask(3)[0]
    life.append_rows(2, materialized_timestamp=6)
    assert life.active_mask(0).shape == (102,)
    assert not life.materialized_mask(0)[100:].any()
    assert life.never_open_mask(6)[100:].all()
    assert life.state_snapshots.frames[0] is original
    assert life.state_snapshots.statistics()['payload_bytes'] > 0


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf')])
def test_cached_queries_reject_invalid_timestamp(timestamp):
    life = DetectorReplayLifespan(1, max_states=2, device=torch.device('cpu'), cache_states=True)
    with pytest.raises(TypeError, match='finite scalar'):
        life.active_mask(timestamp)


def test_repeated_snapshot_lookup_never_uses_interval_fallback():
    life = DetectorReplayLifespan(4, max_states=4, device=torch.device('cpu'), cache_states=True)
    life.open_rows([0], 0)
    life.seal_snapshot(0)
    life.close_rows([0], 1)
    life.seal_snapshot(1)
    life.open_rows([1], 2)
    for _ in range(120):
        for t in [0, 1, 2]:
            life.active_mask(t)
            life.never_open_mask(t)
            life.materialized_mask(t)
    assert life.state_snapshots.interval_fallbacks == 0
    assert life.state_snapshots.hits > 1000


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_snapshot_masks_match_interval_oracle_after_topology_events():
    life = DetectorReplayLifespan(512, max_states=4, device=torch.device('cuda'), cache_states=True)
    life.open_rows(list(range(0, 512, 3)), 0)
    life.seal_snapshot(0)
    life.close_rows(list(range(0, 512, 6)), 1)
    life.append_rows(32, materialized_timestamp=1)
    life.open_rows(list(range(512, 544)), 1)
    life.seal_snapshot(1)
    for t in [-1, 0, .5, 1, 2]:
        actual = [life.active_mask(t), life.never_open_mask(t), life.materialized_mask(t)]
        cache = life.state_snapshots
        life.state_snapshots = None
        expected = [life.active_mask(t), life.never_open_mask(t), life.materialized_mask(t)]
        life.state_snapshots = cache
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_cache_capacity_validation():
    with pytest.raises(ValueError):
        LifespanStateSnapshots(0)
