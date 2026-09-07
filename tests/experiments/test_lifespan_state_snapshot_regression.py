"""Independent state-ledger oracle for the snapshot timing ablation."""
import random

import torch
import pytest

from experiments.view_bayesian_detector_steps import DetectorReplayLifespan


def state_codes(life, timestamp):
    material = life.materialized_mask(timestamp)
    active = life.active_mask(timestamp)
    never = life.never_open_mask(timestamp)
    closed = life.closed_mask(timestamp)
    assert not bool((active & never).any())
    assert torch.equal(active | never | closed, material)
    return (never.to(torch.uint8) + 2 * active.to(torch.uint8)
            + 3 * closed.to(torch.uint8))


@pytest.mark.parametrize("cache_states", [False, True])
def test_same_timestamp_child_retirement_and_future_padding(cache_states):
    life = DetectorReplayLifespan(2, max_states=4, device=torch.device('cpu'), cache_states=cache_states)
    life.open_rows([0], 0)
    assert state_codes(life, 0).tolist() == [2, 1]
    life.seal_snapshot(0)
    life.seal_snapshot(2)  # A pending snapshot must be invalidated by topology below.
    assert life.append_rows(2, materialized_timestamp=2).tolist() == [2, 3]
    life.open_rows([2, 3], 2)
    life.close_rows([0], 2)
    assert state_codes(life, 2).tolist() == [3, 1, 2, 2]
    assert state_codes(life, 0).tolist() == [2, 1, 0, 0]
    assert not bool((life.active_mask(0) & life.active_mask(2)).any())
    life.open_rows([0], 3)
    assert state_codes(life, 0).tolist() == [2, 1, 0, 0]
    assert state_codes(life, 2).tolist() == [3, 1, 2, 2]
    assert life.get_active_state_indices(3).tolist() == [1, -1, 0, 0]
    assert life.validate_lifecycle()


@pytest.mark.parametrize("cache_states", [False, True])
def test_random_online_events_match_independent_timestamp_ledger(cache_states):
    rng = random.Random(16)
    life = DetectorReplayLifespan(9, max_states=32, device=torch.device('cpu'), cache_states=cache_states)
    current = [1] * 9
    history = []
    for t in range(24):
        if t % 3 == 0:
            first = len(current)
            assert life.append_rows(3, materialized_timestamp=t).tolist() == list(range(first, first + 3))
            current.extend([1] * 3)
        closes = [i for i, code in enumerate(current) if code == 2 and rng.random() < .25]
        opens = [i for i, code in enumerate(current) if code != 2 and rng.random() < .25]
        if closes:
            life.close_rows(closes, t)
            for i in closes:
                current[i] = 3
        if opens:
            life.open_rows(opens, t)
            for i in opens:
                current[i] = 2
        life.seal_snapshot(t)
        history.append(current.copy())
        for past, expected in enumerate(history):
            padded = expected + [0] * (len(current) - len(expected))
            assert state_codes(life, past).tolist() == padded
        assert life.validate_lifecycle()


@pytest.mark.parametrize('test_name', [
    'test_append_typed_seeds_aligns_append_only_identity_and_never_open_rows',
    'test_density_split_retires_parent_without_compaction_and_preserves_optimizer_state',
    'test_prune_typed_seeds_only_retires_old_visible_low_opacity_open_rows',
    'test_unlimited_archive_grows_past_twenty_thousand_without_reusing_closed_rows',
])
def test_existing_topology_contract_with_snapshots(monkeypatch, test_name):
    import importlib.util
    from functools import partial
    from pathlib import Path

    path = Path(__file__).with_name('test_panel10_seed_topology.py')
    spec = importlib.util.spec_from_file_location('snapshot_topology_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'DetectorReplayLifespan', partial(DetectorReplayLifespan, cache_states=True))
    getattr(module, test_name)()


def test_cached_replay_preserves_exact_losses_gradients_and_parameter_updates(monkeypatch):
    import importlib.util
    from pathlib import Path
    from experiments import panel10_split_training as split

    path = Path(__file__).with_name('test_panel10_split_training.py')
    spec = importlib.util.spec_from_file_location('snapshot_partition_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(split, 'render_change', module.fake_render)
    replays = []
    for enabled in [False, True]:
        replay = module.make_replay(current=1)
        for name in ['lifecycle', 'seed_lifecycle']:
            life = DetectorReplayLifespan(3, max_states=4, device=torch.device('cpu'), cache_states=enabled)
            life.open_rows([0, 1], 0)
            life.seal_snapshot(0)
            life.close_rows([1], 1)
            setattr(replay, name, life)
        replays.append(replay)
    for timestamp in [0, 1, 0, 1, 1, 0]:
        results = [split.train_partition_update(r, module.item(timestamp=timestamp), current_timestamp=1)
                   for r in replays]
        assert results[0] == results[1]
        for optimizer_name in ['base_optimizer', 'seed_optimizer']:
            left, right = [getattr(r, optimizer_name) for r in replays]
            assert torch.equal(left.step_count, right.step_count)
            for name, parameter in left.params.items():
                other = right.params[name]
                assert torch.equal(parameter, other)
                if parameter.grad is None:
                    assert other.grad is None
                else:
                    assert torch.equal(parameter.grad, other.grad)
