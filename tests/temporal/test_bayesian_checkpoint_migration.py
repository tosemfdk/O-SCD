import torch

from temporal.checkpoint import migrate_temporal_state_dict


def test_old_dc_checkpoint_infers_lifecycle_buffers():
    legacy = {"state_change_dc": torch.zeros(2, 3, 1, 3), "state_start": torch.zeros(2, 3), "state_end": torch.tensor([[1.0, float("inf"), float("inf")], [float("inf"), float("inf"), float("inf")]]), "state_valid": torch.tensor([[True, True, False], [False, False, False]])}
    m = migrate_temporal_state_dict(legacy)
    assert m["state_status"].tolist() == [[2, 1, 0], [0, 0, 0]]
    assert m["num_states"].tolist() == [2, 0]
    assert m["current_state_index"].tolist() == [1, -1]


def test_old_state_specific_geometry_checkpoint_infers_lifecycle_buffers():
    legacy = {"state_change_dc": torch.zeros(1, 2, 1, 3), "state_xyz_delta": torch.zeros(1, 2, 3), "state_start": torch.zeros(1, 2), "state_end": torch.full((1,2), float("inf")), "state_valid": torch.tensor([[True, False]])}
    assert migrate_temporal_state_dict(legacy)["current_state_index"].tolist() == [0]


def test_old_shared_geometry_checkpoint_gets_both_migrations():
    legacy = {"state_change_dc": torch.zeros(3, 2, 1, 3), "shared_xyz_delta": torch.zeros(3, 3), "state_start": torch.zeros(3,2), "state_end": torch.full((3,2), float("inf")), "state_valid": torch.zeros(3,2, dtype=torch.bool)}
    m = migrate_temporal_state_dict(legacy)
    assert "geometry_frozen" in m and "state_status" in m

from types import SimpleNamespace
from torch import nn
from temporal.change_model import TemporalChangeModel


def _base(n=1):
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(n, 3)),
        _features_dc=nn.Parameter(torch.ones(n, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(torch.zeros(n, 4)),
    )


def test_new_bayesian_lifecycle_checkpoint_roundtrip_preserves_close_reopen_metadata():
    model = TemporalChangeModel.from_gaussians(_base(), max_states=3)
    model.reset_all_lifespans_closed()
    model.open_rows([0], 1.0)
    model.close_rows([0], 2.0)
    model.open_rows([0], 4.0)
    state = migrate_temporal_state_dict(model.state_dict())
    clone = TemporalChangeModel.from_gaussians(_base(), max_states=3)
    clone.load_state_dict(state, strict=True)
    assert clone.validate_lifecycle()
    assert clone.state_status.tolist() == [[2, 1, 0]]
    assert clone.num_states.tolist() == [2]
    assert clone.current_state_index.tolist() == [1]
    assert clone.state_start[0, :2].tolist() == [1.0, 4.0]
    assert clone.state_end[0, 0].item() == 2.0
    assert torch.isinf(clone.state_end[0, 1])


def test_legacy_gap_remains_empty_and_is_the_next_reopen_slot():
    legacy = {
        "state_change_dc": torch.zeros(1, 3, 1, 3),
        "state_start": torch.tensor([[0.0, 0.0, 2.0]]),
        "state_end": torch.tensor([[1.0, float("inf"), 3.0]]),
        "state_valid": torch.tensor([[True, False, True]]),
    }
    migrated = migrate_temporal_state_dict(legacy)
    assert migrated["state_status"].tolist() == [[2, 0, 2]]
    assert migrated["num_states"].tolist() == [2]

    model = TemporalChangeModel.from_gaussians(_base(), max_states=3)
    model.load_state_dict(migrated, strict=True)
    assert model.validate_lifecycle()
    slot = model.open_rows([0], timestamp=4.0)
    assert slot.tolist() == [1]
    assert model.num_states.tolist() == [3]
    assert model.current_state_index.tolist() == [1]
