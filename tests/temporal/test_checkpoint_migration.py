import torch

from temporal.checkpoint import migrate_temporal_state_dict


def test_legacy_shared_geometry_checkpoint_gets_unfrozen_default():
    legacy = {
        "state_change_dc": torch.zeros(3, 2, 1, 3),
        "shared_xyz_delta": torch.zeros(3, 3),
    }

    migrated = migrate_temporal_state_dict(legacy)

    assert "geometry_frozen" not in legacy
    assert torch.equal(
        migrated["geometry_frozen"], torch.zeros(3, dtype=torch.bool)
    )


def test_current_shared_geometry_checkpoint_preserves_frozen_metadata():
    frozen = torch.tensor([True, False, True])
    current = {
        "state_change_dc": torch.zeros(3, 2, 1, 3),
        "shared_xyz_delta": torch.zeros(3, 3),
        "geometry_frozen": frozen,
    }

    migrated = migrate_temporal_state_dict(current)

    assert migrated["geometry_frozen"] is frozen
