import copy

import pytest
import torch

from temporal.mcmc_state import FixedCapacityChangeState, OracleBoundaryStateManager, StateArchive
from .conftest import make_base


def test_archive_records_are_immutable_copies_with_checksum_verification(tmp_path):
    archive = StateArchive(metadata={"run": "unit"})
    tensors = {"xyz": torch.ones(2, 3), "raw_change_opacity": torch.zeros(2, 1)}

    record = archive.append(state_id=0, start_time=0.0, end_time=5.0, tensors=tensors, metadata={"segment": 0})
    tensors["xyz"].fill_(99.0)
    archived = archive.tensors(0)
    archived["xyz"].fill_(123.0)

    assert archive.records == (record,)
    assert torch.equal(archive.tensors(0)["xyz"], torch.ones(2, 3))
    archive.verify_checksums()

    path = tmp_path / "archive.pt"
    archive.save(path)
    loaded = StateArchive.load(path)
    assert loaded.records[0].checksum == record.checksum

    tampered_state = copy.deepcopy(archive.state_dict())
    tampered_state["records"][0]["tensors"]["xyz"][0, 0] = 42.0
    with pytest.raises(ValueError, match="(checksum|tensor_hashes) mismatch"):
        StateArchive.from_state_dict(tampered_state)


def test_oracle_boundary_manager_uses_half_open_boundaries_and_archives_closed_states():
    state = FixedCapacityChangeState.from_gaussians(make_base(n=2), capacity=2)
    manager = OracleBoundaryStateManager(state, boundaries=(5.0, 10.0), warm_start=True)

    assert manager.ensure_state(4.999) == 0
    assert manager.ensure_state(5.0) == 1
    assert manager.ensure_state(9.999) == 1
    assert manager.ensure_state(10.0) == 2

    records = manager.archive.records
    assert [(record.state_id, record.start_time, record.end_time) for record in records] == [
        (0, 0.0, 5.0),
        (1, 5.0, 10.0),
    ]
    manager.archive.verify_checksums()


def test_archive_rejects_non_half_open_or_non_finite_time_ranges():
    archive = StateArchive()
    with pytest.raises(ValueError, match="end_time must be greater"):
        archive.append(state_id=0, start_time=2.0, end_time=2.0, tensors={"x": torch.ones(1)})
    with pytest.raises(ValueError, match="finite"):
        archive.append(state_id=0, start_time=0.0, end_time=float("inf"), tensors={"x": torch.ones(1)})
