import pytest
import torch
from torch import nn

from temporal.geometry_freeze import (
    capture_frozen_rows,
    mask_frozen_row_gradients,
    max_frozen_row_drift,
    restore_frozen_rows,
)


def geometry_items():
    xyz = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    opacity = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(4, 1))
    return (("xyz", xyz), ("opacity", opacity))


def test_frozen_rows_are_projected_back_to_their_first_anchor():
    items = geometry_items()
    frozen = torch.tensor([True, False, True, False])
    anchors = capture_frozen_rows(items, frozen)

    with torch.no_grad():
        for _name, parameter in items:
            parameter.add_(10)

    restore_frozen_rows(items, frozen, anchors)

    assert max(max_frozen_row_drift(items, frozen, anchors).values()) == 0.0
    for name, parameter in items:
        assert torch.equal(parameter[frozen], anchors[name])
        assert torch.all(parameter[~frozen] >= 10)


def test_only_frozen_row_gradients_are_masked():
    items = geometry_items()
    frozen = torch.tensor([True, False, True, False])
    for _name, parameter in items:
        parameter.grad = torch.ones_like(parameter)

    pre_mask = mask_frozen_row_gradients(items, frozen)

    assert pre_mask == {"xyz": 1.0, "opacity": 1.0}
    for _name, parameter in items:
        assert torch.count_nonzero(parameter.grad[frozen]) == 0
        assert torch.all(parameter.grad[~frozen] == 1)


def test_freeze_helpers_reject_row_count_mismatch():
    items = geometry_items()
    with pytest.raises(ValueError, match="different row counts"):
        capture_frozen_rows(items, torch.ones(3, dtype=torch.bool))
