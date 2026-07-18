# Gate S1: FrameAccessGuard blocks candidate image content during selection.
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.types import FrameAccessGuard, LeakageError  # noqa: E402


def _cams(n=4):
    return [SimpleNamespace(original_image=torch.rand(3, 4, 4),
                            candidate_map=torch.rand(1, 4, 4)) for _ in range(n)]


def test_guard_blocks_and_restores():
    cams = _cams()
    keep = cams[2].original_image
    with FrameAccessGuard(cams, allowed_ids=[1]):
        _ = cams[1].original_image[0, 0, 0]  # allowed
        with pytest.raises(LeakageError):
            _ = cams[0].original_image[0, 0, 0]
        with pytest.raises(LeakageError):
            _ = cams[3].candidate_map.mean()
        with pytest.raises(LeakageError):
            cams[0].original_image[:3, ...]
    assert torch.equal(cams[2].original_image, keep)  # restored
    assert cams[0].original_image.shape == (3, 4, 4)


def test_guard_restores_on_exception():
    cams = _cams()
    with pytest.raises(RuntimeError):
        with FrameAccessGuard(cams):
            raise RuntimeError("boom")
    assert cams[0].original_image.shape == (3, 4, 4)
