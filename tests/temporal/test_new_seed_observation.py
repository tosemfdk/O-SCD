from dataclasses import FrozenInstanceError

import pytest
import torch

from temporal.new_seed_observation import (
    SignedXFeatObservation,
    XFeatObservationBuffer,
    inside_eroded_mask,
    keypoints_to_mask_indices,
    masked_xfeat_subset,
    signed_mask_selection,
)


def _obs(
    frame_index: int,
    *,
    keypoints: torch.Tensor | None = None,
    descriptors: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    plus_mask64: torch.Tensor | None = None,
    minus_mask64: torch.Tensor | None = None,
) -> SignedXFeatObservation:
    if keypoints is None:
        keypoints = torch.tensor([[20.0, 20.0], [4.0, 4.0], [32.0, 48.0]])
    if descriptors is None:
        descriptors = torch.arange(keypoints.shape[0] * 4, dtype=torch.float32).reshape(keypoints.shape[0], 4)
    if valid is None:
        valid = torch.ones(keypoints.shape[0], dtype=torch.bool)
    if plus_mask64 is None:
        plus_mask64 = torch.zeros(64, 64, dtype=torch.bool)
        plus_mask64[19:22, 19:22] = True
        plus_mask64[31:34, 47:50] = True
    if minus_mask64 is None:
        minus_mask64 = torch.zeros(64, 64, dtype=torch.bool)
        minus_mask64[3:6, 3:6] = True
    return SignedXFeatObservation(
        frame_index=frame_index,
        timestamp=float(frame_index) + 0.25,
        frame_name=f"frame_{frame_index:03d}.png",
        w2c=torch.eye(4, device=keypoints.device),
        K=torch.eye(3, device=keypoints.device),
        image_size=(64, 64),
        keypoints=keypoints,
        descriptors=descriptors,
        valid=valid,
        plus_mask64=plus_mask64,
        minus_mask64=minus_mask64,
        cue_strength64=torch.ones(64, 64, device=keypoints.device),
        pca_margin64=torch.zeros(64, 64, device=keypoints.device),
    )


def test_observation_contract_is_frozen_cpu_detached_and_dtype_normalized():
    keypoints = torch.tensor([[20.0, 20.0]], requires_grad=True)
    descriptors = torch.ones(1, 8, dtype=torch.float32, requires_grad=True)
    obs = _obs(1, keypoints=keypoints, descriptors=descriptors)

    assert obs.keypoints.device.type == "cpu"
    assert obs.keypoints.dtype == torch.float32
    assert obs.descriptors.device.type == "cpu"
    assert obs.descriptors.dtype == torch.float16
    assert obs.valid.dtype == torch.bool
    assert obs.w2c.dtype == torch.float32
    assert not obs.keypoints.requires_grad
    assert not obs.descriptors.requires_grad

    keypoints.data.fill_(0.0)
    descriptors.data.fill_(9.0)
    assert torch.equal(obs.keypoints, torch.tensor([[20.0, 20.0]]))
    assert torch.equal(obs.descriptors, torch.ones(1, 8, dtype=torch.float16))

    with pytest.raises(FrozenInstanceError):
        obs.frame_index = 2


@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        ({"frame_index": -1}, ValueError, "frame_index"),
        ({"timestamp": float("inf")}, ValueError, "timestamp"),
        ({"image_size": (0, 64)}, ValueError, "image_size"),
        ({"keypoints": torch.zeros(3, 3)}, ValueError, "keypoints"),
        ({"descriptors": torch.zeros(2, 4)}, ValueError, "same N"),
        ({"valid": torch.ones(3, 1, dtype=torch.bool)}, ValueError, "valid"),
        ({"plus_mask64": torch.ones(32, 64, dtype=torch.bool)}, ValueError, "plus_mask64"),
    ],
)
def test_observation_rejects_invalid_contract(kwargs, exc, match):
    base = dict(
        frame_index=1,
        timestamp=1.0,
        frame_name="f.png",
        w2c=torch.eye(4),
        K=torch.eye(3),
        image_size=(64, 64),
        keypoints=torch.zeros(3, 2),
        descriptors=torch.zeros(3, 4),
        valid=torch.ones(3, dtype=torch.bool),
        plus_mask64=torch.zeros(64, 64, dtype=torch.bool),
        minus_mask64=torch.zeros(64, 64, dtype=torch.bool),
        cue_strength64=torch.zeros(64, 64),
        pca_margin64=torch.zeros(64, 64),
    )
    base.update(kwargs)
    with pytest.raises(exc, match=match):
        SignedXFeatObservation(**base)


def test_keypoints_to_mask_indices_uses_xy_and_height_width_contract():
    keypoints = torch.tensor(
        [
            [0.0, 0.0],
            [319.9, 239.9],
            [160.0, 120.0],
            [320.0, 120.0],
            [-1.0, 0.0],
        ]
    )

    rows, cols, inside = keypoints_to_mask_indices(keypoints, image_size=(240, 320))

    assert torch.equal(rows, torch.tensor([0, 63, 32, 32, 0]))
    assert torch.equal(cols, torch.tensor([0, 63, 32, 63, 0]))
    assert torch.equal(inside, torch.tensor([True, True, True, False, False]))


def test_inside_eroded_mask_requires_full_three_by_three_neighborhood():
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[9:12, 9:12] = True
    mask[20, 20] = True
    mask[0:2, 0:2] = True
    keypoints = torch.tensor([[10.0, 10.0], [20.0, 20.0], [0.0, 0.0], [10.0, 12.0]])

    selected = inside_eroded_mask(keypoints, (64, 64), mask, erosion_cells=1)

    assert torch.equal(selected, torch.tensor([True, False, False, False]))
    assert torch.equal(
        inside_eroded_mask(keypoints, (64, 64), mask, erosion_cells=0),
        torch.tensor([True, True, True, False]),
    )


def test_signed_selection_and_subset_preserve_absolute_indices():
    valid = torch.tensor([True, True, False])
    obs = _obs(7, valid=valid)

    plus_selection = signed_mask_selection(obs, "+")
    minus_selection = signed_mask_selection(obs, "-")

    assert torch.equal(plus_selection, torch.tensor([True, False, False]))
    assert torch.equal(minus_selection, torch.tensor([False, True, False]))

    subset = masked_xfeat_subset(obs, "+")
    assert subset.frame_index == 7
    assert subset.sign == "+"
    assert subset.count == 1
    assert torch.equal(subset.keypoint_indices, torch.tensor([0]))
    assert torch.equal(subset.keypoints, torch.tensor([[20.0, 20.0]]))
    assert subset.descriptors.dtype == torch.float16

    empty_for_min = masked_xfeat_subset(obs, "+", min_keypoints=2)
    assert empty_for_min.count == 0
    assert empty_for_min.keypoint_indices.dtype == torch.long

    with pytest.raises(ValueError, match="sign"):
        signed_mask_selection(obs, "new")  # type: ignore[arg-type]


def test_buffer_keeps_chronological_bounded_history_and_eviction_is_deterministic():
    buffer = XFeatObservationBuffer(max_frames=3)

    assert buffer.add(_obs(3)) == tuple()
    assert buffer.add(_obs(1)) == tuple()
    assert buffer.add(_obs(2)) == tuple()
    evicted = buffer.add(_obs(4))

    assert [obs.frame_index for obs in evicted] == [1]
    assert [obs.frame_index for obs in buffer.frames()] == [2, 3, 4]
    assert [obs.frame_index for obs in buffer.backfill()] == [2, 3, 4]
    assert [obs.frame_index for obs in buffer.backfill(up_to_frame_index=3)] == [2, 3]
    assert [obs.frame_index for obs in buffer.backfill(max_frames=2)] == [3, 4]
    assert [obs.frame_index for obs in buffer.latest(2)] == [3, 4]


def test_buffer_replaces_same_frame_without_extra_eviction():
    buffer = XFeatObservationBuffer(max_frames=2)
    buffer.add(_obs(1))
    buffer.add(_obs(2))
    replacement = _obs(2, keypoints=torch.tensor([[8.0, 8.0]]))

    evicted = buffer.add(replacement)

    assert evicted == tuple()
    frames = buffer.frames()
    assert [obs.frame_index for obs in frames] == [1, 2]
    assert torch.equal(frames[1].keypoints, torch.tensor([[8.0, 8.0]]))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_buffer_rejects_invalid_capacity(bad):
    with pytest.raises((TypeError, ValueError), match="max_frames"):
        XFeatObservationBuffer(max_frames=bad)  # type: ignore[arg-type]
