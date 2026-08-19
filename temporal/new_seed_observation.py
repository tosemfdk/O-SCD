"""Observation buffering and signed-mask XFeat subsetting for NEW seed birth.

This module intentionally contains only the causal observation contract used by
E4 NEW-Gaussian seeding.  It does not own matching, triangulation, promotion, or
renderer integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
from torch import Tensor

SignedMask = Literal["+", "-"]


_MASK_SHAPE = (64, 64)


@dataclass(frozen=True)
class SignedXFeatObservation:
    """CPU-detached pose-time XFeat observation plus 64x64 signed cues.

    Coordinate contract:
        * ``keypoints`` are detector image pixels in ``[x, y]`` order.
        * ``image_size`` is ``(height, width)`` for the same detector image.
        * signed masks are 64x64 tensors indexed as ``[row=y_cell, col=x_cell]``.
        * ``w2c`` is an explicit OpenCV world-to-camera transform.

    The dataclass is frozen and clones/detaches tensors to CPU during
    construction so the bounded buffer cannot retain autograd graphs or GPU
    storage from the online pose/matching pipeline.
    """

    frame_index: int
    timestamp: float
    frame_name: str
    w2c: Tensor
    K: Tensor
    image_size: tuple[int, int]
    keypoints: Tensor
    descriptors: Tensor
    valid: Tensor
    plus_mask64: Tensor
    minus_mask64: Tensor
    cue_strength64: Tensor
    pca_margin64: Tensor

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be numeric")
        timestamp = float(self.timestamp)
        if not torch.isfinite(torch.tensor(timestamp)):
            raise ValueError("timestamp must be finite")
        if not isinstance(self.frame_name, str):
            raise TypeError("frame_name must be a string")
        image_size = _normalize_image_size(self.image_size)

        keypoints = _cpu_clone(self.keypoints, torch.float32, "keypoints")
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError("keypoints must have shape [N,2] in [x,y] order")
        n_keypoints = keypoints.shape[0]

        descriptors = _cpu_clone(self.descriptors, torch.float16, "descriptors")
        if descriptors.ndim != 2:
            raise ValueError("descriptors must have shape [N,D]")
        if descriptors.shape[0] != n_keypoints:
            raise ValueError("descriptors and keypoints must have the same N")

        valid = _cpu_clone(self.valid, torch.bool, "valid")
        if valid.ndim != 1 or valid.shape[0] != n_keypoints:
            raise ValueError("valid must have shape [N]")

        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "w2c", _matrix_cpu(self.w2c, (4, 4), "w2c"))
        object.__setattr__(self, "K", _matrix_cpu(self.K, (3, 3), "K"))
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "plus_mask64", _mask_cpu(self.plus_mask64, torch.bool, "plus_mask64"))
        object.__setattr__(self, "minus_mask64", _mask_cpu(self.minus_mask64, torch.bool, "minus_mask64"))
        object.__setattr__(self, "cue_strength64", _mask_cpu(self.cue_strength64, torch.float32, "cue_strength64"))
        object.__setattr__(self, "pca_margin64", _mask_cpu(self.pca_margin64, torch.float32, "pca_margin64"))


@dataclass(frozen=True)
class MaskedXFeatSubset:
    """A sign-mask-filtered view of one observation's keypoints/descriptors."""

    frame_index: int
    sign: SignedMask
    keypoint_indices: Tensor  # CPU int64 [M], absolute indices into observation.keypoints
    keypoints: Tensor  # CPU float32 [M,2]
    descriptors: Tensor  # CPU float16 [M,D]

    @property
    def count(self) -> int:
        return int(self.keypoint_indices.numel())


def keypoints_to_mask_indices(
    keypoints: Tensor,
    image_size: tuple[int, int],
    mask_shape: tuple[int, int] = _MASK_SHAPE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map image ``[x,y]`` keypoints to integer mask ``(row, col)`` cells.

    Returns ``(rows, cols, inside)``.  Points are inside only for
    ``0 <= x < width`` and ``0 <= y < height``; coordinates on the far image
    edge are outside instead of being silently clamped.
    """

    image_h, image_w = _normalize_image_size(image_size)
    mask_h, mask_w = _normalize_mask_shape(mask_shape)
    kpts = _cpu_clone(keypoints, torch.float32, "keypoints")
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError("keypoints must have shape [N,2] in [x,y] order")

    x = kpts[:, 0]
    y = kpts[:, 1]
    inside = (x >= 0.0) & (x < float(image_w)) & (y >= 0.0) & (y < float(image_h))

    cols = torch.floor(x * float(mask_w) / float(image_w)).to(torch.long)
    rows = torch.floor(y * float(mask_h) / float(image_h)).to(torch.long)
    # Keep rows/cols index-safe for downstream masked indexing while preserving
    # the separate inside flag for correctness at/outside boundaries.
    cols = cols.clamp(0, mask_w - 1)
    rows = rows.clamp(0, mask_h - 1)
    return rows.cpu(), cols.cpu(), inside.cpu()


def inside_eroded_mask(
    keypoints: Tensor,
    image_size: tuple[int, int],
    mask64: Tensor,
    erosion_cells: int = 1,
) -> Tensor:
    """Return keypoints whose mapped mask cell has a full eroded neighborhood.

    With the primary E4 setting ``erosion_cells=1``, a keypoint passes only when
    all cells in its 3x3 neighborhood are true.  Boundary cells cannot satisfy a
    positive erosion radius because their neighborhood would leave the mask.
    """

    if erosion_cells < 0:
        raise ValueError("erosion_cells must be non-negative")
    mask = _mask_cpu(mask64, torch.bool, "mask64")
    rows, cols, inside = keypoints_to_mask_indices(keypoints, image_size, tuple(mask.shape))
    if rows.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)

    mask_h, mask_w = mask.shape
    selected = torch.zeros_like(inside, dtype=torch.bool)
    if erosion_cells == 0:
        selected[inside] = mask[rows[inside], cols[inside]]
        return selected

    interior = (
        inside
        & (rows >= erosion_cells)
        & (rows < mask_h - erosion_cells)
        & (cols >= erosion_cells)
        & (cols < mask_w - erosion_cells)
    )
    for idx in torch.nonzero(interior, as_tuple=False).flatten().tolist():
        r = int(rows[idx])
        c = int(cols[idx])
        window = mask[
            r - erosion_cells : r + erosion_cells + 1,
            c - erosion_cells : c + erosion_cells + 1,
        ]
        selected[idx] = bool(torch.all(window))
    return selected


def signed_mask_selection(
    observation: SignedXFeatObservation,
    sign: SignedMask,
    erosion_cells: int = 1,
) -> Tensor:
    """Select valid keypoints inside the eroded signed NEW mask."""

    mask = _mask_for_sign(observation, sign)
    return observation.valid & inside_eroded_mask(
        observation.keypoints,
        observation.image_size,
        mask,
        erosion_cells=erosion_cells,
    )


def masked_xfeat_subset(
    observation: SignedXFeatObservation,
    sign: SignedMask,
    erosion_cells: int = 1,
    min_keypoints: int = 0,
) -> MaskedXFeatSubset:
    """Build an absolute-index-preserving masked XFeat subset for matching."""

    if min_keypoints < 0:
        raise ValueError("min_keypoints must be non-negative")
    selection = signed_mask_selection(observation, sign, erosion_cells=erosion_cells)
    indices = torch.nonzero(selection, as_tuple=False).flatten().to(torch.long).cpu()
    if indices.numel() < min_keypoints:
        indices = torch.empty((0,), dtype=torch.long)
    return MaskedXFeatSubset(
        frame_index=observation.frame_index,
        sign=sign,
        keypoint_indices=indices,
        keypoints=observation.keypoints[indices].clone(),
        descriptors=observation.descriptors[indices].clone(),
    )


class XFeatObservationBuffer:
    """Bounded chronological CPU observation buffer.

    The buffer is deterministic: adding an observation with an existing
    ``frame_index`` replaces that frame; storage is sorted by ``frame_index``;
    overflow evicts the oldest frame(s).  ``backfill`` returns a chronological
    tuple over already-observed frames only.
    """

    def __init__(self, max_frames: int = 32) -> None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            raise TypeError("max_frames must be an integer")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = max_frames
        self._observations: dict[int, SignedXFeatObservation] = {}

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self) -> Iterable[SignedXFeatObservation]:
        return iter(self.frames())

    def add(self, observation: SignedXFeatObservation) -> tuple[SignedXFeatObservation, ...]:
        """Insert/replace one observation and return evicted frames oldest-first."""

        if not isinstance(observation, SignedXFeatObservation):
            raise TypeError("observation must be a SignedXFeatObservation")
        self._observations[observation.frame_index] = observation
        evicted: list[SignedXFeatObservation] = []
        while len(self._observations) > self.max_frames:
            oldest = min(self._observations)
            evicted.append(self._observations.pop(oldest))
        return tuple(evicted)

    def frames(self) -> tuple[SignedXFeatObservation, ...]:
        """Return all buffered observations in chronological order."""

        return tuple(self._observations[i] for i in sorted(self._observations))

    def backfill(
        self,
        up_to_frame_index: int | None = None,
        max_frames: int | None = None,
    ) -> tuple[SignedXFeatObservation, ...]:
        """Return chronological bounded history for causal gate opening.

        ``up_to_frame_index`` filters out future frames if callers are replaying
        a prefix.  ``max_frames`` trims to the most recent selected frames while
        preserving chronological order.
        """

        frames = self.frames()
        if up_to_frame_index is not None:
            if isinstance(up_to_frame_index, bool) or not isinstance(up_to_frame_index, int):
                raise TypeError("up_to_frame_index must be an integer or None")
            frames = tuple(obs for obs in frames if obs.frame_index <= up_to_frame_index)
        if max_frames is not None:
            if isinstance(max_frames, bool) or not isinstance(max_frames, int):
                raise TypeError("max_frames must be an integer or None")
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
            frames = frames[-max_frames:]
        return frames

    def latest(self, count: int) -> tuple[SignedXFeatObservation, ...]:
        """Return the latest ``count`` buffered frames in chronological order."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return tuple()
        return self.frames()[-count:]

    def clear(self) -> None:
        self._observations.clear()


def _mask_for_sign(observation: SignedXFeatObservation, sign: SignedMask) -> Tensor:
    if sign == "+":
        return observation.plus_mask64
    if sign == "-":
        return observation.minus_mask64
    raise ValueError("sign must be '+' or '-'")


def _cpu_clone(tensor: Tensor, dtype: torch.dtype, name: str) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor.detach().to(device="cpu", dtype=dtype).clone()


def _matrix_cpu(tensor: Tensor, shape: Sequence[int], name: str) -> Tensor:
    out = _cpu_clone(tensor, torch.float32, name)
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    return out


def _mask_cpu(tensor: Tensor, dtype: torch.dtype, name: str) -> Tensor:
    out = _cpu_clone(tensor, dtype, name)
    if tuple(out.shape) != _MASK_SHAPE:
        raise ValueError(f"{name} must have shape {_MASK_SHAPE}")
    return out


def _normalize_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise TypeError("image_size must be a (height, width) tuple")
    h, w = image_size
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (h, w)):
        raise TypeError("image_size values must be integers")
    if h <= 0 or w <= 0:
        raise ValueError("image_size values must be positive")
    return h, w


def _normalize_mask_shape(mask_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(mask_shape, tuple) or len(mask_shape) != 2:
        raise TypeError("mask_shape must be a (height, width) tuple")
    h, w = mask_shape
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (h, w)):
        raise TypeError("mask_shape values must be integers")
    if h <= 0 or w <= 0:
        raise ValueError("mask_shape values must be positive")
    return h, w
