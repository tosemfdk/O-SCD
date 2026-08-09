"""Fixed-capacity current-state and archive utilities for temporal MCMC change fields.

The current state is a fixed-size set of Gaussian slots for the *currently valid*
change hypothesis. Cue support is metadata only: unsupported slots are still
rendered with their current opacity so MCMC proposals can be evaluated, but the
metadata is available to energy terms and runners.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn

_CUE_SUPPORT_KEY = "cue_support_mask"
_LEGACY_SUPPORT_KEY = "support_mask"
_BASE_ABSENT_OPACITY = 0.001
_CUE_INITIAL_OPACITY = 0.1
_TRAINABLE_PARAM_NAMES = (
    "xyz",
    "features_dc",
    "raw_change_opacity",
    "scaling",
    "rotation",
)
_SNAPSHOT_PARAM_NAMES = _TRAINABLE_PARAM_NAMES
_EVIDENCE_BUFFER_NAMES = (
    "slot_observation_count",
    "slot_cue_support_count",
    "carryover_protected_mask",
    "removal_protected_mask",
    "tentative_mask",
    "slot_relocation_count",
    "last_relocation_step",
)


@dataclass(frozen=True)
class ArchiveRecord:
    """Immutable metadata for one closed current-state snapshot."""

    state_id: int
    start_time: float
    end_time: float
    checksum: str
    metadata: Mapping[str, Any]


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_time(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _logit_probability(probability: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be inside (0, 1)")
    p = torch.as_tensor(probability, device=device, dtype=dtype)
    return torch.log(p / (1.0 - p))


def _tensor_hash(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(value.shape)).encode("utf8"))
    digest.update(str(value.dtype).encode("utf8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_hashes(tensors: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {key: _tensor_hash(value) for key, value in sorted(tensors.items())}




def _clone_archive_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_archive_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_archive_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_archive_value(item) for item in value)
    return copy.deepcopy(value)


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor")
            digest.update(_tensor_hash(value).encode("utf8"))
        elif isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value):
                digest.update(str(key).encode("utf8"))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence")
            for item in value:
                update(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf8"))

    update(payload)
    return digest.hexdigest()


class FixedCapacityChangeState(nn.Module):
    """Trainable fixed-capacity Gaussian state for current-scene change.

    Raw change opacity is separate from cue support. ``cue_support_mask`` records
    which slots have direct current cue evidence, but render attributes always
    expose all slots with ``opacity = sigmoid(current_raw_change_opacity)``.
    Higher-order SH/rest features are fixed to the base model and are not part of
    ``current_parameter_items``.
    """

    absent_opacity: float = _BASE_ABSENT_OPACITY
    cue_initial_opacity: float = _CUE_INITIAL_OPACITY

    def __init__(
        self,
        base_change_gaussians,
        capacity: int | None = None,
        *,
        base_zero: bool = True,
    ):
        super().__init__()
        self._validate_base(base_change_gaussians)
        object.__setattr__(self, "base", base_change_gaussians)

        inferred_capacity = base_change_gaussians._xyz.shape[0] if capacity is None else capacity
        self.capacity = _validate_positive_int("capacity", inferred_capacity)
        self._base_zero = bool(base_zero)
        self._freeze_base_tensors(base_change_gaussians)

        device = base_change_gaussians._xyz.device
        self.register_buffer("slot_ids", torch.arange(self.capacity, device=device, dtype=torch.long))
        self.register_buffer(_CUE_SUPPORT_KEY, torch.zeros(self.capacity, device=device, dtype=torch.bool))
        self.register_buffer(
            "slot_observation_count",
            torch.zeros(self.capacity, device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "slot_cue_support_count",
            torch.zeros(self.capacity, device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "carryover_protected_mask",
            torch.zeros(self.capacity, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "removal_protected_mask",
            torch.zeros(self.capacity, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "tentative_mask",
            torch.zeros(self.capacity, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "slot_relocation_count",
            torch.zeros(self.capacity, device=device, dtype=torch.int32),
        )
        self.register_buffer(
            "last_relocation_step",
            torch.full((self.capacity,), -1, device=device, dtype=torch.int64),
        )

        params = self._initial_parameters(base_change_gaussians, self.capacity, base_zero=base_zero)
        self.current_xyz = nn.Parameter(params["xyz"])
        self.current_features_dc = nn.Parameter(params["features_dc"])
        self.current_raw_change_opacity = nn.Parameter(params["raw_change_opacity"])
        self.current_scaling = nn.Parameter(params["scaling"])
        self.current_rotation = nn.Parameter(params["rotation"])

    @classmethod
    def from_gaussians(
        cls,
        base_change_gaussians,
        capacity: int | None = None,
        *,
        base_zero: bool = True,
    ) -> "FixedCapacityChangeState":
        return cls(base_change_gaussians, capacity=capacity, base_zero=base_zero)

    @property
    def support_mask(self) -> torch.Tensor:
        """Backward-compatible alias; cue support is metadata, not render gating."""
        return self.cue_support_mask

    @staticmethod
    def _validate_base(base_change_gaussians) -> None:
        required = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
        for name in required:
            if not hasattr(base_change_gaussians, name):
                raise AttributeError(f"base_change_gaussians must expose {name}")
            value = getattr(base_change_gaussians, name)
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                raise TypeError(f"base {name} must be a floating tensor")
        if base_change_gaussians._xyz.ndim != 2 or base_change_gaussians._xyz.shape[1] != 3:
            raise ValueError("base _xyz must have shape [N, 3]")
        if base_change_gaussians._features_dc.ndim != 3 or base_change_gaussians._features_dc.shape[1:] != (1, 3):
            raise ValueError("base _features_dc must have shape [N, 1, 3]")
        if base_change_gaussians._opacity.ndim != 2 or base_change_gaussians._opacity.shape[1] != 1:
            raise ValueError("base _opacity must have shape [N, 1]")
        if base_change_gaussians._scaling.ndim != 2 or base_change_gaussians._scaling.shape[1] != 3:
            raise ValueError("base _scaling must have shape [N, 3]")
        if base_change_gaussians._rotation.ndim != 2 or base_change_gaussians._rotation.shape[1] != 4:
            raise ValueError("base _rotation must have shape [N, 4]")

    @staticmethod
    def _freeze_base_tensors(base_change_gaussians) -> None:
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            value = getattr(base_change_gaussians, name, None)
            if isinstance(value, torch.Tensor):
                value.requires_grad_(False)

    @staticmethod
    def _take_or_pad(source: torch.Tensor, capacity: int, *, fill: float = 0.0) -> torch.Tensor:
        out = torch.full((capacity, *source.shape[1:]), fill, device=source.device, dtype=source.dtype)
        n = min(capacity, source.shape[0])
        if n:
            out[:n].copy_(source.detach()[:n])
        return out

    @classmethod
    def _initial_parameters(cls, base, capacity: int, *, base_zero: bool) -> dict[str, torch.Tensor]:
        device = base._xyz.device
        dtype = base._xyz.dtype
        absent_logit = _logit_probability(cls.absent_opacity, device=device, dtype=dtype)

        xyz = cls._take_or_pad(base._xyz, capacity)
        scaling = cls._take_or_pad(base._scaling, capacity)
        rotation = cls._take_or_pad(base._rotation, capacity)
        missing = capacity - min(capacity, base._rotation.shape[0])
        if missing > 0:
            rotation[-missing:] = 0.0
            rotation[-missing:, 0] = 1.0

        if base_zero:
            features_dc = torch.zeros((capacity, 1, 3), device=device, dtype=dtype)
            raw_opacity = absent_logit.expand(capacity, 1).clone()
        else:
            features_dc = cls._take_or_pad(base._features_dc, capacity)
            raw_opacity = absent_logit.expand(capacity, 1).clone()
        return {
            "xyz": xyz,
            "features_dc": features_dc,
            "raw_change_opacity": raw_opacity,
            "scaling": scaling,
            "rotation": rotation,
        }

    @staticmethod
    def _slot_ids_from_input(slot_ids, *, device: torch.device, capacity: int) -> torch.Tensor:
        if isinstance(slot_ids, torch.Tensor) and slot_ids.dtype == torch.bool:
            if slot_ids.shape != (capacity,):
                raise ValueError(f"boolean slot mask must have shape {(capacity,)}")
            return slot_ids.to(device=device).nonzero(as_tuple=False).flatten()
        ids = torch.as_tensor(slot_ids, device=device, dtype=torch.long)
        if ids.numel() == 0:
            return ids.flatten()
        if torch.any((ids < 0) | (ids >= capacity)):
            raise IndexError("slot_ids contain an out-of-range slot")
        return ids.flatten()

    def reset_current(
        self,
        *,
        warm_start: Mapping[str, torch.Tensor] | None = None,
        base_zero: bool | None = None,
        preserve_cue_support: bool = False,
    ) -> None:
        """Reset current parameters either from a snapshot or from base-zero init.

        Warm-start copies only Gaussian parameters by default. Cue support is
        current-state evidence metadata and resets unless explicitly preserved
        for checkpoint restore-style callers.
        """
        if base_zero is None:
            base_zero = self._base_zero
        params = (
            self._coerce_snapshot(warm_start, device=self.current_xyz.device, dtype=self.current_xyz.dtype)
            if warm_start is not None
            else self._initial_parameters(self.base, self.capacity, base_zero=base_zero)
        )
        with torch.no_grad():
            self.current_xyz.copy_(params["xyz"])
            self.current_features_dc.copy_(params["features_dc"])
            self.current_raw_change_opacity.copy_(params["raw_change_opacity"])
            self.current_scaling.copy_(params["scaling"])
            self.current_rotation.copy_(params["rotation"])
            support_key = _CUE_SUPPORT_KEY if _CUE_SUPPORT_KEY in (warm_start or {}) else _LEGACY_SUPPORT_KEY
            if preserve_cue_support and warm_start is not None and support_key in warm_start:
                support = warm_start[support_key].to(device=self.cue_support_mask.device, dtype=torch.bool)
                if support.shape != self.cue_support_mask.shape:
                    raise ValueError(
                        f"warm_start {support_key} has shape {tuple(support.shape)}, "
                        f"expected {tuple(self.cue_support_mask.shape)}"
                    )
                self.cue_support_mask.copy_(support)
            else:
                self.cue_support_mask.zero_()
            previous_support = None
            if warm_start is not None and support_key in warm_start:
                previous_support = warm_start[support_key].to(
                    device=self.carryover_protected_mask.device,
                    dtype=torch.bool,
                )
            previous_protection = None
            if warm_start is not None and "carryover_protected_mask" in warm_start:
                previous_protection = warm_start["carryover_protected_mask"].to(
                    device=self.carryover_protected_mask.device,
                    dtype=torch.bool,
                )
            self.carryover_protected_mask.zero_()
            if previous_support is not None:
                self.carryover_protected_mask |= previous_support
            if previous_protection is not None:
                self.carryover_protected_mask |= previous_protection
            self.slot_observation_count.zero_()
            self.slot_cue_support_count.zero_()
            self.removal_protected_mask.zero_()
            self.tentative_mask.zero_()
            self.slot_relocation_count.zero_()
            self.last_relocation_step.fill_(-1)

    def record_slot_evidence(
        self,
        visible_mask: torch.Tensor,
        cue_supported_mask: torch.Tensor,
    ) -> None:
        """Accumulate causal per-view observation/support evidence for source selection."""

        visible = visible_mask.to(device=self.slot_observation_count.device, dtype=torch.bool)
        supported = cue_supported_mask.to(device=self.slot_cue_support_count.device, dtype=torch.bool)
        expected = (self.capacity,)
        if visible.shape != expected or supported.shape != expected:
            raise ValueError(f"slot evidence masks must have shape {expected}")
        if bool((supported & ~visible).any()):
            # The rasterizer's metric count should be a subset of its visibility
            # result. Fail closed instead of recording inconsistent evidence.
            raise ValueError("cue-supported slots must also be visible")
        with torch.no_grad():
            self.slot_observation_count.add_(visible.to(torch.int32))
            self.slot_cue_support_count.add_(supported.to(torch.int32))
            self.cue_support_mask |= supported
            # A second causal view that directly supports a proposal promotes it
            # out of the tentative lifecycle without consulting opacity.
            confirmed = self.tentative_mask & (self.slot_cue_support_count >= 2)
            self.tentative_mask[confirmed] = False

    def promote_cue_supported_slots(
        self,
        slot_ids: torch.Tensor | list[int] | tuple[int, ...],
        *,
        opacity: float = _CUE_INITIAL_OPACITY,
    ) -> None:
        """Mark cue-supported slots and initialize their change opacity to 0.1 by default."""
        ids = self._slot_ids_from_input(slot_ids, device=self.cue_support_mask.device, capacity=self.capacity)
        if ids.numel() == 0:
            return
        raw = _logit_probability(opacity, device=self.current_raw_change_opacity.device, dtype=self.current_raw_change_opacity.dtype)
        with torch.no_grad():
            self.cue_support_mask[ids] = True
            self.current_raw_change_opacity[ids] = raw

    def activate_slots(self, slot_ids: torch.Tensor | list[int] | tuple[int, ...]) -> None:
        """Backward-compatible alias for ``promote_cue_supported_slots``."""
        self.promote_cue_supported_slots(slot_ids)

    def deactivate_slots(self, slot_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        with torch.no_grad():
            if slot_ids is None:
                self.cue_support_mask.zero_()
                return
            ids = self._slot_ids_from_input(slot_ids, device=self.cue_support_mask.device, capacity=self.capacity)
            self.cue_support_mask[ids] = False

    def current_parameter_items(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return trainable current-state params; fixed rest features are excluded."""
        return (
            ("xyz", self.current_xyz),
            ("features_dc", self.current_features_dc),
            ("raw_change_opacity", self.current_raw_change_opacity),
            ("scaling", self.current_scaling),
            ("rotation", self.current_rotation),
        )

    def snapshot(self, *, detach: bool = True, cpu: bool = True) -> dict[str, torch.Tensor]:
        tensors = {
            "slot_ids": self.slot_ids,
            _CUE_SUPPORT_KEY: self.cue_support_mask,
            _LEGACY_SUPPORT_KEY: self.cue_support_mask,
            "xyz": self.current_xyz,
            "features_dc": self.current_features_dc,
            "raw_change_opacity": self.current_raw_change_opacity,
            "scaling": self.current_scaling,
            "rotation": self.current_rotation,
            **{name: getattr(self, name) for name in _EVIDENCE_BUFFER_NAMES},
        }
        out: dict[str, torch.Tensor] = {}
        for key, value in tensors.items():
            tensor = value.detach().clone() if detach else value.clone()
            out[key] = tensor.cpu() if cpu else tensor
        return out

    def _coerce_snapshot(
        self,
        snapshot: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        out = {}
        expected = self.snapshot(detach=True, cpu=False)
        for name in _SNAPSHOT_PARAM_NAMES:
            if name not in snapshot:
                raise KeyError(f"warm_start snapshot missing {name!r}")
            value = snapshot[name].to(device=device, dtype=dtype)
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"warm_start {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected[name].shape)}"
                )
            out[name] = value
        return out

    def get_active_render_attributes(self, timestamp: float | None = None) -> dict[str, torch.Tensor]:
        """Return renderer override tensors for all current-state slots.

        ``active`` is all-ones because current-state MCMC does not lifespan-gate
        slots at render time. ``cue_support_mask`` is returned only as metadata.
        """
        del timestamp
        active = torch.ones_like(self.cue_support_mask, dtype=torch.bool)
        return {
            "dc": self.current_features_dc,
            "xyz": self.current_xyz,
            "opacity": self.base.opacity_activation(self.current_raw_change_opacity),
            "raw_change_opacity": self.current_raw_change_opacity,
            "scaling": self.base.scaling_activation(self.current_scaling),
            "rotation": self.base.rotation_activation(self.current_rotation),
            # Change rendering is SH degree 0. Higher-order coefficients remain
            # immutable on the reference Gaussian model and are deliberately
            # not duplicated in every current-state snapshot/archive.
            "features_rest": self.base._features_rest,
            "active": active,
            _CUE_SUPPORT_KEY: self.cue_support_mask,
            _LEGACY_SUPPORT_KEY: self.cue_support_mask,
            "slot_ids": self.slot_ids,
        }


class StateArchive:
    """Append-only archive for closed fixed-capacity change states."""

    format_version = 1

    def __init__(self, records: list[dict[str, Any]] | None = None, metadata: Mapping[str, Any] | None = None):
        self._records = _clone_archive_value(list(records or []))
        self.metadata = _clone_archive_value(dict(metadata or {}))

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[ArchiveRecord, ...]:
        return tuple(
            ArchiveRecord(
                state_id=int(record["state_id"]),
                start_time=float(record["start_time"]),
                end_time=float(record["end_time"]),
                checksum=str(record["checksum"]),
                metadata=_clone_archive_value(record.get("metadata", {})),
            )
            for record in self._records
        )

    def tensors(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._records[index]["tensors"].items()}

    def append(
        self,
        *,
        state_id: int,
        start_time: float,
        end_time: float,
        tensors: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] | None = None,
    ) -> ArchiveRecord:
        state_id = _validate_nonnegative_int("state_id", state_id)
        start_time = _validate_time("start_time", start_time)
        end_time = _validate_time("end_time", end_time)
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        tensor_copy = {key: value.detach().cpu().clone() for key, value in tensors.items()}
        tensor_hashes = _tensor_hashes(tensor_copy)
        immutable_metadata = {"tensor_hashes": tensor_hashes, **_clone_archive_value(dict(metadata or {}))}
        payload = {
            "state_id": state_id,
            "start_time": start_time,
            "end_time": end_time,
            "metadata": immutable_metadata,
            "tensors": tensor_copy,
        }
        checksum = _payload_checksum(payload)
        record = {**payload, "checksum": checksum}
        self._records.append(record)
        return ArchiveRecord(
            state_id,
            start_time,
            end_time,
            checksum,
            _clone_archive_value(immutable_metadata),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "metadata": _clone_archive_value(self.metadata),
            "records": _clone_archive_value(self._records),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "StateArchive":
        if int(state.get("format_version", -1)) != cls.format_version:
            raise ValueError("unsupported StateArchive format_version")
        archive = cls(records=_clone_archive_value(list(state.get("records", []))), metadata=_clone_archive_value(state.get("metadata", {})))
        archive.verify_checksums()
        return archive

    def verify_checksums(self) -> None:
        for record in self._records:
            tensors = record["tensors"]
            metadata = dict(record.get("metadata", {}))
            expected_hashes = metadata.get("tensor_hashes")
            actual_hashes = _tensor_hashes(tensors)
            if expected_hashes is not None and dict(expected_hashes) != actual_hashes:
                raise ValueError(f"archive tensor_hashes mismatch for state {record.get('state_id')}")
            payload = {
                "state_id": int(record["state_id"]),
                "start_time": float(record["start_time"]),
                "end_time": float(record["end_time"]),
                "metadata": metadata,
                "tensors": tensors,
            }
            if _payload_checksum(payload) != record.get("checksum"):
                raise ValueError(f"archive checksum mismatch for state {record.get('state_id')}")

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save the archive as a PyTorch checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.verify_checksums()
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        try:
            torch.save(self.state_dict(), tmp_name)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, map_location: str | torch.device = "cpu") -> "StateArchive":
        state = torch.load(path, map_location=map_location)
        return cls.from_state_dict(state)


class OracleBoundaryStateManager:
    """Lifecycle helper that closes/starts current states at oracle boundaries."""

    def __init__(
        self,
        current_state: FixedCapacityChangeState,
        boundaries: list[float] | tuple[float, ...] = (),
        *,
        archive: StateArchive | None = None,
        warm_start: bool = True,
        base_zero: bool = True,
        initial_time: float = 0.0,
    ):
        self.current_state = current_state
        boundaries = tuple(sorted(_validate_time("boundary", b) for b in boundaries))
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("boundaries must be unique")
        self.boundaries = boundaries
        self.archive = archive if archive is not None else StateArchive()
        self.warm_start = bool(warm_start)
        self.base_zero = bool(base_zero)
        self.initial_time = _validate_time("initial_time", initial_time)
        self.current_state_id = 0
        self.current_start_time: float | None = None
        self._next_boundary_index = 0
        self._last_timestamp: float | None = None
        self._closed = False

    def segment_id_for(self, timestamp: float) -> int:
        timestamp = _validate_time("timestamp", timestamp)
        return sum(timestamp >= boundary for boundary in self.boundaries)

    def _reset_for_new_state(self, warm_snapshot: Mapping[str, torch.Tensor] | None) -> None:
        self.current_state.reset_current(
            warm_start=warm_snapshot if self.warm_start else None,
            base_zero=self.base_zero,
            preserve_cue_support=False,
        )

    def ensure_state(self, timestamp: float, *, metadata: Mapping[str, Any] | None = None) -> int:
        """Archive/reset at each configured boundary crossed by ``timestamp``.

        If frames skip over one or more oracle boundaries, each skipped segment is
        explicitly archived with its configured half-open interval rather than
        being closed at the later observed frame timestamp.
        """
        if self._closed:
            raise RuntimeError("cannot ensure_state after close")
        timestamp = _validate_time("timestamp", timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("OracleBoundaryStateManager only supports non-decreasing timestamps")
        self._last_timestamp = timestamp

        if self.current_start_time is None:
            self.current_state_id = 0
            self.current_start_time = self.initial_time
            self.current_state.reset_current(base_zero=self.base_zero)

        while self._next_boundary_index < len(self.boundaries) and timestamp >= self.boundaries[self._next_boundary_index]:
            boundary = self.boundaries[self._next_boundary_index]
            warm_snapshot = self.current_state.snapshot(detach=True, cpu=False) if self.warm_start else None
            self.archive.append(
                state_id=self.current_state_id,
                start_time=self.current_start_time,
                end_time=boundary,
                tensors=self.current_state.snapshot(detach=True, cpu=True),
                metadata={
                    "closed_by": "oracle_boundary",
                    "boundary": boundary,
                    "skipped_by_timestamp": timestamp if timestamp > boundary else None,
                    **dict(metadata or {}),
                },
            )
            self.current_state_id += 1
            self.current_start_time = boundary
            self._next_boundary_index += 1
            self._reset_for_new_state(warm_snapshot)

        return self.current_state_id

    def close(self, end_time: float, *, metadata: Mapping[str, Any] | None = None) -> ArchiveRecord:
        if self._closed:
            raise RuntimeError("current state has already been closed")
        if self.current_start_time is None:
            self.current_state_id = 0
            self.current_start_time = self.initial_time
            self.current_state.reset_current(base_zero=self.base_zero)
        end_time = _validate_time("end_time", end_time)
        if end_time <= self.current_start_time:
            raise ValueError("end_time must be greater than current_start_time")
        record = self.archive.append(
            state_id=self.current_state_id,
            start_time=self.current_start_time,
            end_time=end_time,
            tensors=self.current_state.snapshot(detach=True, cpu=True),
            metadata={"closed_by": "close", **dict(metadata or {})},
        )
        self._closed = True
        self.current_start_time = None
        return record
