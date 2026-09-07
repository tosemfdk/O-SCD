"""Compact replay-state snapshots; learned parameters and visibility are not cached."""
from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PackedLifespanState:
    """A stable-ID prefix: 0 unborn, 1 NEVER_OPEN, 2 OPEN, 3 CLOSED."""

    count: int
    encoding: str
    values: np.ndarray
    starts: np.ndarray

    @classmethod
    def encode(cls, states: np.ndarray) -> "PackedLifespanState":
        states = np.asarray(states)
        if states.ndim != 1 or states.dtype != np.uint8 or np.any(states > 3):
            raise ValueError("states must be uint8 [N] with codes in [0,3]")
        count = int(states.size)
        starts = (np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
                  .astype(np.uint32) if count else np.empty(0, np.uint32))
        run_values = states[starts].copy()
        packed_bytes = (count + 3) // 4
        if starts.nbytes + run_values.nbytes < packed_bytes:
            values, encoding = run_values, "rle"
        else:
            padded = np.zeros(packed_bytes * 4, dtype=np.uint8)
            padded[:count] = states
            groups = padded.reshape(-1, 4)
            values = (groups[:, 0] | (groups[:, 1] << 2)
                      | (groups[:, 2] << 4) | (groups[:, 3] << 6))
            starts, encoding = np.empty(0, np.uint32), "2bit"
        values.flags.writeable = False
        starts.flags.writeable = False
        return cls(count, encoding, values, starts)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.starts.nbytes)

    def decode(self) -> np.ndarray:
        if self.encoding == "rle":
            lengths = np.diff(np.r_[self.starts.astype(np.int64), self.count])
            return np.repeat(self.values, lengths)
        shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8)
        return ((self.values[:, None] >> shifts) & 3).reshape(-1)[:self.count].copy()


class LifespanStateSnapshots:
    """CPU frame snapshots plus a bounded decoded device-mask LRU.

    Invalidation is driven by successful lifecycle mutations, including density
    retirement and child birth at the current timestamp. Old frame prefixes stay
    immutable when rows are appended later. Cached masks are internal read-only
    tensors; lifecycle public getters return copies to preserve ownership.
    """

    def __init__(self, decoded_capacity: int = 16):
        if decoded_capacity < 1:
            raise ValueError("decoded_capacity must be positive")
        self.decoded_capacity = int(decoded_capacity)
        self.frames: dict[float, PackedLifespanState] = {}
        self.decoded: OrderedDict[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.latest_mutation = -math.inf
        self.hits = 0
        self.misses = 0
        self.interval_fallbacks = 0

    def invalidate(self, timestamp: int | float) -> None:
        timestamp = float(timestamp)
        self.latest_mutation = max(self.latest_mutation, timestamp)
        # Clears device masks also on append (their archive length has changed).
        self.decoded.clear()
        for key in tuple(self.frames):
            if key >= timestamp:
                del self.frames[key]

    def masks(self, life: Any, timestamp: int | float):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise TypeError("timestamp must be a finite scalar")
        if timestamp in self.decoded:
            self.hits += 1
            self.decoded.move_to_end(timestamp)
            return self.decoded[timestamp]
        self.misses += 1
        if timestamp in self.frames:
            frame = self.frames[timestamp]
            if frame.count > life.count:
                raise RuntimeError("snapshot row IDs cannot survive archive compaction")
            states = np.zeros(life.count, dtype=np.uint8)
            states[:frame.count] = frame.decode()
            codes = torch.from_numpy(states).to(device=life.current_state_index.device)
            result = (codes == 2, codes == 1, codes != 0)
        elif timestamp >= self.latest_mutation:
            materialized = life.materialized_timestamp <= timestamp
            result = ((life.current_state_index >= 0) & materialized,
                      (life.num_states == 0) & materialized, materialized)
        else:
            # Non-sealed historical/diagnostic queries retain exact interval
            # semantics; normal replay uses sealed frame snapshots instead.
            self.interval_fallbacks += 1
            materialized = life.materialized_timestamp <= timestamp
            slots = (life.state_valid & (life.state_start <= timestamp)
                     & (timestamp < life.state_end) & materialized[:, None])
            if bool((slots.sum(dim=1) > 1).any()):
                raise RuntimeError("lifespan intervals overlap at the replay timestamp")
            ever = (life.state_valid & (life.state_start <= timestamp)).any(dim=1)
            result = (slots.any(dim=1), materialized & ~ever, materialized)
        self.decoded[timestamp] = result
        while len(self.decoded) > self.decoded_capacity:
            self.decoded.popitem(last=False)
        return result

    def seal(self, life: Any, timestamp: int | float) -> None:
        active, never, materialized = self.masks(life, timestamp)
        codes = (never.to(torch.uint8) + 2 * active.to(torch.uint8)
                 + 3 * (materialized & ~(active | never)).to(torch.uint8))
        self.frames[float(timestamp)] = PackedLifespanState.encode(codes.cpu().numpy())

    def statistics(self) -> dict[str, int]:
        return dict(frames=len(self.frames),
                    payload_bytes=sum(frame.nbytes for frame in self.frames.values()),
                    decoded_bytes=sum(t.numel() * t.element_size()
                                      for masks in self.decoded.values() for t in masks),
                    rle_frames=sum(frame.encoding == "rle" for frame in self.frames.values()),
                    bit2_frames=sum(frame.encoding == "2bit" for frame in self.frames.values()),
                    hits=self.hits, misses=self.misses,
                    interval_fallbacks=self.interval_fallbacks)
