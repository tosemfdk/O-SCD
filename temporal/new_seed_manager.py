"""Online NEW-seed candidate and active-track lifecycle management.

This module intentionally owns only identity/lifecycle bookkeeping.  It does
not import feature extraction, matching, or triangulation code; callers pass
already validated pair edges and an injectable geometry resolver for the current
track observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class SeedObservationId:
    """Stable identity for one detected keypoint in one inference frame."""

    frame_index: int
    keypoint_index: int

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, Integral):
            raise TypeError("frame_index must be an integer")
        if isinstance(self.keypoint_index, bool) or not isinstance(self.keypoint_index, Integral):
            raise TypeError("keypoint_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if self.keypoint_index < 0:
            raise ValueError("keypoint_index must be nonnegative")


@dataclass(frozen=True)
class SeedGeometryResult:
    """Validated geometry attached to a candidate/promotion.

    The caller is responsible for known-pose epipolar, cheirality, ray-angle,
    reprojection, and mask-inside validation before returning ``passed=True``.
    """

    xyz: tuple[float, float, float]
    passed: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.xyz) != 3:
            raise ValueError("xyz must contain exactly 3 coordinates")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)) for v in self.xyz):
            raise ValueError("xyz must contain finite real coordinates")
        object.__setattr__(self, "xyz", tuple(float(v) for v in self.xyz))


@dataclass(frozen=True)
class SeedMatchEdge:
    """A validated pairwise association between two inference observations."""

    obs_a: SeedObservationId
    obs_b: SeedObservationId
    score: float = 1.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.obs_a == self.obs_b:
            raise ValueError("a match edge needs two distinct observations")
        if isinstance(self.score, bool) or not isinstance(self.score, Real) or not math.isfinite(float(self.score)):
            raise ValueError("score must be a finite real scalar")
        object.__setattr__(self, "score", float(self.score))
        if self.obs_b < self.obs_a:
            obs_a, obs_b = self.obs_a, self.obs_b
            object.__setattr__(self, "obs_a", obs_b)
            object.__setattr__(self, "obs_b", obs_a)

    @property
    def key(self) -> tuple[SeedObservationId, SeedObservationId]:
        return (self.obs_a, self.obs_b)


@dataclass(frozen=True)
class SeedTrack:
    """Current union-find track snapshot."""

    track_id: int
    observations: tuple[SeedObservationId, ...]
    source_sign: str
    first_observed_time: float
    last_observed_time: float
    best_edge_score: float

    @property
    def unique_frame_count(self) -> int:
        return len({obs.frame_index for obs in self.observations})


@dataclass(frozen=True)
class SeedCandidate:
    """Two-or-more-view pre-Gaussian candidate."""

    track_id: int
    source_sign: str
    support_observations: tuple[SeedObservationId, ...]
    xyz: tuple[float, float, float]
    first_observed_time: float
    last_observed_time: float
    created_time: float
    expires_after_frame: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unique_frame_count(self) -> int:
        return len({obs.frame_index for obs in self.support_observations})


@dataclass(frozen=True)
class SeedPromotion:
    """Active fixed-geometry NEW seed row metadata.

    ``start_time`` is the promotion time, not the first observation time.  A
    mapping flip closes the half-open interval by setting ``end_time`` to the
    flip timestamp; rows remain present in ``active_promotions``.
    """

    seed_id: int
    track_id: int
    source_sign: str
    support_observations: tuple[SeedObservationId, ...]
    xyz: tuple[float, float, float]
    first_observed_time: float
    promotion_time: float
    start_time: float
    end_time: float = math.inf
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unique_frame_count(self) -> int:
        return len({obs.frame_index for obs in self.support_observations})

    @property
    def is_active(self) -> bool:
        return math.isinf(self.end_time)

    def close(self, end_time: float) -> "SeedPromotion":
        _validate_time(end_time, "end_time")
        if end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        return SeedPromotion(
            seed_id=self.seed_id,
            track_id=self.track_id,
            source_sign=self.source_sign,
            support_observations=self.support_observations,
            xyz=self.xyz,
            first_observed_time=self.first_observed_time,
            promotion_time=self.promotion_time,
            start_time=self.start_time,
            end_time=float(end_time),
            diagnostics=self.diagnostics,
        )


GeometryResolver = Callable[[Sequence[SeedObservationId]], SeedGeometryResult | None]


@dataclass(frozen=True)
class SeedManagerUpdate:
    """Result returned by manager updates."""

    candidates: tuple[SeedCandidate, ...]
    promotions: tuple[SeedPromotion, ...]
    rejected_edges: tuple[tuple[SeedMatchEdge, str], ...] = ()


@dataclass(frozen=True)
class NewSeedManagerConfig:
    candidate_support_views: int = 2
    promotion_support_views: int = 3
    candidate_ttl_frames: int = 20

    def __post_init__(self) -> None:
        for name in ("candidate_support_views", "promotion_support_views", "candidate_ttl_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.candidate_support_views < 2:
            raise ValueError("candidate_support_views must be at least 2")
        if self.promotion_support_views < self.candidate_support_views:
            raise ValueError("promotion_support_views must be >= candidate_support_views")


class NewSeedManager:
    """Deterministic online manager for NEW seed tracks.

    Gate semantics:
    - ``new_sign=None`` pauses creation; buffered/matched observations can be
      supplied but no candidate/active seed is created.
    - reopening the same sign resumes creation using retained track evidence.
    - switching to the opposite sign drops unpromoted candidates/edges for the
      old sign and closes existing active rows at the flip time; no row is
      deleted.
    """

    def __init__(self, config: NewSeedManagerConfig | None = None) -> None:
        self.config = config or NewSeedManagerConfig()
        self.current_sign: str | None = None
        self.gate_open: bool = False
        self._edges: dict[tuple[SeedObservationId, SeedObservationId], SeedMatchEdge] = {}
        self._tracks: dict[int, SeedTrack] = {}
        self._track_signature_to_id: dict[tuple[SeedObservationId, ...], int] = {}
        self._candidates: dict[int, SeedCandidate] = {}
        self._promoted_track_ids: set[int] = set()
        self._promotions: list[SeedPromotion] = []
        self._next_track_id = 0
        self._next_seed_id = 0
        self.rejection_histogram: dict[str, int] = {}

    @property
    def active_promotions(self) -> tuple[SeedPromotion, ...]:
        return tuple(self._promotions)

    @property
    def active_seed_count(self) -> int:
        return sum(1 for p in self._promotions if p.is_active)

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def tracks(self) -> tuple[SeedTrack, ...]:
        return tuple(sorted(self._tracks.values(), key=lambda t: t.track_id))

    @property
    def candidates(self) -> tuple[SeedCandidate, ...]:
        return tuple(sorted(self._candidates.values(), key=lambda c: c.track_id))

    def set_gate(self, new_sign: str | None, timestamp: float) -> None:
        """Open, pause, or flip the seed-creation gate."""
        _validate_time(timestamp, "timestamp")
        if new_sign is not None:
            _validate_sign(new_sign)

        if new_sign is None:
            self.gate_open = False
            return

        if self.current_sign is None:
            self.current_sign = new_sign
            self.gate_open = True
            return

        if new_sign == self.current_sign:
            self.gate_open = True
            return

        # Opposite sign consensus: old unpromoted evidence no longer supports
        # active creation; active rows become closed half-open intervals.
        self._close_active_promotions(timestamp)
        self._edges.clear()
        self._tracks.clear()
        self._track_signature_to_id.clear()
        self._candidates.clear()
        self._promoted_track_ids.clear()
        self.current_sign = new_sign
        self.gate_open = True

    def pause_gate(self, timestamp: float | None = None) -> None:
        if timestamp is not None:
            _validate_time(timestamp, "timestamp")
        self.gate_open = False

    def ingest_edges(
        self,
        *,
        frame_index: int,
        timestamp: float,
        source_sign: str,
        edges: Iterable[SeedMatchEdge | tuple[SeedObservationId, SeedObservationId] | tuple[SeedObservationId, SeedObservationId, float]],
        geometry_resolver: GeometryResolver | None = None,
    ) -> SeedManagerUpdate:
        """Ingest validated pair edges and update candidates/promotions.

        ``geometry_resolver`` is called with the sorted support observations of
        any track that has enough unique frames for candidate/promotion status.
        Returning ``None`` or ``passed=False`` leaves the track unmaterialized.
        """
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral) or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        _validate_time(timestamp, "timestamp")
        _validate_sign(source_sign)

        if not self.gate_open or self.current_sign != source_sign:
            self._expire_candidates(frame_index)
            return SeedManagerUpdate(candidates=(), promotions=())

        self._prune_stale_edges(frame_index)
        normalized_edges = tuple(_coerce_edge(edge) for edge in edges)
        for edge in normalized_edges:
            old = self._edges.get(edge.key)
            if old is None or _edge_sort_key(edge) > _edge_sort_key(old):
                self._edges[edge.key] = edge

        rejected = self._rebuild_tracks(timestamp, source_sign)
        self._expire_candidates(frame_index)

        new_candidates: list[SeedCandidate] = []
        new_promotions: list[SeedPromotion] = []
        for track in sorted(self._tracks.values(), key=lambda t: (t.unique_frame_count, t.track_id)):
            if track.unique_frame_count < self.config.candidate_support_views:
                continue
            geometry = self._resolve_geometry(track.observations, geometry_resolver)
            if geometry is None or not geometry.passed:
                self._candidates.pop(track.track_id, None)
                reason = "geometry_rejected"
                if geometry is not None:
                    reasons = geometry.diagnostics.get("rejection_reasons")
                    if isinstance(reasons, (tuple, list)) and reasons:
                        reason = "geometry_" + "+".join(str(item) for item in reasons)
                    else:
                        reason = str(geometry.diagnostics.get("reason", reason))
                self.rejection_histogram[reason] = self.rejection_histogram.get(reason, 0) + 1
                continue

            candidate = SeedCandidate(
                track_id=track.track_id,
                source_sign=source_sign,
                support_observations=track.observations,
                xyz=geometry.xyz,
                first_observed_time=track.first_observed_time,
                last_observed_time=track.last_observed_time,
                created_time=timestamp,
                expires_after_frame=int(frame_index) + self.config.candidate_ttl_frames,
                diagnostics=dict(geometry.diagnostics),
            )
            is_new_candidate = track.track_id not in self._candidates
            self._candidates[track.track_id] = candidate
            if is_new_candidate:
                new_candidates.append(candidate)

            if (
                track.unique_frame_count >= self.config.promotion_support_views
                and track.track_id not in self._promoted_track_ids
            ):
                promotion = SeedPromotion(
                    seed_id=self._next_seed_id,
                    track_id=track.track_id,
                    source_sign=source_sign,
                    support_observations=track.observations,
                    xyz=geometry.xyz,
                    first_observed_time=track.first_observed_time,
                    promotion_time=timestamp,
                    start_time=timestamp,
                    end_time=math.inf,
                    diagnostics=dict(geometry.diagnostics),
                )
                self._next_seed_id += 1
                self._promoted_track_ids.add(track.track_id)
                self._promotions.append(promotion)
                new_promotions.append(promotion)

        for _, reason in rejected:
            self.rejection_histogram[reason] = self.rejection_histogram.get(reason, 0) + 1
        return SeedManagerUpdate(
            candidates=tuple(new_candidates),
            promotions=tuple(new_promotions),
            rejected_edges=tuple(rejected),
        )

    def _resolve_geometry(
        self,
        observations: Sequence[SeedObservationId],
        geometry_resolver: GeometryResolver | None,
    ) -> SeedGeometryResult:
        if geometry_resolver is None:
            # Unit-test/default bookkeeping path: geometry has intentionally
            # already been validated by the caller, but no coordinates are
            # needed by the lifecycle assertions.
            return SeedGeometryResult((0.0, 0.0, 0.0), passed=True, diagnostics={"source": "default"})
        result = geometry_resolver(tuple(observations))
        if result is None:
            return SeedGeometryResult((0.0, 0.0, 0.0), passed=False, diagnostics={"reason": "geometry_none"})
        if not isinstance(result, SeedGeometryResult):
            if isinstance(result, Mapping):
                result = SeedGeometryResult(
                    xyz=tuple(result["xyz"]),
                    passed=bool(result.get("passed", True)),
                    diagnostics=result.get("diagnostics", {}),
                )
            else:
                result = SeedGeometryResult(tuple(result))  # type: ignore[arg-type]
        return result

    def _rebuild_tracks(self, timestamp: float, source_sign: str) -> list[tuple[SeedMatchEdge, str]]:
        parent: dict[SeedObservationId, SeedObservationId] = {}
        members: dict[SeedObservationId, set[SeedObservationId]] = {}
        best_score: dict[SeedObservationId, float] = {}
        rejected: list[tuple[SeedMatchEdge, str]] = []

        def make(obs: SeedObservationId) -> None:
            if obs not in parent:
                parent[obs] = obs
                members[obs] = {obs}
                best_score[obs] = float("-inf")

        def find(obs: SeedObservationId) -> SeedObservationId:
            root = parent[obs]
            if root != obs:
                parent[obs] = find(root)
            return parent[obs]

        def union(edge: SeedMatchEdge) -> None:
            make(edge.obs_a)
            make(edge.obs_b)
            ra, rb = find(edge.obs_a), find(edge.obs_b)
            if ra == rb:
                best_score[ra] = max(best_score[ra], edge.score)
                return
            merged = members[ra] | members[rb]
            if not _has_unique_frames(merged):
                rejected.append((edge, "same_frame_conflict"))
                return
            # Deterministic root by smallest observation id.
            root, child = (ra, rb) if ra < rb else (rb, ra)
            parent[child] = root
            members[root] = merged
            best_score[root] = max(best_score[ra], best_score[rb], edge.score)
            del members[child]
            del best_score[child]

        for edge in sorted(self._edges.values(), key=_edge_sort_key, reverse=True):
            union(edge)

        new_tracks: dict[int, SeedTrack] = {}
        for root, obs_set in sorted(members.items(), key=lambda item: tuple(sorted(item[1]))):
            root = find(root)
            observations = tuple(sorted(obs_set))
            if len(observations) < 2:
                continue
            track_id = self._track_signature_to_id.get(observations)
            if track_id is None:
                obs_set_for_id = set(observations)
                overlapping_old_ids = [
                    old.track_id
                    for old in self._tracks.values()
                    if set(old.observations).issubset(obs_set_for_id)
                ]
                if overlapping_old_ids:
                    track_id = min(overlapping_old_ids)
                else:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                self._track_signature_to_id[observations] = track_id
            frames = [obs.frame_index for obs in observations]
            track = SeedTrack(
                track_id=track_id,
                observations=observations,
                source_sign=source_sign,
                first_observed_time=float(min(frames)),
                last_observed_time=float(max(frames)),
                best_edge_score=best_score[root],
            )
            new_tracks[track_id] = track
        self._tracks = new_tracks
        return rejected

    def _expire_candidates(self, frame_index: int) -> None:
        expired = [tid for tid, candidate in self._candidates.items() if candidate.expires_after_frame < frame_index]
        for tid in expired:
            if tid not in self._promoted_track_ids:
                del self._candidates[tid]

    def _prune_stale_edges(self, frame_index: int) -> None:
        min_live_frame = int(frame_index) - self.config.candidate_ttl_frames
        stale = [
            key
            for key, edge in self._edges.items()
            if max(edge.obs_a.frame_index, edge.obs_b.frame_index) < min_live_frame
        ]
        for key in stale:
            del self._edges[key]

    def _close_active_promotions(self, timestamp: float) -> None:
        self._promotions = [p.close(timestamp) if p.is_active else p for p in self._promotions]


def _validate_time(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real scalar")


def _validate_sign(value: str) -> None:
    if not isinstance(value, str) or value not in {"+", "-"}:
        raise ValueError("source/new sign must be '+' or '-'")


def _coerce_edge(
    edge: SeedMatchEdge | tuple[SeedObservationId, SeedObservationId] | tuple[SeedObservationId, SeedObservationId, float]
) -> SeedMatchEdge:
    if isinstance(edge, SeedMatchEdge):
        return edge
    if len(edge) == 2:
        return SeedMatchEdge(edge[0], edge[1])
    if len(edge) == 3:
        return SeedMatchEdge(edge[0], edge[1], edge[2])
    raise ValueError("edge tuples must have length 2 or 3")


def _edge_sort_key(edge: SeedMatchEdge) -> tuple[float, SeedObservationId, SeedObservationId]:
    return (edge.score, edge.obs_a, edge.obs_b)


def _has_unique_frames(observations: Iterable[SeedObservationId]) -> bool:
    frames = [obs.frame_index for obs in observations]
    return len(frames) == len(set(frames))


__all__ = [
    "GeometryResolver",
    "NewSeedManagerConfig",
    "NewSeedManager",
    "NewSeedTrackManager",
    "SeedCandidate",
    "SeedGeometryResult",
    "SeedManagerUpdate",
    "SeedMatchEdge",
    "SeedObservationId",
    "SeedPromotion",
    "SeedTrack",
]

# Backward-compatible descriptive alias.
NewSeedTrackManager = NewSeedManager
