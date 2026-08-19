import math

import pytest

from temporal.new_seed_manager import (
    NewSeedManagerConfig,
    NewSeedTrackManager,
    SeedGeometryResult,
    SeedMatchEdge,
    SeedObservationId,
)


def oid(frame, keypoint=0):
    return SeedObservationId(frame, keypoint)


def resolver(observations):
    # Prove the manager accepts already-validated geometry without importing a
    # triangulation module.  Coordinates are deterministic functions of support.
    frames = [obs.frame_index for obs in observations]
    return SeedGeometryResult(
        xyz=(float(sum(frames)), float(len(frames)), 1.0),
        passed=True,
        diagnostics={"frames": tuple(frames), "validated": True},
    )


def test_gate_before_open_never_creates_candidates_or_active_seeds():
    manager = NewSeedTrackManager()

    update = manager.ingest_edges(
        frame_index=1,
        timestamp=1.0,
        source_sign="+",
        edges=[(oid(0), oid(1), 0.99)],
        geometry_resolver=resolver,
    )

    assert update.candidates == ()
    assert update.promotions == ()
    assert manager.candidate_count == 0
    assert manager.active_seed_count == 0


def test_two_views_make_candidate_but_not_active_seed():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=10.0)

    update = manager.ingest_edges(
        frame_index=11,
        timestamp=11.0,
        source_sign="+",
        edges=[SeedMatchEdge(oid(10, 4), oid(11, 5), score=0.91)],
        geometry_resolver=resolver,
    )

    assert len(update.candidates) == 1
    assert update.promotions == ()
    candidate = update.candidates[0]
    assert candidate.unique_frame_count == 2
    assert candidate.support_observations == (oid(10, 4), oid(11, 5))
    assert candidate.diagnostics["validated"] is True
    assert manager.candidate_count == 1
    assert manager.active_seed_count == 0


def test_third_unique_view_promotes_with_causal_support_and_promotion_start():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=10.0)
    manager.ingest_edges(
        frame_index=11,
        timestamp=11.0,
        source_sign="+",
        edges=[(oid(10), oid(11), 0.9)],
        geometry_resolver=resolver,
    )

    update = manager.ingest_edges(
        frame_index=12,
        timestamp=12.0,
        source_sign="+",
        edges=[(oid(11), oid(12), 0.95)],
        geometry_resolver=resolver,
    )

    assert update.candidates == ()  # existing two-view candidate was updated.
    assert len(update.promotions) == 1
    promotion = update.promotions[0]
    assert promotion.seed_id == 0
    assert promotion.unique_frame_count == 3
    assert promotion.first_observed_time == 10.0
    assert promotion.promotion_time == 12.0
    assert promotion.start_time == 12.0
    assert math.isinf(promotion.end_time)
    assert all(obs.frame_index <= promotion.promotion_time for obs in promotion.support_observations)
    assert manager.active_seed_count == 1


def test_unique_frame_conflict_keeps_higher_score_edge_deterministically():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=0.0)

    update = manager.ingest_edges(
        frame_index=2,
        timestamp=2.0,
        source_sign="+",
        edges=[
            # If accepted first, this lower-score edge would block the better
            # frame-1 association below.  The rebuild sorts by score so the
            # higher-score same-frame alternative wins.
            (oid(0, 0), oid(1, 1), 0.1),
            (oid(0, 0), oid(1, 2), 0.9),
            (oid(1, 2), oid(2, 0), 0.8),
        ],
        geometry_resolver=resolver,
    )

    assert len(update.promotions) == 1
    support = set(update.promotions[0].support_observations)
    assert oid(1, 2) in support
    assert oid(1, 1) not in support
    assert any(reason == "same_frame_conflict" for _, reason in update.rejected_edges)


def test_candidate_ttl_expires_unpromoted_records():
    manager = NewSeedTrackManager(NewSeedManagerConfig(candidate_ttl_frames=1))
    manager.set_gate("+", timestamp=0.0)
    manager.ingest_edges(
        frame_index=1,
        timestamp=1.0,
        source_sign="+",
        edges=[(oid(0), oid(1), 0.9)],
        geometry_resolver=resolver,
    )
    assert manager.candidate_count == 1

    manager.ingest_edges(
        frame_index=3,
        timestamp=3.0,
        source_sign="+",
        edges=[],
        geometry_resolver=resolver,
    )

    assert manager.candidate_count == 0
    assert manager.active_seed_count == 0


def test_gate_pause_stops_new_candidates_without_closing_active_rows():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=0.0)
    manager.ingest_edges(
        frame_index=2,
        timestamp=2.0,
        source_sign="+",
        edges=[(oid(0), oid(1), 0.9), (oid(1), oid(2), 0.9)],
        geometry_resolver=resolver,
    )
    assert manager.active_seed_count == 1

    manager.pause_gate(timestamp=3.0)
    update = manager.ingest_edges(
        frame_index=4,
        timestamp=4.0,
        source_sign="+",
        edges=[(oid(3), oid(4), 0.9)],
        geometry_resolver=resolver,
    )

    assert update.candidates == ()
    assert update.promotions == ()
    assert manager.candidate_count == 1  # existing promoted candidate metadata remains.
    assert manager.active_seed_count == 1
    assert math.isinf(manager.active_promotions[0].end_time)


def test_mapping_flip_closes_active_half_open_lifespan_without_deleting_rows():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=0.0)
    manager.ingest_edges(
        frame_index=2,
        timestamp=2.0,
        source_sign="+",
        edges=[(oid(0), oid(1), 0.9), (oid(1), oid(2), 0.9)],
        geometry_resolver=resolver,
    )
    assert len(manager.active_promotions) == 1

    manager.set_gate("-", timestamp=5.0)

    assert len(manager.active_promotions) == 1
    closed = manager.active_promotions[0]
    assert closed.source_sign == "+"
    assert closed.start_time == 2.0
    assert closed.end_time == 5.0
    assert closed.is_active is False
    assert manager.active_seed_count == 0
    assert manager.candidate_count == 0

    update = manager.ingest_edges(
        frame_index=7,
        timestamp=7.0,
        source_sign="-",
        edges=[(oid(5), oid(6), 0.9), (oid(6), oid(7), 0.9)],
        geometry_resolver=resolver,
    )
    assert len(update.promotions) == 1
    assert len(manager.active_promotions) == 2
    assert manager.active_seed_count == 1


def test_geometry_rejection_reason_is_counted():
    manager = NewSeedTrackManager()
    manager.set_gate("+", timestamp=0.0)

    update = manager.ingest_edges(
        frame_index=1,
        timestamp=1.0,
        source_sign="+",
        edges=[(oid(0), oid(1), 0.9)],
        geometry_resolver=lambda _: SeedGeometryResult(
            xyz=(0.0, 0.0, 1.0),
            passed=False,
            diagnostics={"rejection_reasons": ("reprojection", "mask")},
        ),
    )

    assert update.candidates == ()
    assert manager.rejection_histogram["geometry_reprojection+mask"] == 1


@pytest.mark.parametrize("bad_sign", ["", "plus", "new", 1])
def test_gate_rejects_invalid_sign(bad_sign):
    manager = NewSeedTrackManager()
    with pytest.raises(ValueError):
        manager.set_gate(bad_sign, timestamp=0.0)
