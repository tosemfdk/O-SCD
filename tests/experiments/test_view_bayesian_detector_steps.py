from argparse import Namespace
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.view_bayesian_detector_steps import (
    BayesianDetectorReplay,
    BayesianDetectorViewer,
    DA3SeedReplay,
    DISPLAY_LIFECYCLE,
    DetectorReplayLifespan,
    ReplayStepSummary,
    build_add_remove_gt_index,
    build_binary_change_gt_index,
    binary_change_mask_image,
    compose_gt_change_image,
    compose_horizontal_dashboard,
    compose_two_row_dashboard,
    CueTypeStats,
    causal_training_view_index,
    constrain_base_representation_geometry_,
    constrain_da3_seed_geometry_,
    da3_seed_detector_eligible_rows,
    format_status,
    letterbox_image,
    load_causal_sam_sign_trace,
    load_da3_metric_depth_replay,
    load_learned_sigmoid_boundaries,
    load_da3_seed_replay,
    load_union_mask,
    overlay_da3_seed_centers,
    part19_da3_detector_cue,
    parse_args,
    q_weighted_signed_sam_feature_diff_image,
    representation_dc_target,
    representation_geometry_target,
    rendered_gs_online_da3_depth_difference_image,
    typed_change_cue_image,
    typed_change_cue_masks,
    panel10_da3_seed_proposals,
    signed_reference_da3_depth_difference_image,
)


def test_panel10_raw_masks_partition_targets_without_display_quantization():
    q = torch.tensor([[0.0001, 0.2, 0.8, 1.0]])
    new, removed, appearance = typed_change_cue_masks(
        q, torch.tensor([[0.2, -0.2, 0.2, 0.0]]),
        torch.tensor([[0.1, -0.1, -0.1, 0.1]]),
        torch.ones((1, 4), dtype=torch.bool),
    )
    assert new.tolist() == [[True, False, False, False]]
    assert removed.tolist() == [[False, True, False, False]]
    assert appearance.tolist() == [[False, False, True, True]]
    q_new = q * new
    q_base = q - q_new
    assert torch.equal(q_new + q_base, q)
    assert not bool(((q_new > 0) & (q_base > 0)).any())
    assert q_new[0, 0] > 0  # uint8 display would be black, supervision is not.


def test_panel10_seed_proposals_unproject_only_new_at_scaled_intrinsics():
    q = torch.full((1, 4, 4), 0.7)
    support = torch.zeros((4, 4), dtype=torch.bool)
    support[1, 3] = True
    K = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    batch = panel10_da3_seed_proposals(
        q, support, torch.full((2, 2), 2.0), torch.full((2, 2), 3.0),
        K, torch.eye(4), stride=1, maximum=10,
    )
    assert batch.pixels_xy.tolist() == [[3, 1]]
    assert torch.allclose(batch.xyz, torch.tensor([[0.5, -0.5, 2.0]]))
    assert bool(support[batch.pixels_xy[:, 1], batch.pixels_xy[:, 0]].all())
    assert torch.isfinite(batch.log_scaling).all()
    # No NEW support never invents seeds in REMOVE/appearance regions.
    empty = panel10_da3_seed_proposals(
        q, torch.zeros_like(support), torch.full((2, 2), 2.0),
        torch.full((2, 2), 3.0), K, torch.eye(4), stride=1, maximum=10,
    )
    assert len(empty.xyz) == 0


def test_typed_new_target_uses_cached_raw_mask_not_rgb(monkeypatch):
    replay = BayesianDetectorReplay.__new__(BayesianDetectorReplay)
    replay.args = Namespace(training_partition="panel10_new")
    replay.da3_metric_depth = object()
    replay.current_index = 2
    replay.current_view = Namespace(image_height=1, image_width=2)
    replay._typed_new_mask_cache = {2: torch.tensor([[True, False]])}
    monkeypatch.setattr(replay, "rendered_gs_online_da3_depth_difference_rgb", lambda: None)
    cue = torch.tensor([[[0.0001, 0.9]]])
    target = replay._new_seed_target(2, replay.current_view, cue)
    assert torch.equal(target, torch.tensor([[[0.0001, 0.0]]]))
    assert torch.equal(cue, torch.tensor([[[0.0001, 0.9]]]))


def test_panel10_cli_requires_depth_and_frozen_never_open(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--training-partition", "panel10_new"])
    checkpoint = tmp_path / "replay.pt"
    checkpoint.touch()
    common = ["--training-partition", "panel10_new", "--da3-seed-checkpoint",
              str(checkpoint), "--sam-sign-trace-root", str(tmp_path)]
    args = parse_args(common)
    assert args.training_partition == "panel10_new"
    assert not args.train_never_open_geometry
    assert args.da3_max_rows == 20000
    assert parse_args(common + ["--da3-max-rows", "0"]).da3_max_rows == 0
    with pytest.raises(SystemExit):
        parse_args(common + ["--da3-max-rows", "-1"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--train-never-open-geometry"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--da3-detector-cue-source", "part19_binary"])


@pytest.mark.parametrize("max_rows", [0, 20000])
def test_panel10_birth_budget_is_applied_after_covered_cells_are_rejected(monkeypatch, max_rows):
    import experiments.view_bayesian_detector_steps as viewer_module
    import experiments.panel10_seed_topology as topology_module

    replay = BayesianDetectorReplay.__new__(BayesianDetectorReplay)
    replay.args = parse_args([])
    replay.args.da3_max_rows = max_rows
    replay.args.training_partition = "panel10_new"
    replay.args.da3_birth_max_per_frame = 1
    replay.args.da3_birth_stride = 1
    replay.device = torch.device("cpu")
    replay.current_index = 1
    replay.current_view = Namespace(candidate_map=torch.ones((1, 2, 2)))
    replay.records = [None, Namespace(name="frame1")]
    replay._typed_new_mask_cache = {1: torch.ones((2, 2), dtype=torch.bool)}
    replay._typed_depth_cache = {1: (None,) * 4}
    replay.seed_model = Namespace(num_gaussians=1, get_xyz=torch.zeros((1, 3)),
                                  get_scaling=torch.ones((1, 3)))
    replay.seed_geometry_update_counts = torch.zeros(1, dtype=torch.long)
    replay.seed_retired = torch.zeros(1, dtype=torch.bool)
    monkeypatch.setattr(replay, "rendered_gs_online_da3_depth_difference_rgb", lambda: None)
    monkeypatch.setattr(replay, "_seed_lifecycle_masks",
                        lambda t: (torch.tensor([False]), torch.tensor([True])))

    def proposals(*args, **kwargs):
        assert kwargs["maximum"] == 4  # not the 1-row birth budget
        return Namespace(xyz=torch.tensor([[0., 0., 1.], [1., 0., 1.]]),
                         log_scaling=torch.zeros((2, 3)),
                         pixels_xy=torch.tensor([[0, 0], [1, 0]]),
                         confidence=torch.tensor([.9, .8]))

    def coverage(*args, **kwargs):
        assert args[3].tolist() == [replay.args.da3_coverage_min_updates]
        return torch.tensor([False, True])

    def append(_replay, *, xyz, scaling, timestamp, metadata):
        assert xyz.tolist() == [[1., 0., 1.]]
        assert metadata[0]["pixel_xy"] == [1, 0]
        return 1

    monkeypatch.setattr(viewer_module, "panel10_da3_seed_proposals", proposals)
    monkeypatch.setattr(viewer_module, "uncovered_by_learned_gaussian_support", coverage)
    monkeypatch.setattr(topology_module, "append_typed_seeds", append)
    assert replay._append_panel10_da3_seeds(1) == {
        "proposed": 2,
        "accepted": 1,
        "coverage_rejected": 1,
        "coverage_3d_rejected": 1,
        "coverage_2d_existing_rejected": 0,
        "coverage_2d_same_frame_rejected": 0,
        "budget_rejected": 0,
        "occupancy_2d_pixels": 0,
        "occupancy_2d_seed_rows": 0,
    }


def test_replay_lifespan_tracks_open_close_and_reopen_slots() -> None:
    state = DetectorReplayLifespan(3, max_states=2, device=torch.device("cpu"))

    assert state.open_rows([1], timestamp=0).tolist() == [0]
    assert state.current_state_index.tolist() == [-1, 0, -1]
    assert state.close_rows([1], timestamp=1).tolist() == [0]
    assert state.current_state_index.tolist() == [-1, -1, -1]
    assert state.open_rows([1], timestamp=2).tolist() == [1]
    assert state.num_states.tolist() == [0, 2, 0]
    assert state.get_active_state_indices(0).tolist() == [-1, 0, -1]
    assert state.get_active_state_indices(1).tolist() == [-1, -1, -1]
    assert state.get_active_state_indices(2).tolist() == [-1, 1, -1]
    assert state.never_open_mask(0).tolist() == [True, False, True]
    assert state.closed_mask(1).tolist() == [False, True, False]


def test_replay_lifespan_excludes_future_rows_and_preserves_reopen_history() -> None:
    state = DetectorReplayLifespan(0, max_states=2, device=torch.device("cpu"))
    rows = state.append_rows(2, materialized_timestamp=3)

    assert rows.tolist() == [0, 1]
    assert state.materialized_mask(2).tolist() == [False, False]
    assert state.never_open_mask(3).tolist() == [True, True]

    state.open_rows([0], timestamp=4)
    state.close_rows([0], timestamp=5)
    state.open_rows([0], timestamp=7)

    assert state.get_active_state_indices(4).tolist() == [0, -1]
    assert state.closed_mask(6).tolist() == [True, False]
    assert state.get_active_state_indices(7).tolist() == [1, -1]
    assert state.validate_lifecycle()


def test_replay_lifespan_rejects_same_timestamp_close() -> None:
    state = DetectorReplayLifespan(1, max_states=2, device=torch.device("cpu"))
    state.open_rows([0], timestamp=3)

    with pytest.raises(RuntimeError, match="after OPEN"):
        state.close_rows([0], timestamp=3)


def test_initialization_status_makes_unconsumed_cue_explicit() -> None:
    text = format_status(
        None,
        consumed_frames=0,
        total_frames=304,
        bayes_factor_threshold=30.0,
    )

    assert "Reference initialization" in text
    assert "0 / 304" in text
    assert "BF ≥ 30" in text


def test_step_status_contains_requested_color_counts() -> None:
    summary = ReplayStepSummary(
        timestamp=0,
        frame_name="frame.png",
        observed=10,
        never_open=6,
        uncertain=1,
        open=2,
        closed=1,
        opened_now=2,
        closed_now=1,
        candidate_started=3,
        candidate_continued=2,
        candidate_rejected=1,
        candidate_committed=3,
        positive_pseudocount_mass=4.0,
        negative_pseudocount_mass=5.0,
    )

    text = format_status(
        summary,
        consumed_frames=1,
        total_frames=304,
        bayes_factor_threshold=30.0,
    )

    for label in ("black NEVER_OPEN", "yellow candidate", "green OPEN", "red CLOSED"):
        assert label in text


def test_step_status_reports_learned_boundary_when_present() -> None:
    summary = ReplayStepSummary(
        timestamp=0,
        frame_name="frame.png",
        observed=1,
        never_open=1,
        uncertain=0,
        open=0,
        closed=0,
        opened_now=0,
        closed_now=0,
        candidate_started=0,
        candidate_continued=0,
        candidate_rejected=0,
        candidate_committed=0,
        positive_pseudocount_mass=0.0,
        negative_pseudocount_mass=1.0,
        cue_tau=0.31,
        cue_width=0.08,
    )

    text = format_status(
        summary,
        consumed_frames=1,
        total_frames=1,
        bayes_factor_threshold=30.0,
    )

    assert "0.3100 / 0.0800" in text


def test_parser_defaults_to_current_bf30_detector_contract() -> None:
    args: Namespace = parse_args([])

    assert args.bayes_factor_threshold == 30.0
    assert args.stable_flip_prior == 1.0
    assert args.stable_keep_prior == 10.0
    assert args.resolution == 4.0
    assert args.dashboard_width == 1920
    assert args.cue_mode == "binary"
    assert args.cue_scale == 2.0
    assert args.cue_fusion == "sum"
    assert args.product_exponent == 0.3
    assert args.cue_remap == "identity"
    assert args.soft_band_low == 0.15
    assert args.soft_band_high == 0.35
    assert args.sam_cue_threshold == 0.5
    assert args.representation_updates == 16
    assert args.current_view_probability == 0.33
    assert args.representation_cue_amplitude == 1.0
    assert args.geometry_cue_amplitude == 1.0
    assert args.seed_dc_supervision == "joint"
    assert args.seed_dc_loss_weight == 1.0
    assert args.dc_replay_mode == "sampled"
    assert args.da3_detector_cue_source == "shared"
    assert args.da3_detector_cue_threshold == 0.5
    assert args.da3_coverage_min_updates == 4
    assert args.da3_coverage_sigma == 2.0
    assert not args.train_never_open_geometry
    assert args.base_geometry_scope == "frozen"
    assert args.base_geometry_target == "change"
    assert args.sam_new_sign is None
    assert not args.train_never_open_base_opacity


def test_representation_dc_target_restores_original_cue_amplitude_without_clamp() -> None:
    unit_q = torch.tensor([[[0.0, 0.25, 1.0]]])

    target = representation_dc_target(unit_q, amplitude=2.0)

    assert torch.equal(target, torch.tensor([[[0.0, 0.5, 2.0]]]))
    assert torch.equal(unit_q, torch.tensor([[[0.0, 0.25, 1.0]]]))


def test_representation_geometry_target_accepts_two_q_without_clamp() -> None:
    unit_q = torch.tensor([[[0.0, 0.25, 1.0]]])

    target = representation_geometry_target(unit_q, amplitude=2.0)

    assert torch.equal(target, torch.tensor([[[0.0, 0.5, 2.0]]]))
    assert torch.equal(unit_q, torch.tensor([[[0.0, 0.25, 1.0]]]))


def test_parser_accepts_seed_local_dc_ablation() -> None:
    args = parse_args(
        [
            "--representation-cue-amplitude",
            "2",
            "--seed-dc-supervision",
            "projected_bce",
            "--seed-dc-loss-weight",
            "0.5",
            "--dc-replay-mode",
            "current",
        ]
    )

    assert args.representation_cue_amplitude == 2.0
    assert args.seed_dc_supervision == "projected_bce"
    assert args.seed_dc_loss_weight == 0.5
    assert args.dc_replay_mode == "current"


def test_parser_requires_signed_sam_contract_for_signed_new_geometry() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--base-geometry-target", "signed_new"])


def test_parser_requires_open_and_never_scope_for_never_open_opacity() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--train-never-open-base-opacity"])

    args = parse_args(
        [
            "--base-geometry-scope",
            "open_and_never_open",
            "--train-never-open-base-opacity",
        ]
    )
    assert args.train_never_open_base_opacity


def test_part19_da3_detector_uses_untouched_cached_sum_before_learned_q() -> None:
    cached_sum = torch.tensor([[[0.2, 0.8]]])
    learned_q = torch.tensor([[[0.99, 0.01]]])
    view = SimpleNamespace(
        cached_sum_candidate_map=cached_sum,
        pre_remap_candidate_map=torch.full_like(cached_sum, 0.4),
        candidate_map=learned_q,
    )

    assert part19_da3_detector_cue(view) is cached_sum


def test_part19_da3_detector_recovers_sum_before_remap() -> None:
    raw_sum = torch.tensor([[[0.3, 0.7]]])
    view = SimpleNamespace(
        pre_remap_candidate_map=raw_sum,
        candidate_map=torch.tensor([[[0.9, 0.1]]]),
    )

    assert part19_da3_detector_cue(view) is raw_sum


def test_da3_seed_detector_skips_birth_frame_and_starts_on_next_frame() -> None:
    births = [3, 4, 4]

    assert da3_seed_detector_eligible_rows(
        births, timestamp=4
    ).tolist() == [0]
    assert da3_seed_detector_eligible_rows(
        births, timestamp=5
    ).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("source", "expected"),
    [("shared", [0.9, 0.1]), ("part19_binary", [0.0, 1.0])],
)
def test_da3_detector_uses_shared_soft_q_unless_legacy_is_explicit(
    monkeypatch, source, expected
) -> None:
    from temporal.change_evidence import cue_to_change_probability
    import temporal.new_seed_gaussians as seed_module

    replay = object.__new__(BayesianDetectorReplay)
    replay.args = parse_args(["--cue-mode", "soft"])
    replay.args.da3_detector_cue_source = source
    replay.device = torch.device("cpu")
    replay.count = 1
    replay.base = SimpleNamespace(_features_dc=torch.zeros((1, 1, 3)))
    replay.pipe = SimpleNamespace()
    replay.evidence_background = torch.zeros(3)
    replay.seed_detector_probe = SimpleNamespace(num_seeds=2)
    replay.seed_lifecycle = DetectorReplayLifespan(
        2, max_states=2, device=replay.device, materialized_timestamp=0
    )
    replay.seed_model = SimpleNamespace(active_mask=replay.seed_lifecycle.active_mask)
    replay.accepted_da3_source_rows = [4, 7]
    replay.accepted_da3_birth_global = [0, 0]
    calls = []
    observations = []
    probe = object()

    def concatenate(base, seeds, **kwargs):
        assert base is replay.base
        assert seeds is replay.seed_detector_probe
        assert kwargs["seed_attribute_mode"] == "all"
        return probe

    def accumulate(view, model, pipe, background, cue, **kwargs):
        assert model is probe
        calls.append((cue, kwargs))
        probability = cue_to_change_probability(
            cue, mode=kwargs["cue_mode"],
            threshold=kwargs["cue_threshold"], scale=kwargs["cue_scale"],
        ).flatten()
        return SimpleNamespace(
            delta_a=torch.cat((torch.zeros(1), probability)),
            delta_b=torch.cat((torch.zeros(1), 1.0 - probability)),
            total_mass=torch.tensor([0.0, 1.0, 1.0]),
        )

    def update(delta_a, delta_b, **kwargs):
        observations.append((delta_a, delta_b, kwargs))
        return SimpleNamespace(
            candidate_committed=torch.zeros(2, dtype=torch.bool),
            observed=torch.ones(2, dtype=torch.bool),
        )

    monkeypatch.setattr(seed_module, "build_concatenated_change_view", concatenate)
    replay._accumulate_change_evidence = accumulate
    replay.seed_tracker = SimpleNamespace(update=update)
    view = SimpleNamespace(
        candidate_map=torch.tensor([[[1.8, 0.2]]]),  # stored 2Q after sigmoid
        cached_sum_candidate_map=torch.tensor([[[0.2, 0.8]]]),
        pre_remap_candidate_map=torch.tensor([[[0.4, 0.6]]]),
    )

    replay._update_da3_seed_detector(view, timestamp=0)
    assert not calls  # birth frame is proposal-only
    result = replay._update_da3_seed_detector(view, timestamp=1)

    assert len(calls) == len(observations) == 1
    delta_a, delta_b, kwargs = observations[0]
    assert torch.allclose(delta_a, torch.tensor(expected))
    assert torch.allclose(delta_b, 1.0 - torch.tensor(expected))
    assert kwargs["row_indices"].tolist() == [4, 7]
    assert kwargs["timestamp"] == 1
    assert result["observed"] == 2
    if source == "shared":
        assert calls[0][0] is view.candidate_map
        assert calls[0][1]["cue_mode"] == replay.args.cue_mode
        assert calls[0][1]["cue_scale"] == replay.args.cue_scale
        assert torch.allclose(delta_a, replay._normalized_cue_target(view).flatten())


def test_joint_lifecycle_view_keeps_base_and_open_seed_dc_trainable() -> None:
    from temporal.active_new_gaussians import ActiveNewGaussianModel
    from temporal.new_seed_gaussians import build_concatenated_change_view

    base_dc = torch.nn.Parameter(torch.zeros((1, 1, 3)))
    base = SimpleNamespace(
        active_sh_degree=0,
        max_sh_degree=0,
        scaling_activation=torch.exp,
        opacity_activation=torch.sigmoid,
        rotation_activation=lambda value: torch.nn.functional.normalize(value),
        covariance_activation=lambda scaling, modifier, rotation: scaling,
        _features_dc=base_dc,
        _features_rest=torch.zeros((1, 0, 3)),
        get_xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        get_opacity=torch.full((1, 1), 0.5),
        get_scaling=torch.full((1, 3), 0.1),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    seeds = ActiveNewGaussianModel(device="cpu")
    seeds.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        start=0.0,
        scaling=torch.full((2, 3), math.log(0.1)),
        opacity=0.5,
    )
    seeds.start[1] = float("inf")
    seeds.end[1] = float("inf")

    joint = build_concatenated_change_view(
        base,
        seeds,
        timestamp=0.0,
        base_dc=base_dc,
        detach_base_dc=False,
        seed_attribute_mode="lifecycle",
    )
    joint.get_features.sum().backward()

    assert base_dc.grad is not None
    assert float(base_dc.grad.abs().sum()) > 0.0
    assert seeds.new_dc.grad is not None
    assert float(seeds.new_dc.grad[0].abs().sum()) > 0.0
    assert float(seeds.new_dc.grad[1].abs().sum()) == 0.0


def test_joint_lifecycle_view_uses_explicit_historical_seed_population() -> None:
    from temporal.active_new_gaussians import ActiveNewGaussianModel
    from temporal.new_seed_gaussians import build_concatenated_change_view

    base_dc = torch.nn.Parameter(torch.zeros((1, 1, 3)))
    base = SimpleNamespace(
        active_sh_degree=0,
        max_sh_degree=0,
        scaling_activation=torch.exp,
        opacity_activation=torch.sigmoid,
        rotation_activation=lambda value: torch.nn.functional.normalize(value),
        covariance_activation=lambda scaling, modifier, rotation: scaling,
        _features_dc=base_dc,
        _features_rest=torch.zeros((1, 0, 3)),
        get_xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        get_opacity=torch.full((1, 1), 0.5),
        get_scaling=torch.full((1, 3), 0.1),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    seeds = ActiveNewGaussianModel(device="cpu")
    seeds.append_xfeat_anchors(
        xyz=torch.tensor(
            [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]]
        ),
        start=0.0,
        scaling=torch.full((3, 3), math.log(0.1)),
        opacity=0.5,
    )
    historical_active = torch.tensor([True, False, False])
    historical_never_open = torch.tensor([False, True, False])

    joint = build_concatenated_change_view(
        base,
        seeds,
        timestamp=9.0,
        base_dc=base_dc,
        detach_base_dc=False,
        seed_attribute_mode="lifecycle",
        seed_lifecycle_active=historical_active,
        seed_lifecycle_never_open=historical_never_open,
    )
    joint.get_features.sum().backward()

    assert joint.seed_active_mask.tolist() == [True, True, False]
    assert joint.get_xyz.shape[0] == 3  # base + historical OPEN + historical pending
    assert seeds.new_dc.grad is not None
    assert float(seeds.new_dc.grad[0].abs().sum()) > 0.0
    assert float(seeds.new_dc.grad[1].abs().sum()) == 0.0
    assert float(seeds.new_dc.grad[2].abs().sum()) == 0.0


def test_base_change_attributes_follow_replay_timestamp_not_current_state() -> None:
    replay = object.__new__(BayesianDetectorReplay)
    replay.current_index = 4
    replay.args = SimpleNamespace(train_never_open_base_opacity=False)
    replay.change_dc = torch.nn.Parameter(
        torch.tensor(
            [
                [[1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0]],
            ]
        )
    )
    replay.base = SimpleNamespace(get_opacity=torch.ones((3, 1)))
    replay.lifecycle = DetectorReplayLifespan(
        3, max_states=2, device=torch.device("cpu")
    )
    replay.lifecycle.open_rows([0], timestamp=0)
    replay.lifecycle.close_rows([0], timestamp=2)
    replay.lifecycle.open_rows([1], timestamp=3)

    dc_at_one, opacity_at_one, active_at_one = (
        replay._effective_base_change_attributes(timestamp=1)
    )
    dc_at_four, opacity_at_four, active_at_four = (
        replay._effective_base_change_attributes(timestamp=4)
    )

    assert active_at_one.tolist() == [True, False, False]
    assert active_at_four.tolist() == [False, True, False]
    assert dc_at_one[0].tolist() == [[1.0, 1.0, 1.0]]
    assert dc_at_four[1].tolist() == [[2.0, 2.0, 2.0]]
    assert opacity_at_one[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert opacity_at_four[:, 0].tolist() == [0.0, 1.0, 1.0]


def test_da3_seed_detector_rejects_future_materialization() -> None:
    with pytest.raises(ValueError, match="future DA3 seed proposals"):
        da3_seed_detector_eligible_rows([3, 5], timestamp=4)


def test_da3_seed_visual_state_uses_committed_sidecar_lifecycle() -> None:
    replay = object.__new__(BayesianDetectorReplay)
    replay.device = torch.device("cpu")
    replay.current_index = 5
    replay.args = SimpleNamespace(bayes_factor_threshold=30.0)
    replay.accepted_da3_source_rows = [2, 0, 3]
    replay.seed_model = SimpleNamespace(
        num_gaussians=3,
        active_mask=lambda timestamp: torch.tensor([False, True, False]),
        never_open_mask=lambda: torch.tensor([True, False, False]),
    )
    replay.seed_lifecycle = DetectorReplayLifespan(
        3,
        max_states=1,
        device=torch.device("cpu"),
        materialized_timestamp=0,
    )
    replay.seed_lifecycle.open_rows([1, 2], timestamp=1)
    replay.seed_lifecycle.close_rows([2], timestamp=2)
    replay.seed_tracker = SimpleNamespace(
        candidate_active=torch.tensor([False, False, True, False]),
        last_log_bayes_factor=torch.zeros(4),
    )

    state = replay._seed_visual_state()

    assert state.never_open.tolist() == [True, False, False]
    assert state.open.tolist() == [False, True, False]
    assert state.closed.tolist() == [False, False, True]
    assert state.colors.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]


def test_parser_accepts_soft_fractional_cue_mode() -> None:
    args: Namespace = parse_args(
        [
            "--cue-mode",
            "soft",
            "--cue-scale",
            "1.5",
            "--cue-fusion",
            "l1_power_product",
            "--product-exponent",
            "0.4",
            "--cue-remap",
            "smoothstep_band",
            "--soft-band-low",
            "0.1",
            "--soft-band-high",
            "0.4",
        ]
    )

    assert args.cue_mode == "soft"
    assert args.cue_scale == 1.5
    assert args.cue_fusion == "l1_power_product"
    assert args.product_exponent == 0.4
    assert args.cue_remap == "smoothstep_band"
    assert args.soft_band_low == 0.1
    assert args.soft_band_high == 0.4


def test_parser_enables_never_open_xyz_scale_rotation_refinement() -> None:
    args = parse_args(["--train-never-open-geometry"])

    assert args.train_never_open_geometry

    frozen = parse_args(
        ["--train-never-open-geometry", "--no-train-never-open-geometry"]
    )
    assert not frozen.train_never_open_geometry


def test_parser_accepts_two_q_geometry_target() -> None:
    args = parse_args(["--geometry-cue-amplitude", "2"])

    assert args.geometry_cue_amplitude == 2.0


def test_base_geometry_constraint_changes_only_selected_rows() -> None:
    anchor_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchor_scaling = torch.zeros(2, 3)
    xyz = anchor_xyz.clone()
    xyz[0, 0] = 10.0
    xyz[1, 0] = 11.0
    scaling = anchor_scaling.clone()
    scaling[:] = math.log(10.0)
    rotation = torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(2, 1)

    changed = constrain_base_representation_geometry_(
        xyz=xyz,
        scaling=scaling,
        rotation=rotation,
        anchor_xyz=anchor_xyz,
        anchor_scaling=anchor_scaling,
        selected_rows=torch.tensor([True, False]),
        max_displacement_ratio=2.0,
        min_scale_ratio=0.5,
        max_scale_ratio=2.0,
    )

    assert changed["xyz"].tolist() == [True, False]
    assert xyz[0, 0].item() == pytest.approx(2.0)
    assert xyz[1, 0].item() == pytest.approx(11.0)
    assert torch.allclose(scaling[0], torch.full((3,), math.log(2.0)))
    assert torch.allclose(scaling[1], torch.full((3,), math.log(10.0)))
    assert torch.allclose(rotation[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(rotation[1], torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_parser_rejects_reversed_soft_band() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--soft-band-low", "0.6", "--soft-band-high", "0.4"])


def test_load_learned_sigmoid_boundaries(tmp_path) -> None:
    path = tmp_path / "boundaries.json"
    path.write_text(
        json.dumps(
            {
                "remap": "sigmoid",
                "edge_probability": 0.05,
                "frames": {"frame.png": {"tau": 0.3, "width": 0.08}},
            }
        ),
        encoding="utf-8",
    )

    boundaries, edge_probability = load_learned_sigmoid_boundaries(path)

    assert edge_probability == 0.05
    assert boundaries["frame.png"].tau == 0.3
    assert boundaries["frame.png"].width == 0.08


def test_parser_accepts_learned_sigmoid_artifact(tmp_path) -> None:
    artifact = tmp_path / "boundaries.json"
    artifact.write_text("{}", encoding="utf-8")

    args = parse_args(
        [
            "--cue-remap",
            "learned_sigmoid",
            "--cue-boundary-json",
            str(artifact),
        ]
    )

    assert args.cue_remap == "learned_sigmoid"
    assert args.cue_boundary_json == artifact


def test_q_weighted_signed_sam_feature_diff_uses_continuous_q() -> None:
    score = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    cue = torch.tensor([[1.0, 0.5], [0.0, 1.0]])

    image, stats = q_weighted_signed_sam_feature_diff_image(
        score,
        cue,
        width=2,
        height=2,
    )

    assert image.tolist() == [
        [[255, 0, 0], [0, 0, 128]],
        [[0, 0, 0], [0, 0, 128]],
    ]
    assert stats.positive == 1
    assert stats.negative == 2
    assert stats.neutral == 1
    assert stats.normalization_scale == pytest.approx(1.0)


def test_binary_change_mask_uses_inclusive_point_five_threshold() -> None:
    score = torch.tensor([[[0.0, 0.4999], [0.5, 1.0]]])

    image = binary_change_mask_image(score, threshold=0.5)

    assert image.tolist() == [
        [[0, 0, 0], [0, 0, 0]],
        [[255, 255, 255], [255, 255, 255]],
    ]


def test_q_weighted_signed_sam_feature_diff_keeps_all_magnitudes() -> None:
    score = torch.tensor([[4.0, 2.0], [-3.0, -1.0]])
    cue = torch.ones((2, 2))

    image, stats = q_weighted_signed_sam_feature_diff_image(
        score,
        cue,
        width=2,
        height=2,
    )

    assert image[0, 0].tolist() == [255, 0, 0]
    assert image[0, 1].tolist() == [128, 0, 0]
    assert image[1, 0].tolist() == [0, 0, 191]
    assert image[1, 1].tolist() == [0, 0, 64]
    assert stats.positive == 2
    assert stats.negative == 2
    assert stats.neutral == 0
    assert stats.normalization_scale == pytest.approx(4.0)


def test_typed_change_cue_classifies_panel3_with_matching_panel8_and_panel9_signs() -> None:
    learned_q = torch.tensor(
        [[[1.0, 0.5, 0.25], [0.75, 1.0, 0.0]]], dtype=torch.float64
    )
    weighted_sam = torch.tensor(
        [[0.2, -0.2, 0.2], [-0.2, 0.2, -0.2]], dtype=torch.float64
    )
    signed_depth = torch.tensor(
        [[0.04, -0.04, -0.04], [0.04, 0.03, -0.04]], dtype=torch.float64
    )
    depth_valid = torch.ones((2, 3), dtype=torch.bool)

    image, stats = typed_change_cue_image(
        learned_q,
        weighted_sam,
        signed_depth,
        depth_valid,
    )

    assert isinstance(stats, CueTypeStats)
    assert image.tolist() == [
        [[255, 0, 0], [0, 0, 128], [64, 64, 0]],
        [[191, 191, 0], [255, 255, 0], [0, 0, 0]],
    ]
    assert stats.new == 1
    assert stats.removed == 1
    assert stats.appearance == 3
    assert stats.background == 1
    assert stats.new + stats.removed + stats.appearance + stats.background == 6


def test_typed_change_cue_treats_threshold_equality_invalid_and_nonfinite_as_appearance() -> None:
    learned_q = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    weighted_sam = torch.tensor([[0.1, 0.2, float("nan"), -0.2]])
    signed_depth = torch.tensor([[0.04, 0.03, 0.04, float("nan")]])
    depth_valid = torch.tensor([[True, True, True, False]])

    image, stats = typed_change_cue_image(
        learned_q,
        weighted_sam,
        signed_depth,
        depth_valid,
    )

    assert image.tolist() == [
        [[26, 26, 0], [51, 51, 0], [76, 76, 0], [102, 102, 0]]
    ]
    assert stats.new == 0
    assert stats.removed == 0
    assert stats.appearance == 4
    assert stats.background == 0
    assert stats.new + stats.removed + stats.appearance + stats.background == 4


def test_typed_change_cue_preserves_low_q_brightness_without_reweighting() -> None:
    learned_q = torch.tensor([[0.01, 0.49, 0.501, 1.0]])
    weighted_sam = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    signed_depth = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    depth_valid = torch.ones((1, 4), dtype=torch.bool)

    image, stats = typed_change_cue_image(
        learned_q,
        weighted_sam,
        signed_depth,
        depth_valid,
    )

    assert image.tolist() == [
        [[3, 0, 0], [125, 0, 0], [0, 0, 128], [0, 0, 255]]
    ]
    assert stats.new == 2
    assert stats.removed == 2
    assert stats.appearance == 0
    assert stats.background == 0


def test_typed_change_cue_resizes_depth_sign_support_nearest_to_q_grid() -> None:
    learned_q = torch.ones((4, 4))
    weighted_sam = torch.tensor(
        [
            [0.2, 0.2, -0.2, -0.2],
            [0.2, 0.2, -0.2, -0.2],
            [-0.2, -0.2, 0.2, 0.2],
            [-0.2, -0.2, 0.2, 0.2],
        ]
    )
    signed_depth = torch.tensor([[0.04, -0.04], [-0.04, 0.04]])
    depth_valid = torch.ones((2, 2), dtype=torch.bool)

    image, stats = typed_change_cue_image(
        learned_q,
        weighted_sam,
        signed_depth,
        depth_valid,
    )

    assert image.tolist() == [
        [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]],
        [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]],
        [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
        [[0, 0, 255], [0, 0, 255], [255, 0, 0], [255, 0, 0]],
    ]
    assert stats.new == 8
    assert stats.removed == 8
    assert stats.appearance == 0
    assert stats.background == 0


def test_typed_change_cue_rejects_dimension_mismatches() -> None:
    learned_q = torch.ones((2, 2))
    weighted_sam = torch.ones((2, 2))
    signed_depth = torch.ones((1, 1))
    depth_valid = torch.ones((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="depth"):
        typed_change_cue_image(learned_q, weighted_sam, signed_depth, depth_valid)

    with pytest.raises(ValueError, match="weighted_sam"):
        typed_change_cue_image(
            learned_q,
            torch.ones((1, 2)),
            torch.ones((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="learned_q"):
        typed_change_cue_image(
            torch.ones((1, 2, 2, 1)),
            weighted_sam,
            signed_depth,
            torch.ones((1, 1), dtype=torch.bool),
        )


def test_typed_change_cue_does_not_mutate_inputs() -> None:
    learned_q = torch.tensor([[0.2, 0.8]])
    weighted_sam = torch.tensor([[0.2, -0.2]])
    signed_depth = torch.tensor([[0.04, -0.04]])
    depth_valid = torch.ones((1, 2), dtype=torch.bool)
    originals = tuple(value.clone() for value in (learned_q, weighted_sam, signed_depth, depth_valid))

    typed_change_cue_image(learned_q, weighted_sam, signed_depth, depth_valid)

    for current, original in zip((learned_q, weighted_sam, signed_depth, depth_valid), originals):
        assert torch.equal(current, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_q_weighted_signed_sam_feature_diff_accepts_cuda_score_and_cpu_q() -> None:
    image, stats = q_weighted_signed_sam_feature_diff_image(
        torch.tensor([[1.0, -1.0], [0.5, -0.5]], device="cuda"),
        torch.ones((2, 2)),
        width=2,
        height=2,
    )

    assert image[0].tolist() == [[255, 0, 0], [0, 0, 255]]
    assert stats.positive == 2
    assert stats.negative == 2


def test_signed_reference_da3_depth_difference_marks_front_red_and_behind_blue() -> None:
    reference = torch.full((2, 2), 2.0)
    da3 = torch.tensor([[1.0, 3.0], [2.0, 1.5]])
    valid = torch.ones((2, 2), dtype=torch.bool)

    image, stats = signed_reference_da3_depth_difference_image(
        reference,
        da3,
        valid,
        width=2,
        height=2,
    )

    assert image.tolist() == [
        [[255, 0, 0], [0, 0, 255]],
        [[0, 0, 0], [128, 0, 0]],
    ]
    assert stats.valid_pixels == 4
    assert stats.median_absolute_difference == pytest.approx(0.5)
    assert stats.normalization_scale == pytest.approx(1.0)


def test_signed_depth_difference_normalizes_only_inside_valid_overlap() -> None:
    reference = torch.tensor([[2.0, 100.0], [2.0, 2.0]])
    da3 = torch.ones((2, 2))
    overlap = torch.tensor([[True, False], [False, False]])

    image, stats = signed_reference_da3_depth_difference_image(
        reference,
        da3,
        overlap,
        width=2,
        height=2,
    )

    assert image[0, 0].tolist() == [255, 0, 0]
    assert image[0, 1].tolist() == [0, 0, 0]
    assert image[1].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert stats.valid_pixels == 1
    assert stats.normalization_scale == pytest.approx(1.0)


def test_rendered_gs_online_da3_difference_fits_only_stable_pixels() -> None:
    rendered = torch.full((5, 5), 2.0)
    online_da3 = torch.full((5, 5), 0.5)
    online_da3[-1] = torch.tensor([0.25, 0.75, 0.25, 0.75, 0.5])
    stable = torch.ones((5, 5), dtype=torch.bool)
    stable[-1] = False
    display = ~stable

    image, stats = rendered_gs_online_da3_depth_difference_image(
        online_da3,
        rendered,
        stable,
        display,
        width=5,
        height=5,
    )

    assert stats.alignment_scale == pytest.approx(4.0)
    assert stats.alignment_samples == 20
    assert stats.alignment_inliers == 20
    assert image[-1, 0, 0] > 0
    assert image[-1, 0, 2] == 0
    assert image[-1, 1, 2] > 0
    assert image[-1, 1, 0] == 0


def test_rendered_gs_online_da3_difference_hides_weak_depth_or_sam_support() -> None:
    rendered = torch.full((4, 4), 2.0)
    online_da3 = torch.full((4, 4), 0.5)
    stable = torch.ones((4, 4), dtype=torch.bool)
    sam_support = torch.zeros((4, 4), dtype=torch.bool)
    sam_support[-1, 0] = True
    sam_support[-1, 1] = True
    online_da3[-1, 0] = 0.49  # aligned residual +0.04: visible
    online_da3[-1, 1] = 0.495  # aligned residual +0.02: hidden

    image, stats = rendered_gs_online_da3_depth_difference_image(
        online_da3,
        rendered,
        stable,
        sam_support,
        width=4,
        height=4,
        depth_difference_threshold=0.03,
    )

    assert image[-1, 0, 0] == 255
    assert image[-1, 1].tolist() == [0, 0, 0]
    assert int((image != 0).any(axis=2).sum()) == 1
    assert stats.valid_pixels == 1


def test_load_causal_sam_sign_trace_aligns_axes_across_scenes(tmp_path) -> None:
    axis1 = np.zeros((1, 256), dtype=np.float32)
    axis1[0, 0] = 1.0
    axis2 = -axis1
    for scene, name, axis, negative, positive in (
        (1, "scene_change1_frame_000001.png", axis1, -2.0, 1.0),
        (2, "scene_change2_frame_000001.png", axis2, -3.0, 4.0),
    ):
        directory = tmp_path / f"scene_change{scene}"
        directory.mkdir()
        np.savez_compressed(
            directory / "causal_pca_posterior_arrays.npz",
            frame_names=np.asarray([name]),
            pc1_axes=axis,
            epsilon_negative=np.asarray([negative]),
            epsilon_positive=np.asarray([positive]),
        )
    records = [
        SimpleNamespace(name="scene_change1_frame_000001.png"),
        SimpleNamespace(name="scene_change2_frame_000001.png"),
    ]

    trace = load_causal_sam_sign_trace(tmp_path, records)

    second = trace["scene_change2_frame_000001.png"]
    assert second.axis[0] == 1.0
    assert second.epsilon_negative == -4.0
    assert second.epsilon_positive == 3.0


def test_load_da3_seed_replay_audits_and_preserves_birth_frames(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "seed_sidecar": {
                "xyz": torch.tensor(
                    [[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.4, 0.0, 2.0]]
                ),
                "metadata": [
                    {
                        "frame_global": 4,
                        "frame_local": 2,
                        "frame_name": "a.png",
                        "causal_window": [1, 2],
                    },
                    {
                        "frame_global": 5,
                        "frame_local": 3,
                        "frame_name": "b.png",
                        "causal_window": [1, 2, 3],
                    },
                    {
                        "frame_global": 95,
                        "frame_local": 1,
                        "frame_name": "scene2.png",
                        "causal_window_global": [88, 89, 90, 91, 92, 93, 94, 95],
                    },
                ],
            }
        },
        checkpoint,
    )

    replay = load_da3_seed_replay(checkpoint)

    assert replay.xyz.shape == (3, 3)
    assert replay.birth_global.tolist() == [4, 5, 95]
    assert replay.birth_name == ("a.png", "b.png", "scene2.png")
    assert replay.log_scaling.shape == (3, 3)
    assert np.allclose(np.exp(replay.log_scaling), 0.01)
    assert replay.source_sign == ("+", "+", "+")


@pytest.mark.parametrize(
    "depth_scale_source",
    ["da3metric_reference_bank_fixed", "da3metric_reference_render"],
)
def test_load_da3_metric_depth_replay_resolves_metric_cache(
    tmp_path, depth_scale_source: str
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    cache_root = tmp_path / "da3metric_cache"
    cache_root.mkdir()
    torch.save(
        {
            "configuration": {
                "depth_scale_source": depth_scale_source,
                "metric_fixed_scene_scale": 5.75,
                "metric_cache_root": None,
                "metric_model": "depth-anything/DA3METRIC-LARGE",
                "process_res": 448,
            }
        },
        checkpoint,
    )

    replay = load_da3_metric_depth_replay(checkpoint)

    assert replay is not None
    assert replay.cache_root == cache_root
    assert replay.model_name == "depth-anything/DA3METRIC-LARGE"
    assert replay.process_res == 448


def test_causal_training_sampler_reproduces_seeded_point33_current_branch() -> None:
    draws = [
        causal_training_view_index(9, update, seed=7)
        for update in range(200)
    ]

    assert draws == [
        causal_training_view_index(9, update, seed=7)
        for update in range(200)
    ]
    assert all(0 <= index <= 9 for index, _ in draws)
    latest_branches = sum(int(latest) for _, latest in draws)
    assert 45 <= latest_branches <= 85


def test_da3_geometry_constraint_clips_birth_relative_extent() -> None:
    from temporal.active_new_gaussians import ActiveNewGaussianModel

    model = ActiveNewGaussianModel(device="cpu")
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        start=0.0,
        scaling=torch.log(torch.tensor([[0.01, 0.01, 0.01]])),
        opacity=0.1,
    )
    with torch.no_grad():
        model._xyz[0, 0] = 1.0
        model._scaling[0] = math.log(1.0)
        model._opacity[0] = 50.0
        model._rotation[0].zero_()

    clipped = constrain_da3_seed_geometry_(
        model,
        torch.tensor([True]),
        max_displacement_ratio=4.0,
        min_scale_ratio=0.25,
        max_scale_ratio=4.0,
        min_opacity=0.01,
        max_opacity=0.99,
    )

    assert torch.linalg.vector_norm(model._xyz[0]).item() == pytest.approx(0.04)
    assert model.get_scaling[0].max().item() == pytest.approx(0.04)
    assert model.get_opacity[0].item() == pytest.approx(0.99)
    assert model.get_rotation[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert all(bool(mask[0]) for mask in clipped.values())


def test_da3_seed_overlay_shows_prior_green_and_current_red() -> None:
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    seeds = DA3SeedReplay(
        xyz=np.asarray([[0.0, 0.0, 2.0], [0.8, 0.0, 2.0]], dtype=np.float32),
        birth_global=np.asarray([4, 5], dtype=np.int64),
        birth_name=("a.png", "b.png"),
    )
    camera = {
        "rotation": np.eye(3).tolist(),
        "position": [0.0, 0.0, 0.0],
        "fx": 10.0,
        "fy": 10.0,
        "width": 20,
        "height": 20,
    }

    image, stats = overlay_da3_seed_centers(
        rgb, seeds, global_index=5, camera=camera, marker_radius=1
    )

    assert stats.accepted_so_far == 2
    assert stats.visible == 2
    assert stats.born_now == 1
    assert stats.born_now_visible == 1
    assert image[10, 10, 1] > image[10, 10, 0]
    assert image[10, 14, 0] > image[10, 14, 1]


def test_add_remove_gt_index_and_union_are_partitioned(tmp_path) -> None:
    source = tmp_path / "scene_change1_2_3"
    scene = tmp_path / "scene_change1"
    (scene / "object_masks" / "add").mkdir(parents=True)
    (scene / "object_masks" / "remove").mkdir(parents=True)
    add_path = scene / "object_masks" / "add" / "frame.png"
    remove_path = scene / "object_masks" / "remove" / "frame.png"
    add = np.zeros((4, 4), dtype=np.uint8)
    remove = np.zeros((4, 4), dtype=np.uint8)
    add[:2, :2] = 255
    remove[2:, 2:] = 255
    Image.fromarray(add).save(add_path)
    Image.fromarray(remove).save(remove_path)
    objects = []
    for state, relative_path in (
        ("NEW", "object_masks/add/frame.png"),
        ("REMOVED", "object_masks/remove/frame.png"),
    ):
        objects.append(
            {
                "attributes": {"change_state": state},
                "segmentation": {
                    "masks": [
                        {"frame_name": "frame.png", "mask_path": relative_path}
                    ]
                },
            }
        )
    (scene / "object_change_annotations.json").write_text(
        json.dumps({"objects": objects}), encoding="utf-8"
    )
    records = [SimpleNamespace(segment_name="scene_change1", name="frame.png")]

    index = build_add_remove_gt_index(source, records)
    add_union = load_union_mask(index["frame.png"].add, width=4, height=4)
    remove_union = load_union_mask(index["frame.png"].remove, width=4, height=4)

    assert np.array_equal(add_union, add > 0)
    assert np.array_equal(remove_union, remove > 0)


def test_gt_change_image_is_single_white_add_remove_union() -> None:
    add = np.array([[True, False], [False, False]])
    remove = np.array([[False, False], [False, True]])

    image = compose_gt_change_image(add, remove)

    assert image.tolist() == [
        [[255, 255, 255], [0, 0, 0]],
        [[0, 0, 0], [255, 255, 255]],
    ]


def test_letterbox_preserves_source_aspect_ratio() -> None:
    source = torch.zeros((40, 20, 3), dtype=torch.uint8).numpy()
    source[:, :, 1] = 255

    fitted = letterbox_image(source, width=200, height=100)

    foreground = fitted[:, :, 1] == 255
    rows, columns = foreground.nonzero()
    assert rows.max() - rows.min() + 1 == 100
    assert columns.max() - columns.min() + 1 == 50


def test_horizontal_dashboard_matches_browser_aspect_and_has_four_columns() -> None:
    panels = []
    for index in range(4):
        image = torch.zeros((80, 40, 3), dtype=torch.uint8).numpy()
        image[:, :, index % 3] = 255
        panels.append((f"panel {index}", image))

    dashboard = compose_horizontal_dashboard(panels, width=1600, aspect=2.0)

    assert dashboard.shape == (800, 1600, 3)
    for x in (200, 600, 1000, 1400):
        assert dashboard[400, x].max() == 255


def test_two_row_dashboard_places_one_to_seven_above_eight_to_fourteen() -> None:
    panels = []
    for index in range(14):
        image = np.zeros((80, 40, 3), dtype=np.uint8)
        image[:, :, 0 if index < 7 else 2] = 255
        panels.append((f"panel {index + 1}", image))

    dashboard = compose_two_row_dashboard(panels, width=1400, aspect=1.75)

    assert dashboard.shape == (800, 1400, 3)
    assert dashboard[200, 100].tolist() == [255, 0, 0]
    assert dashboard[600, 100].tolist() == [0, 0, 255]


def test_viewer_inserts_binary_prediction_at_six_and_shifts_later_panels() -> None:
    viewer = object.__new__(BayesianDetectorViewer)
    viewer.display = DISPLAY_LIFECYCLE
    viewer.replay = SimpleNamespace(da3_seeds=None, sam_model=None)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    viewer.main_image = image
    viewer.input_image = image
    viewer.cue_image = image
    viewer.detector_state_image = image
    viewer.learned_change_image = image
    viewer.predicted_change_mask_image = image
    viewer.gt_change_image = image
    viewer.sam_feature_diff_image = image
    viewer.depth_difference_image = image
    viewer.cue_type_image = image
    viewer.cue_type_stats = CueTypeStats(new=0, removed=0, appearance=0, background=4)
    viewer.da3_seed_stats = SimpleNamespace(accepted_so_far=0, born_now=0)
    viewer.sam_feature_diff_stats = SimpleNamespace(normalization_scale=0.0)
    viewer.depth_difference_stats = SimpleNamespace(
        alignment_scale=1.0,
        alignment_inliers=0,
        alignment_samples=0,
        median_absolute_difference=0.0,
        normalization_scale=0.0,
    )
    viewer.args = SimpleNamespace(cue_mode="soft", cue_remap="learned_sigmoid")

    panels = viewer._dashboard_panels()

    assert len(panels) == 14
    assert panels[4][0].startswith("5. Current-valid learned R_change")
    assert panels[5][0] == "6. Predicted change mask (raw R_change >= 0.5)"
    assert panels[6][0] == "7. GT Change (ADD union REMOVE)"
    assert panels[7][0].startswith("8. Signed SAM diff")
    assert panels[8][0].startswith("9. GS-sDA3")
    assert panels[9][0].startswith("10. Cue types")
    assert panels[9][1] is image
    for index, panel_number in enumerate(range(11, 15), start=10):
        assert panels[index][0] == f"{panel_number}. Reserved"


def test_binary_change_gt_index_requires_exact_frame_masks_without_object_labels(tmp_path):
    (tmp_path / 'gt_mask').mkdir()
    path = tmp_path / 'gt_mask' / 'Inst_1_test_IMG_1.png'
    Image.fromarray(np.array([[0, 255], [0, 0]], dtype=np.uint8)).save(path)
    records = [SimpleNamespace(name=path.name)]
    index = build_binary_change_gt_index(tmp_path, records)
    assert index == {path.name: path}
    assert build_binary_change_gt_index(tmp_path, [SimpleNamespace(name=path.with_suffix('.jpg').name)]) == {path.with_suffix('.jpg').name: path}
    assert load_union_mask((index[path.name],), width=2, height=2).sum() == 1
    with pytest.raises(FileNotFoundError):
        build_binary_change_gt_index(tmp_path, [SimpleNamespace(name='missing.png')])
    assert parse_args(['--gt-format', 'binary']).gt_format == 'binary'
    assert parse_args([]).gt_format == 'objects'


def test_load_causal_sam_sign_trace_accepts_scene_local_root(tmp_path):
    axis = np.zeros((2, 256), dtype=np.float32)
    axis[:, 0] = [1., -1.]
    np.savez_compressed(tmp_path / 'causal_pca_posterior_arrays.npz',
                        frame_names=np.array(['a.png', 'b.png']), pc1_axes=axis,
                        epsilon_negative=np.array([-2., -3.]),
                        epsilon_positive=np.array([1., 4.]))
    records = [SimpleNamespace(name=n) for n in ['a.png', 'b.png']]
    trace = load_causal_sam_sign_trace(tmp_path, records)
    assert trace['b.png'].axis[0] == 1.
    assert trace['b.png'].epsilon_negative == -4.
    with pytest.raises(KeyError, match='missing'):
        load_causal_sam_sign_trace(tmp_path, [SimpleNamespace(name='missing.png')])
