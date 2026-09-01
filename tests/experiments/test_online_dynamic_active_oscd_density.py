from pathlib import Path
from types import SimpleNamespace
import math

from experiments.run_online_dynamic_active_oscd_density import (
    _candidate_render_gate_state,
    _candidate_transition_progress,
    _candidate_dc_optimizer_masks,
    _config,
    _cue_local_support,
    _cue_mixture_score,
    _initialize_first_open_dc,
    _learned_dc_change_magnitude,
    _lifespan_gate_semantic_colors,
    _optimizer_row_masks,
    _positive_growth_map,
    _render_change_probability,
    _render_to_rgb_u8,
    _save_binary_mask,
    _sample_training_view,
    _close_only_lifespan_change_weight,
    _single_candidate_close_candidate_mask,
    _single_candidate_reset_change_probability,
    _soft_learned_dc_opacity,
    _soft_lifespan_change_weight,
    _soft_lifespan_gate_overrides,
    _validate_args,
    _zero_frozen_dc_gradients,
    first_open_close_latency_diagnostics,
    parse_args,
)
from experiments.run_online_binary_state_lifespan_thaw import LifecycleEvent
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from temporal.masked_optimizer import MaskedRowAdam
from utils.sh_utils import RGB2SH, SH2RGB


def test_dynamic_runner_defaults_lock_bf30_close_only_contract():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density")),
        ]
    )
    config = _config(args)
    assert args.density_policy == "active_oscd"
    assert args.render_support_mode == "open_or_never_open"
    assert args.detector_probe_scaling == "native"
    assert args.detector_mode == "lifespan_gate_beta"
    assert args.candidate_bayes_factor_threshold == 30.0
    assert args.optimizer_selection == "all_open"
    assert args.first_open_dc_initialization == "preserve"
    assert args.change_color_mode == "lifespan_gate"
    assert args.candidate_render_gate == "close_only_log_bf_progress"
    assert args.loss_regularization_mode == "local_growth_replay"
    assert args.local_support_scale == 2.0
    assert args.growth_replay_weight == 7.5
    assert args.gate_stable_flip_prior == 1.0
    assert args.gate_stable_keep_prior == 10.0
    assert args.gate_reset_flip_prior == 1.0
    assert args.gate_reset_keep_prior == 1.0
    assert args.agreement_stable_flip_prior == 1.0
    assert args.agreement_stable_keep_prior == 10.0
    assert args.agreement_reset_flip_prior == 1.0
    assert args.agreement_reset_keep_prior == 1.0
    assert args.agreement_confirmation_views == 3
    assert args.agreement_directional_margin == 0.0
    assert args.agreement_candidate_dc_policy == "adapt"
    assert config.updates_per_frame == 16
    assert args.densify_update_index == 4
    assert args.oscd_grad_threshold == 0.001
    assert args.cue_mixture_threshold == 0.5
    assert args.black_child_prune_threshold == 0.5
    assert args.black_child_prune_min_age_frames == 1
    assert args.percent_dense == 0.01
    assert args.min_opacity == 0.0
    assert args.max_screen_size == 0.0
    assert config.lifecycle_controller == "view_consistent"
    assert config.transition_confirmation_views == 3
    assert config.min_transition_bayes_factor == 3.0
    _validate_args(args)


def test_runner_exposes_open_or_never_open_render_ablation():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density")),
            "--render-support-mode",
            "open_or_never_open",
        ]
    )
    assert args.render_support_mode == "open_or_never_open"


def test_runner_exposes_frozen_black_never_open_occluders():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-black-never-open")),
            "--render-support-mode",
            "open_or_never_open_black",
            "--optimizer-selection",
            "all_open",
        ]
    )

    _validate_args(args)

    assert args.render_support_mode == "open_or_never_open_black"
    masks = _optimizer_row_masks(
        args.render_support_mode,
        active_rows=torch.tensor([True, False]),
        active_visible=torch.tensor([True, False]),
        never_open_visible=torch.tensor([False, True]),
        selection_policy=args.optimizer_selection,
    )
    assert isinstance(masks, torch.Tensor)
    assert masks.tolist() == [True, False]


def test_runner_exposes_detector_only_isotropic_min_probe():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density")),
            "--detector-probe-scaling",
            "isotropic_min",
        ]
    )
    assert args.detector_probe_scaling == "isotropic_min"


def test_runner_accepts_active_oscd_cue_mixture_density():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-active-oscd-cue-mixture")),
            "--density-policy",
            "active_oscd_cue_mixture",
            "--cue-mixture-threshold",
            "0.5",
        ]
    )
    _validate_args(args)
    assert args.density_policy == "active_oscd_cue_mixture"
    assert args.cue_mixture_threshold == 0.5


def test_runner_accepts_child_only_black_pruning_after_cue_mixture_split():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-active-oscd-cue-mixture-black-child-prune")),
            "--density-policy",
            "active_oscd_cue_mixture_black_child_prune",
            "--cue-mixture-threshold",
            "0.5",
            "--black-child-prune-threshold",
            "0.5",
            "--black-child-prune-min-age-frames",
            "1",
        ]
    )
    _validate_args(args)
    assert args.density_policy == "active_oscd_cue_mixture_black_child_prune"
    assert args.cue_mixture_threshold == 0.5
    assert args.black_child_prune_threshold == 0.5
    assert args.black_child_prune_min_age_frames == 1


def test_cue_mixture_score_requires_both_sides_and_observation_mass():
    score = _cue_mixture_score(
        torch.tensor([1.0, 0.9, 0.75, 0.5, 0.05, 9.0]),
        torch.tensor([0.0, 0.1, 0.25, 0.5, 0.05, 1.0]),
    )
    assert torch.allclose(
        score,
        torch.tensor([0.0, 0.2, 0.5, 1.0, 0.1, 0.2]),
    )


def test_runner_accepts_single_candidate_beta_raw_bayes_factor_threshold():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-single-candidate-beta")),
            "--detector-mode",
            "single_candidate_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
            "--candidate-bayes-factor-threshold",
            "30",
        ]
    )
    _validate_args(args)
    assert args.detector_mode == "single_candidate_beta"
    assert args.candidate_bayes_factor_threshold == 30.0


def test_runner_accepts_fixed_never_open_all_open_render_one_contract():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-first-open-render-one")),
            "--render-support-mode",
            "open_or_never_open",
            "--optimizer-selection",
            "all_open",
            "--first-open-dc-initialization",
            "render_one",
            "--detector-mode",
            "single_candidate_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
        ]
    )
    _validate_args(args)
    assert args.render_support_mode == "open_or_never_open"
    assert args.optimizer_selection == "all_open"
    assert args.first_open_dc_initialization == "render_one"


def test_runner_accepts_local_growth_replay_for_bf30_all_open_contract():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-local-growth-replay")),
            "--detector-mode",
            "single_candidate_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
            "--candidate-bayes-factor-threshold",
            "30",
            "--render-support-mode",
            "open_or_never_open",
            "--optimizer-selection",
            "all_open",
            "--loss-regularization-mode",
            "local_growth_replay",
            "--growth-replay-weight",
            "7.5",
        ]
    )
    _validate_args(args)
    assert args.loss_regularization_mode == "local_growth_replay"
    assert args.local_support_scale == 2.0
    assert args.growth_replay_weight == 7.5


def test_runner_accepts_lifespan_gate_semantic_dc_bf30_contract():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-lifespan-gate")),
            "--detector-mode",
            "lifespan_gate_beta",
            "--change-color-mode",
            "lifespan_gate",
            "--candidate-render-gate",
            "none",
            "--candidate-bayes-factor-threshold",
            "30",
            "--render-support-mode",
            "open_or_never_open",
            "--optimizer-selection",
            "all_open",
            "--loss-regularization-mode",
            "local_growth_replay",
        ]
    )
    _validate_args(args)
    assert args.detector_mode == "lifespan_gate_beta"
    assert args.change_color_mode == "lifespan_gate"


def test_runner_accepts_output_only_log_bf_candidate_render_gate():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-soft-candidate-render")),
            "--detector-mode",
            "lifespan_gate_beta",
            "--change-color-mode",
            "lifespan_gate",
            "--candidate-render-gate",
            "log_bf_progress",
        ]
    )

    _validate_args(args)

    assert args.candidate_render_gate == "log_bf_progress"


def test_runner_accepts_close_only_log_bf_candidate_render_gate():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-close-only-candidate-render")),
            "--detector-mode",
            "lifespan_gate_beta",
            "--change-color-mode",
            "lifespan_gate",
            "--candidate-render-gate",
            "close_only_log_bf_progress",
        ]
    )

    _validate_args(args)

    assert args.candidate_render_gate == "close_only_log_bf_progress"


def test_runner_accepts_raw_cue_close_only_gate_with_learned_dc():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-raw-close-only-learned-dc")),
            "--detector-mode",
            "single_candidate_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "close_only_log_bf_progress",
        ]
    )

    _validate_args(args)


def test_runner_rejects_candidate_render_gate_outside_lifespan_gate_contract():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-soft-candidate-render")),
            "--detector-mode",
            "single_candidate_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "log_bf_progress",
        ]
    )

    with pytest.raises(ValueError, match="requires lifespan_gate_beta"):
        _validate_args(args)


@pytest.mark.parametrize("policy", ["adapt", "freeze"])
def test_runner_accepts_learned_dc_agreement_bf30_contract(policy):
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-learned-dc-agreement")),
            "--detector-mode",
            "learned_dc_agreement_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
            "--agreement-candidate-dc-policy",
            policy,
            "--candidate-bayes-factor-threshold",
            "30",
            "--render-support-mode",
            "open_or_never_open",
            "--optimizer-selection",
            "all_open",
            "--loss-regularization-mode",
            "local_growth_replay",
        ]
    )
    _validate_args(args)
    assert args.detector_mode == "learned_dc_agreement_beta"
    assert args.agreement_candidate_dc_policy == policy


def test_runner_accepts_anchored_dc_agreement_bf30_k3_with_optimizer_adapting():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-bf30-anchored-dc-agreement")),
            "--detector-mode",
            "anchored_dc_agreement_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
            "--candidate-bayes-factor-threshold",
            "30",
            "--agreement-confirmation-views",
            "3",
            "--agreement-directional-margin",
            "0",
            "--render-support-mode",
            "open_or_never_open",
            "--optimizer-selection",
            "all_open",
            "--loss-regularization-mode",
            "local_growth_replay",
        ]
    )

    _validate_args(args)

    assert args.detector_mode == "anchored_dc_agreement_beta"
    assert args.agreement_candidate_dc_policy == "adapt"
    assert args.agreement_confirmation_views == 3
    assert args.agreement_directional_margin == 0.0


def test_runner_rejects_anchored_agreement_candidate_dc_freeze():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-anchored-freeze")),
            "--detector-mode",
            "anchored_dc_agreement_beta",
            "--change-color-mode",
            "learned_dc",
            "--candidate-render-gate",
            "none",
            "--agreement-candidate-dc-policy",
            "freeze",
        ]
    )

    with pytest.raises(ValueError, match="requires candidate DC optimization"):
        _validate_args(args)


def test_runner_rejects_agreement_detector_with_semantic_gate_color():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-agreement-color")),
            "--detector-mode",
            "learned_dc_agreement_beta",
            "--change-color-mode",
            "lifespan_gate",
        ]
    )
    with pytest.raises(ValueError):
        _validate_args(args)


@pytest.mark.parametrize(
    ("detector", "color"),
    [
        ("lifespan_gate_beta", "learned_dc"),
        ("single_candidate_beta", "lifespan_gate"),
    ],
)
def test_runner_rejects_mismatched_gate_detector_and_color(detector, color):
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-gate-contract")),
            "--detector-mode",
            detector,
            "--change-color-mode",
            color,
        ]
    )
    with pytest.raises(ValueError, match="must be enabled together"):
        _validate_args(args)


def test_lifespan_gate_semantic_colors_are_exact_binary_rgb():
    colors = _lifespan_gate_semantic_colors(
        torch.tensor([True, False, True]), dtype=torch.float32
    )
    assert colors.tolist() == [
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]


def test_candidate_transition_progress_uses_only_live_normalized_log_bf():
    log_threshold = math.log(30.0)
    log_bf = torch.tensor(
        [2.0 * log_threshold, 0.0, 0.5 * log_threshold, log_threshold],
        requires_grad=True,
    )
    progress = _candidate_transition_progress(
        torch.tensor([False, True, True, True]),
        log_bf,
        bayes_factor_threshold=30.0,
    )

    assert torch.allclose(progress, torch.tensor([0.0, 0.0, 0.5, 1.0]))
    assert not progress.requires_grad


def test_soft_lifespan_change_weight_cross_fades_toward_opposite_state():
    progress = torch.tensor([0.0, 0.25, 0.0, 0.75], requires_grad=True)
    weight = _soft_lifespan_change_weight(
        torch.tensor([False, False, True, True]), progress
    )

    assert torch.allclose(weight, torch.tensor([0.0, 0.25, 1.0, 0.25]))
    assert not weight.requires_grad


def test_close_only_lifespan_weight_never_previews_open_candidates():
    progress = torch.tensor([0.0, 0.25, 0.0, 0.75], requires_grad=True)
    weight = _close_only_lifespan_change_weight(
        torch.tensor([False, False, True, True]), progress
    )

    assert torch.allclose(weight, torch.tensor([0.0, 0.0, 1.0, 0.25]))
    assert not weight.requires_grad


def test_close_only_gate_selects_only_live_candidates_on_open_rows():
    log_threshold = math.log(30.0)
    progress, weight, gated_rows = _candidate_render_gate_state(
        "close_only_log_bf_progress",
        torch.tensor([False, True, True, False]),
        torch.tensor([True, True, False, False]),
        torch.tensor(
            [0.5 * log_threshold, 0.5 * log_threshold, 0.0, log_threshold]
        ),
        bayes_factor_threshold=30.0,
    )

    assert torch.allclose(progress, torch.tensor([0.5, 0.5, 0.0, 0.0]))
    assert torch.allclose(weight, torch.tensor([0.0, 0.5, 1.0, 0.0]))
    assert gated_rows.tolist() == [False, True, False, False]


def test_raw_single_candidate_close_direction_uses_fresh_candidate_beta():
    tracker = SimpleNamespace(
        config=SimpleNamespace(prior_a=1.0, prior_b=1.0),
        candidate_delta_a=torch.tensor([0.0, 3.0, 1.0, 0.0, 0.0]),
        candidate_delta_b=torch.tensor([3.0, 0.0, 1.0, 4.0, 4.0]),
        candidate_active=torch.tensor([True, True, True, False, True]),
    )
    active = torch.tensor([True, True, True, True, False])

    probability = _single_candidate_reset_change_probability(tracker)
    close_candidates = _single_candidate_close_candidate_mask(
        tracker, active, close_probability=0.4
    )

    assert torch.allclose(
        probability, torch.tensor([0.2, 0.8, 0.5, 1.0 / 6.0, 1.0 / 6.0])
    )
    assert close_candidates.tolist() == [True, False, False, False, False]

    log_threshold = math.log(30.0)
    _, weight, gated_rows = _candidate_render_gate_state(
        "close_only_log_bf_progress",
        active,
        tracker.candidate_active,
        torch.full((5,), 0.5 * log_threshold),
        bayes_factor_threshold=30.0,
        close_candidate_mask=close_candidates,
    )
    assert torch.allclose(weight, torch.tensor([0.5, 1.0, 1.0, 1.0, 0.0]))
    assert gated_rows.tolist() == [True, False, False, False, False]


def test_soft_lifespan_gate_scales_opacity_without_mutating_parameters():
    base_opacity = torch.tensor([[0.8], [0.6], [0.4], [0.2]], requires_grad=True)
    soft_weight = torch.tensor([0.0, 0.25, 0.75, 0.0], requires_grad=True)
    colors, opacity = _soft_lifespan_gate_overrides(
        base_opacity,
        torch.tensor([True, True, False, False]),
        soft_weight,
        include_never_open_occluders=True,
    )

    assert colors.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    assert torch.allclose(opacity, torch.tensor([[0.8], [0.15], [0.3], [0.0]]))
    assert not colors.requires_grad
    assert not opacity.requires_grad
    assert torch.equal(base_opacity.detach(), torch.tensor([[0.8], [0.6], [0.4], [0.2]]))

    _, open_only_opacity = _soft_lifespan_gate_overrides(
        base_opacity,
        torch.tensor([True, True, False, False]),
        soft_weight,
        include_never_open_occluders=False,
    )
    assert torch.allclose(
        open_only_opacity, torch.tensor([[0.0], [0.15], [0.3], [0.0]])
    )


def test_soft_learned_dc_opacity_preserves_never_open_occluders_and_closed_rows():
    render_opacity = torch.tensor(
        [[0.8], [0.6], [0.0], [0.4]], requires_grad=True
    )
    opacity = _soft_learned_dc_opacity(
        render_opacity,
        torch.tensor([True, True, False, False]),
        torch.tensor([0.25, 1.0, 0.0, 0.0]),
    )

    assert torch.allclose(opacity, torch.tensor([[0.2], [0.6], [0.0], [0.4]]))
    assert not opacity.requires_grad
    assert torch.equal(
        render_opacity.detach(), torch.tensor([[0.8], [0.6], [0.0], [0.4]])
    )


def test_local_growth_replay_rejects_visibility_restricted_optimizer():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-local-growth-replay")),
            "--loss-regularization-mode",
            "local_growth_replay",
            "--optimizer-selection",
            "active_visible",
        ]
    )
    with pytest.raises(ValueError, match="requires all_open"):
        _validate_args(args)


def test_local_support_and_positive_growth_match_previous_ablation_formula():
    cue = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    support = _cue_local_support(cue, 2.0)
    assert support.tolist() == [[[0.0, 0.5, 1.0, 1.0]]]

    pre = torch.tensor([[[0.2, 0.8, 0.4]]])
    post = torch.tensor([[[0.5, 0.6, 0.9]]])
    growth = _positive_growth_map(pre, post)
    assert torch.allclose(growth, torch.tensor([[[0.3, 0.0, 0.5]]]))
    assert not growth.requires_grad


def test_render_probability_matches_ssf_sigmoid_channel_mean():
    rendered = torch.tensor(
        [
            [[-1.0, 1.0]],
            [[0.0, 2.0]],
            [[1.0, 3.0]],
        ]
    )
    expected = torch.sigmoid(rendered.mean(dim=0, keepdim=True))
    assert torch.equal(_render_change_probability(rendered), expected)


def test_learned_dc_change_magnitude_maps_raw_zero_to_zero_and_white_to_one():
    model = SimpleNamespace(
        change_dc=nn.Parameter(
            torch.cat(
                (
                    torch.zeros(1, 1, 3),
                    RGB2SH(torch.ones(1, 1, 3)),
                    RGB2SH(torch.zeros(1, 1, 3)),
                ),
                dim=0,
            )
        )
    )
    values = _learned_dc_change_magnitude(
        model, torch.tensor([0, 1, 2], dtype=torch.long)
    )
    assert values.tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_all_open_selection_rejects_trainable_never_open_appearance():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("/tmp/test-invalid-all-open")),
            "--render-support-mode",
            "open_or_never_open_dc_opacity",
            "--optimizer-selection",
            "all_open",
        ]
    )
    with pytest.raises(ValueError, match="trainable NEVER_OPEN"):
        _validate_args(args)


def test_runner_accepts_full_continuous_escd_scope():
    args = parse_args(
        [
            "--scope",
            "continuous",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density-continuous")),
        ]
    )
    assert args.scope == "continuous"
    assert args.max_frames is None


def test_runner_accepts_optional_online_visual_capture():
    args = parse_args(
        [
            "--scope",
            "scene_change1",
            "--output-dir",
            str(Path("outputs/test-dynamic-active-density")),
            "--visualization-dir",
            str(Path("outputs/test-dynamic-active-density/causal_visuals")),
        ]
    )
    assert args.visualization_dir == Path(
        "outputs/test-dynamic-active-density/causal_visuals"
    )


def test_render_to_rgb_u8_clamps_and_reorders_channels():
    rendered = torch.tensor(
        [
            [[-1.0, 0.5]],
            [[0.0, 1.0]],
            [[0.25, 2.0]],
        ]
    )
    image = _render_to_rgb_u8(rendered)
    assert image.shape == (1, 2, 3)
    assert image.tolist() == [[[0, 0, 64], [128, 255, 255]]]


def test_save_binary_mask_writes_zero_and_255(tmp_path: Path):
    path = tmp_path / "binary.png"
    _save_binary_mask(path, np.array([[False, True], [True, False]]))

    with Image.open(path) as image:
        stored = np.asarray(image)
    assert stored.tolist() == [[0, 255], [255, 0]]


def test_never_open_appearance_mode_uses_parameter_specific_masks():
    active = np.array([True, False, False])
    never_open = np.array([False, True, False])

    masks = _optimizer_row_masks(
        "open_or_never_open_dc_opacity",
        active_visible=torch.from_numpy(active),
        never_open_visible=torch.from_numpy(never_open),
    )
    assert masks["dc"].tolist() == [True, True, False]
    assert masks["opacity"].tolist() == [True, True, False]
    assert masks["xyz"].tolist() == [True, False, False]
    assert masks["features_rest"].tolist() == [True, False, False]
    assert masks["scaling"].tolist() == [True, False, False]
    assert masks["rotation"].tolist() == [True, False, False]


def test_all_open_optimizer_selection_ignores_visibility_and_freezes_never_open():
    active = torch.tensor([True, False, True, False])
    masks = _optimizer_row_masks(
        "open_or_never_open",
        active_rows=active,
        active_visible=torch.tensor([True, False, False, False]),
        never_open_visible=torch.tensor([False, True, False, True]),
        selection_policy="all_open",
    )
    assert isinstance(masks, torch.Tensor)
    assert masks.tolist() == [True, False, True, False]


def test_candidate_freeze_masks_only_live_candidate_dc_rows():
    selected = torch.tensor([True, True, False, True])
    candidate = torch.tensor([False, True, True, False])
    masks = _candidate_dc_optimizer_masks(
        selected, candidate, policy="freeze"
    )

    assert isinstance(masks, dict)
    assert masks["dc"].tolist() == [True, False, False, True]
    for name in ("xyz", "features_rest", "opacity", "scaling", "rotation"):
        assert torch.equal(masks[name], selected)


def test_candidate_freeze_zeros_only_live_candidate_dc_gradient():
    model = SimpleNamespace(
        change_dc=nn.Parameter(torch.zeros(3, 1, 3))
    )
    model.change_dc.grad = torch.tensor(
        [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0]]]
    )
    maximum = _zero_frozen_dc_gradients(
        model, torch.tensor([False, True, False]), policy="freeze"
    )

    assert maximum == 6.0
    assert torch.count_nonzero(model.change_dc.grad[1]) == 0
    assert torch.equal(
        model.change_dc.grad[[0, 2]],
        torch.tensor([[[1.0, 2.0, 3.0]], [[7.0, 8.0, 9.0]]]),
    )


def _event(
    row: int, timestamp: int, action: str, old_slot: int, new_slot: int
) -> LifecycleEvent:
    return LifecycleEvent(
        gaussian_index=row,
        decision_timestamp=timestamp,
        old_binary_label=int(action == "CLOSE"),
        new_binary_label=int(action == "OPEN"),
        action=action,
        old_slot=old_slot,
        new_current_slot=new_slot,
        p_active=0.9,
        p_01=0.0,
        p_10=0.0,
        p_flip=0.9,
        visible_observation_count=3,
    )


def test_render_one_initializes_only_first_open_and_resets_only_its_dc_state():
    dc = nn.Parameter(torch.zeros(3, 1, 3))
    optimizer = MaskedRowAdam(
        {"dc": dc}, thaw_names=("dc",), lrs={"dc": 0.01}
    )
    dc.grad = torch.ones_like(dc)
    optimizer.step(torch.tensor([True, True, True]))
    with torch.no_grad():
        dc.zero_()
    state_before = {
        name: value.detach().clone()
        for name, value in optimizer.state[dc].items()
        if isinstance(value, torch.Tensor)
    }
    model = SimpleNamespace(change_dc=dc)

    result = _initialize_first_open_dc(
        model,
        optimizer,
        [_event(0, 4, "OPEN", -1, 0), _event(1, 4, "OPEN", 0, 1)],
        torch.tensor([10, 11, 12]),
        "render_one",
    )

    assert result["rows"].tolist() == [0]
    assert result["stable_ids"].tolist() == [10]
    assert result["adam_reset_count"] == 1
    assert torch.allclose(dc[0], RGB2SH(torch.ones_like(dc[0])))
    assert torch.allclose(SH2RGB(dc[0]), torch.ones_like(dc[0]))
    assert torch.equal(dc[1], torch.zeros_like(dc[1]))
    assert result["before"].tolist() == pytest.approx([0.5])
    assert result["immediate"].tolist() == pytest.approx([1.0])
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[dc][name][0]) == 0
        assert torch.equal(optimizer.state[dc][name][1:], state_before[name][1:])


def test_first_open_close_latency_diagnostics_counts_short_lived_rows():
    diagnostics = first_open_close_latency_diagnostics(
        [
            _event(10, 2, "OPEN", -1, 0),
            _event(11, 3, "OPEN", -1, 0),
            _event(10, 3, "CLOSE", 0, -1),
            _event(10, 8, "OPEN", 0, 1),
            _event(11, 8, "CLOSE", 0, -1),
            _event(12, 9, "OPEN", -1, 0),
        ]
    )
    assert diagnostics["first_open_count"] == 3
    assert diagnostics["subsequently_closed_count"] == 2
    assert diagnostics["not_closed_by_end_count"] == 1
    assert diagnostics["closed_same_frame_count"] == 0
    assert diagnostics["closed_within_1_frame_count"] == 1
    assert diagnostics["closed_within_5_frames_count"] == 2


def test_online_replay_samples_only_already_processed_views():
    views = [object() for _ in range(7)]
    rng = np.random.default_rng(3)
    for _ in range(100):
        selected, index = _sample_training_view(views, rng, 0.33)
        assert selected is views[index]
        assert 0 <= index < len(views)


def test_current_probability_one_always_selects_current_view():
    views = [object() for _ in range(4)]
    rng = np.random.default_rng(9)
    for _ in range(20):
        _selected, index = _sample_training_view(views, rng, 1.0)
        assert index == 3
