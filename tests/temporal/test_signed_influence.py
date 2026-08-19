from types import SimpleNamespace

import torch
from torch import nn

from experiments.render_state_signed_influence_multiview import role_colors
from temporal import (
    TemporalChangeModel,
    classify_influence,
    forced_state_render_attributes,
    opacity_removal_influence,
    split_signed_influence,
)


def make_base(n: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        _xyz=nn.Parameter(torch.zeros(n, 3)),
        _features_dc=nn.Parameter(torch.zeros(n, 1, 3)),
        _features_rest=nn.Parameter(torch.zeros(n, 2, 3)),
        _opacity=nn.Parameter(torch.zeros(n, 1)),
        _scaling=nn.Parameter(torch.zeros(n, 3)),
        _rotation=nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)),
        get_xyz=torch.zeros(n, 3),
        get_opacity=torch.full((n, 1), 0.5),
        get_scaling=torch.ones(n, 3),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
    )


def test_forced_state_attributes_ignore_existing_validity_gate():
    model = TemporalChangeModel.from_gaussians(make_base(), max_states=2)
    with torch.no_grad():
        model.state_change_dc[:, 1].fill_(2.0)
        model.state_valid[:, 1] = False

    attributes = forced_state_render_attributes(model, 1)

    assert torch.equal(attributes["dc"], torch.full((2, 1, 3), 2.0))
    assert torch.equal(attributes["opacity"], torch.full((2, 1), 0.5))


def test_opacity_removal_influence_preserves_additive_and_occluding_signs():
    opacity = torch.tensor([[0.5], [0.25]], requires_grad=True)
    scalar = 2.0 * opacity[0, 0] - 4.0 * opacity[1, 0]
    rendered = scalar.expand(3, 1, 1)

    signed = opacity_removal_influence(rendered, opacity)
    additive, occluding = split_signed_influence(signed)

    assert torch.equal(signed, torch.tensor([1.0, -1.0]))
    assert torch.equal(additive, torch.tensor([1.0, 0.0]))
    assert torch.equal(occluding, torch.tensor([0.0, 1.0]))


def test_classify_influence_keeps_both_roles():
    result = classify_influence(
        additive=torch.tensor([2.0, 0.0, 1.0, 0.0]),
        occluding=torch.tensor([0.0, 3.0, 2.0, 0.0]),
        contributing_views=torch.tensor([2, 2, 2, 2]),
        min_mean_influence=0.5,
        min_views=2,
        view_count=2,
    )

    assert result["additive_mask"].tolist() == [True, False, True, False]
    assert result["occluding_mask"].tolist() == [False, True, True, False]
    assert result["both_mask"].tolist() == [False, False, True, False]
    assert result["valid_mask"].tolist() == [True, True, True, False]


def test_soft_binary_influence_keeps_signed_roles_near_mask_threshold():
    opacity = torch.tensor([[0.5], [0.25]], requires_grad=True)
    scalar = 0.5 + 0.1 * opacity[0, 0] - 0.2 * opacity[1, 0]
    rendered = scalar.expand(3, 1, 1)

    signed = opacity_removal_influence(
        rendered,
        opacity,
        mask_threshold=0.5,
        mask_temperature=0.05,
    )

    assert signed[0] > 0
    assert signed[1] < 0


def test_candidate_opacity_scores_insertion_from_zero_opacity_baseline():
    effective = torch.tensor([[0.0]], requires_grad=True)
    candidate = torch.tensor([[0.75]])
    rendered = (-2.0 * effective[0, 0]).expand(3, 1, 1)

    signed = opacity_removal_influence(
        rendered,
        effective,
        influence_opacity=candidate,
    )

    assert torch.equal(signed, torch.tensor([-1.5]))


def test_multiview_role_colors_match_viewer_legend():
    colors = role_colors(
        additive=torch.tensor([True, False, True, False]),
        occluding=torch.tensor([False, True, True, False]),
    )

    assert torch.allclose(colors[0], torch.tensor([1.0, 0.20, 0.05]))
    assert torch.allclose(colors[1], torch.tensor([0.05, 0.45, 1.0]))
    assert torch.allclose(colors[2], torch.tensor([0.85, 0.10, 0.85]))
    assert torch.equal(colors[3], torch.zeros(3))
