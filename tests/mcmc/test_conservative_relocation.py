from types import SimpleNamespace

import pytest
import torch

from temporal.conservative_relocation import (
    ConservativeRowSnapshot,
    conservative_acceptance,
    initialize_row_local_adam,
    residual_candidates_from_views,
    restore_rows_,
    row_local_adam_step_,
    select_conservative_sources,
    select_residual_destinations,
    source_deactivation_safety,
    transport_sources_,
)
from temporal.mcmc_state import FixedCapacityChangeState

from .conftest import make_base


def _camera(*, center_x: float, width: int = 100, height: int = 100, focal: float = 50.0):
    world_view = torch.eye(4)
    world_view[3, 0] = -float(center_x)
    return SimpleNamespace(
        image_width=width,
        image_height=height,
        FoVx=2.0 * torch.atan(torch.tensor(width / (2.0 * focal))).item(),
        FoVy=2.0 * torch.atan(torch.tensor(height / (2.0 * focal))).item(),
        world_view_transform=world_view,
        camera_center=torch.tensor([center_x, 0.0, 0.0]),
    )


def test_source_selection_is_observation_and_support_based_not_opacity_based():
    observations = torch.tensor([0, 12, 15, 20], dtype=torch.int32)
    cue_support = torch.tensor([0, 0, 2, 0], dtype=torch.int32)
    protected = torch.zeros(4, dtype=torch.bool)
    tentative = torch.zeros(4, dtype=torch.bool)

    selected = select_conservative_sources(
        observation_count=observations,
        cue_support_count=cue_support,
        protected_mask=protected,
        tentative_mask=tentative,
        min_observations=10,
        max_sources=8,
        seed=0,
    )

    assert selected.tolist() == [1, 3]


def test_source_selection_excludes_historical_and_tentative_slots():
    selected = select_conservative_sources(
        observation_count=torch.tensor([20, 20, 20, 20]),
        cue_support_count=torch.zeros(4, dtype=torch.int32),
        protected_mask=torch.tensor([False, True, False, False]),
        tentative_mask=torch.tensor([False, False, True, False]),
        min_observations=3,
        max_sources=8,
        seed=0,
    )

    assert selected.tolist() == [0, 3]


def test_multiview_residual_rays_triangulate_a_disconnected_destination():
    first = _camera(center_x=0.0)
    second = _camera(center_x=1.0)
    residual_first = torch.zeros(1, 100, 100)
    residual_second = torch.zeros(1, 100, 100)
    # World point [0, 0, 5] projects to (50, 50) and (40, 50).
    residual_first[0, 50, 50] = 1.0
    residual_second[0, 50, 40] = 1.0

    candidates = residual_candidates_from_views(
        [first, second],
        [residual_first, residual_second],
        per_view_topk=1,
        residual_threshold=0.5,
        min_support_views=2,
        max_ray_distance=0.05,
        scene_bounds=(torch.tensor([-2.0, -2.0, 1.0]), torch.tensor([2.0, 2.0, 8.0])),
        max_candidates=16,
    )

    assert candidates.xyz.shape[0] == 1
    assert candidates.support_views.tolist() == [2]
    assert torch.allclose(candidates.xyz[0], torch.tensor([0.0, 0.0, 5.0]), atol=0.12)


def test_destination_requires_multiview_support():
    camera = _camera(center_x=0.0)
    residual = torch.zeros(1, 100, 100)
    residual[0, 50, 50] = 1.0

    candidates = residual_candidates_from_views(
        [camera],
        [residual],
        per_view_topk=1,
        residual_threshold=0.5,
        min_support_views=2,
        max_ray_distance=0.05,
        scene_bounds=(torch.tensor([-2.0, -2.0, 1.0]), torch.tensor([2.0, 2.0, 8.0])),
        max_candidates=16,
    )

    assert candidates.xyz.shape == (0, 3)


def test_destination_selection_filters_single_view_candidates():
    candidates = SimpleNamespace(
        xyz=torch.tensor([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        scores=torch.tensor([0.9, 0.95]),
        support_views=torch.tensor([3, 1]),
        ray_distance=torch.tensor([0.01, 0.01]),
    )

    chosen = select_residual_destinations(
        candidates,
        count=1,
        min_support_views=2,
        seed=0,
    )

    assert torch.equal(chosen, torch.tensor([0]))


def test_transport_is_fixed_capacity_and_fully_reversible():
    state = FixedCapacityChangeState.from_gaussians(make_base(4), capacity=4, base_zero=True)
    source = torch.tensor([1, 3])
    destination = torch.tensor([[10.0, 0.0, 2.0], [11.0, 0.0, 2.0]])
    before_ids = {name: id(param) for name, param in state.current_parameter_items()}
    snapshot = ConservativeRowSnapshot.capture(state, source)

    transport_sources_(state, source, destination, tentative_opacity=0.005, global_step=7)
    state.removal_protected_mask[source] = True

    assert state.capacity == 4
    assert {name: id(param) for name, param in state.current_parameter_items()} == before_ids
    assert torch.equal(state.current_xyz[source], destination)
    assert torch.equal(state.current_features_dc[source], torch.zeros_like(state.current_features_dc[source]))
    assert torch.allclose(
        torch.sigmoid(state.current_raw_change_opacity[source]),
        torch.full((2, 1), 0.005),
        atol=1e-7,
    )
    assert state.tentative_mask[source].all()
    assert not state.cue_support_mask[source].any()

    restore_rows_(state, snapshot)

    assert snapshot.matches(state)


def test_deactivation_safety_protects_zero_dc_occluder_when_negative_mass_rises():
    decision = source_deactivation_safety(
        pre_energy=1.0,
        deactivated_energy=0.99,
        pre_negative_mass=0.10,
        deactivated_negative_mass=0.11,
        max_relative_energy_increase=0.0,
        max_negative_mass_increase=0.0,
    )

    assert not decision.safe
    assert decision.reason == "source_occlusion_responsibility"
    assert decision.negative_mass_delta == pytest.approx(0.01)


def test_deactivation_safety_allows_truly_unused_zero_dc_slot():
    decision = source_deactivation_safety(
        pre_energy=1.0,
        deactivated_energy=0.99,
        pre_negative_mass=0.10,
        deactivated_negative_mass=0.10,
        max_relative_energy_increase=0.0,
        max_negative_mass_increase=0.0,
    )

    assert decision.safe
    assert decision.reason == "source_removal_safe"


def test_row_local_adam_stages_attributes_and_never_updates_unselected_rows():
    parameters = {
        "features_dc": torch.nn.Parameter(torch.zeros(5, 1, 3)),
        "raw_change_opacity": torch.nn.Parameter(torch.zeros(5, 1)),
        "xyz": torch.nn.Parameter(torch.zeros(5, 3)),
    }
    selected = torch.tensor([1, 3])
    learning_rates = {name: 0.1 for name in parameters}
    local = initialize_row_local_adam(parameters, selected)

    sum(parameter.sum() for parameter in parameters.values()).backward()
    row_local_adam_step_(
        parameters,
        selected,
        learning_rates,
        local,
        allowed_names=("features_dc", "raw_change_opacity"),
    )

    assert torch.count_nonzero(parameters["features_dc"][[0, 2, 4]]) == 0
    assert torch.count_nonzero(parameters["raw_change_opacity"][[0, 2, 4]]) == 0
    assert torch.count_nonzero(parameters["features_dc"][selected]) > 0
    assert torch.count_nonzero(parameters["raw_change_opacity"][selected]) > 0
    assert torch.count_nonzero(parameters["xyz"]) == 0
    assert local.steps["xyz"] == 0

    for parameter in parameters.values():
        parameter.grad = None
    sum(parameter.sum() for parameter in parameters.values()).backward()
    row_local_adam_step_(parameters, selected, learning_rates, local)

    assert torch.count_nonzero(parameters["xyz"][[0, 2, 4]]) == 0
    assert torch.count_nonzero(parameters["xyz"][selected]) > 0
    assert local.steps["xyz"] == 1


def test_row_local_burnin_changes_are_exactly_reversible():
    state = FixedCapacityChangeState.from_gaussians(make_base(4), capacity=4, base_zero=True)
    selected = torch.tensor([1, 3])
    snapshot = ConservativeRowSnapshot.capture(state, selected)
    parameters = dict(state.current_parameter_items())
    local = initialize_row_local_adam(parameters, selected)
    learning_rates = {name: 0.1 for name in parameters}

    sum(parameter.sum() for parameter in parameters.values()).backward()
    row_local_adam_step_(parameters, selected, learning_rates, local)
    assert not snapshot.matches(state)

    restore_rows_(state, snapshot)
    assert snapshot.matches(state)


@pytest.mark.parametrize(
    "pre_energy,post_energy,pre_coverage,post_coverage,accepted,reason",
    [
        (10.0, 9.8, 0.3, 0.4, True, "accepted"),
        (10.0, 10.2, 0.3, 0.4, False, "energy_increase"),
        (10.0, 9.9, 0.6, 0.2, False, "coverage_drop"),
    ],
)
def test_conservative_acceptance_requires_energy_and_coverage(
    pre_energy,
    post_energy,
    pre_coverage,
    post_coverage,
    accepted,
    reason,
):
    decision = conservative_acceptance(
        pre_energy=pre_energy,
        post_energy=post_energy,
        pre_coverage=pre_coverage,
        post_coverage=post_coverage,
        max_relative_energy_increase=0.01,
        min_coverage_gain=0.0,
    )

    assert decision.accepted is accepted
    assert decision.reason == reason


def test_conservative_acceptance_strict_defaults_reject_zero_gain_and_energy_increase():
    zero_gain = conservative_acceptance(
        pre_energy=10.0,
        post_energy=9.9,
        pre_coverage=0.3,
        post_coverage=0.3,
        max_relative_energy_increase=0.0,
        min_coverage_gain=1e-6,
    )
    energy_increase = conservative_acceptance(
        pre_energy=10.0,
        post_energy=10.0001,
        pre_coverage=0.3,
        post_coverage=0.4,
        max_relative_energy_increase=0.0,
        min_coverage_gain=1e-6,
    )

    assert not zero_gain.accepted
    assert zero_gain.reason == "coverage_drop"
    assert not energy_increase.accepted
    assert energy_increase.reason == "energy_increase"


def test_conservative_acceptance_rejects_negative_region_leakage():
    decision = conservative_acceptance(
        pre_energy=10.0,
        post_energy=9.9,
        pre_coverage=0.3,
        post_coverage=0.4,
        max_relative_energy_increase=0.0,
        min_coverage_gain=1e-6,
        pre_negative_mass=0.1,
        post_negative_mass=0.11,
        max_negative_mass_increase=0.0,
    )

    assert not decision.accepted
    assert decision.reason == "negative_mass_increase"
