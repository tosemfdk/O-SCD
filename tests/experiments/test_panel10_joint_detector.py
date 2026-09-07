from types import SimpleNamespace

import pytest
import torch

from experiments.panel10_seed_topology import append_typed_seeds
from experiments.view_bayesian_detector_steps import BayesianDetectorReplay, parse_args
from temporal.new_seed_gaussians import NewSeedGaussianModel


def make_replay(device="cpu", *, max_rows=12):
    replay = BayesianDetectorReplay.__new__(BayesianDetectorReplay)
    replay.args = parse_args([])
    replay.args.training_partition = "panel10_new"
    replay.args.da3_max_rows = max_rows
    replay.args.filter_chunk_size = 3  # One chunk straddles the base/seed boundary.
    replay.args.cue_mode = "soft"
    replay.args.cue_scale = 1.0
    replay.device = torch.device(device)
    replay.base = NewSeedGaussianModel(device=device)
    replay.base.append(
        xyz=torch.tensor([[0., 0., 3.], [0.5, 0., 3.]], device=device),
        start=0., scaling=torch.full((2, 3), -2., device=device), opacity=0.9,
    )
    replay.count = 2
    replay.current_index = -1
    replay.latest_open_rows = torch.zeros(2, device=device, dtype=torch.bool)
    replay.latest_close_rows = torch.zeros_like(replay.latest_open_rows)
    replay.pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    replay.evidence_background = torch.zeros(3, device=device)
    replay.records = [SimpleNamespace(name=f"frame{i}") for i in range(36)]
    replay._reset_detector_state()
    replay._initialize_representation_state()
    return replay


def append_seed(replay, timestamp, *, x=0., z=2.):
    return append_typed_seeds(
        replay, torch.tensor([[x, 0., z]], device=replay.device),
        torch.full((1, 3), -2., device=replay.device), timestamp,
    )


def test_panel10_uses_one_tracker_with_base_prefix_and_seed_capacity():
    replay = make_replay()
    assert replay.tracker.stable_a.numel() == replay.count + replay.args.da3_max_rows
    assert replay.seed_tracker is None
    before = {n: getattr(replay.tracker, n).clone() for n in replay.tracker.topology_buffer_names}
    append_seed(replay, 0)
    # Birth initializes geometry/lifespan only, not a second detector observation.
    for name, value in before.items():
        assert torch.equal(getattr(replay.tracker, name), value)


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
@pytest.mark.parametrize("max_rows", [0, 12])
@pytest.mark.parametrize("first_bf", [None, 10.0])
@pytest.mark.parametrize("pixel_cue,gaussian_cue", [("unchanged", "unchanged"), ("binary_q05", "unchanged"), ("unchanged", "binary_q05")])
def test_birth_precedes_single_shared_evidence_and_updates_each_row_once(monkeypatch, chunk_size, max_rows, first_bf, pixel_cue, gaussian_cue):
    replay = make_replay(max_rows=max_rows)
    replay.args.filter_chunk_size = chunk_size
    replay.args.first_open_bayes_factor_threshold = first_bf
    replay.args.detector_pixel_cue = pixel_cue
    replay.args.detector_gaussian_cue = gaussian_cue
    events = []
    view = SimpleNamespace(candidate_map=torch.ones(1, 2, 2))
    monkeypatch.setattr(replay, "_load_view", lambda _: view)

    def new_target(timestamp, current, cue):
        assert replay.current_index == timestamp and replay.current_view is current
        events.append("target")
        return cue * 0.5

    def birth(timestamp):
        events.append("birth")
        count = append_seed(replay, timestamp) if timestamp == 0 else 0
        return dict(proposed=count, accepted=count, coverage_rejected=0)

    def accumulate(current, probe, _pipe, _bg, cue, **kwargs):
        events.append("evidence")
        assert probe.get_xyz.shape == (3, 3)
        assert torch.equal(cue, view.candidate_map)  # Full Q, never NEW target.
        assert kwargs['cue_mode'] == ('binary' if pixel_cue == 'binary_q05' else 'soft')
        assert kwargs['count_mode'] == ('capped_binary' if gaussian_cue == 'binary_q05' else 'capped')
        assert replay.accepted_da3_birth_global == [0]
        positive = torch.full((3,), float(cue.mean()))
        return SimpleNamespace(delta_a=positive, delta_b=1. - positive, total_mass=torch.ones(3))

    def train(timestamp):
        events.append("train")
        assert replay.tracker.last_timestamp[:3].tolist() == [timestamp] * 3
        assert replay.tracker.visible_observations[:3].tolist() == [timestamp + 1] * 3
        assert (replay.tracker.last_timestamp[3:] < 0).all()
        return dict(updates=0, latest_branch_count=0, sampled_oldest=None,
                    sampled_newest=None, historical_lifespan_replay_updates=0,
                    lifespan_render_violations=0, loss=0., trainable_open_rows=0,
                    visible_seed_rows=0, visible_pending_seed_rows=0, future_view_accesses=0)

    monkeypatch.setattr(replay, "_new_seed_target", new_target)
    monkeypatch.setattr(replay, "_append_current_da3_seeds", birth)
    monkeypatch.setattr(replay, "_accumulate_change_evidence", accumulate, raising=False)
    monkeypatch.setattr(replay, "_train_representation", train)
    monkeypatch.setattr(replay, "_update_da3_seed_detector", lambda *a, **kw: pytest.fail("separate seed detector ran"))
    states = []
    for timestamp in range(36):
        events.clear()
        view.candidate_map.fill_(0. if 12 <= timestamp < 24 else 1.)
        replay.step()
        assert events == ["target", "birth", "evidence", "train"]
        assert replay._projected_evidence_score().shape == (2,)
        assert replay._bayes_factor_progress_score().shape == (2,)
        base_active = replay.lifecycle.active_mask(timestamp)
        seed_active = replay.seed_lifecycle.active_mask(timestamp)
        assert base_active.tolist() == seed_active.tolist() * 2
        states.append(bool(seed_active[0]))
    assert any(states[:12]) and not states[23] and states[-1]
    assert states.index(True) == (2 if first_bf is None else 1)
    assert replay.lifecycle.num_states.tolist() == [2, 2]
    assert replay.seed_lifecycle.num_states.tolist() == [2]
    assert replay._seed_visual_state().open.tolist() == [True]


def test_binary_detector_thresholds_normalized_pixels_only_and_preserves_soft_training():
    from temporal.change_evidence import cue_to_change_probability

    replay = make_replay()
    replay.args.cue_scale = 2.0
    replay.args.detector_pixel_cue = 'binary_q05'
    q = torch.tensor([[[0.2, 0.499, 0.5, 0.501, 0.8]]])
    view = SimpleNamespace(candidate_map=2 * q)
    before = view.candidate_map.clone()
    cue, mode, threshold, scale = replay._detector_cue_inputs(view)
    binary = cue_to_change_probability(cue, mode=mode, threshold=threshold, scale=scale)
    assert binary.tolist() == [[[0., 0., 0., 1., 1.]]]
    assert torch.equal(replay._normalized_cue_target(view), q)
    assert torch.equal(view.candidate_map, before)


def test_gaussian_binary_cli_requires_soft_pixels_and_does_not_modify_q():
    common = ['--cue-mode', 'soft', '--detector-gaussian-cue', 'binary_q05']
    args = parse_args(common)
    assert args.detector_pixel_cue == 'unchanged'
    for bad in (['--cue-mode', 'binary'], ['--detector-pixel-cue', 'binary_q05']):
        with pytest.raises(SystemExit):
            parse_args(common + bad)
    replay = make_replay()
    replay.args = args
    q = torch.tensor([[[0.2, 0.49, 0.51, 0.8]]])
    view = SimpleNamespace(candidate_map=2*q)
    cue, mode, _, scale = replay._detector_cue_inputs(view)
    assert cue is view.candidate_map and mode == 'soft' and scale == 2.
    assert torch.equal(replay._normalized_cue_target(view), q)


def test_joint_probe_keeps_closed_but_excludes_retired_and_future_rows():
    replay = make_replay()
    for t in (0, 0, 0):
        append_seed(replay, t)
    replay.seed_lifecycle.open_rows([0, 1], 0)
    replay.seed_model.open_rows(torch.tensor([0, 1]), 0)
    replay.seed_lifecycle.close_rows([0, 1], 1)
    replay.seed_model.close_rows(torch.tensor([0, 1]), 1)
    replay.seed_retired[1] = True
    probe, rows = replay._joint_detector_probe(timestamp=1)
    assert rows.tolist() == [0, 1, 2, 4]  # CLOSED row0 survives; retired row1 does not occlude.
    assert probe.get_xyz.shape == (4, 3)
    expected = {n: getattr(probe, n).clone() for n in ("get_xyz", "get_opacity", "get_scaling", "get_rotation")}
    with torch.no_grad():
        replay.change_dc.fill_(999.)
        for p in replay.seed_model.parameters():
            p.add_(17.)
    actual, _ = replay._joint_detector_probe(timestamp=1)
    for name, value in expected.items():
        assert torch.equal(getattr(actual, name), value)
        assert not getattr(actual, name).requires_grad
    replay.accepted_da3_birth_global[2] = 2
    with pytest.raises(ValueError, match="future"):
        replay._joint_detector_probe(timestamp=1)


@pytest.mark.parametrize("max_rows", [0, 12])
def test_reset_clears_joint_seed_suffix_and_restarts_birth_indices(monkeypatch, max_rows):
    replay = make_replay(max_rows=max_rows)
    append_seed(replay, 0)
    replay.tracker.update(torch.ones(3), torch.zeros(3), total_mass=torch.ones(3),
                          current_active=torch.zeros(3, dtype=torch.bool),
                          row_indices=torch.arange(3), timestamp=0)
    monkeypatch.setattr(replay, "_load_view", lambda _: SimpleNamespace())
    replay.reset()
    assert replay.current_index == -1
    assert replay.seed_tracker is None
    assert replay.seed_model.num_gaussians == replay.seed_lifecycle.count == 0
    assert replay.accepted_da3_birth_global == []
    assert replay.tracker.visible_observations.eq(0).all()
    assert replay.tracker.last_timestamp.eq(-1).all()
    assert replay.tracker.stable_a.shape == (replay.count + replay.args.da3_max_rows,)
    append_seed(replay, 0)
    _, rows = replay._joint_detector_probe(timestamp=0)
    assert rows.tolist() == [0, 1, 2]


@pytest.mark.cuda
def test_cuda_new_seed_reduces_base_evidence_in_same_joint_render():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("diff_gaussian_rasterization_fastgs")
    from experiments.temporal_lifespan_smoke import build_synthetic_temporal_scene
    from temporal.change_evidence import accumulate_change_evidence

    replay = make_replay("cuda")
    _, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    q = torch.ones((1, 32, 32), device="cuda")
    base_only = accumulate_change_evidence(camera, replay.base, pipe, background, q, cue_mode="soft", cue_scale=1.)
    append_seed(replay, timestamp=0)
    with torch.no_grad():
        replay.seed_detector_probe._opacity.fill_(8.)
        replay.seed_detector_probe._scaling.fill_(-0.8)
    probe, rows = replay._joint_detector_probe(timestamp=0)
    joint = accumulate_change_evidence(camera, probe, pipe, background, q, cue_mode="soft", cue_scale=1.)
    assert rows.tolist() == [0, 1, 2]
    assert joint.total_mass[0] < base_only.total_mass[0] * 0.5
    assert joint.total_mass[2] > 0


@pytest.mark.cuda
def test_cuda_pixel_binarization_retains_alpha_t_weighting_and_one_view_mass_cap():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("diff_gaussian_rasterization_fastgs")
    from experiments.temporal_lifespan_smoke import build_synthetic_temporal_scene
    from temporal.change_evidence import accumulate_change_evidence

    replay = make_replay("cuda")
    _, camera, pipe, background = build_synthetic_temporal_scene(image_size=32)
    q = torch.full((1, 32, 32), 0.8, device='cuda')
    q[..., ::2] = 0.2
    soft = accumulate_change_evidence(camera, replay.base, pipe, background, q,
        cue_mode='soft', cue_scale=1., count_mode='capped')
    binary = accumulate_change_evidence(camera, replay.base, pipe, background, q,
        cue_mode='binary', cue_threshold=0.5, count_mode='capped')
    assert torch.equal(binary.cue, (q[0] > 0.5).float())
    assert torch.allclose(binary.total_mass, soft.total_mass, rtol=1e-5, atol=1e-6)
    assert bool(((binary.e_plus > 0) & (binary.e_minus > 0)).all())
    assert bool((binary.delta_a + binary.delta_b <= 1.000001).all())
