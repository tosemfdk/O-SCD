import pytest

from experiments.run_online_xfeat_new_seed import (
    FEATURE_CACHE_META_KEY,
    active_new_constrains_xfeat_anchor,
    active_new_uses_density,
    active_new_uses_pruning,
    active_new_uses_seed_only_dc,
    active_seed_render_sign,
    aggregate_across_scenes,
    aggregate_metric_scope,
    object_scope_prediction,
    new_sidecar_projected_dc_loss,
    validate_feature_cache,
    write_e4d_comparison_artifacts,
    write_seed_comparison_artifacts,
    seed_only_coverage_loss,
    _e4d_training_schedule,
    e4d_full_target,
)


COUNT_NAMES = ("tp", "tn", "fp", "fn", "pred_positive", "gt_positive", "pixels")


def metric_row(branch, scope, *, tp, tn, fp, fn):
    counts = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_positive": tp + fp,
        "gt_positive": tp + fn,
        "pixels": tp + tn + fp + fn,
    }
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    out = {f"{branch}_{scope}_{key}": value for key, value in counts.items()}
    out.update(
        {
            f"{branch}_{scope}_precision": precision,
            f"{branch}_{scope}_recall": recall,
            f"{branch}_{scope}_iou": iou,
            f"{branch}_{scope}_f1": f1,
            f"{branch}_{scope}_accuracy": (tp + tn) / counts["pixels"],
        }
    )
    return out


def test_metric_aggregation_and_no_seed_equivalence():
    rows = []
    for counts in ((2, 5, 1, 2), (1, 7, 2, 0)):
        row = {}
        row.update(metric_row("baseline", "full", tp=counts[0], tn=counts[1], fp=counts[2], fn=counts[3]))
        row.update(metric_row("seed", "full", tp=counts[0], tn=counts[1], fp=counts[2], fn=counts[3]))
        rows.append(row)

    baseline = aggregate_metric_scope(rows, "baseline", "full")
    seeded = aggregate_metric_scope(rows, "seed", "full")
    assert {key: baseline[key] for key in COUNT_NAMES} == {
        key: seeded[key] for key in COUNT_NAMES
    }
    assert baseline["iou"] == seeded["iou"]
    assert baseline["f1"] == seeded["f1"]

    reports = [{"metric_scopes": {"baseline": {"full": baseline}, "seed": {"full": seeded}}}]
    combined = aggregate_across_scenes(reports, "seed", "full")
    assert combined["tp"] == 3
    assert combined["fp"] == 3
    assert combined["fn"] == 2


def test_feature_cache_configuration_is_explicit_and_mismatch_is_fatal():
    cache = {}
    validate_feature_cache(cache, top_k=512, width=528, height=941)
    assert cache[FEATURE_CACHE_META_KEY]["top_k"] == 512
    validate_feature_cache(cache, top_k=512, width=528, height=941)
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_feature_cache(cache, top_k=4096, width=528, height=941)


def test_legacy_nonempty_feature_cache_without_metadata_is_rejected():
    cache = {"scene2:frame.png": {}}
    with pytest.raises(ValueError, match="no configuration metadata"):
        validate_feature_cache(cache, top_k=512, width=528, height=941)


def test_neutral_gate_pauses_birth_but_keeps_active_seed_rendering():
    assert active_seed_render_sign("+", "+", 3) == "+"
    assert active_seed_render_sign(None, "+", 3) == "+"
    assert active_seed_render_sign(None, None, 0) is None
    with pytest.raises(RuntimeError, match="without an owning PCA sign"):
        active_seed_render_sign(None, None, 1)


def test_seed_comparison_artifacts_include_visual_table(tmp_path):
    rows = []
    for branch in ("baseline", "seed"):
        row = metric_row(branch, "full", tp=3, tn=7, fp=1, fn=2)
        rows.append((branch, aggregate_metric_scope([row], branch, "full")))
    report = {
        "scene": 1,
        "frames": 1,
        "seed_branch_contract": {"seed_count": 2},
        "metric_scopes": {
            branch: {
                scope: dict(metric, scope=scope)
                for scope in ("full", "all", "geometry", "new", "remove")
            }
            for branch, metric in rows
        },
    }

    outputs = write_seed_comparison_artifacts([report], tmp_path)

    assert (tmp_path / "baseline_vs_xfeat_new_seed_comparison.csv").is_file()
    assert (tmp_path / "baseline_vs_xfeat_new_seed_comparison.md").is_file()
    assert (tmp_path / "baseline_vs_xfeat_new_seed_comparison.png").is_file()
    assert outputs["visual_table"].endswith("baseline_vs_xfeat_new_seed_comparison.png")


def test_seed_only_coverage_loss_keeps_gradient_independent_of_base_memory():
    import torch

    target = torch.zeros((1, 4, 4))
    target[:, 1:3, 1:3] = 1.0
    seed_logits = torch.zeros((3, 4, 4), requires_grad=True)

    loss, parts = seed_only_coverage_loss(target, seed_logits)
    loss.backward()

    assert seed_logits.grad is not None
    assert seed_logits.grad[:, 1:3, 1:3].abs().sum() > 0
    assert parts["positive"] > 0


def test_e4d_replay_is_current_biased_and_never_selects_future():
    replay = [{"timestamp": value} for value in range(9)]
    selected = _e4d_training_schedule(replay, 12)
    assert [row["timestamp"] for row in selected[::3]] == [8, 8, 8, 8]
    assert sum(row["timestamp"] == 8 for row in selected) == 4
    assert all(row["timestamp"] <= 8 for row in selected)


def test_e4d_full_target_is_signed_cue_intersection_without_gt():
    import torch
    from types import SimpleNamespace

    cue = torch.ones((1, 4, 4))
    cue[:, :, 0] = 0.1
    view = SimpleNamespace(image_height=4, image_width=4, candidate_map=cue)
    signed = torch.ones((2, 2), dtype=torch.bool)
    target, stable = e4d_full_target(view, signed)
    assert target.shape == (1, 4, 4)
    assert not bool(target[:, :, 0].any())
    assert bool(target[:, :, 1:].all())
    assert bool(stable[:, 0].all())


def test_new_sidecar_projected_dc_loss_trains_only_dc_without_base_rows():
    import math
    from types import SimpleNamespace

    import torch

    from temporal.active_new_gaussians import ActiveNewGaussianModel

    model = ActiveNewGaussianModel(device="cpu")
    model.append_xfeat_anchors(
        xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        start=0.0,
        scaling=torch.full((1, 3), math.log(0.1)),
        opacity=0.5,
    )
    view = SimpleNamespace(
        image_height=16,
        image_width=16,
        FoVx=math.radians(60.0),
        FoVy=math.radians(60.0),
        world_view_transform=torch.eye(4),
    )
    target = torch.zeros((1, 16, 16))
    target[:, 6:11, 6:11] = 1.0

    loss, parts = new_sidecar_projected_dc_loss(
        model, view, target, timestamp=0
    )
    loss.backward()

    assert parts["visible_rows"] == 1
    assert parts["cue_target_mean"] > 0.0
    assert model.new_dc.grad is not None
    assert float(model.new_dc.grad.abs().sum()) > 0.0
    for parameter in (model._xyz, model._opacity, model._scaling, model._rotation):
        assert parameter.grad is None


def test_object_metrics_select_matching_prediction_bank():
    import numpy as np

    full = np.array([[True, True, True]])
    new = np.array([[True, False, False]])
    remove = np.array([[False, True, False]])

    assert np.array_equal(
        object_scope_prediction(
            "new_full",
            full_prediction=full,
            new_prediction=new,
            remove_prediction=remove,
        ),
        new,
    )
    assert np.array_equal(
        object_scope_prediction(
            "remove_full",
            full_prediction=full,
            new_prediction=new,
            remove_prediction=remove,
        ),
        remove,
    )
    assert np.array_equal(
        object_scope_prediction(
            "full",
            full_prediction=full,
            new_prediction=new,
            remove_prediction=remove,
        ),
        full,
    )


def test_e4e_variant_factorial_contract():
    assert {
        variant: (
            active_new_uses_seed_only_dc(variant),
            active_new_constrains_xfeat_anchor(variant),
            active_new_uses_density(variant),
            active_new_uses_pruning(variant),
        )
        for variant in ("C0", "C1", "C2", "C3")
    } == {
        "C0": (False, False, True, True),
        "C1": (True, False, True, True),
        "C2": (False, True, True, True),
        "C3": (True, True, True, True),
    }
    assert active_new_uses_density("D2")
    assert active_new_uses_pruning("D3")
    assert not active_new_uses_seed_only_dc("D3")


def test_e4e_comparison_artifacts_include_sidecar_scope(tmp_path):
    metric = {
        "frames": 1,
        "tp": 3,
        "tn": 7,
        "fp": 1,
        "fn": 2,
        "precision": 0.75,
        "recall": 0.6,
        "iou": 0.5,
        "f1": 2.0 / 3.0,
    }
    report = {
        "scene": 1,
        "e4d_metric_scopes": {
            variant: {
                scope: metric
                for scope in (
                    "full",
                    "new_full",
                    "remove_full",
                    "sidecar_alpha_new_full",
                )
            }
            for variant in ("C0", "C1", "C2", "C3")
        },
    }

    outputs = write_e4d_comparison_artifacts(
        [report], tmp_path, ("C0", "C1", "C2", "C3")
    )

    assert (tmp_path / "C0_C1_C2_C3_comparison.csv").is_file()
    assert "sidecar_alpha_new_full" in (
        tmp_path / "C0_C1_C2_C3_comparison.md"
    ).read_text()
    assert outputs["plot"].endswith("C0_C1_C2_C3_comparison.png")
