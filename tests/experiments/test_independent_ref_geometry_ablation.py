from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.run_independent_ref_geometry_ablation import (
    aggregate_metrics,
    aggregate_scene_summaries,
    binary_metrics,
    build_one_state_model,
    independent_dataset_audit,
    scene_ranges_from_manifest,
)
from experiments.train_geometry_temporal_rchange import make_geometry_optimizer


def test_scene_ranges_follow_manifest_order() -> None:
    manifest = {
        "sequence_order": ["scene_change1", "scene_change2", "scene_change3"],
        "counts": {
            "inference_images": 304,
            "per_source_scene": {
                "scene_change1": 95,
                "scene_change2": 104,
                "scene_change3": 105,
            },
        },
    }

    assert scene_ranges_from_manifest(manifest) == {
        "scene_change1": (0, 95),
        "scene_change2": (95, 199),
        "scene_change3": (199, 304),
    }


def test_scene_ranges_reject_inconsistent_total() -> None:
    manifest = {
        "sequence_order": ["scene_change1"],
        "counts": {
            "inference_images": 2,
            "per_source_scene": {"scene_change1": 1},
        },
    }

    with pytest.raises(ValueError, match="expected 2"):
        scene_ranges_from_manifest(manifest)


def test_independent_dataset_audit_is_exact() -> None:
    assert independent_dataset_audit(104, 120) == {
        "expected_dataset_frames": 104,
        "actual_dataset_frames": 104,
        "actual_train_records": 104,
        "actual_train_views": 104,
        "expected_exact_total_updates": 12480,
        "dataset_contract_errors": [],
        "dataset_contract_passed": True,
        "independent_reference_start": True,
    }


def test_dc_opacity_matches_dc_only_initialization_and_freezes_geometry() -> None:
    base = SimpleNamespace(
        _xyz=nn.Parameter(torch.randn(3, 3)),
        _features_dc=nn.Parameter(torch.randn(3, 1, 3)),
        _features_rest=nn.Parameter(torch.randn(3, 2, 3)),
        _opacity=nn.Parameter(torch.randn(3, 1)),
        _scaling=nn.Parameter(torch.randn(3, 3)),
        _rotation=nn.Parameter(F.normalize(torch.randn(3, 4), dim=-1)),
        opacity_activation=torch.sigmoid,
        scaling_activation=torch.exp,
        rotation_activation=F.normalize,
    )
    expected_dc = base._features_dc.detach().clone()
    expected_opacity = torch.sigmoid(base._opacity.detach()).clone()
    expected_xyz = base._xyz.detach().clone()

    model = build_one_state_model(base, "dc_opacity")
    parameters = dict(model.state_parameter_items())
    attributes = model.get_active_render_attributes(0.0)

    assert torch.equal(model.state_change_dc[:, 0].detach(), expected_dc)
    assert torch.count_nonzero(model.state_opacity_delta) == 0
    assert torch.equal(attributes["opacity"].detach(), expected_opacity)
    assert torch.equal(attributes["xyz"].detach(), expected_xyz)
    assert {
        name: parameter.requires_grad for name, parameter in parameters.items()
    } == {
        "dc": True,
        "xyz": False,
        "opacity": True,
        "scaling": False,
        "rotation": False,
    }
    optimizer = make_geometry_optimizer(
        model,
        SimpleNamespace(
            dc_lr=0.0025,
            xyz_lr=0.0,
            opacity_lr=0.025,
            scaling_lr=0.0,
            rotation_lr=0.0,
        ),
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "dc",
        "opacity",
    ]


def test_binary_and_aggregate_metrics_match_manual_counts() -> None:
    pred = torch.tensor([[[0.8, 0.8], [0.1, 0.1]]])
    gt = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    counts, metrics, confusion = binary_metrics(pred, gt, threshold=0.5)

    assert counts == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "pred_positive": 2,
        "gt_positive": 2,
        "pixels": 4,
    }
    assert metrics == {
        "iou": pytest.approx(1 / 3),
        "f1": pytest.approx(0.5),
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
    }
    assert tuple(confusion[0, 0]) == (0, 200, 0)
    assert tuple(confusion[0, 1]) == (255, 105, 180)
    assert tuple(confusion[1, 0]) == (0, 90, 255)
    assert np.array_equal(confusion[1, 1], np.zeros(3, dtype=np.uint8))

    aggregate = aggregate_metrics([{**counts, **metrics}, {**counts, **metrics}])
    assert aggregate["frames"] == 2
    assert aggregate["mean_frame_iou"] == pytest.approx(1 / 3)
    assert aggregate["aggregate_iou"] == pytest.approx(1 / 3)
    assert aggregate["mean_frame_f1"] == pytest.approx(0.5)
    assert aggregate["aggregate_f1"] == pytest.approx(0.5)


def test_aggregate_scene_summaries_uses_frame_weighted_means() -> None:
    rows = [
        {
            "condition": "dc_only",
            "frames": 1,
            "tp": 1,
            "tn": 3,
            "fp": 0,
            "fn": 0,
            "pred_positive": 1,
            "gt_positive": 1,
            "pixels": 4,
            "precision": 1.0,
            "recall": 1.0,
            "aggregate_iou": 1.0,
            "aggregate_f1": 1.0,
            "mean_frame_iou": 1.0,
            "mean_frame_f1": 1.0,
            "post_train_ssf_loss": 0.2,
            "runtime_seconds": 2.0,
        },
        {
            "condition": "dc_only",
            "frames": 3,
            "tp": 3,
            "tn": 9,
            "fp": 3,
            "fn": 0,
            "pred_positive": 6,
            "gt_positive": 3,
            "pixels": 15,
            "precision": 0.5,
            "recall": 1.0,
            "aggregate_iou": 0.5,
            "aggregate_f1": 2 / 3,
            "mean_frame_iou": 0.5,
            "mean_frame_f1": 2 / 3,
            "post_train_ssf_loss": 0.4,
            "runtime_seconds": 4.0,
        },
    ]

    overall = aggregate_scene_summaries(rows, "dc_only")

    assert overall["frames"] == 4
    assert overall["tp"] == 4
    assert overall["fp"] == 3
    assert overall["precision"] == pytest.approx(4 / 7)
    assert overall["recall"] == pytest.approx(1.0)
    assert overall["mean_frame_iou"] == pytest.approx(0.625)
    assert overall["mean_frame_f1"] == pytest.approx(0.75)
    assert overall["post_train_ssf_loss"] == pytest.approx(0.35)
    assert overall["runtime_seconds"] == pytest.approx(6.0)
