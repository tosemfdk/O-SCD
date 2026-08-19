import pytest

from experiments.run_online_xfeat_new_seed import (
    FEATURE_CACHE_META_KEY,
    active_seed_render_sign,
    aggregate_across_scenes,
    aggregate_metric_scope,
    validate_feature_cache,
    write_seed_comparison_artifacts,
    seed_only_coverage_loss,
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
