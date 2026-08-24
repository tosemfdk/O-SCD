import json
from pathlib import Path

import pytest

from experiments.summarize_ref_sc1_change_cue_density import (
    aggregate_summaries,
    load_summaries,
)


def _write_summary(root: Path, seed: int, condition: str, k: int | None, miou: float, splits: int) -> None:
    output = root / f"seed{seed}" / (f"cue_vcd_k{k}" if k else condition)
    output.mkdir(parents=True)
    summary = {
        "scope": "ref -> scene_change1 only",
        "condition": condition,
        "k_views": k,
        "gt_used_in_causal_loop": False,
        "future_view_access_count": 0,
        "immutable_reference_unchanged": True,
        "topology_integrity": {"passed": True},
        "initial_gaussian_count": 100,
        "final_gaussian_count": 100 + splits,
        "total_clones": 0,
        "total_splits": splits,
        "total_pruned": 0,
        "runtime_seconds": 10.0 + seed,
        "peak_cuda_memory_bytes": 2 * 1024**3,
        "metrics": {
            "mean_frame_iou": miou,
            "mean_frame_f1": miou + 0.1,
            "precision": 0.8,
            "recall": 0.9,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregate_repeats_and_compare_cue_gate_to_fastgs_gradient(tmp_path: Path):
    for seed, offset in ((0, 0.0), (1, 0.02), (2, 0.04)):
        _write_summary(tmp_path, seed, "fastgs_gradient_only", None, 0.60 + offset, 300)
        _write_summary(tmp_path, seed, "cue_vcd", 10, 0.62 + offset, 200)
    comparison = aggregate_summaries(load_summaries(tmp_path))
    assert comparison["summary_count"] == 6
    cue = next(row for row in comparison["conditions"] if row["condition"] == "cue_vcd_k10")
    assert cue["metrics"]["mIoU"]["mean"] == pytest.approx(0.64)
    assert cue["metrics"]["splits"]["mean"] == pytest.approx(200.0)
    delta = comparison["comparisons"]["cue_vcd_k10"]
    assert delta["mean_mIoU_delta_vs_fastgs_gradient_only"] == pytest.approx(0.02)
    assert delta["mean_split_reduction_fraction_vs_fastgs_gradient_only"] == pytest.approx(1 / 3)


def test_failed_causal_audit_is_rejected(tmp_path: Path):
    _write_summary(tmp_path, 0, "cue_vcd", 1, 0.6, 10)
    path = next(tmp_path.glob("seed*/*/summary.json"))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["future_view_access_count"] = 1
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="future-view"):
        load_summaries(tmp_path)
