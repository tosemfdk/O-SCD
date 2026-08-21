import csv
import json
from pathlib import Path

import numpy as np

from experiments.plot_binary_state_lifespan_results import (
    COMPARISON_FILES,
    PER_RUN_PLOTS,
    compare_runs,
    main,
    plot_run,
)


def _write_run(path: Path, *, miou: float = 0.5, f1: float = 0.6, opens=(1, 0, 0), closes=(0, 1, 0), reopens=(0, 0, 1)):
    path.mkdir(parents=True)
    summary = {
        "runtime_seconds": 12.5,
        "run_config": {"open_probability": 0.6, "close_probability": 0.4},
        "metrics": {
            "mean_frame_iou": miou,
            "mean_frame_f1": f1,
            "precision": 0.7,
            "recall": 0.8,
            "pre_opt": {"mean_frame_iou": miou - 0.1, "mean_frame_f1": f1 - 0.1},
            "post_opt": {"mean_frame_iou": miou, "mean_frame_f1": f1},
            "open_frame_pre_post": {"count": 1, "mean_delta_iou": 0.1, "mean_delta_f1": 0.1},
        },
        "open_count": int(sum(opens)),
        "close_count": int(sum(closes)),
        "reopen_count": int(sum(reopens)),
        "final_active_gs": 3,
        "base_max_drift": 0.0,
        "closed_slot_max_drift": 0.0,
        "inactive_gradient_first_step_violations": 0.0,
        "active_to_active_false_split_count": 0,
        "reused_slot_violations": 0,
        "cuda_peak_memory_bytes": 123,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    for t in range(3):
        rows.append({
            "timestamp": t,
            "iou": miou + 0.01 * t,
            "f1": f1 + 0.01 * t,
            "pre_iou": miou - 0.1,
            "pre_f1": f1 - 0.1,
            "pre_predicted_positive_fraction": 0.1 + t * 0.01,
            "post_predicted_positive_fraction": 0.2 + t * 0.01,
            "open_count": opens[t],
            "close_count": closes[t],
            "keep_count": t,
            "uncertain_count": 2 - t,
            "reopen_count": reopens[t],
            "active_lifespan_count": 1 + t,
        })
    with (path / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    stat = np.asarray([
        [10, 0.2, 0.1, 0.2, 0.3],
        [10, 0.5, 0.4, 0.5, 0.6],
        [10, 0.8, 0.7, 0.8, 0.9],
    ], dtype=np.float64)
    np.savez_compressed(
        path / "per_frame_binary_state_stats.npz",
        timestamp=np.arange(3),
        p_active=stat,
        p_flip=stat * 0.5,
        p_01=stat * 0.25,
        p_10=stat * 0.125,
        open_count=np.asarray(opens),
        close_count=np.asarray(closes),
        keep_count=np.arange(3),
        uncertain_count=np.asarray([2, 1, 0]),
        reopen_count=np.asarray(reopens),
        active_lifespan_count=np.arange(1, 4),
        pre_predicted_positive_fraction=np.asarray([0.1, 0.11, 0.12]),
        post_predicted_positive_fraction=np.asarray([0.2, 0.21, 0.22]),
        frame_runtime_seconds=np.ones(3),
        cuda_peak_memory_bytes=np.ones(3, dtype=np.int64),
    )


def test_plot_run_writes_required_pngs(tmp_path: Path):
    run = tmp_path / "run"
    _write_run(run)

    result = plot_run(run, boundaries=(1,))

    assert [Path(p).name for p in result["plots"]] == list(PER_RUN_PLOTS)
    for name in PER_RUN_PLOTS:
        path = run / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_compare_runs_writes_plot_and_machine_readable_outputs(tmp_path: Path):
    dc = tmp_path / "dc"
    geo = tmp_path / "geo"
    out = tmp_path / "cmp"
    _write_run(dc, miou=0.4, f1=0.5)
    _write_run(geo, miou=0.6, f1=0.7)

    comparison = compare_runs(dc, geo, output_dir=out)

    assert sorted(p.name for p in out.iterdir()) == sorted(COMPARISON_FILES)
    assert comparison["conditions"][0]["label"] == "direct-binary DC-only"
    assert comparison["conditions"][1]["mIoU"] == 0.6
    assert (out / "dc_only_vs_all_geometry.png").stat().st_size > 0
    loaded = json.loads((out / "comparison.json").read_text())
    assert loaded["conditions"][0]["open_count"] == 1


def test_compare_runs_derives_legacy_beam_counts_from_frame_metrics(tmp_path: Path):
    dc = tmp_path / "dc"
    geo = tmp_path / "geo"
    beam = tmp_path / "beam"
    out = tmp_path / "cmp"
    _write_run(dc)
    _write_run(geo)
    _write_run(beam, opens=(2, 3, 4), closes=(0, 5, 6), reopens=(0, 0, 7))
    summary_path = beam / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("open_count")
    summary.pop("close_count")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    comparison = compare_runs(
        dc,
        geo,
        output_dir=out,
        beam2_summary=summary_path,
    )

    legacy = comparison["conditions"][2]
    assert legacy["open_count"] == 9
    assert legacy["close_count"] == 11
    assert legacy["final_active_gs"] == 3


def test_plot_cli_single_run_and_comparison(tmp_path: Path):
    dc = tmp_path / "dc"
    geo = tmp_path / "geo"
    _write_run(dc)
    _write_run(geo)

    assert main(["--run-dir", str(dc), "--boundaries", "1"]) == 0
    assert (dc / "binary_state_filter_metrics.png").exists()
    out = tmp_path / "comparison"
    assert main(["--dc-only-dir", str(dc), "--all-geometry-dir", str(geo), "--output-dir", str(out)]) == 0
    assert (out / "comparison.md").exists()
