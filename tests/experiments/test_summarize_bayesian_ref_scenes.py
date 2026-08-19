import csv
import json
from pathlib import Path

from experiments.summarize_bayesian_ref_scenes import main, summarize_scene


def _write_run(path: Path, scene: str, iou: float, f1: float):
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "frames": 2,
                "algorithm": "map_reset",
                "optimized_parameters": ["dc"],
                "event_count": 3,
                "reopen_count": 0,
                "active_to_active_false_split_count": 0,
                "runtime_seconds": 1.5,
                "cuda_peak_memory_bytes": 10,
                "run_arguments": {"source_path": f"data/Instance_1/{scene}"},
                "metrics": {
                    "aggregate_iou": iou,
                    "aggregate_f1": f1,
                    "precision": 0.7,
                    "recall": 0.8,
                    "mean_frame_iou": iou / 2,
                    "mean_frame_f1": f1 / 2,
                },
            }
        ),
        encoding="utf-8",
    )
    with (path / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["active_lifespan_count", "open_count", "keep_count", "predicted_positive_fraction"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"active_lifespan_count": 1, "open_count": 1, "keep_count": 0, "predicted_positive_fraction": 0.1},
                {"active_lifespan_count": 2, "open_count": 0, "keep_count": 2, "predicted_positive_fraction": 0.2},
            ]
        )


def test_summarize_scene_reads_metrics_and_frame_counts(tmp_path: Path):
    run = tmp_path / "sc1"
    _write_run(run, "scene_change1", 0.3, 0.4)

    row = summarize_scene(run)

    assert row["scene"] == "scene_change1"
    assert row["aggregate_iou"] == 0.3
    assert row["mean_active_lifespan_count"] == 1.5
    assert row["total_keep_count"] == 2.0


def test_comparison_main_writes_csv_json_png(tmp_path: Path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write_run(run1, "scene_change1", 0.3, 0.4)
    _write_run(run2, "scene_change2", 0.5, 0.6)
    out = tmp_path / "summary"

    assert main([str(run1), str(run2), "--labels", "sc1", "sc2", "--output-dir", str(out)]) == 0
    assert (out / "comparison_metrics.csv").is_file()
    assert (out / "comparison_summary.json").is_file()
    assert (out / "comparison_metrics.png").is_file()
    payload = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    assert [row["scene"] for row in payload["scenes"]] == ["sc1", "sc2"]
