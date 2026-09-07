import csv
import json
from pathlib import Path

from PIL import Image

from experiments.analyze_signed_lifespan_object_density import (
    ObjectSpec,
    analyze_events,
    load_event_rows,
    load_mask_series,
    main,
)


def _write_mask(path: Path, pixels: list[tuple[int, int]], size: tuple[int, int] = (4, 4)) -> None:
    image = Image.new("L", size, 0)
    for x, y in pixels:
        image.putpixel((x, y), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_coordinate_scaling_and_sign_action_aggregation(tmp_path: Path):
    new_dir = tmp_path / "new"
    removed_dir = tmp_path / "removed"
    _write_mask(new_dir / "000.png", [(1, 1)])
    _write_mask(removed_dir / "000.png", [(2, 2)])
    rows = [
        {"frame_name": "000.png", "projected_x": 2, "projected_y": 2, "width": 8, "height": 8, "density_score": 0.7, "action": "split", "generation": 0},
        {"frame_name": "000.png", "projected_x": 4, "projected_y": 4, "width": 8, "height": 8, "density_score": -0.4, "action": "child_prune", "generation": 1},
        {"frame_name": "000.png", "projected_x": 4, "projected_y": 4, "width": 8, "height": 8, "density_score": 0.2, "action": "clone", "generation": 0},
    ]
    summary = analyze_events(
        rows,
        [
            load_mask_series(ObjectSpec("new", new_dir, 1)),
            load_mask_series(ObjectSpec("removed", removed_dir, -1)),
        ],
    )
    assert summary["objects"]["new"]["inside_event_count"] == 1
    assert summary["objects"]["new"]["correct_sign_count"] == 1
    assert summary["objects"]["new"]["positive_growth_count"] == 1
    assert summary["objects"]["removed"]["inside_event_count"] == 2
    assert summary["objects"]["removed"]["correct_sign_count"] == 1
    assert summary["objects"]["removed"]["wrong_sign_count"] == 1
    assert summary["objects"]["removed"]["negative_suppress_count"] == 1
    assert summary["objects"]["removed"]["child_generation_count"] == 1


def test_empty_masks_and_events_are_reported(tmp_path: Path):
    mask_dir = tmp_path / "obj"
    _write_mask(mask_dir / "000.png", [])
    summary = analyze_events(
        [{"timestamp": 0, "x": 1, "y": 1, "density_score": 1, "action": "split"}],
        [load_mask_series(ObjectSpec("obj", mask_dir, 1))],
    )
    assert summary["event_count"] == 1
    assert summary["matched_inside_event_rows"] == 0
    assert summary["objects"]["obj"]["inside_event_count"] == 0
    assert summary["objects"]["obj"]["sign_correct_fraction"] is None


def test_cli_loads_csv_and_writes_json_and_per_frame_csv(tmp_path: Path):
    mask_dir = tmp_path / "obj"
    _write_mask(mask_dir / "frame_a.png", [(0, 0)], size=(2, 2))
    events = tmp_path / "events.csv"
    with events.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "u", "v", "signed_support", "event"])
        writer.writeheader()
        writer.writerow({"image_name": "frame_a.png", "u": 0, "v": 0, "signed_support": -0.5, "event": "suppress"})
    out_json = tmp_path / "summary.json"
    out_csv = tmp_path / "frames.csv"
    main([
        "--events",
        str(events),
        "--output-json",
        str(out_json),
        "--output-csv",
        str(out_csv),
        "--object",
        f"removed={mask_dir}:-",
    ])
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["objects"]["removed"]["correct_sign_count"] == 1
    assert out_csv.read_text(encoding="utf-8").splitlines()[0].startswith("object,frame_name")


def test_jsonl_loader(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"timestamp": 0, "x": 1}\n\n{"timestamp": 1, "x": 2}\n', encoding="utf-8")
    assert len(load_event_rows(path)) == 2


def test_optional_prediction_mask_object_support(tmp_path: Path):
    mask_dir = tmp_path / "obj"
    pred_dir = tmp_path / "pred"
    _write_mask(mask_dir / "frame_a.png", [(0, 0), (1, 0)], size=(2, 2))
    _write_mask(pred_dir / "frame_a.png", [(0, 0), (0, 1)], size=(2, 2))
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    out_json = tmp_path / "summary.json"
    main([
        "--events",
        str(events),
        "--output-json",
        str(out_json),
        "--prediction-mask-dir",
        str(pred_dir),
        "--object",
        f"new={mask_dir}:+",
    ])
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    pred = summary["prediction_masks"]["objects"]["new"]
    assert pred["object_pixels"] == 2
    assert pred["predicted_inside_object_pixels"] == 1
    assert pred["predicted_object_fraction"] == 0.5
