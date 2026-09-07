import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.compose_panel10_lifecycle_video import (
    compose_frame,
    event_counts,
    extract_dashboard_panel,
    load_panels,
    load_records,
    ordered_panel_titles,
    PANEL_TITLES,
)


SIDEcars = (
    "main",
    "detector_state",
    "learned_rchange",
    "prediction_mask",
    "sam_feature_diff",
    "depth_difference",
    "cue_types",
)


def _write_png(path: Path, color, size=(8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_dashboard(path: Path, source_size=(16, 8), width=707, height=260) -> list[tuple[int, int, int]]:
    """Write a synthetic 7x2 dashboard with title bands and letterboxed cells."""
    label_height = 44
    colors = [
        (240, 0, 0),
        (0, 200, 0),
        (0, 0, 220),
        (230, 230, 0),
        (230, 0, 230),
        (0, 220, 220),
        (255, 128, 0),
        (128, 0, 255),
        (128, 128, 128),
        (30, 180, 90),
        (90, 30, 180),
        (180, 90, 30),
        (10, 90, 180),
        (180, 10, 90),
    ]
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    col_edges = np.rint(np.linspace(0, width, 8)).astype(int)
    row_edges = np.rint(np.linspace(0, height, 3)).astype(int)
    sw, sh = source_size
    for index, color in enumerate(colors):
        row = index // 7
        col = index % 7
        left, right = int(col_edges[col]), int(col_edges[col + 1])
        top, bottom = int(row_edges[row]), int(row_edges[row + 1])
        # Title band must not leak into extracted content.
        title = Image.new("RGB", (right - left, label_height), (77, 77, 77))
        canvas.paste(title, (left, top))
        content_w = right - left
        content_h = bottom - top - label_height
        source = Image.new("RGB", source_size, color)
        scale = min(content_w / sw, content_h / sh)
        resized = (max(1, min(content_w, int(round(sw * scale)))), max(1, min(content_h, int(round(sh * scale)))))
        letterbox = Image.new("RGB", (content_w, content_h), (0, 0, 0))
        letterbox.paste(source.resize(resized, Image.Resampling.NEAREST), ((content_w - resized[0]) // 2, (content_h - resized[1]) // 2))
        canvas.paste(letterbox, (left, top + label_height))
    canvas.save(path)
    return colors


def _make_complete_run(root: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with (root / "frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    captures = root / "captures"
    for row in rows:
        timestamp = int(row["timestamp"])
        stem = f"{timestamp:06d}"
        summary = {
            "timestamp": timestamp,
            "frame_name": row["frame_name"],
            "opened_now": int(row["base_opened_now"]),
            "closed_now": int(row["base_closed_now"]),
            "da3_seed_opened_now": int(row["seed_opened_now"]),
            "da3_seed_closed_now": int(row["seed_closed_now"]),
        }
        (captures / f"{stem}_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (captures / f"{stem}_summary.json").write_text(json.dumps(summary) + "\n")
        _write_dashboard(captures / f"{stem}_dashboard.png", source_size=(8, 6))
        for i, suffix in enumerate(SIDEcars):
            _write_png(captures / f"{stem}_{suffix}.png", (i * 25, i * 25, i * 25), size=(8, 6))


def test_load_records_maps_columns_and_keeps_contiguous_timestamp_order(tmp_path):
    rows = [
        {
            "scene": "SC1",
            "timestamp": "1",
            "frame_name": "frame_001.png",
            "base_opened_now": "2",
            "base_closed_now": "3",
            "seed_opened_now": "5",
            "seed_closed_now": "7",
            "gt_iou": "0.25",
        },
        {
            "scene": "SC1",
            "timestamp": "0",
            "frame_name": "frame_000.png",
            "base_opened_now": "0",
            "base_closed_now": "0",
            "seed_opened_now": "0",
            "seed_closed_now": "0",
            "gt_iou": "0.5",
        },
    ]
    _make_complete_run(tmp_path, rows)

    records = load_records(tmp_path)

    assert [int(record["timestamp"]) for record in records] == [0, 1]
    assert int(records[0]["base_opened_now"]) == 0
    assert int(records[1]["base_opened_now"]) == 2
    assert int(records[1]["base_closed_now"]) == 3
    assert int(records[1]["seed_opened_now"]) == 5
    assert int(records[1]["seed_closed_now"]) == 7
    assert records[1]["scene"] == "SC1"
    assert float(records[1]["gt_iou"]) == pytest.approx(0.25)


def test_event_counts_sums_per_frame_base_and_seed_events_not_current_stock():
    records = [
        {"base_opened_now": 0, "base_closed_now": 0, "seed_opened_now": 0, "seed_closed_now": 0, "base_open": 999, "seed_open": 999},
        {"base_opened_now": 2, "base_closed_now": 3, "seed_opened_now": 5, "seed_closed_now": 7, "base_open": 999, "seed_open": 999},
    ]

    opens, closes = event_counts(records)

    assert opens == [0, 7]
    assert closes == [0, 10]


@pytest.mark.parametrize("timestamps", [(0, 0), (0, 2)])
def test_load_records_rejects_duplicate_or_missing_timestamps(tmp_path, timestamps):
    rows = [
        {"scene": "SC1", "timestamp": str(t), "frame_name": f"frame_{i}.png", "base_opened_now": "0", "base_closed_now": "0", "seed_opened_now": "0", "seed_closed_now": "0"}
        for i, t in enumerate(timestamps)
    ]
    _make_complete_run(tmp_path, rows)

    with pytest.raises(ValueError, match="contiguous|duplicate|timestamp"):
        load_records(tmp_path)


@pytest.mark.parametrize(
    "csv_patch, summary_patch, error",
    [
        ({"base_opened_now": "-1"}, {}, "nonnegative|negative"),
        ({"base_opened_now": "1.5"}, {}, "integer|invalid"),
        ({}, {"frame_name": "different.png"}, "frame_name|summary"),
    ],
)
def test_load_records_rejects_negative_noninteger_or_mismatched_summary(tmp_path, csv_patch, summary_patch, error):
    rows = [{"scene": "SC1", "timestamp": "0", "frame_name": "frame_000.png", "base_opened_now": "1", "base_closed_now": "0", "seed_opened_now": "0", "seed_closed_now": "0"}]
    _make_complete_run(tmp_path, rows)
    if csv_patch:
        rows[0].update(csv_patch)
        with (tmp_path / "frame_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if summary_patch:
        summary_path = tmp_path / "captures" / "000000_summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(summary_patch)
        summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises((ValueError, FileNotFoundError), match=error):
        load_records(tmp_path)


def test_load_records_rejects_missing_required_capture(tmp_path):
    rows = [{"scene": "SC1", "timestamp": "0", "frame_name": "frame_000.png", "base_opened_now": "0", "base_closed_now": "0", "seed_opened_now": "0", "seed_closed_now": "0"}]
    _make_complete_run(tmp_path, rows)
    (tmp_path / "captures" / "000000_depth_difference.png").unlink()

    with pytest.raises(FileNotFoundError, match="depth_difference"):
        load_records(tmp_path)


def test_extract_dashboard_panel_removes_title_and_letterbox_and_rejects_invalid_index(tmp_path):
    source_size = (16, 8)
    colors = _write_dashboard(tmp_path / "000000_dashboard.png", source_size=source_size, width=707, height=260)
    dashboard = Image.open(tmp_path / "000000_dashboard.png")

    extracted = extract_dashboard_panel(dashboard, 6, source_size)

    assert extracted.size[0] > source_size[0]
    assert extracted.size[1] > source_size[1]
    assert np.asarray(extracted)[0, 0].tolist() == list(colors[6])
    assert not (np.asarray(extracted) == [77, 77, 77]).all(axis=2).any()
    assert not (np.asarray(extracted) == [0, 0, 0]).all(axis=2).any()
    with pytest.raises(ValueError, match="index"):
        extract_dashboard_panel(dashboard, 10, source_size)


def test_load_panels_uses_raw_sidecars_and_dashboard_rgb_q_gt_order(tmp_path):
    rows = [{"scene": "SC1", "timestamp": "0", "frame_name": "frame_000.png", "base_opened_now": "0", "base_closed_now": "0", "seed_opened_now": "0", "seed_closed_now": "0"}]
    _make_complete_run(tmp_path, rows)

    panels = load_panels(tmp_path, 0)

    assert len(panels) == 10
    assert np.asarray(panels[0])[0, 0].tolist() == [0, 0, 0]
    assert np.asarray(panels[1])[0, 0].tolist() == [0, 200, 0]
    assert np.asarray(panels[2])[0, 0].tolist() == [0, 0, 220]
    assert np.asarray(panels[3])[0, 0].tolist() == [25, 25, 25]
    assert np.asarray(panels[4])[0, 0].tolist() == [50, 50, 50]
    assert np.asarray(panels[5])[0, 0].tolist() == [75, 75, 75]
    assert np.asarray(panels[6])[0, 0].tolist() == [255, 128, 0]
    assert np.asarray(panels[7])[0, 0].tolist() == [100, 100, 100]
    assert np.asarray(panels[8])[0, 0].tolist() == [125, 125, 125]
    assert np.asarray(panels[9])[0, 0].tolist() == [150, 150, 150]


def test_compose_frame_places_ten_panels_row_major_and_adds_timeline(tmp_path):
    colors = [(i * 20, 255 - i * 20, 30 + i * 10) for i in range(10)]
    panels = [Image.new("RGB", (12, 8), color) for color in colors]
    record = {"timestamp": 1, "frame_name": "frame_001.png", "scene": "SC1", "gt_iou": 0.5, "gt_f1": 0.66}

    frame = compose_frame(panels, record, [0, 4, 1], [0, 2, 3], width=1920, chart_height=320, boundaries=(1,), segment_names=("A", "B"), run_label="unit")

    assert frame.size[0] == 1920
    arr = np.asarray(frame)
    header, label, gap = 72, 56, 8
    content_height = round((1920 / 5) * panels[0].height / panels[0].width)
    tile_height = label + content_height
    col_centers = [int(round((i + 0.5) * frame.size[0] / 5)) for i in range(5)]
    row_centers = [header + label + content_height // 2, header + tile_height + gap + label + content_height // 2]
    for i, color in enumerate(colors):
        y = row_centers[i // 5]
        x = col_centers[i % 5]
        assert np.abs(arr[y, x].astype(int) - np.array(color)).max() <= 3
    assert arr[-160].max() > 0


def test_compose_frame_rejects_non_ten_panel_input():
    with pytest.raises(ValueError, match="10|ten"):
        compose_frame([Image.new("RGB", (2, 2), "red")] * 9, {"timestamp": 0, "frame_name": "f.png", "scene": "S"}, [0], [0])


def test_requested_reorder_moves_content_and_renumbers_titles(monkeypatch):
    from experiments import compose_panel10_lifecycle_video as compositor

    order = (1, 2, 3, 4, 5, 8, 9, 10, 6, 7)
    colors = [(i * 20, 255 - i * 20, 30 + i * 10) for i in range(10)]
    panels = [Image.new("RGB", (12, 8), color) for color in colors]
    original_contain = compositor.ImageOps.contain
    methods = []

    def capture_resampling(image, size, *, method):
        methods.append(method)
        return original_contain(image, size, method=method)

    monkeypatch.setattr(compositor.ImageOps, "contain", capture_resampling)
    record = {"timestamp": 0, "frame_name": "frame_000.png", "scene": "SC1"}
    frame = compose_frame(panels, record, [1], [0], panel_order=order)
    # Content height=256, tile=312; unchanged header/label/gap dimensions.
    for destination, source in enumerate(order):
        row, col = divmod(destination, 5)
        assert frame.getpixel((col * 384 + 192, 72 + row * 320 + 56 + 128)) == colors[source - 1]
        expected_method = Image.Resampling.NEAREST if source in (6, 7, 10) else Image.Resampling.BILINEAR
        assert methods[destination] == expected_method
    assert [item[0] for item in ordered_panel_titles(order)[5:]] == [
        "6. Signed SAM difference", "7. Reference - DA3 depth", "8. Cue types",
        "9. Predicted change mask", "10. Ground-truth change",
    ]
    assert ordered_panel_titles() == PANEL_TITLES


@pytest.mark.parametrize("order", [tuple(range(10)), (1,) * 10, tuple(range(1, 10))])
def test_panel_order_rejects_missing_duplicate_or_out_of_range_ids(order):
    with pytest.raises(ValueError, match="permutation"):
        compose_frame([Image.new("RGB", (2, 2))] * 10,
                      {"timestamp": 0, "frame_name": "f", "scene": "S"},
                      [0], [0], panel_order=order)
