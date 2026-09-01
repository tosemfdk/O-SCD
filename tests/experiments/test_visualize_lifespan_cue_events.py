import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.visualize_lifespan_cue_events import (
    CLOSE_COLOR,
    OPEN_COLOR,
    compose_frame_panel,
    event_probe_tensors,
    load_evaluation_mask_rgb,
    load_transition_rows,
    open_close_timeline_chart,
    threshold_render_rgb,
    turbo_heatmap,
)


def test_transition_rows_include_only_exact_timestamp_open_and_close(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    events = [
        {"decision_timestamp": 4, "gaussian_index": 7, "action": "OPEN"},
        {"decision_timestamp": 4, "gaussian_index": 8, "action": "KEEP"},
        {"decision_timestamp": 5, "gaussian_index": 7, "action": "CLOSE"},
        {"decision_timestamp": 6, "gaussian_index": 9, "action": "OPEN"},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    grouped = load_transition_rows(path)

    assert grouped == {
        4: {"OPEN": (7,), "CLOSE": ()},
        5: {"OPEN": (), "CLOSE": (7,)},
        6: {"OPEN": (9,), "CLOSE": ()},
    }


def test_event_probe_colors_only_current_open_and_close_rows():
    colors, selected = event_probe_tensors(
        5,
        open_rows=[1],
        close_rows=[3],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert selected.tolist() == [False, True, False, True, False]
    assert torch.equal(colors[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(colors[3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.count_nonzero(colors[[0, 2, 4]]).item() == 0


def test_event_probe_rejects_same_row_open_and_close_and_bad_indices():
    kwargs = dict(gaussian_count=3, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ValueError, match="cannot OPEN and CLOSE"):
        event_probe_tensors(open_rows=[1], close_rows=[1], **kwargs)
    with pytest.raises(IndexError, match="out-of-range"):
        event_probe_tensors(open_rows=[3], close_rows=[], **kwargs)


def test_panel_uses_turbo_cue_and_exact_event_palette():
    cue = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    heat = turbo_heatmap(cue)
    assert heat.shape == (2, 2, 3)
    assert len({tuple(pixel) for pixel in heat.reshape(-1, 3)}) == 4

    rgb = np.zeros((4, 3, 3), dtype=np.uint8)
    raw = np.zeros_like(rgb)
    event = np.zeros_like(rgb)
    event[0, 0] = OPEN_COLOR
    event[1, 1] = CLOSE_COLOR
    panel = compose_frame_panel(
        rgb,
        np.zeros((4, 3), dtype=np.float32),
        raw,
        event,
        timestamp=95,
        segment="scene_change2",
        open_count=1,
        close_count=1,
        panel_width=30,
    )
    assert panel.width == 30 * 4 + 8 * 3
    assert panel.height > 30


def test_threshold_render_uses_channel_mean_at_half():
    raw = np.array(
        [[[255, 255, 0], [255, 0, 0], [128, 128, 127], [127, 127, 128]]],
        dtype=np.uint8,
    )

    thresholded = threshold_render_rgb(raw, threshold=0.5)

    assert thresholded.tolist() == [
        [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]]
    ]


def test_saved_evaluation_mask_is_used_instead_of_rethresholding_raw(tmp_path: Path):
    raw = np.full((2, 2, 3), 255, dtype=np.uint8)
    saved = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    from PIL import Image

    Image.fromarray(saved).save(path)

    rendered, source = load_evaluation_mask_rgb(
        path,
        raw_render=raw,
        threshold=0.5,
    )

    assert source == "saved_metric_prediction"
    assert rendered[..., 0].tolist() == saved.tolist()
    assert np.array_equal(rendered[..., 0], rendered[..., 1])
    assert np.array_equal(rendered[..., 1], rendered[..., 2])


def test_extended_panel_adds_threshold_mask_and_absolute_count_timeline():
    panel_width = 70
    total_width = panel_width * 5 + 8 * 4
    chart = open_close_timeline_chart(
        [0, 4, 2, 7],
        [0, 1, 3, 2],
        current_timestamp=2,
        width=total_width,
        height=180,
        boundaries=[2],
        segment_names=["SC1", "SC2"],
    )
    rgb = np.zeros((4, 3, 3), dtype=np.uint8)
    panel = compose_frame_panel(
        rgb,
        np.zeros((4, 3), dtype=np.float32),
        rgb,
        rgb,
        timestamp=2,
        segment="scene_change2",
        open_count=2,
        close_count=3,
        panel_width=panel_width,
        thresholded_render=rgb,
        lifecycle_chart=chart,
    )

    assert chart.size == (total_width, 180)
    assert panel.width == total_width
    assert panel.height > chart.height
