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
    load_transition_rows,
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
