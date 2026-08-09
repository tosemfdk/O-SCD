import numpy as np
import torch
from PIL import Image

from experiments.render_temporal_confusion_maps import PALETTE, confusion_arrays, metrics_from_counts, save_gif


def test_requested_palette_leaves_tn_uncolored_and_uses_blue_for_fn():
    assert PALETTE["tn"] == (0, 0, 0)
    assert PALETTE["fn"] == (0, 90, 255)


def test_confusion_arrays_assign_requested_colors_and_counts():
    pred = torch.tensor([[[0.5, 0.0], [1.0, 0.49]]], dtype=torch.float32)
    gt = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)

    image, counts = confusion_arrays(pred, gt, threshold=0.5)

    assert counts == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "pred_positive": 2,
        "gt_positive": 2,
        "pixels": 4,
    }
    assert tuple(image[0, 0]) == PALETTE["tp"]
    assert tuple(image[0, 1]) == PALETTE["tn"]
    assert tuple(image[1, 0]) == PALETTE["fp"]
    assert tuple(image[1, 1]) == PALETTE["fn"]


def test_metrics_from_counts_aggregates_binary_segmentation_scores():
    metrics = metrics_from_counts({"tp": 2, "tn": 5, "fp": 1, "fn": 3})

    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 5)
    assert np.isclose(metrics["iou"], 2 / 6)
    assert np.isclose(metrics["f1"], 0.5)
    assert np.isclose(metrics["accuracy"], 7 / 11)


def test_save_gif_preserves_frame_order(tmp_path):
    paths = []
    for index, color in enumerate((PALETTE["tp"], PALETTE["fn"])):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (8, 6), color).save(path)
        paths.append(path)

    output = tmp_path / "result.gif"
    save_gif(paths, output, width=8, duration_ms=100)

    with Image.open(output) as gif:
        assert gif.n_frames == 2
        gif.seek(0)
        assert gif.convert("RGB").getpixel((0, 0)) == PALETTE["tp"]
        gif.seek(1)
        assert gif.convert("RGB").getpixel((0, 0)) == PALETTE["fn"]
