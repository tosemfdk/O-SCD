import csv

from PIL import Image
import pytest

from experiments.view_saved_panel10_results import LAYERS, capture_path, load_records


def make_run(root, timestamps=(1, 0)):
    rows = [dict(instance='Instance_1', scene='Cantina', timestamp=t) for t in timestamps]
    with (root/'frame_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        for suffix in LAYERS.values():
            path = capture_path(root, 'Instance_1/Cantina', row, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (4, 2), 'red').save(path)
    return rows


def test_saved_viewer_sorts_frames_and_uses_exact_capture_paths(tmp_path):
    make_run(tmp_path)
    rows = load_records(tmp_path)['Instance_1/Cantina']
    assert [int(row['timestamp']) for row in rows] == [0, 1]
    assert capture_path(tmp_path, 'Instance_1/Cantina', rows[1], 'prediction_mask').name == '000001_prediction_mask.png'


def test_saved_viewer_rejects_missing_images(tmp_path):
    rows = make_run(tmp_path)
    capture_path(tmp_path, 'Instance_1/Cantina', rows[0], 'cue_types').unlink()
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path)


@pytest.mark.parametrize('timestamps', [(0, 0), (0, 2)])
def test_saved_viewer_rejects_duplicate_or_missing_frames(tmp_path, timestamps):
    make_run(tmp_path, timestamps)
    with pytest.raises(ValueError, match='non-contiguous'):
        load_records(tmp_path)
