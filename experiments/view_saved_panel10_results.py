"""Read-only Viser browser for completed Panel10 evaluation captures.

No model, detector or optimizer is initialized. Navigation displays the exact
saved arrival-time images and metrics, rather than rerendering later GS state.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import viser

from experiments.view_bayesian_detector_steps import letterbox_image

DEFAULT_ROOT = Path('outputs/paslcd_panel10_gaussian_binary_20260907')
CONDITIONS = {
    'First OPEN BF10 | Gaussian binary': 'first10_gaussian_binary',
    'BF30 | Gaussian binary': 'bf30_gaussian_binary',
}
LAYERS = {
    'All panels': 'dashboard',
    '6. Prediction mask': 'prediction_mask',
    '5. Learned R_change': 'learned_rchange',
    '10. NEW / REMOVE / appearance': 'cue_types',
    '1. Gaussian lifecycle': 'main',
    '4. Bayes factor': 'detector_state',
    '8. Signed SAM': 'sam_feature_diff',
    '9. Depth difference': 'depth_difference',
}


def load_records(root: Path) -> dict[str, list[dict[str, str]]]:
    scenes: dict[str, list[dict[str, str]]] = {}
    with (root / 'frame_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            for field in ('instance', 'scene'):
                if Path(row[field]).name != row[field] or row[field] in ('.', '..'):
                    raise ValueError(f'invalid scene component: {row[field]}')
            key = f'{row["instance"]}/{row["scene"]}'
            scenes.setdefault(key, []).append(row)
    if not scenes:
        raise ValueError(f'no completed frames in {root}')
    for key, rows in scenes.items():
        rows.sort(key=lambda row: int(row['timestamp']))
        if [int(row['timestamp']) for row in rows] != list(range(len(rows))):
            raise ValueError(f'non-contiguous/duplicate frames: {key}')
        for row in rows:
            for suffix in LAYERS.values():
                if not capture_path(root, key, row, suffix).is_file():
                    raise FileNotFoundError(capture_path(root, key, row, suffix))
    return scenes


def capture_path(root: Path, scene: str, row: dict[str, str], suffix: str) -> Path:
    return root / scene / 'captures' / f'{int(row["timestamp"]):06d}_{suffix}.png'


class SavedPanel10Viewer:
    def __init__(self, root: Path, *, host: str = '127.0.0.1', port: int = 8091):
        self.root = root
        self.runs = {label: load_records(root / name) for label, name in CONDITIONS.items()}
        self.lock = threading.RLock()
        self.image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(dark_mode=True, control_layout='collapsible',
                                        control_width='medium', show_share_button=False)
        self.server.gui.add_markdown('## PASLCD — saved evaluation viewer\n'
            'Gaussian aggregation → binary observation.\n\n'
            '**Read-only:** no retraining or detector updates. Panel3/loss/birth used soft Q.')
        self.condition = self.server.gui.add_dropdown('Condition', tuple(CONDITIONS))
        scenes = tuple(self.runs[self.condition.value])
        self.scene = self.server.gui.add_dropdown('Scene', scenes)
        count = len(self.runs[self.condition.value][self.scene.value])
        self.frame = self.server.gui.add_slider('Frame', min=1, max=count, step=1,
                                               initial_value=min(15, count))
        self.previous = self.server.gui.add_button('Previous frame')
        self.next = self.server.gui.add_button('Next frame')
        self.layer = self.server.gui.add_dropdown('View', tuple(LAYERS))
        self.status = self.server.gui.add_markdown('Loading saved frame...')
        self.condition.on_update(self._change_scene)
        self.scene.on_update(self._change_scene)
        self.frame.on_update(self.refresh)
        self.layer.on_update(self.refresh)
        self.previous.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))

        @self.server.on_client_connect
        def connect(client):
            self._push(client)
            client.camera.on_update(lambda _: self._push(client))

        self.refresh()
        print(f'READY http://localhost:{port} | 2 conditions / 20 scenes / 500 frames each', flush=True)

    def _change_scene(self, _=None):
        with self.lock:
            scenes = tuple(self.runs[self.condition.value])
            self.scene.options = scenes
            if self.scene.value not in scenes:
                self.scene.value = scenes[0]
            count = len(self.runs[self.condition.value][self.scene.value])
            self.frame.max = count
            self.frame.value = min(self.frame.value, count)
            self.refresh()

    def _move(self, offset: int):
        with self.lock:
            self.frame.value = max(1, min(int(self.frame.max), self.frame.value + offset))
            self.refresh()

    def refresh(self, _=None):
        with self.lock:
            run = self.root / CONDITIONS[self.condition.value]
            rows = self.runs[self.condition.value][self.scene.value]
            row = rows[int(self.frame.value) - 1]
            path = capture_path(run, self.scene.value, row, LAYERS[self.layer.value])
            with Image.open(path) as image:
                self.image = np.asarray(image.convert('RGB')).copy()
            self.previous.disabled = self.frame.value == 1
            self.next.disabled = self.frame.value == len(rows)
            self.status.content = (
                f'### {self.scene.value} — {self.frame.value}/{len(rows)}\n'
                f'`{row["frame_name"]}`\n\n'
                f'**Frame IoU {float(row["gt_iou"]):.4f} · F1 {float(row["gt_f1"]):.4f}**\n\n'
                f'Base OPEN: {int(row["open"]):,} · Seed OPEN: {int(row["active_da3_seed_rows"]):,}\n\n'
                f'Seed archive: {int(row["seed_archived"]):,}\n\n'
                '**Saved arrival-time result; browsing does not change state.**'
            )
            for client in self.server.get_clients().values():
                self._push(client)
            print('VIEW', json.dumps({'condition': self.condition.value, 'scene': self.scene.value,
                  'frame': self.frame.value, 'layer': self.layer.value, 'file': str(path)}), flush=True)

    def _push(self, client):
        with self.lock:
            aspect = float(client.camera.aspect)
            if not math.isfinite(aspect) or aspect <= 0:
                aspect = 16 / 9
            fitted = letterbox_image(self.image, width=1920,
                                     height=max(192, min(4096, round(1920 / aspect))))
            client.scene.set_background_image(fitted, format='jpeg', jpeg_quality=95)

    def run(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    SavedPanel10Viewer(args.root, host=args.host, port=args.port).run()


if __name__ == '__main__':
    main()
