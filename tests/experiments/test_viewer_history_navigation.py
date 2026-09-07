from types import SimpleNamespace
import threading

import numpy as np

from experiments.view_bayesian_detector_steps import (
    BayesianDetectorViewer,
    CueTypeStats,
    DISPLAY_LIFECYCLE,
    ViewerFrameSnapshot,
)


class Handle:
    def __init__(self, *, value=None):
        self.disabled = False
        self.value = value
        self.content = ""

    def on_click(self, callback):
        self.callback = callback
        return callback

    on_update = on_click


class Replay:
    current_index = -1
    total_frames = 2
    da3_seeds = None
    sam_model = None
    step_calls = 0
    reset_calls = 0

    @property
    def has_next(self):
        return self.current_index + 1 < self.total_frames

    def step(self):
        if not self.has_next:
            raise StopIteration
        self.step_calls += 1
        self.current_index += 1

    def reset(self):
        self.reset_calls += 1
        self.current_index = -1


def make_viewer():
    viewer = BayesianDetectorViewer.__new__(BayesianDetectorViewer)
    viewer.replay = Replay()
    viewer.lock = threading.Lock()
    viewer.display = DISPLAY_LIFECYCLE
    viewer.args = SimpleNamespace(
        representation_updates=1, cue_mode="soft", cue_remap="learned_sigmoid"
    )
    viewer.server = SimpleNamespace(
        on_client_connect=lambda callback: callback, get_clients=lambda: {}
    )
    viewer.gui_previous = Handle()
    viewer.gui_next = Handle()
    viewer.gui_reset = Handle()
    viewer.gui_display = Handle(value=DISPLAY_LIFECYCLE)
    viewer.gui_status = Handle()
    viewer.gui_cue_types = Handle()
    viewer._frame_history = {}
    viewer._history_panels = None
    viewer._viewed_index = -1
    viewer._client_aspects = {}
    viewer.da3_seed_stats = SimpleNamespace(accepted_so_far=0, born_now=0)
    viewer.sam_feature_diff_stats = SimpleNamespace(normalization_scale=0.0)
    viewer.depth_difference_stats = SimpleNamespace(
        alignment_scale=1., alignment_inliers=0, alignment_samples=0,
        median_absolute_difference=0., normalization_scale=0.,
    )
    viewer.cue_type_stats = CueTypeStats()

    def refresh():
        value = 20 * (viewer.replay.current_index + 2)
        for name in (
            "main_image", "input_image", "cue_image", "detector_state_image",
            "learned_change_image", "predicted_change_mask_image", "gt_change_image",
            "sam_feature_diff_image", "depth_difference_image", "cue_type_image",
        ):
            setattr(viewer, name, np.full((2, 3, 3), value, dtype=np.uint8))
        viewer.gui_status.content = f"status-{viewer.replay.current_index}"
        viewer.gui_cue_types.content = f"cue-{viewer.replay.current_index}"
        viewer._remember_current_frame()

    viewer._refresh_images = refresh
    refresh()
    viewer._setup_callbacks()
    return viewer


def test_snapshot_is_lossless_and_does_not_alias_live_images():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    before = image.copy()
    snapshot = ViewerFrameSnapshot.capture((("panel", image),), "status", "cue")
    image[:] = 0
    assert isinstance(snapshot.panels[0][1], bytes)
    decoded = snapshot.decoded_panels()
    assert decoded[0][0] == "panel"
    assert np.array_equal(decoded[0][1], before)
    decoded[0][1][:] = 255
    assert np.array_equal(snapshot.decoded_panels()[0][1], before)


def test_previous_and_forward_browse_saved_pixels_without_reprocessing():
    viewer = make_viewer()
    assert viewer.gui_previous.disabled
    viewer.gui_next.callback(None)  # Process frame 0.
    first = viewer.main_image.copy()
    viewer.gui_next.callback(None)  # Process frame 1, end of stream.
    latest = viewer.main_image.copy()
    assert viewer.replay.step_calls == 2
    assert viewer.gui_next.disabled

    viewer.gui_previous.callback(None)
    assert viewer._viewed_index == 0
    assert viewer.replay.current_index == 1
    assert viewer.replay.step_calls == 2
    assert np.array_equal(viewer._dashboard_panels()[0][1], first)
    assert "status-0" in viewer.gui_status.content
    assert "processed through t=1" in viewer.gui_status.content
    assert viewer.gui_cue_types.content == "cue-0"
    assert viewer.gui_display.disabled
    assert not viewer.gui_next.disabled  # Browsing works even after end-of-stream.

    viewer.gui_next.callback(None)
    assert viewer._viewed_index == 1
    assert viewer.replay.step_calls == 2
    assert np.array_equal(viewer._dashboard_panels()[0][1], latest)
    assert viewer.gui_status.content == "status-1"
    assert viewer.gui_cue_types.content == "cue-1"
    assert not viewer.gui_display.disabled
    assert viewer.gui_next.disabled


def test_previous_reaches_initialization_and_stops_at_boundary():
    viewer = make_viewer()
    viewer.gui_previous.callback(None)
    assert viewer._viewed_index == -1
    viewer.gui_next.callback(None)
    viewer.gui_previous.callback(None)
    assert viewer._viewed_index == -1
    assert viewer.replay.current_index == 0
    assert viewer.gui_previous.disabled
    viewer.gui_next.callback(None)  # Return to saved frame 0, not a new observation.
    assert viewer.replay.step_calls == 1
    viewer.gui_next.callback(None)  # Only now consume frame 1.
    assert viewer.replay.step_calls == 2


def test_history_disables_current_state_rerender_and_respects_lock():
    viewer = make_viewer()
    viewer.gui_next.callback(None)
    viewer.gui_previous.callback(None)
    old_display = viewer.display
    viewer.gui_display.value = "not a historical render mode"
    viewer.gui_display.callback(None)  # Must not call a replay renderer.
    assert viewer.display == old_display
    with viewer.lock:
        viewer.gui_next.callback(None)
        viewer.gui_previous.callback(None)
    assert viewer._viewed_index == -1
    assert viewer.replay.step_calls == 1


def test_reset_discards_history_instead_of_mixing_runs():
    viewer = make_viewer()
    viewer.gui_next.callback(None)
    viewer.gui_next.callback(None)
    viewer.gui_previous.callback(None)
    viewer.gui_reset.callback(None)
    assert viewer.replay.reset_calls == 1
    assert viewer.replay.current_index == -1
    assert viewer._viewed_index == -1
    assert set(viewer._frame_history) == {-1}
    assert viewer._history_panels is None
    assert viewer.gui_previous.disabled
    assert not viewer.gui_next.disabled
    assert not viewer.gui_display.disabled
