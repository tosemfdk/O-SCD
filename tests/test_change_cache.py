# Gate S1: information cache hit/miss/corruption semantics (spec §10.1).
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_selection.cache import load_frame, save_frame  # noqa: E402

KEY = {"checkpoint": "abc", "camera": "h1", "resolution": 4,
       "alpha_threshold": 0.5, "output_space": "raw", "weight_mode": "pose",
       "num_probes": 4, "probe_seed_version": 1, "channel_projection": True,
       "rasterizer": "fastgs-r1"}


def test_roundtrip_hit(tmp_path):
    d = torch.rand(64)
    save_frame(str(tmp_path), "Garden", KEY, 3, d, {"probes_ran": 4})
    got = load_frame(str(tmp_path), "Garden", KEY, 3)
    assert got is not None
    diag, meta = got
    assert torch.allclose(diag, d) and meta["probes_ran"] == 4


def test_key_mutation_misses(tmp_path):
    save_frame(str(tmp_path), "Garden", KEY, 3, torch.rand(8), {})
    for field, val in [("num_probes", 8), ("output_space", "sigmoid"),
                       ("checkpoint", "zzz"), ("alpha_threshold", 0.6)]:
        assert load_frame(str(tmp_path), "Garden", {**KEY, field: val}, 3) is None
    assert load_frame(str(tmp_path), "Garden", KEY, 4) is None  # other frame
    assert load_frame(str(tmp_path), "Zen", KEY, 3) is None     # other scene


def test_corruption_invalidates(tmp_path):
    from view_selection.cache import frame_path
    save_frame(str(tmp_path), "Garden", KEY, 3, torch.rand(8), {})
    path = frame_path(str(tmp_path), "Garden", KEY, 3)
    with open(path, "wb") as f:
        f.write(b"not a torch file")
    assert load_frame(str(tmp_path), "Garden", KEY, 3) is None
    bad = torch.full((4,), float("nan"))
    save_frame(str(tmp_path), "Garden", KEY, 5, bad, {})
    assert load_frame(str(tmp_path), "Garden", KEY, 5) is None
