# Global all-view change context R_global (Part 2 cycle 2, spec §2A/§10).
#
# R_global is the FIXED change scene produced by fusing all 25 inference frames
# through the standard all-25 pipeline (subset_oscd.py --frames_method all with
# the exact settings behind the recorded all-25 baseline). It is context, not a
# teacher: selection only reads its rendered soft change masks and its position
# Jacobian — no loss ever pulls a student toward it, and its Gaussian indices
# are never compared with any other model's.
#
# This module owns the cache layout and the c-preserving reload. Building the
# checkpoint (an all-25 subprocess run) and validating it against the recorded
# all-25 numbers live in experiments/run_keyframe_gl_eval.py — this package
# must stay free of GT access.
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np
import torch

from view_selection.cache import config_hash

CACHE_SCHEMA_VERSION = 1


def _git_commit(path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:16] if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def global_context_key(scene: str, instance: str,
                       reference_checkpoint_hash: str,
                       frame_names: list[str], global_seed: int,
                       resolution: int, alpha_threshold: float,
                       repo_root: str, fusion_iterations: int = 16) -> dict:
    """Everything that changes R_global (spec §10). frame_names is the ordered
    all-25 inference frame list — its hash pins the frame manifest."""
    manifest_hash = hashlib.sha256(
        json.dumps(list(frame_names)).encode()).hexdigest()[:16]
    return {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "scene": scene,
        "instance": instance,
        "reference_checkpoint_hash": reference_checkpoint_hash,
        "all25_frame_manifest_hash": manifest_hash,
        "fusion_iterations": fusion_iterations,
        "global_seed": global_seed,
        "resolution": resolution,
        "alpha_threshold": alpha_threshold,
        "code_commit": _git_commit(repo_root),
        "rasterizer_commit": _git_commit(os.path.join(
            repo_root, "submodules", "diff-gaussian-rasterization_fastgs")),
    }


def global_context_dir(root: str, scene: str, instance: str,
                       key: dict) -> str:
    """outputs/change_nbv/global_context/<scene>/<instance>/<hash>/"""
    return os.path.join(root, scene, instance, config_hash(key))


def find_cached_context(root: str, scene: str, instance: str,
                        key: dict) -> str | None:
    """Returns the cache dir when a complete, key-matching R_global exists."""
    d = global_context_dir(root, scene, instance, key)
    needed = ("r_global.ply", "all25_rendered_soft_masks.pt",
              "alpha_masks.pt", "metadata.json")
    if not all(os.path.exists(os.path.join(d, n)) for n in needed):
        return None
    try:
        meta = json.load(open(os.path.join(d, "metadata.json")))
    except Exception:
        return None
    return d if meta.get("key") == key else None


def write_metadata(cache_dir: str, key: dict, extra: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump({"key": key, "config_hash": config_hash(key), **extra},
                  f, indent=2)


def load_frozen_change_model(ply_path: str, sh_degree: int = 3):
    """Reload a saved change model (save_ply_change output) WITH its change
    scalar c. load_ply_change deliberately zeroes f_dc (fresh-fusion loading),
    so the dc channels are read back from the PLY and restored here. Render /
    autograd only — no optimizer state."""
    from plyfile import PlyData

    from scene import GaussianModel

    model = GaussianModel(sh_degree, 0)
    model.load_ply_change(ply_path)

    ply = PlyData.read(ply_path)
    n = model.get_xyz.shape[0]
    f_dc = np.zeros((n, 3, 1))
    f_dc[:, 0, 0] = np.asarray(ply.elements[0]["f_dc_0"])
    f_dc[:, 1, 0] = np.asarray(ply.elements[0]["f_dc_1"])
    f_dc[:, 2, 0] = np.asarray(ply.elements[0]["f_dc_2"])
    with torch.no_grad():
        model._features_dc.data = (
            torch.tensor(f_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous())
    return model
