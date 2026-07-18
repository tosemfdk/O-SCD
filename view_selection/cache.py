# Persistent M1 information cache (spec §10.1). Key covers everything that
# changes b_v; any mismatch or corruption invalidates silently (recompute).
from __future__ import annotations

import hashlib
import json
import os

import torch

SCHEMA_VERSION = 1


def config_hash(key: dict) -> str:
    payload = json.dumps({"schema_version": SCHEMA_VERSION, **key},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_dir(root: str, scene: str, key: dict) -> str:
    return os.path.join(root, scene, config_hash(key))


def frame_path(root: str, scene: str, key: dict, frame_id: int) -> str:
    return os.path.join(cache_dir(root, scene, key), f"frame_{frame_id:05d}.pt")


def save_frame(root: str, scene: str, key: dict, frame_id: int,
               diagonal: torch.Tensor, metadata: dict) -> str:
    path = frame_path(root, scene, key, frame_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"schema_version": SCHEMA_VERSION, "key": key,
                "frame_id": frame_id, "metadata": metadata,
                "diagonal": diagonal.detach().float().cpu()}, tmp)
    os.replace(tmp, path)  # atomic
    return path


def load_frame(root: str, scene: str, key: dict, frame_id: int):
    """Returns (diagonal, metadata) or None on miss/mismatch/corruption."""
    path = frame_path(root, scene, key, frame_id)
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if (blob.get("schema_version") != SCHEMA_VERSION
            or blob.get("key") != key or blob.get("frame_id") != frame_id):
        return None
    d = blob.get("diagonal")
    if not isinstance(d, torch.Tensor) or not torch.isfinite(d).all():
        return None
    return d, blob.get("metadata", {})
