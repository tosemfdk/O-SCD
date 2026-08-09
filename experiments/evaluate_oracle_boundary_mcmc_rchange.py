"""Evaluate oracle-boundary MCMC R_change checkpoints against GT masks.

This script is evaluation-only: GT masks are loaded here, never by the training
runner. It reuses the fixed-camera/cue construction used by the oracle-boundary
runner and the panel/metric helpers from ``render_temporal_confusion_maps``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.render_temporal_confusion_maps import (  # noqa: E402
    PALETTE,
    aggregate_rows,
    build_all_records,
    confusion_arrays,
    metrics_from_counts,
    save_contact_sheet,
    save_panel,
    source_scene_name,
    write_csv,
)
from experiments.train_cue_temporal_rchange import (  # noqa: E402
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_real_temporal_rchange import (  # noqa: E402
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    boundaries_from_manifest,
    file_checksum,
    list_images,
    read_manifest,
    seed_everything,
    segment_id,
    tensor_checksum,
    validate_boundaries,
)
from experiments.visualize_temporal_state_switch import load_temporal_model  # noqa: E402
from gaussian_renderer import render_change, render_change_temporal  # noqa: E402
from scene import GaussianModel  # noqa: E402
from temporal.mcmc_state import StateArchive  # noqa: E402
from temporal import TemporalChangeModel  # noqa: E402


DEFAULT_RUN_DIR = Path("outputs/oracle_boundary_mcmc_rchange")
DEFAULT_OUTPUT_DIR = Path("outputs/oracle_boundary_mcmc_rchange_eval")
DEFAULT_AUTHORITATIVE_EVALUATOR = REPO_ROOT.parent / "O-SCD" / "utils" / "evaluate.py"
MCMC_REQUIRED_TENSORS = ("xyz", "features_dc", "raw_change_opacity", "scaling", "rotation")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def tensor_payload_checksum(tensors: Mapping[str, torch.Tensor]) -> str:
    return sha256_json({key: tensor_checksum(value.detach().cpu()) for key, value in sorted(tensors.items())})


def load_gt_mask(path: Path, resolution: float) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if resolution > 0.0 and resolution != 1.0:
        width = int(round(image.width / resolution))
        height = int(round(image.height / resolution))
        image = image.resize((width, height), Image.Resampling.NEAREST)
    arr = (np.asarray(image, dtype=np.float32) / 255.0)[None, ...]
    return torch.from_numpy(arr).contiguous()


def records_with_gt(source_path: Path, boundaries: tuple[int, ...], max_frames: int | None) -> tuple[list[FrameRecord], list[str]]:
    names = list_images(source_path / "inference_scene" / "images")
    validate_boundaries(len(names), boundaries)
    records = build_all_records(source_path, boundaries, max_frames)
    return records, names


def _record_from_name(source_path: Path, name: str, global_index: int, boundaries: tuple[int, ...]) -> FrameRecord:
    stem = Path(name).stem
    return FrameRecord(
        global_index=int(global_index),
        segment_id=segment_id(int(global_index), boundaries),
        name=name,
        image_path=str(source_path / "inference_scene" / "images" / name),
        mask_path=str(source_path / "gt_mask" / f"{stem}.png"),
    )


def selected_records_with_gt(
    source_path: Path,
    boundaries: tuple[int, ...],
    *,
    max_frames: int | None = None,
    frames_per_state: int | None = None,
    training_frames_only: bool = False,
    train_summary: Mapping[str, Any] | None = None,
) -> tuple[list[FrameRecord], list[str], dict[str, Any]]:
    names = list_images(source_path / "inference_scene" / "images")
    validate_boundaries(len(names), boundaries)
    selection: dict[str, Any] = {
        "max_frames": max_frames,
        "frames_per_state": frames_per_state,
        "training_frames_only": training_frames_only,
        "policy": "all_frames",
    }
    if training_frames_only:
        train_frames = list((train_summary or {}).get("train_frames", []))
        if not train_frames:
            raise ValueError("--training-frames-only requested but run summary has no train_frames")
        records = []
        for item in train_frames:
            name = str(item["name"])
            global_index = int(item["global_index"])
            records.append(_record_from_name(source_path, name, global_index, boundaries))
        selection["policy"] = "training_frames_only"
    else:
        records = [_record_from_name(source_path, name, index, boundaries) for index, name in enumerate(names)]
        selection["policy"] = "all_frames"

    if frames_per_state is not None:
        if frames_per_state <= 0:
            raise ValueError("--frames-per-state must be positive")
        counts: dict[int, int] = {}
        limited = []
        for record in records:
            state = int(record.segment_id)
            if counts.get(state, 0) < frames_per_state:
                limited.append(record)
                counts[state] = counts.get(state, 0) + 1
        records = limited
        selection["policy"] = f"{selection['policy']}_first_{frames_per_state}_per_state"
        selection["selected_per_state"] = {str(k): v for k, v in sorted(counts.items())}

    if max_frames is not None:
        records = records[: max(0, int(max_frames))]
        selection["policy"] = f"{selection['policy']}_max_{max_frames}"

    if not records:
        raise ValueError("No frames selected")
    return records, names, selection


def load_gt_masks(records: list[FrameRecord], resolution: float) -> dict[str, torch.Tensor]:
    return {record.name: load_gt_mask(Path(record.mask_path), resolution) for record in records}


def load_base_gaussians(checkpoint: Mapping[str, Any]) -> GaussianModel:
    base_ply = Path(checkpoint["base_ply"])
    if not base_ply.exists():
        raise FileNotFoundError(base_ply)
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    return base


def checkpoint_mode(checkpoint: Mapping[str, Any]) -> str:
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint.get("metadata"), Mapping) else {}
    mode = metadata.get("mode")
    if mode:
        return str(mode)
    contract = str(checkpoint.get("contract", ""))
    if "dc_only" in contract or "temporal_dc" in contract:
        return "dc_only"
    return "mcmc"


class MCMCArchiveReplay:
    """Half-open timestamp/state selector for MCMC archive + current snapshot."""

    def __init__(self, checkpoint: Mapping[str, Any], *, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.mode = checkpoint_mode(checkpoint)
        self.base = load_base_gaussians(checkpoint)
        self.archive = StateArchive.from_state_dict(checkpoint["state_archive"])
        self.archive.verify_checksums()
        self.current_snapshot = checkpoint.get("current_snapshot")
        if self.current_snapshot is None:
            raise ValueError("MCMC checkpoint missing current_snapshot")
        self.boundaries = tuple(float(b) for b in checkpoint.get("boundaries", []))
        self.device = torch.device(device)
        self.capacity = self._infer_capacity()
        self.base_tensors_to_device_and_capacity()
        self.records = self._build_records()
        self._cached_record_key: tuple[Any, ...] | None = None
        self._cached_tensors: dict[str, torch.Tensor] = {}
        self._cached_support_count = 0

    def _infer_capacity(self) -> int:
        for source in [*(self.archive.tensors(i) for i in range(len(self.archive))), self.current_snapshot]:
            if isinstance(source, Mapping) and isinstance(source.get("xyz"), torch.Tensor):
                return int(source["xyz"].shape[0])
        raise ValueError("Cannot infer MCMC archive capacity from checkpoint tensors")

    def base_tensors_to_device_and_capacity(self) -> None:
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            value = getattr(self.base, name, None)
            if isinstance(value, torch.Tensor):
                if value.shape[:1] and value.shape[0] >= self.capacity:
                    value = value[: self.capacity].contiguous()
                setattr(self.base, name, value.to(self.device))
        for name in ("max_radii2D", "xyz_gradient_accum", "xyz_gradient_accum_abs", "denom"):
            value = getattr(self.base, name, None)
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] >= self.capacity:
                setattr(self.base, name, value[: self.capacity].contiguous().to(self.device))

    def _build_records(self) -> list[dict[str, Any]]:
        records = []
        for index, record in enumerate(self.archive.records):
            records.append(
                {
                    "source": "archive",
                    "archive_index": index,
                    "state_id": int(record.state_id),
                    "start_time": float(record.start_time),
                    "end_time": float(record.end_time),
                    "checksum": record.checksum,
                    "metadata": dict(record.metadata),
                    "tensors": self.archive.tensors(index),
                }
            )
        current_start = max(self.boundaries) if self.boundaries else 0.0
        current_state = len(self.boundaries)
        if records:
            current_start = max(current_start, max(float(r["end_time"]) for r in records))
            current_state = max(int(r["state_id"]) for r in records) + 1
        records.append(
            {
                "source": "current_snapshot",
                "archive_index": None,
                "state_id": current_state,
                "start_time": current_start,
                "end_time": float("inf"),
                "checksum": tensor_payload_checksum({k: v for k, v in self.current_snapshot.items() if isinstance(v, torch.Tensor)}),
                "metadata": {"tensor_hashes": {k: tensor_checksum(v) for k, v in self.current_snapshot.items() if isinstance(v, torch.Tensor)}},
                "tensors": self.current_snapshot,
            }
        )
        records.sort(key=lambda r: (float(r["start_time"]), int(r["state_id"])))
        return records

    def select(self, timestamp: float) -> dict[str, Any]:
        for record in self.records:
            if float(record["start_time"]) <= float(timestamp) < float(record["end_time"]):
                return record
        raise KeyError(f"No archived/current state covers timestamp {timestamp}")

    def _tensor(self, tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        if name not in tensors:
            raise KeyError(f"state tensors missing {name!r}")
        return tensors[name].to(self.device, non_blocking=True)

    def _activate_record(self, record: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], int]:
        key = (record["source"], record["archive_index"], record["state_id"], record["checksum"])
        if key != self._cached_record_key:
            tensors = record["tensors"]
            self._cached_tensors = {
                name: self._tensor(tensors, name)
                for name in MCMC_REQUIRED_TENSORS
            }
            cue_support = tensors.get("cue_support_mask", tensors.get("support_mask"))
            self._cached_support_count = int(cue_support.detach().bool().sum().item()) if isinstance(cue_support, torch.Tensor) else 0
            self._cached_record_key = key
        return self._cached_tensors, self._cached_support_count

    def render(self, view, timestamp: float, pipe: SimpleNamespace, background: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.select(timestamp)
        tensors = record["tensors"]
        for name in MCMC_REQUIRED_TENSORS:
            if name not in tensors:
                raise KeyError(f"state {record['state_id']} missing required tensor {name!r}")
        active_tensors, support_count = self._activate_record(record)
        raw_opacity = active_tensors["raw_change_opacity"]
        package = render_change(
            view,
            self.base,
            pipe,
            background,
            override_dc=active_tensors["features_dc"],
            override_opacity=self.base.opacity_activation(raw_opacity),
            override_xyz=active_tensors["xyz"],
            override_scaling=self.base.scaling_activation(active_tensors["scaling"]),
            override_rotation=self.base.rotation_activation(active_tensors["rotation"]),
        )
        rendered = package["render"].mean(dim=0, keepdim=True).clamp(0, 1)
        audit = {
            "checkpoint_kind": "mcmc_archive",
            "state_source": record["source"],
            "replay_state_id": int(record["state_id"]),
            "replay_state_start": float(record["start_time"]),
            "replay_state_end": float(record["end_time"]),
            "archive_checksum": record["checksum"],
            "cue_support_count": support_count,
            "rendered_count": int((package["radii"] > 0).sum().detach().item()),
            "replay_checksum": record["checksum"],
        }
        return rendered, audit

    def audit(self) -> dict[str, Any]:
        return {
            "kind": "mcmc_archive",
            "archive_records": [
                {
                    "state_id": int(r["state_id"]),
                    "start_time": float(r["start_time"]),
                    "end_time": None if not np.isfinite(float(r["end_time"])) else float(r["end_time"]),
                    "source": r["source"],
                    "checksum": r["checksum"],
                    "metadata": r["metadata"],
                }
                for r in self.records
            ],
            "archive_verified": True,
            "render_capacity": int(self.capacity),
        }


class LegacyTemporalReplay:
    def __init__(self, checkpoint_path: Path, checkpoint: Mapping[str, Any] | None = None):
        checkpoint = dict(checkpoint or torch.load(checkpoint_path, map_location="cpu"))
        state_dict = checkpoint.get("state_dict", {})
        state_change_dc = state_dict.get("state_change_dc")
        if not isinstance(state_change_dc, torch.Tensor):
            self.model = load_temporal_model(checkpoint_path)
            self.capacity = int(self.model.state_change_dc.shape[0])
            return
        capacity = int(state_change_dc.shape[0])
        max_states = int(state_change_dc.shape[1])
        base = load_base_gaussians(checkpoint)
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            value = getattr(base, name, None)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                setattr(base, name, value[:capacity].detach().clone())
        self.model = TemporalChangeModel.from_gaussians(base, max_states=max_states, initial_time=0.0)
        device = base._xyz.device
        self.model.load_state_dict(
            {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in state_dict.items()},
            strict=True,
        )
        self.capacity = capacity

    def render(self, view, timestamp: float, pipe: SimpleNamespace, background: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        package = render_change_temporal(view, self.model, pipe, background, timestamp=timestamp)
        rendered = package["render"].mean(dim=0, keepdim=True).clamp(0, 1)
        state = int(view.segment_id)
        return rendered, {
            "checkpoint_kind": "dc_only_temporal",
            "state_source": "temporal_model",
            "replay_state_id": state,
            "replay_state_start": None,
            "replay_state_end": None,
            "archive_checksum": "",
            "cue_support_count": 0,
            "rendered_count": int((package["radii"] > 0).sum().detach().item()),
            "replay_checksum": tensor_checksum(self.model.state_change_dc[:, state].detach().cpu()),
        }

    def audit(self) -> dict[str, Any]:
        return {"kind": "dc_only_temporal", "archive_records": [], "archive_verified": None, "render_capacity": self.capacity}


def load_replay(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("state_archive") is not None and checkpoint.get("current_snapshot") is not None:
        return MCMCArchiveReplay(checkpoint), checkpoint
    return LegacyTemporalReplay(checkpoint_path, checkpoint), checkpoint


def boundary_start(timestamp: int, boundaries: tuple[int, ...]) -> int:
    start = 0
    for boundary in boundaries:
        if timestamp >= int(boundary):
            start = int(boundary)
        else:
            break
    return start


def coverage_values(rendered: torch.Tensor, cue: torch.Tensor, threshold: float) -> dict[str, float | int]:
    rendered_cpu = rendered.detach().float().cpu().clamp(0, 1)
    cue_cpu = cue.detach().float().cpu().clamp(0, 1)
    cue_positive = cue_cpu >= float(threshold)
    pred_positive = rendered_cpu >= float(threshold)
    cue_mass = float(cue_cpu.sum().item())
    rendered_mass = float(rendered_cpu.sum().item())
    cue_positive_mass = float(cue_cpu[cue_positive].sum().item()) if cue_positive.any() else 0.0
    covered_mass = float(rendered_cpu[cue_positive].sum().item()) if cue_positive.any() else 0.0
    return {
        "cue_positive_pixels": int(cue_positive.sum().item()),
        "cue_positive_mass": cue_positive_mass,
        "cue_positive_mass_coverage": covered_mass / cue_positive_mass if cue_positive_mass > 0 else 0.0,
        "cue_total_mass": cue_mass,
        "rendered_alpha_mass": rendered_mass,
        "rendered_alpha_mean": float(rendered_cpu.mean().item()),
        "rendered_alpha_positive_pixels": int(pred_positive.sum().item()),
        "rendered_alpha_coverage": float(pred_positive.float().mean().item()),
    }


def with_headline_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Add authoritative O-SCD headline aliases: arithmetic per-frame means."""
    out = dict(summary)
    out["mIoU"] = float(out.get("mean_frame_iou", 0.0))
    out["mF1"] = float(out.get("mean_frame_f1", 0.0))
    out["micro_iou"] = float(out.get("iou", 0.0))
    out["micro_f1"] = float(out.get("f1", 0.0))
    return out


def load_online_prediction_index(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = run_dir / "online_predictions" / "index.jsonl"
    if not path.exists():
        return {}, {"available": False, "reason": f"missing {path}"}
    entries: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            timestamp = payload.get("global_index", payload.get("timestamp", payload.get("frame_index")))
            if timestamp is None:
                raise ValueError(f"online prediction index line {line_no} has no global_index/timestamp")
            entries[int(float(timestamp))] = payload
    return entries, {"available": True, "index_path": str(path), "entries": len(entries)}


def _resolve_online_path(run_dir: Path, payload: Mapping[str, Any]) -> Path | None:
    for key in ("pred_binary_png", "prediction_png", "render_png", "path", "prediction_path", "tensor_path", "pt_path"):
        value = payload.get(key)
        if value:
            path = Path(str(value))
            if path.is_absolute():
                return path
            direct = run_dir / path
            if direct.exists():
                return direct
            return run_dir / "online_predictions" / path
    return None


def load_online_prediction(run_dir: Path, payload: Mapping[str, Any], expected_hw: tuple[int, int]) -> torch.Tensor:
    path = _resolve_online_path(run_dir, payload)
    if path is None:
        raise ValueError("online prediction entry has no supported prediction path field")
    if path.suffix.lower() == ".pt":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, Mapping):
            for key in ("pred", "prediction", "render", "mask", "rendered"):
                if isinstance(value.get(key), torch.Tensor):
                    value = value[key]
                    break
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"online prediction tensor file did not contain a tensor: {path}")
        tensor = value.detach().float().cpu()
        if tensor.ndim == 2:
            tensor = tensor[None]
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = tensor.mean(dim=0, keepdim=True)
    else:
        image = Image.open(path).convert("L")
        if (image.height, image.width) != expected_hw:
            image = image.resize((expected_hw[1], expected_hw[0]), Image.Resampling.NEAREST)
        tensor = torch.from_numpy((np.asarray(image, dtype=np.float32) / 255.0)[None])
    if tuple(tensor.shape[-2:]) != tuple(expected_hw):
        raise ValueError(f"online prediction shape {tuple(tensor.shape)} does not match {expected_hw}")
    return tensor.clamp(0, 1)


def compute_online_arrival_rows(
    rows: list[dict[str, Any]],
    gt_masks: Mapping[str, torch.Tensor],
    online_index: Mapping[int, Mapping[str, Any]],
    run_dir: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not online_index:
        return [], {"available": False, "reason": "online_predictions/index.jsonl unavailable or empty"}
    online_rows: list[dict[str, Any]] = []
    for base_row in sorted(rows, key=lambda r: int(r["global_index"])):
        timestamp = int(base_row["global_index"])
        payload = online_index.get(timestamp)
        if payload is None:
            continue
        gt = gt_masks[str(base_row["frame"])]
        pred = load_online_prediction(run_dir, payload, tuple(gt.shape[-2:]))
        _confusion, counts = confusion_arrays(pred, gt, threshold)
        metrics = metrics_from_counts(counts)
        online_rows.append({
            "frame": base_row["frame"],
            "source_scene": base_row["source_scene"],
            "global_index": timestamp,
            "state": int(base_row["state"]),
            "online_frames_since_boundary": int(payload.get("frames_since_boundary", base_row["frames_since_boundary"])),
            "prediction_source": str(_resolve_online_path(run_dir, payload)),
            **counts,
            **metrics,
        })
    if not online_rows:
        return [], {"available": False, "reason": "index exists but contains no selected frame predictions", "index_entries": len(online_index)}
    by_state: dict[int, list[dict[str, Any]]] = {}
    for row in online_rows:
        state = int(row["state"])
        by_state.setdefault(state, []).append(row)
        seen = by_state[state]
        row["online_seen_prefix_mIoU"] = float(np.mean([r["iou"] for r in seen]))
        row["online_seen_prefix_mF1"] = float(np.mean([r["f1"] for r in seen]))
    return online_rows, {
        "available": True,
        "matched_selected_frames": len(online_rows),
        "seen_prefix_mIoU": float(np.mean([r["online_seen_prefix_mIoU"] for r in online_rows])),
        "seen_prefix_mF1": float(np.mean([r["online_seen_prefix_mF1"] for r in online_rows])),
    }


def save_prediction_png(pred_binary: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = pred_binary.detach().to(torch.uint8).mul(255).squeeze().cpu().numpy()
    Image.fromarray(arr).save(path)


def _replace_link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.symlink_to(target.resolve())


def authoritative_oscd_metrics(
    evaluator: Path,
    records: list[FrameRecord],
    prediction_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run the existing O-SCD evaluator on an exact selected-frame link set."""
    if not evaluator.exists():
        raise FileNotFoundError(f"authoritative O-SCD evaluator not found: {evaluator}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    gt_dir = work_dir / "gt"
    pred_dir = work_dir / "pred_binary"
    matched = 0
    for record in records:
        gt = Path(record.mask_path)
        pred = prediction_dir / f"{Path(record.name).stem}.png"
        if not gt.exists() or not pred.exists():
            raise FileNotFoundError(f"authoritative evaluator input missing: gt={gt}, pred={pred}")
        _replace_link(gt_dir / gt.name, gt)
        _replace_link(pred_dir / pred.name, pred)
        matched += 1
    completed = subprocess.run(
        [sys.executable, str(evaluator), "--gt", str(gt_dir), "--pred_binary", str(pred_dir)],
        check=True,
        text=True,
        capture_output=True,
    )
    miou_match = re.search(r"Mean IoU:\s*([-+0-9.eE]+)", completed.stdout)
    f1_match = re.search(r"Mean F1:\s*([-+0-9.eE]+)", completed.stdout)
    if miou_match is None or f1_match is None:
        raise RuntimeError(f"could not parse authoritative evaluator output: {completed.stdout!r}")
    return {
        "mIoU": float(miou_match.group(1)),
        "mF1": float(f1_match.group(1)),
        "matched_frames": matched,
        "evaluator": str(evaluator),
        "stdout": completed.stdout.strip(),
    }


def apply_authoritative_headline(summary: dict[str, Any], authoritative: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out["diagnostic_resolution_mean_frame_iou"] = out.get("mean_frame_iou")
    out["diagnostic_resolution_mean_frame_f1"] = out.get("mean_frame_f1")
    out["mean_frame_iou"] = float(authoritative["mIoU"])
    out["mean_frame_f1"] = float(authoritative["mF1"])
    out["mIoU"] = float(authoritative["mIoU"])
    out["mF1"] = float(authoritative["mF1"])
    out["headline_source"] = "existing O-SCD utils/evaluate.py on native-resolution GT"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle-boundary MCMC R_change archive/current checkpoints")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <run-dir>/temporal_rchange_checkpoint.pt")
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--boundaries", nargs="+", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frames-per-state", type=int, default=None, help="Evaluate first K selected frames from each oracle state, preserving global timestamps")
    parser.add_argument("--training-frames-only", action="store_true", help="Evaluate only train_frames recorded in the run summary")
    parser.add_argument("--panel-width", type=int, default=260)
    parser.add_argument("--contact-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--authoritative-evaluator", type=Path, default=DEFAULT_AUTHORITATIVE_EVALUATOR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing Gaussian renderer")
    seed_everything(args.seed)

    checkpoint_path = args.checkpoint or (args.run_dir / "temporal_rchange_checkpoint.pt")
    replay, checkpoint = load_replay(checkpoint_path)
    summary_path = args.run_dir / "summary.json"
    train_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if args.resolution is None:
        args.resolution = float(train_summary.get("resolution", checkpoint.get("metadata", {}).get("resolution", 8.0)))

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.source_path)
    boundaries = tuple(args.boundaries) if args.boundaries is not None else tuple(checkpoint.get("boundaries") or train_summary.get("boundaries") or boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES))
    records, all_names, selection_audit = selected_records_with_gt(
        args.source_path,
        boundaries,
        max_frames=args.max_frames,
        frames_per_state=args.frames_per_state,
        training_frames_only=args.training_frames_only,
        train_summary=train_summary,
    )
    validate_cue_cache(args.cue_cache_root, Path(checkpoint["base_ply"]), float(args.resolution))
    views, poses, intrinsics = build_fixed_cue_views(records, load_fixed_camera_index(args.fixed_cameras_json), args.cue_cache_root, float(args.resolution))
    gt_masks = load_gt_masks(records, float(args.resolution))

    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    record_by_stem = {Path(record.name).stem: record for record in records}
    panel_paths_by_scene: dict[str, list[Path]] = {}
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for view in views:
            record = record_by_stem[view.image_name]
            timestamp = float(record.global_index)
            rendered, replay_audit = replay.render(view, timestamp, pipe, background)
            gt = gt_masks[record.name]
            confusion, counts = confusion_arrays(rendered, gt, args.threshold)
            metric_vals = metrics_from_counts(counts)
            pred_binary = (rendered >= args.threshold).float().detach().cpu()
            scene = source_scene_name(record.name)
            stem = Path(record.name).stem
            raw_path = args.output_dir / scene / "raw_confusion" / f"{record.global_index:06d}_{stem}_confusion.png"
            panel_path = args.output_dir / scene / "panels" / f"{record.global_index:06d}_{stem}_panel.png"
            pred_path = args.output_dir / "pred_binary" / f"{stem}.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(confusion).save(raw_path)
            save_prediction_png(pred_binary, pred_path)
            save_panel(
                view.original_image[:3].detach().cpu(),
                gt,
                pred_binary,
                confusion,
                panel_path,
                f"{scene} | t={record.global_index} | GT S{record.segment_id} | replay S{replay_audit['replay_state_id']}",
                metric_vals,
                args.panel_width,
            )
            panel_paths_by_scene.setdefault(scene, []).append(panel_path)
            cov = coverage_values(rendered, view.candidate_map, args.threshold)
            pose = poses[record.name]
            row = {
                "frame": record.name,
                "source_scene": scene,
                "global_index": int(record.global_index),
                "state": int(record.segment_id),
                "frames_since_boundary": int(record.global_index) - boundary_start(int(record.global_index), boundaries),
                "pose_ok": True,
                "pose_reference": pose.reference_name,
                "pose_matches": pose.matches,
                "pose_inliers": pose.inliers,
                "pose_reprojection_rmse": pose.reprojection_rmse,
                "raw_confusion_png": str(raw_path),
                "panel_png": str(panel_path),
                "pred_binary_png": str(pred_path),
                **replay_audit,
                **cov,
                **counts,
                **metric_vals,
            }
            rows.append(row)

    online_index, online_index_audit = load_online_prediction_index(args.run_dir)
    online_rows, online_arrival_audit = compute_online_arrival_rows(
        rows, gt_masks, online_index, args.run_dir, float(args.threshold)
    )
    online_arrival_audit = {**online_index_audit, **online_arrival_audit}

    for scene, paths in panel_paths_by_scene.items():
        selected = paths
        if len(paths) > args.contact_samples:
            idx = np.linspace(0, len(paths) - 1, args.contact_samples).round().astype(int).tolist()
            selected = [paths[i] for i in sorted(set(idx))]
        save_contact_sheet(selected, args.output_dir / scene / f"{scene}_contact_sheet.png")
    all_panels = [p for paths in panel_paths_by_scene.values() for p in paths]
    if all_panels:
        selected = all_panels
        if len(all_panels) > args.contact_samples:
            idx = np.linspace(0, len(all_panels) - 1, args.contact_samples).round().astype(int).tolist()
            selected = [all_panels[i] for i in sorted(set(idx))]
        save_contact_sheet(selected, args.output_dir / "contact_sheet.png")

    scene_summaries = [with_headline_metrics(aggregate_rows([r for r in rows if r["source_scene"] == scene], scene)) for scene in sorted({r["source_scene"] for r in rows})]
    state_ids = sorted({int(r["state"]) for r in rows})
    state_summaries = [with_headline_metrics(aggregate_rows([r for r in rows if int(r["state"]) == state], f"state_{state}")) for state in state_ids]
    overall = with_headline_metrics(aggregate_rows(rows, "overall"))
    diagnostic_overall = dict(overall)
    authoritative_root = args.output_dir / "authoritative_evaluator_inputs"
    authoritative_overall = authoritative_oscd_metrics(
        args.authoritative_evaluator,
        records,
        args.output_dir / "pred_binary",
        authoritative_root / "overall",
    )
    authoritative_by_state = {}
    for state in state_ids:
        state_records = [record for record in records if int(record.segment_id) == state]
        authoritative_by_state[state] = authoritative_oscd_metrics(
            args.authoritative_evaluator,
            state_records,
            args.output_dir / "pred_binary",
            authoritative_root / f"state_{state}",
        )
    overall = apply_authoritative_headline(overall, authoritative_overall)
    state_summaries = [
        apply_authoritative_headline(summary, authoritative_by_state[state])
        for state, summary in zip(state_ids, state_summaries)
    ]
    full_state_posthoc_miou = float(overall["mIoU"])
    full_state_posthoc_mf1 = float(overall["mF1"])
    online_seen_prefix_miou = online_arrival_audit.get("seen_prefix_mIoU") if online_arrival_audit.get("available") else None
    online_seen_prefix_mf1 = online_arrival_audit.get("seen_prefix_mF1") if online_arrival_audit.get("available") else None
    replay_checksums = {f"{r['global_index']:06d}_{r['frame']}": r["replay_checksum"] for r in rows}
    parameter_checksums_expected = train_summary.get("parameter_checksums", checkpoint.get("metadata", {}).get("parameter_checksums", {}))
    current_snapshot = checkpoint.get("current_snapshot")
    parameter_checksums_actual = {
        key: tensor_checksum(value)
        for key, value in (current_snapshot or checkpoint.get("state_dict", {})).items()
        if isinstance(value, torch.Tensor)
    }
    checksum_audit = {
        "checkpoint_sha256": file_checksum(checkpoint_path),
        "base_ply_sha256": file_checksum(Path(checkpoint["base_ply"])),
        "parameter_checksums_expected": parameter_checksums_expected,
        "parameter_checksums_actual": parameter_checksums_actual,
        "parameter_checksum_match": {
            key: parameter_checksums_actual.get(key) == value
            for key, value in parameter_checksums_expected.items()
        }
        if isinstance(parameter_checksums_expected, Mapping)
        else {},
        "replay_checksums_sha256": sha256_json(replay_checksums),
        "replay_checksums": replay_checksums,
    }

    summary_out = {
        "script": "experiments/evaluate_oracle_boundary_mcmc_rchange.py",
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "source_path": str(args.source_path),
        "output_dir": str(args.output_dir),
        "threshold": float(args.threshold),
        "resolution": float(args.resolution),
        "boundaries": list(boundaries),
        "frames_requested": len(records),
        "frames_evaluated": len(rows),
        "selection": selection_audit,
        "gt_used_in_evaluator_only": True,
        "gt_mask_pixels_loaded": int(sum(int(gt.numel()) for gt in gt_masks.values())),
        "online_arrival_seen_prefix_mIoU_semantics": "True online-arrival metric from <run-dir>/online_predictions/index.jsonl; each row uses the prediction saved at frame arrival time, then reports cumulative arithmetic mean of per-frame IoU/F1 over seen frames within that state.",
        "full_state_posthoc_mIoU_semantics": "Posthoc metric from final archived/current half-open state replay; headline mIoU/mF1 are arithmetic means of per-frame IoU/F1 over selected frames, while micro_iou/micro_f1 retain count-aggregated values.",
        "authoritative_prediction_compatibility": "Binary masks are written as pred_binary/<GT stem>.png to match utils/evaluate.py naming expectations; panels/confusion may include global-index prefixes.",
        "seen_prefix_mIoU": online_seen_prefix_miou,
        "seen_prefix_mF1": online_seen_prefix_mf1,
        "seen_prefix_status": "available" if online_arrival_audit.get("available") else "unavailable",
        "online_arrival_audit": online_arrival_audit,
        "full_state_posthoc_mIoU": full_state_posthoc_miou,
        "full_state_posthoc_mF1": full_state_posthoc_mf1,
        "overall": overall,
        "diagnostic_resolution_overall": diagnostic_overall,
        "per_state": state_summaries,
        "per_scene": scene_summaries,
        "authoritative_oscd_evaluator": {
            "overall": authoritative_overall,
            "per_state": {str(key): value for key, value in authoritative_by_state.items()},
        },
        "archive_audit": replay.audit(),
        "checksum_audit": checksum_audit,
        "input_audit": {
            "all_frame_names_sha256": sha256_json(all_names),
            "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
            "cue_cache_metadata_sha256": file_checksum(args.cue_cache_root / "metadata.json"),
            "camera_intrinsics": intrinsics.tolist(),
        },
        "runtime_seconds": time.time() - started,
        "palette": PALETTE,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "per_frame_metrics.csv", rows)
    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "scene_summary.csv", [overall, *state_summaries, *scene_summaries])
    if online_rows:
        write_csv(args.output_dir / "online_arrival_metrics.csv", online_rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "frames_evaluated": len(rows),
                "overall_mIoU": overall.get("mIoU"),
                "overall_mF1": overall.get("mF1"),
                "overall_micro_iou": overall.get("micro_iou"),
                "overall_micro_f1": overall.get("micro_f1"),
                "seen_prefix_mIoU": online_seen_prefix_miou,
                "seen_prefix_mF1": online_seen_prefix_mf1,
                "seen_prefix_status": "available" if online_arrival_audit.get("available") else "unavailable",
                "full_state_posthoc_mIoU": full_state_posthoc_miou,
                "full_state_posthoc_mF1": full_state_posthoc_mf1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
