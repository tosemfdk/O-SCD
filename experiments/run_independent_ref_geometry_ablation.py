"""Compare independent Ref->SCn DC-only, DC+opacity, and DC+geometry.

Each scene-change sequence starts from the same immutable reference Gaussian
PLY, a fresh one-state temporal sidecar, and a fresh Adam optimizer.  Training
uses only the cached O-SCD pixel + SAM2.1 cue.  GT masks are loaded after
training solely for evaluation.

This differs from the temporal lifespan experiment in one deliberate way:
there is no cross-state initialization or S0 geometry anchor because SC1, SC2,
and SC3 are separate one-step problems.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.train_cue_temporal_rchange import (
    DEFAULT_CUE_CACHE,
    DEFAULT_FIXED_CAMERAS,
    build_fixed_cue_views,
    load_fixed_camera_index,
    validate_cue_cache,
)
from experiments.train_geometry_temporal_rchange import (
    geometry_movement_summary,
    train_geometry_slots,
)
from experiments.train_real_temporal_rchange import (
    BASE_PLY_REL,
    DEFAULT_BOUNDARIES,
    DEFAULT_SOURCE,
    FrameRecord,
    PoseResult,
    boundaries_from_manifest,
    build_frame_records,
    compute_state_support,
    file_checksum,
    load_mask_tensor,
    read_manifest,
    seed_everything,
    summarize_losses,
    tensor_checksum,
    train_temporal_slots,
)
from gaussian_renderer import render_change_temporal
from scene import GaussianModel
from scene.cameras import Camera
from temporal import TemporalChangeModel, TemporalGeometryChangeModel


DEFAULT_OUTPUT = Path(
    "outputs/instance1_independent_ref_scene_dc_geometry_"
    "oscd_cues_allframes_120"
)
CONDITIONS = ("dc_only", "dc_opacity", "dc_geometry")
CONDITION_LABELS = {
    "dc_only": "DC-only",
    "dc_opacity": "DC + opacity",
    "dc_geometry": "DC + geometry",
}
CONDITION_COLORS = {
    "dc_only": (56, 118, 190),
    "dc_opacity": (86, 160, 86),
    "dc_geometry": (230, 126, 34),
}
CONFUSION_COLORS = {
    "tp": (0, 200, 0),
    "fp": (255, 105, 180),
    "fn": (0, 90, 255),
}


def scene_ranges_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    """Return half-open global frame ranges in manifest sequence order."""
    order = manifest.get("sequence_order")
    counts = manifest.get("counts", {}).get("per_source_scene")
    if not isinstance(order, list) or not order:
        raise ValueError("manifest.sequence_order must be a non-empty list")
    if not isinstance(counts, dict):
        raise ValueError("manifest counts.per_source_scene must be a mapping")

    ranges: dict[str, tuple[int, int]] = {}
    start = 0
    for scene in order:
        count = int(counts[scene])
        if count <= 0:
            raise ValueError(f"Scene {scene} must contain at least one frame")
        ranges[str(scene)] = (start, start + count)
        start += count

    expected = manifest.get("counts", {}).get("inference_images")
    if expected is not None and start != int(expected):
        raise ValueError(
            f"Per-scene counts total {start}, expected {int(expected)}"
        )
    return ranges


def independent_dataset_audit(
    frame_count: int,
    updates_per_frame: int,
) -> dict[str, Any]:
    """Build the exact-schedule contract expected by shared trainers."""
    if frame_count <= 0 or updates_per_frame <= 0:
        raise ValueError("frame_count and updates_per_frame must be positive")
    total_updates = int(frame_count) * int(updates_per_frame)
    return {
        "expected_dataset_frames": int(frame_count),
        "actual_dataset_frames": int(frame_count),
        "actual_train_records": int(frame_count),
        "actual_train_views": int(frame_count),
        "expected_exact_total_updates": total_updates,
        "dataset_contract_errors": [],
        "dataset_contract_passed": True,
        "independent_reference_start": True,
    }


def select_scene_items(
    records: list[FrameRecord],
    views: list[Camera],
    scene_range: tuple[int, int],
) -> tuple[list[FrameRecord], list[Camera]]:
    """Select one scene and rebase only its training state id to zero."""
    if len(records) != len(views):
        raise ValueError("records and views must be one-to-one")
    start, end = scene_range
    selected_records = records[start:end]
    selected_views = views[start:end]
    if len(selected_records) != end - start:
        raise ValueError("scene range exceeds available records")
    for record, view in zip(selected_records, selected_views):
        if int(record.global_index) < start or int(record.global_index) >= end:
            raise ValueError("record does not belong to the selected scene range")
        view.segment_id = 0
    return selected_records, selected_views


def build_one_state_model(
    base: GaussianModel,
    condition: str,
) -> TemporalChangeModel | TemporalGeometryChangeModel:
    """Create a fresh one-state model from the immutable reference scaffold."""
    if condition == "dc_only":
        model: TemporalChangeModel | TemporalGeometryChangeModel = (
            TemporalChangeModel.from_gaussians(
                base,
                max_states=1,
                initial_time=0.0,
            )
        )
    elif condition in {"dc_opacity", "dc_geometry"}:
        model = TemporalGeometryChangeModel.from_gaussians(
            base,
            max_states=1,
            initial_time=0.0,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    with torch.no_grad():
        model.state_change_dc[:, 0].copy_(base._features_dc.detach())
        model.state_start[:, 0].zero_()
        model.state_end[:, 0].fill_(float("inf"))
        model.state_valid[:, 0].fill_(True)
    if condition == "dc_opacity":
        for name, parameter in model.state_parameter_items():
            parameter.requires_grad_(name in {"dc", "opacity"})
    return model


def training_args(args: argparse.Namespace, condition: str) -> SimpleNamespace:
    """Expose one controlled optimizer configuration to both trainers."""
    opacity_only = condition == "dc_opacity"
    return SimpleNamespace(
        updates_per_frame=int(args.updates_per_frame),
        steps_state=None,
        lr=float(args.dc_lr),
        dc_lr=float(args.dc_lr),
        xyz_lr=0.0 if opacity_only else float(args.xyz_lr),
        opacity_lr=float(args.opacity_lr),
        scaling_lr=0.0 if opacity_only else float(args.scaling_lr),
        rotation_lr=0.0 if opacity_only else float(args.rotation_lr),
        xyz_anchor_weight=0.0,
        state0_anchor_min_support_views=1,
        inherit_previous_state=False,
        gradient_audit_interval=int(args.gradient_audit_interval),
        progress_interval=int(args.progress_interval),
    )


def binary_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
) -> tuple[dict[str, int], dict[str, float], np.ndarray]:
    """Return O-SCD-compatible per-frame metrics and a confusion image."""
    pred_b = pred.detach().squeeze().cpu().numpy() > float(threshold)
    gt_b = gt.detach().squeeze().cpu().numpy() > 0.5
    tp = pred_b & gt_b
    tn = ~pred_b & ~gt_b
    fp = pred_b & ~gt_b
    fn = ~pred_b & gt_b
    counts = {
        "tp": int(tp.sum()),
        "tn": int(tn.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "pred_positive": int(pred_b.sum()),
        "gt_positive": int(gt_b.sum()),
        "pixels": int(pred_b.size),
    }
    union = counts["tp"] + counts["fp"] + counts["fn"]
    f1_denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    metrics = {
        "iou": counts["tp"] / union if union else 0.0,
        "f1": 2 * counts["tp"] / f1_denominator if f1_denominator else 0.0,
        "precision": (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else 0.0
        ),
        "recall": (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else 0.0
        ),
    }
    confusion = np.zeros((*pred_b.shape, 3), dtype=np.uint8)
    confusion[tp] = CONFUSION_COLORS["tp"]
    confusion[fp] = CONFUSION_COLORS["fp"]
    confusion[fn] = CONFUSION_COLORS["fn"]
    return counts, metrics, confusion


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        key: int(sum(int(row[key]) for row in rows))
        for key in (
            "tp",
            "tn",
            "fp",
            "fn",
            "pred_positive",
            "gt_positive",
            "pixels",
        )
    }
    union = counts["tp"] + counts["fp"] + counts["fn"]
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = (
        counts["tp"] / precision_denominator if precision_denominator else 0.0
    )
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    return {
        "frames": len(rows),
        **counts,
        "precision": precision,
        "recall": recall,
        "aggregate_iou": counts["tp"] / union if union else 0.0,
        "aggregate_f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "mean_frame_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_frame_f1": float(np.mean([row["f1"] for row in rows])),
    }


def aggregate_scene_summaries(
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    """Aggregate disjoint scene runs without pretending they share a model."""
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        raise ValueError(f"No rows found for condition {condition}")
    frames = int(sum(int(row["frames"]) for row in selected))
    counts = {
        key: int(sum(int(row[key]) for row in selected))
        for key in (
            "tp",
            "tn",
            "fp",
            "fn",
            "pred_positive",
            "gt_positive",
            "pixels",
        )
    }
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = (
        counts["tp"] / precision_denominator if precision_denominator else 0.0
    )
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    union = counts["tp"] + counts["fp"] + counts["fn"]

    def frame_weighted(key: str) -> float:
        return float(
            sum(int(row["frames"]) * float(row[key]) for row in selected)
            / frames
        )

    return {
        "scope": "overall_disjoint_scene_runs",
        "scene": "overall",
        "condition": condition,
        "frames": frames,
        **counts,
        "precision": precision,
        "recall": recall,
        "aggregate_iou": counts["tp"] / union if union else 0.0,
        "aggregate_f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "mean_frame_iou": frame_weighted("mean_frame_iou"),
        "mean_frame_f1": frame_weighted("mean_frame_f1"),
        "post_train_ssf_loss": frame_weighted("post_train_ssf_loss"),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in selected)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: TemporalChangeModel | TemporalGeometryChangeModel,
    records: list[FrameRecord],
    views: list[Camera],
    background: torch.Tensor,
    pipe: SimpleNamespace,
    output_dir: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load GT only after training and persist score/binary/confusion maps."""
    score_dir = output_dir / "pred_score"
    binary_dir = output_dir / "pred_binary"
    confusion_dir = output_dir / "confusion"
    for directory in (score_dir, binary_dir, confusion_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for record, view in zip(records, views):
            rendered = render_change_temporal(
                view,
                model,
                pipe,
                background,
                timestamp=float(view.timestamp),
            )["render"].mean(dim=0, keepdim=True).clamp(0, 1)
            gt = load_mask_tensor(
                Path(record.mask_path),
                int(view.image_height),
                int(view.image_width),
            )
            counts, metrics, confusion = binary_metrics(rendered, gt, threshold)
            stem = Path(record.name).stem
            score_u8 = (
                rendered.mul(255).round().to(torch.uint8).squeeze().cpu().numpy()
            )
            binary_u8 = (
                (rendered > threshold)
                .to(torch.uint8)
                .mul(255)
                .squeeze()
                .cpu()
                .numpy()
            )
            Image.fromarray(score_u8).save(score_dir / f"{stem}.png")
            Image.fromarray(binary_u8).save(binary_dir / f"{stem}.png")
            Image.fromarray(confusion).save(confusion_dir / f"{stem}.png")
            rows.append(
                {
                    "frame": record.name,
                    "global_index": int(record.global_index),
                    **counts,
                    **metrics,
                }
            )
    write_csv(output_dir / "frame_metrics.csv", rows)
    return rows, aggregate_metrics(rows)


def checkpoint_payload(
    model: TemporalChangeModel | TemporalGeometryChangeModel,
    base_ply: Path,
    condition: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "base_ply": str(base_ply),
        "boundaries": [],
        "contract": f"independent_ref_scene_{condition}",
        "metadata": metadata,
    }


def run_condition(
    *,
    scene: str,
    condition: str,
    records: list[FrameRecord],
    views: list[Camera],
    poses: dict[str, PoseResult],
    cue_metadata: dict[str, Any],
    args: argparse.Namespace,
    pipe: SimpleNamespace,
    background: torch.Tensor,
) -> dict[str, Any]:
    output_dir = args.output_dir / scene / condition
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if summary_path.exists() and checkpoint_path.exists() and not args.overwrite:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "scene": scene,
            "condition": condition,
            "updates_per_frame": int(args.updates_per_frame),
            "threshold": float(args.threshold),
        }
        actual = {
            "scene": saved.get("scene"),
            "condition": saved.get("condition"),
            "updates_per_frame": saved.get("config", {}).get(
                "updates_per_frame"
            ),
            "threshold": saved.get("config", {}).get("threshold"),
        }
        if actual != expected:
            raise RuntimeError(
                f"Existing run config mismatch at {summary_path}: "
                f"expected {expected}, found {actual}"
            )
        print(f"[resume] {scene}/{condition}", flush=True)
        return saved

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_everything(args.seed)
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply_change(str(base_ply))
    model = build_one_state_model(base, condition)
    audit = independent_dataset_audit(len(records), args.updates_per_frame)
    support = compute_state_support(
        model,
        views,
        background,
        pipe,
        args.support_threshold,
        map_threshold=args.support_map_threshold,
    )
    pre_loss = summarize_losses(model, views, background, pipe)
    controlled_args = training_args(args, condition)
    if condition == "dc_only":
        train_log, gradient_audit, schedule_audit = train_temporal_slots(
            model,
            views,
            background,
            pipe,
            controlled_args,
            audit,
        )
        geometry_summary = None
        optimized_parameters = ["state_change_dc"]
    else:
        assert isinstance(model, TemporalGeometryChangeModel)
        no_anchor = torch.zeros_like(model.state_valid[:, 0])
        train_log, gradient_audit, schedule_audit = train_geometry_slots(
            model,
            views,
            background,
            pipe,
            controlled_args,
            audit,
            no_anchor,
        )
        geometry_summary = geometry_movement_summary(model, no_anchor)
        optimized_parameters = (
            ["state_change_dc", "state_opacity_delta"]
            if condition == "dc_opacity"
            else [
                "state_change_dc",
                "state_xyz_delta",
                "state_opacity_delta",
                "state_scaling_delta",
                "state_rotation_delta",
            ]
        )
    post_loss = summarize_losses(model, views, background, pipe)
    rows, evaluation = evaluate_model(
        model,
        records,
        views,
        background,
        pipe,
        output_dir,
        args.threshold,
    )

    metadata = {
        "schema_version": 1,
        "scene": scene,
        "condition": condition,
        "independent_reference_start": True,
        "cross_scene_parameter_sharing": False,
        "state_count": 1,
        "state_anchor": False,
        "state_inheritance": False,
        "fixed_topology": True,
        "densification": False,
        "pruning": False,
        "gt_used_for_training": False,
        "base_ply_sha256": file_checksum(base_ply),
        "cue_cache_metadata_sha256": file_checksum(
            args.cue_cache_root / "metadata.json"
        ),
        "fixed_cameras_sha256": file_checksum(args.fixed_cameras_json),
        "training_schedule": schedule_audit,
    }
    torch.save(
        checkpoint_payload(model, base_ply, condition, metadata),
        checkpoint_path,
    )
    summary = {
        "script": "experiments/run_independent_ref_geometry_ablation.py",
        "scene": scene,
        "condition": condition,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "source_path": str(args.source_path.resolve()),
        "base_ply": str(base_ply),
        "base_ply_sha256": file_checksum(base_ply),
        "fixed_cameras_json": str(args.fixed_cameras_json.resolve()),
        "cue_cache_root": str(args.cue_cache_root.resolve()),
        "cue_cache_metadata": cue_metadata,
        "supervision": "O-SCD pixel cue + SAM2.1 Hiera Tiny feature cue",
        "gt_used_for_training": False,
        "gt_loaded_only_after_training": True,
        "independent_reference_start": True,
        "cross_scene_parameter_sharing": False,
        "state_count": 1,
        "state_anchor": False,
        "state_inheritance": False,
        "fixed_topology": True,
        "densify_prune": False,
        "optimized_parameters": optimized_parameters,
        "config": {
            "updates_per_frame": int(args.updates_per_frame),
            "threshold": float(args.threshold),
            "dc_lr": float(args.dc_lr),
            "xyz_lr": float(args.xyz_lr),
            "opacity_lr": float(args.opacity_lr),
            "scaling_lr": float(args.scaling_lr),
            "rotation_lr": float(args.rotation_lr),
            "effective_learning_rates": {
                "dc": float(controlled_args.dc_lr),
                "xyz": float(controlled_args.xyz_lr),
                "opacity": float(controlled_args.opacity_lr)
                if condition != "dc_only"
                else 0.0,
                "scaling": float(controlled_args.scaling_lr),
                "rotation": float(controlled_args.rotation_lr),
            },
            "support_threshold": int(args.support_threshold),
            "support_map_threshold": float(args.support_map_threshold),
            "seed": int(args.seed),
        },
        "frames": len(records),
        "train_frames": [
            {
                "global_index": int(record.global_index),
                "name": record.name,
                "image_path": record.image_path,
            }
            for record in records
        ],
        "pose_policy": "O-SCD fixed canonical pose",
        "pose_results": {
            record.name: asdict(poses[record.name]) for record in records
        },
        "support": support,
        "loss_summary": {"pre_train": pre_loss, "post_train": post_loss},
        "training_schedule": schedule_audit,
        "gradient_audit": gradient_audit,
        "geometry_movement": geometry_summary,
        "evaluation": evaluation,
        "state_valid_count": int(model.state_valid[:, 0].sum().item()),
        "state_change_dc_checksum": tensor_checksum(model.state_change_dc),
        "checkpoint": str(checkpoint_path),
    }
    if isinstance(model, TemporalGeometryChangeModel):
        summary["state_parameter_checksums"] = {
            name: tensor_checksum(parameter)
            for name, parameter in model.state_parameter_items()
        }
    summary["checkpoint_sha256"] = file_checksum(checkpoint_path)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    print(
        json.dumps(
            {
                "scene": scene,
                "condition": condition,
                "frames": len(rows),
                "updates": schedule_audit["actual_total_updates"],
                "mean_frame_iou": evaluation["mean_frame_iou"],
                "mean_frame_f1": evaluation["mean_frame_f1"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def resize_image(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.NEAREST)


def labeled(image: Image.Image, text: str, width: int) -> Image.Image:
    resized = resize_image(image.convert("RGB"), width)
    out = Image.new("RGB", (width, resized.height + 28), "white")
    out.paste(resized, (0, 28))
    ImageDraw.Draw(out).text((5, 7), text, fill="black", font=ImageFont.load_default())
    return out


def horizontal(images: Iterable[Image.Image], gap: int = 8) -> Image.Image:
    values = list(images)
    width = sum(image.width for image in values) + gap * (len(values) - 1)
    height = max(image.height for image in values)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in values:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def vertical(images: Iterable[Image.Image], gap: int = 8) -> Image.Image:
    values = list(images)
    width = max(image.width for image in values)
    height = sum(image.height for image in values) + gap * (len(values) - 1)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for image in values:
        out.paste(image, (0, y))
        y += image.height + gap
    return out


def save_scene_comparison(
    scene: str,
    records: list[FrameRecord],
    output_dir: Path,
    sample_count: int,
    panel_width: int,
) -> Path:
    indices = np.linspace(0, len(records) - 1, min(sample_count, len(records)))
    selected = sorted(set(int(round(index)) for index in indices))
    rows: list[Image.Image] = []
    for index in selected:
        record = records[index]
        stem = Path(record.name).stem
        with Image.open(record.image_path) as rgb, Image.open(record.mask_path) as gt:
            panels = [
                labeled(rgb, f"RGB | {stem}", panel_width),
                labeled(gt, "GT", panel_width),
            ]
        for condition in CONDITIONS:
            label = CONDITION_LABELS[condition]
            prediction_path = (
                output_dir / scene / condition / "pred_binary" / f"{stem}.png"
            )
            confusion_path = (
                output_dir / scene / condition / "confusion" / f"{stem}.png"
            )
            with Image.open(prediction_path) as prediction:
                panels.append(labeled(prediction, label, panel_width))
            with Image.open(confusion_path) as confusion:
                panels.append(
                    labeled(confusion, f"{label} confusion", panel_width)
                )
        rows.append(horizontal(panels))
    path = output_dir / scene / f"{scene}_comparison_contact_sheet.png"
    vertical(rows).save(path)
    return path


def save_metric_chart(rows: list[dict[str, Any]], path: Path) -> None:
    """Save a dependency-free paired bar chart for mIoU and F1."""
    width, height = 1600, 760
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text(
        (margin_left, 24),
        "Independent Ref->SCn: DC-only vs DC+opacity vs DC+geometry",
        fill="black",
        font=font,
    )
    for tick in range(0, 11):
        value = tick / 10
        y = margin_top + plot_h - int(value * plot_h)
        draw.line((margin_left, y, width - margin_right, y), fill=(225, 225, 225))
        draw.text((35, y - 8), f"{value:.1f}", fill="black", font=small)

    available_scenes = {str(row["scene"]) for row in rows}
    scene_order = sorted(available_scenes - {"overall"})
    if "overall" in available_scenes:
        scene_order.append("overall")
    groups: list[tuple[str, str]] = []
    for scene in scene_order:
        groups.extend(((scene, "mean_frame_iou"), (scene, "mean_frame_f1")))
    group_w = plot_w / len(groups)
    bar_w = max(12, int(group_w * 0.28))
    lookup = {(row["scene"], row["condition"]): row for row in rows}
    for group_index, (scene, metric) in enumerate(groups):
        center = margin_left + (group_index + 0.5) * group_w
        pair_values: dict[str, float] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            value = float(lookup[(scene, condition)][metric])
            pair_values[condition] = value
            x0 = int(center + (condition_index - 1) * bar_w)
            x1 = x0 + bar_w
            y0 = margin_top + plot_h - int(value * plot_h)
            draw.rectangle(
                (x0, y0, x1, margin_top + plot_h),
                fill=CONDITION_COLORS[condition],
            )
            draw.text(
                (x0 - 5, y0 - 20),
                f"{value:.3f}",
                fill="black",
                font=small,
            )
        opacity_delta = pair_values["dc_opacity"] - pair_values["dc_only"]
        geometry_delta = pair_values["dc_geometry"] - pair_values["dc_only"]
        delta_y = (
            margin_top
            + plot_h
            - int(max(pair_values.values()) * plot_h)
            - 43
        )
        draw.text(
            (int(center - group_w * 0.22), delta_y),
            f"opacity {opacity_delta:+.4f} | geometry {geometry_delta:+.4f}",
            fill=(45, 45, 45),
            font=small,
        )
        short_scene = (
            "Overall" if scene == "overall" else scene.replace("scene_change", "SC")
        )
        short_metric = "mIoU" if metric == "mean_frame_iou" else "F1"
        draw.text(
            (int(center - group_w * 0.35), margin_top + plot_h + 15),
            f"{short_scene}\n{short_metric}",
            fill="black",
            font=small,
        )
    legend_y = height - 36
    x = margin_left
    for condition in CONDITIONS:
        draw.rectangle(
            (x, legend_y, x + 22, legend_y + 18),
            fill=CONDITION_COLORS[condition],
        )
        draw.text(
            (x + 30, legend_y),
            CONDITION_LABELS[condition],
            fill="black",
            font=small,
        )
        x += 210
    canvas.save(path)


def summarize_experiment(
    summaries: list[dict[str, Any]],
    scene_records: Mapping[str, list[FrameRecord]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [
        {
            "scene": summary["scene"],
            "condition": summary["condition"],
            **summary["evaluation"],
            "post_train_ssf_loss": summary["loss_summary"]["post_train"]["0"][
                "mean_loss"
            ],
            "runtime_seconds": summary["runtime_seconds"],
        }
        for summary in summaries
    ]
    lookup = {(row["scene"], row["condition"]): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for scene in scene_records:
        dc = lookup[(scene, "dc_only")]
        opacity = lookup[(scene, "dc_opacity")]
        geometry = lookup[(scene, "dc_geometry")]
        comparisons.append(
            {
                "scene": scene,
                "delta_mean_frame_iou_opacity_minus_dc": (
                    opacity["mean_frame_iou"] - dc["mean_frame_iou"]
                ),
                "delta_mean_frame_f1_opacity_minus_dc": (
                    opacity["mean_frame_f1"] - dc["mean_frame_f1"]
                ),
                "delta_precision_opacity_minus_dc": (
                    opacity["precision"] - dc["precision"]
                ),
                "delta_recall_opacity_minus_dc": (
                    opacity["recall"] - dc["recall"]
                ),
                "delta_fp_opacity_minus_dc": opacity["fp"] - dc["fp"],
                "delta_fn_opacity_minus_dc": opacity["fn"] - dc["fn"],
                "delta_post_train_ssf_loss_opacity_minus_dc": (
                    opacity["post_train_ssf_loss"] - dc["post_train_ssf_loss"]
                ),
                "delta_mean_frame_iou_geometry_minus_dc": (
                    geometry["mean_frame_iou"] - dc["mean_frame_iou"]
                ),
                "delta_mean_frame_f1_geometry_minus_dc": (
                    geometry["mean_frame_f1"] - dc["mean_frame_f1"]
                ),
                "delta_precision_geometry_minus_dc": (
                    geometry["precision"] - dc["precision"]
                ),
                "delta_recall_geometry_minus_dc": (
                    geometry["recall"] - dc["recall"]
                ),
                "delta_fp_geometry_minus_dc": geometry["fp"] - dc["fp"],
                "delta_fn_geometry_minus_dc": geometry["fn"] - dc["fn"],
                "delta_post_train_ssf_loss_geometry_minus_dc": (
                    geometry["post_train_ssf_loss"] - dc["post_train_ssf_loss"]
                ),
            }
        )
    overall_rows = [
        aggregate_scene_summaries(rows, condition) for condition in CONDITIONS
    ]
    overall_lookup = {row["condition"]: row for row in overall_rows}
    overall_dc = overall_lookup["dc_only"]
    overall_opacity = overall_lookup["dc_opacity"]
    overall_geometry = overall_lookup["dc_geometry"]
    overall_opacity_comparison = {
        "scope": "overall_disjoint_scene_runs",
        "delta_mean_frame_iou_opacity_minus_dc": (
            overall_opacity["mean_frame_iou"] - overall_dc["mean_frame_iou"]
        ),
        "delta_mean_frame_f1_opacity_minus_dc": (
            overall_opacity["mean_frame_f1"] - overall_dc["mean_frame_f1"]
        ),
        "delta_precision_opacity_minus_dc": (
            overall_opacity["precision"] - overall_dc["precision"]
        ),
        "delta_recall_opacity_minus_dc": (
            overall_opacity["recall"] - overall_dc["recall"]
        ),
        "delta_fp_opacity_minus_dc": overall_opacity["fp"] - overall_dc["fp"],
        "delta_fn_opacity_minus_dc": overall_opacity["fn"] - overall_dc["fn"],
        "delta_post_train_ssf_loss_opacity_minus_dc": (
            overall_opacity["post_train_ssf_loss"]
            - overall_dc["post_train_ssf_loss"]
        ),
    }
    overall_comparison = {
        "scope": "overall_disjoint_scene_runs",
        "delta_mean_frame_iou_geometry_minus_dc": (
            overall_geometry["mean_frame_iou"] - overall_dc["mean_frame_iou"]
        ),
        "delta_mean_frame_f1_geometry_minus_dc": (
            overall_geometry["mean_frame_f1"] - overall_dc["mean_frame_f1"]
        ),
        "delta_precision_geometry_minus_dc": (
            overall_geometry["precision"] - overall_dc["precision"]
        ),
        "delta_recall_geometry_minus_dc": (
            overall_geometry["recall"] - overall_dc["recall"]
        ),
        "delta_fp_geometry_minus_dc": overall_geometry["fp"] - overall_dc["fp"],
        "delta_fn_geometry_minus_dc": overall_geometry["fn"] - overall_dc["fn"],
        "delta_post_train_ssf_loss_geometry_minus_dc": (
            overall_geometry["post_train_ssf_loss"]
            - overall_dc["post_train_ssf_loss"]
        ),
    }
    contact_sheets = {
        scene: str(
            save_scene_comparison(
                scene,
                records,
                args.output_dir,
                args.contact_samples,
                args.panel_width,
            )
        )
        for scene, records in scene_records.items()
    }
    chart = args.output_dir / "independent_ref_dc_opacity_geometry_metrics.png"
    save_metric_chart([*rows, *overall_rows], chart)
    write_csv(args.output_dir / "condition_metrics.csv", rows)
    write_csv(args.output_dir / "overall_condition_metrics.csv", overall_rows)
    write_csv(args.output_dir / "condition_deltas.csv", comparisons)
    opacity_comparisons = [
        {
            key: value
            for key, value in comparison.items()
            if key == "scene" or "opacity_minus_dc" in key
        }
        for comparison in comparisons
    ]
    geometry_comparisons = [
        {
            key: value
            for key, value in comparison.items()
            if key == "scene" or "geometry_minus_dc" in key
        }
        for comparison in comparisons
    ]
    result = {
        "script": "experiments/run_independent_ref_geometry_ablation.py",
        "contract": (
            "independent_ref_to_each_scene_dc_only_vs_dc_opacity_vs_dc_geometry"
        ),
        "source_path": str(args.source_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "scene_order": list(scene_records),
        "conditions": list(CONDITIONS),
        "training_supervision": "O-SCD pixel cue + SAM2.1 feature cue",
        "evaluation_supervision": "GT binary masks",
        "gt_used_for_training": False,
        "independent_reference_start": True,
        "cross_scene_parameter_sharing": False,
        "state_anchor": False,
        "state_inheritance": False,
        "fixed_topology": True,
        "densify_prune": False,
        "updates_per_frame": int(args.updates_per_frame),
        "condition_metrics": rows,
        "overall_by_condition": overall_rows,
        "opacity_minus_dc": opacity_comparisons,
        "overall_opacity_minus_dc": overall_opacity_comparison,
        "geometry_minus_dc": geometry_comparisons,
        "overall_geometry_minus_dc": overall_comparison,
        "metric_chart": str(chart),
        "contact_sheets": contact_sheets,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent Ref->SC1/SC2/SC3 DC-only, DC+opacity, and "
            "DC+geometry ablation"
        )
    )
    parser.add_argument("--source-path", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS
    )
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--support-map-threshold", type=float, default=0.5)
    parser.add_argument("--support-threshold", type=positive_int, default=1)
    parser.add_argument(
        "--gradient-audit-interval", type=positive_int, default=100
    )
    parser.add_argument("--progress-interval", type=positive_int, default=2000)
    parser.add_argument("--contact-samples", type=positive_int, default=6)
    parser.add_argument("--panel-width", type=positive_int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian renderer")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    args.source_path = args.source_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.fixed_cameras_json = args.fixed_cameras_json.resolve()
    args.cue_cache_root = args.cue_cache_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.source_path)
    scene_ranges = scene_ranges_from_manifest(manifest)
    boundaries = boundaries_from_manifest(manifest, DEFAULT_BOUNDARIES)
    records, _, all_names = build_frame_records(
        args.source_path,
        frames_per_state=1,
        probes=[],
        boundaries=boundaries,
        all_training_frames=True,
    )
    if len(records) != len(all_names):
        raise RuntimeError("Exact all-frame record construction failed")
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    cue_metadata = validate_cue_cache(
        args.cue_cache_root,
        base_ply,
        args.resolution,
    )
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    views, poses, _intrinsics = build_fixed_cue_views(
        records,
        cameras,
        args.cue_cache_root,
        args.resolution,
    )
    scene_records: dict[str, list[FrameRecord]] = {}
    scene_views: dict[str, list[Camera]] = {}
    for scene, frame_range in scene_ranges.items():
        selected_records, selected_views = select_scene_items(
            records,
            views,
            frame_range,
        )
        scene_records[scene] = selected_records
        scene_views[scene] = selected_views

    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    summaries: list[dict[str, Any]] = []
    for scene in scene_ranges:
        for condition in CONDITIONS:
            summaries.append(
                run_condition(
                    scene=scene,
                    condition=condition,
                    records=scene_records[scene],
                    views=scene_views[scene],
                    poses=poses,
                    cue_metadata=cue_metadata,
                    args=args,
                    pipe=pipe,
                    background=background,
                )
            )
    summary = summarize_experiment(summaries, scene_records, args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary": str(args.output_dir / "summary.json"),
                "conditions": len(summary["condition_metrics"]),
                "metric_chart": summary["metric_chart"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
