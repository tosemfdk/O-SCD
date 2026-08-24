"""Run causal FastGS-style soft change-cue density control on ESCD scopes.

This is deliberately a mutable ``R_change`` ablation rather than a temporal
sidecar integration.  The immutable reference PLY and cached ref--inf cue are
read-only; clone/split operations affect only a separate change bank.  Cue-based
pruning is excluded from this first experiment because high change support has
the opposite semantics of FastGS reconstruction-degradation pruning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gaussian_renderer import render_change
from temporal.change_cue_density import (
    DensificationResult,
    apply_fastgs_change_densification,
    apply_oscd_gradient_only_densification,
    causal_random_view_indices,
    compute_soft_multiview_change_score,
    topology_integrity,
)


DEFAULT_SOURCE = Path("data/Instance_1/scene_change1_2_3")
DEFAULT_FIXED_CAMERAS = Path(
    "/home/rvl/workspace/github/O-SCD/output/ESCD_fixedpose_protocols_res4/scene_change1_2_3/cameras_fixed.json"
)
DEFAULT_CUE_CACHE = Path(
    "/home/rvl/workspace/github/O-SCD/artifacts/escd_396ref/fixed_pose_cues_res4_v1"
)
BASE_PLY_REL = Path("reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply")
FASTGS_SOURCE_COMMIT = "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f"
SCOPE_MAX_FRAMES = {
    "scene_change1": 95,
    "scene_change2": 104,
    "scene_change3": 105,
    "continuous": 304,
}
SCOPE_LABELS = {
    "scene_change1": "ref -> scene_change1 only",
    "scene_change2": "ref -> scene_change2 only",
    "scene_change3": "ref -> scene_change3 only",
    "continuous": "ref -> scene_change1 -> scene_change2 -> scene_change3",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_training_view_index(
    timestamp: int,
    update_index: int,
    *,
    seed: int,
) -> int:
    """Reproduce O-SCD's current-biased causal replay without shared RNG state."""
    if timestamp < 0 or update_index < 0:
        raise ValueError("timestamp and update_index must be nonnegative")
    local = np.random.default_rng(int(seed) + 1_000_003 * timestamp + 1009 * update_index)
    if float(local.random()) <= 0.33:
        return int(timestamp)
    return int(local.integers(0, timestamp + 1))


def select_scope_records(
    records: Sequence[Any],
    *,
    scope: str,
    max_frames: int,
) -> list[Any]:
    """Select an independent segment or the full stream and reindex causally."""
    if scope not in SCOPE_MAX_FRAMES:
        raise ValueError(f"unknown scope: {scope}")
    if max_frames < 1 or max_frames > SCOPE_MAX_FRAMES[scope]:
        raise ValueError(f"max_frames exceeds the {scope} scope")
    selected = (
        list(records[:max_frames])
        if scope == "continuous"
        else [record for record in records if record.segment_name == scope][:max_frames]
    )
    if len(selected) != max_frames:
        raise RuntimeError(
            f"requested {max_frames} frames for {scope}, found {len(selected)}"
        )
    normalized: list[Any] = []
    for index, record in enumerate(selected):
        values = vars(record).copy()
        values["global_index"] = index
        normalized.append(SimpleNamespace(**values))
    return normalized


def oracle_state_local_view_indices(
    processed_records: Sequence[Any],
    current_index: int,
    k: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Sample current plus past views only from the current oracle state."""
    if current_index != len(processed_records) - 1:
        raise ValueError("processed_records must end at current_index")
    segment = processed_records[current_index].segment_name
    start = current_index
    while start > 0 and processed_records[start - 1].segment_name == segment:
        start -= 1
    local_indices = causal_random_view_indices(
        current_index - start,
        k,
        seed=int(seed) + 10_007 * int(processed_records[current_index].segment_id),
    )
    selected = tuple(start + index for index in local_indices)
    if any(processed_records[index].segment_name != segment for index in selected):
        raise RuntimeError("oracle state-local density sampling crossed a boundary")
    return selected


def reference_scene_extent(cameras_json: Path) -> float:
    rows = json.loads(cameras_json.read_text(encoding="utf-8"))
    positions = np.asarray([row["position"] for row in rows], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("reference cameras must contain at least two 3D positions")
    center = positions.mean(axis=0)
    extent = float(np.linalg.norm(positions - center[None], axis=1).max() * 1.1)
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("reference camera extent is invalid")
    return extent


def optimization_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        percent_dense=float(args.percent_dense),
        position_lr_init=float(args.xyz_lr),
        position_lr_final=float(args.xyz_lr_final),
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=float(args.dc_lr),
        lowfeature_lr=float(args.dc_lr),
        highfeature_lr=0.005,
        opacity_lr=float(args.opacity_lr),
        scaling_lr=float(args.scaling_lr),
        rotation_lr=float(args.rotation_lr),
    )


class LineageTracker:
    """Track clone/split identities without altering GaussianModel internals."""

    def __init__(self, gaussian_count: int, device: torch.device) -> None:
        self.current_ids = torch.arange(gaussian_count, device=device, dtype=torch.long)
        self.next_id = int(gaussian_count)
        self.creation_ids: list[np.ndarray] = []
        self.creation_parent_ids: list[np.ndarray] = []
        self.creation_timestamps: list[np.ndarray] = []
        self.creation_kinds: list[np.ndarray] = []
        self.deletion_ids: list[np.ndarray] = []
        self.deletion_timestamps: list[np.ndarray] = []
        self.deletion_reasons: list[np.ndarray] = []

    def _new_ids(self, count: int) -> torch.Tensor:
        ids = torch.arange(
            self.next_id,
            self.next_id + int(count),
            device=self.current_ids.device,
            dtype=torch.long,
        )
        self.next_id += int(count)
        return ids

    def _record_creation(
        self,
        ids: torch.Tensor,
        parents: torch.Tensor,
        timestamp: int,
        kind: int,
    ) -> None:
        if ids.numel() == 0:
            return
        count = int(ids.numel())
        self.creation_ids.append(ids.cpu().numpy())
        self.creation_parent_ids.append(parents.cpu().numpy())
        self.creation_timestamps.append(np.full(count, timestamp, dtype=np.int32))
        self.creation_kinds.append(np.full(count, kind, dtype=np.int8))

    def apply_densification(self, result: DensificationResult, timestamp: int) -> None:
        if self.current_ids.numel() != result.initial_count:
            raise RuntimeError("lineage/model topology diverged before densification")
        clone_mask = result.masks.clone
        clone_parents = self.current_ids[clone_mask]
        clone_ids = self._new_ids(int(clone_parents.numel()))
        self._record_creation(clone_ids, clone_parents, timestamp, kind=1)
        self.current_ids = torch.cat((self.current_ids, clone_ids))

        split_mask = torch.cat(
            (
                result.masks.split,
                torch.zeros(
                    result.clone_count,
                    device=self.current_ids.device,
                    dtype=torch.bool,
                ),
            )
        )
        split_parents = self.current_ids[split_mask]
        split_parent_repeated = split_parents.repeat(2)
        split_ids = self._new_ids(int(split_parent_repeated.numel()))
        self._record_creation(split_ids, split_parent_repeated, timestamp, kind=2)
        if split_parents.numel():
            self.deletion_ids.append(split_parents.cpu().numpy())
            self.deletion_timestamps.append(
                np.full(int(split_parents.numel()), timestamp, dtype=np.int32)
            )
            self.deletion_reasons.append(
                np.full(int(split_parents.numel()), 1, dtype=np.int8)
            )
        all_ids = torch.cat((self.current_ids, split_ids))
        keep = torch.cat(
            (
                ~split_mask,
                torch.ones(split_ids.numel(), device=all_ids.device, dtype=torch.bool),
            )
        )
        self.current_ids = all_ids[keep]
        if self.current_ids.numel() != result.final_count:
            raise RuntimeError("lineage/model topology diverged after densification")

    @staticmethod
    def _concat(chunks: Sequence[np.ndarray], dtype: np.dtype) -> np.ndarray:
        return np.concatenate(chunks).astype(dtype, copy=False) if chunks else np.asarray([], dtype=dtype)

    def save(self, path: Path, final_timestamp: int, initial_count: int) -> dict[str, Any]:
        creation_id = self._concat(self.creation_ids, np.int64)
        parent_id = self._concat(self.creation_parent_ids, np.int64)
        creation_timestamp = self._concat(self.creation_timestamps, np.int32)
        creation_kind = self._concat(self.creation_kinds, np.int8)
        deletion_id = self._concat(self.deletion_ids, np.int64)
        deletion_timestamp = self._concat(self.deletion_timestamps, np.int32)
        deletion_reason = self._concat(self.deletion_reasons, np.int8)
        current_ids = self.current_ids.detach().cpu().numpy()
        new_mask = creation_id >= int(initial_count)
        alive = np.isin(creation_id, current_ids, assume_unique=True)
        ages = np.maximum(0, int(final_timestamp) - creation_timestamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            creation_id=creation_id,
            parent_id=parent_id,
            creation_timestamp=creation_timestamp,
            creation_kind=creation_kind,
            alive=alive,
            age=ages,
            deletion_id=deletion_id,
            deletion_timestamp=deletion_timestamp,
            deletion_reason=deletion_reason,
            final_current_ids=current_ids,
        )
        return {
            "created": int(creation_id.size),
            "created_new": int(new_mask.sum()),
            "alive_created": int(alive.sum()),
            "split_parent_deletions": int(deletion_id.size),
            "lineage_path": str(path),
        }


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {key: 0.0 for key in ("q05", "q50", "q95", "max", "mean")}
    quantiles = torch.quantile(finite.float(), torch.tensor([0.05, 0.5, 0.95], device=finite.device))
    return {
        "q05": float(quantiles[0].item()),
        "q50": float(quantiles[1].item()),
        "q95": float(quantiles[2].item()),
        "max": float(finite.max().item()),
        "mean": float(finite.float().mean().item()),
    }


def sample_importance_for_plot(values: torch.Tensor, limit: int = 10_000) -> np.ndarray:
    positive = values[values > 0].detach().float()
    if positive.numel() > limit:
        indices = torch.linspace(0, positive.numel() - 1, limit, device=positive.device).long()
        positive = positive[indices]
    return positive.cpu().numpy()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
                    for key, value in row.items()
                }
            )


def write_plots(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    importance_samples: Sequence[np.ndarray],
    lineage_path: Path,
) -> list[str]:
    timestamps = np.asarray([row["timestamp"] for row in rows])
    paths: list[str] = []

    def font(size: int) -> ImageFont.ImageFont:
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()

    def line_plot(
        name: str,
        title: str,
        series: dict[str, Sequence[float]],
        *,
        log1p: bool = False,
    ) -> None:
        width, height = 1200, 500
        margin = (80, 65, 25, 65)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        left, top, right, bottom = margin[0], margin[1], width - margin[2], height - margin[3]
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        palette = ((24, 95, 180), (220, 55, 45), (30, 155, 75), (145, 70, 190))
        transformed = {
            key: np.log1p(np.asarray(values, dtype=np.float64)) if log1p else np.asarray(values, dtype=np.float64)
            for key, values in series.items()
        }
        maxima = [float(np.max(values)) for values in transformed.values() if len(values)]
        minima = [float(np.min(values)) for values in transformed.values() if len(values)]
        ymin = min(minima, default=0.0)
        ymax = max(maxima, default=1.0)
        if ymax <= ymin:
            ymax = ymin + 1.0
        xden = max(1, len(timestamps) - 1)
        for color, (label, values) in zip(palette, transformed.items()):
            points = []
            for index, value in enumerate(values):
                x = left + (right - left) * index / xden
                y = bottom - (bottom - top) * (float(value) - ymin) / (ymax - ymin)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            legend_x = left + 215 * list(transformed).index(label)
            draw.rectangle((legend_x, height - 42, legend_x + 18, height - 24), fill=color)
            draw.text((legend_x + 25, height - 45), label, fill="black", font=font(16))
        draw.text((left, 15), title, fill="black", font=font(24))
        draw.text((left, bottom + 12), "global timestamp", fill="black", font=font(16))
        draw.text((5, top), f"range {ymin:.3g}..{ymax:.3g}" + (" (log1p)" if log1p else ""), fill="black", font=font(13))
        path = output_dir / name
        canvas.save(path)
        paths.append(str(path))

    def histogram_plot(name: str, title: str, groups: dict[str, np.ndarray], bins: int = 60) -> None:
        width, height = 1000, 500
        left, top, right, bottom = 75, 65, 975, 430
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        all_values = np.concatenate([value for value in groups.values() if value.size]) if any(value.size for value in groups.values()) else np.asarray([0.0])
        vmin, vmax = float(all_values.min()), float(all_values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        palette = ((24, 95, 180), (220, 55, 45), (30, 155, 75))
        histograms = {
            key: np.histogram(value, bins=bins, range=(vmin, vmax))[0]
            for key, value in groups.items()
        }
        max_count = max((int(hist.max()) for hist in histograms.values()), default=1)
        max_count = max(max_count, 1)
        bar_width = (right - left) / bins
        for color, (label, hist) in zip(palette, histograms.items()):
            for index, count in enumerate(hist):
                x0 = left + index * bar_width
                y = bottom - (bottom - top) * math.log1p(int(count)) / math.log1p(max_count)
                draw.line((x0, bottom, x0, y), fill=color, width=max(1, int(bar_width)))
            lx = left + 220 * list(histograms).index(label)
            draw.rectangle((lx, height - 42, lx + 18, height - 24), fill=color)
            draw.text((lx + 25, height - 45), label, fill="black", font=font(16))
        draw.text((left, 15), title, fill="black", font=font(24))
        draw.text((left, bottom + 12), f"value range {vmin:.3g}..{vmax:.3g}; y=log count", fill="black", font=font(15))
        path = output_dir / name
        canvas.save(path)
        paths.append(str(path))

    line_plot(
        "per_frame_gaussian_count.png",
        "Mutable R_change Gaussian count",
        {"Gaussian count": [row["gaussian_count"] for row in rows]},
    )
    line_plot(
        "densification_candidate_histogram.png",
        "Densification candidates and mutations",
        {
            "clone": [row["clone_count"] for row in rows],
            "split": [row["split_count"] for row in rows],
            "importance": [row["importance_candidate_count"] for row in rows],
        },
        log1p=True,
    )
    samples = np.concatenate(importance_samples) if importance_samples else np.asarray([], dtype=np.float32)
    histogram_plot(
        "multi_view_importance_histogram.png",
        "Soft alpha-T multi-view change importance",
        {"importance": samples},
    )
    line_plot(
        "fn_fp_evolution.png",
        "Post-inference FP/FN evolution",
        {
            "FP": [row.get("fp", 0) for row in rows],
            "FN": [row.get("fn", 0) for row in rows],
        },
    )
    lineage = np.load(lineage_path)
    ages = lineage["age"]
    alive = lineage["alive"].astype(bool)
    histogram_plot(
        "new_gaussian_age_survival_histogram.png",
        "New-GS age and survival",
        {
            "alive": ages[alive],
            "removed/split": ages[~alive],
        },
        bins=40,
    )
    return paths


def validate_args(args: argparse.Namespace) -> None:
    if args.condition not in {"baseline", "fastgs_gradient_only", "cue_vcd"}:
        raise ValueError("condition must be baseline, fastgs_gradient_only, or cue_vcd")
    if args.scope not in SCOPE_MAX_FRAMES:
        raise ValueError("unknown ESCD scope")
    if args.max_frames > SCOPE_MAX_FRAMES[args.scope]:
        raise ValueError(f"max_frames exceeds the {args.scope} scope")
    if args.oracle_state_local_density_views and args.scope != "continuous":
        raise ValueError("oracle state-local density views require continuous scope")
    if not 0 <= args.densify_update_index < args.updates_per_frame:
        raise ValueError("densify_update_index must be within the per-frame update schedule")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_online_bayesian_lifespan_thaw import (
        build_causal_records,
        evaluate_after_inference,
    )
    from experiments.train_cue_temporal_rchange import (
        build_fixed_cue_views,
        load_fixed_camera_index,
        validate_cue_cache,
    )
    from experiments.train_real_temporal_rchange import (
        oscd_positive_sparsity_loss,
        seed_everything,
    )
    from scene import GaussianModel

    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records, _names = build_causal_records(args.source_path)
    records = select_scope_records(
        all_records,
        scope=args.scope,
        max_frames=int(args.max_frames),
    )
    base_ply = (args.source_path / BASE_PLY_REL).resolve()
    immutable_hash_before = file_sha256(base_ply)
    validate_cue_cache(args.cue_cache_root, base_ply, args.resolution)
    cameras = load_fixed_camera_index(args.fixed_cameras_json)
    extent = reference_scene_extent(args.source_path / "reference_reconstruction/cameras.json")

    change_bank = GaussianModel(sh_degree=3, active_sh_degree=0)
    change_bank.load_ply_change(str(base_ply))
    change_bank.spatial_lr_scale = float(args.spatial_lr_scale)
    opt = optimization_config(args)
    if args.condition == "baseline":
        change_bank.training_setup_change(opt)
        change_bank.xyz_gradient_accum_abs = torch.zeros_like(change_bank.xyz_gradient_accum)
    else:
        change_bank.training_setup_update(opt)
    initial_count = int(change_bank.get_xyz.shape[0])
    lineage = LineageTracker(initial_count, change_bank.get_xyz.device)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, device="cuda", dtype=change_bank.get_xyz.dtype)

    processed_views: list[Any] = []
    predictions: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    density_events: list[dict[str, Any]] = []
    importance_samples: list[np.ndarray] = []
    global_update = 0
    total_clones = 0
    total_splits = 0
    cross_state_density_view_access_count = 0

    for record in records:
        frame_started = time.time()
        timestamp = int(record.global_index)
        current_view = build_fixed_cue_views(
            [record], cameras, args.cue_cache_root, args.resolution
        )[0][0]
        processed_views.append(current_view)
        frame_clone = 0
        frame_split = 0
        event: dict[str, Any] | None = None
        last_loss = 0.0

        for update_index in range(int(args.updates_per_frame)):
            training_index = deterministic_training_view_index(
                timestamp, update_index, seed=int(args.seed)
            )
            if training_index > timestamp:
                raise RuntimeError("future training view accessed")
            view = processed_views[training_index]
            change_bank.update_learning_rate(global_update)
            package = render_change(view, change_bank, pipe, background)
            loss, _parts = oscd_positive_sparsity_loss(
                view.training_target, package["render"]
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite R_change loss")
            loss.backward()
            change_bank.optimizer.step()
            change_bank.optimizer.zero_grad(set_to_none=True)
            last_loss = float(loss.detach().item())
            visible = package["visibility_filter"].flatten()
            with torch.no_grad():
                change_bank.max_radii2D[visible] = torch.maximum(
                    change_bank.max_radii2D[visible], package["radii"][visible]
                )
                if args.condition == "baseline":
                    change_bank.add_densification_stats(
                        package["viewspace_points"], visible
                    )
                else:
                    change_bank.add_densification_stats_fastgs(
                        package["viewspace_points"], visible
                    )

            if update_index == int(args.densify_update_index):
                if args.condition == "baseline":
                    density = apply_oscd_gradient_only_densification(
                        change_bank,
                        package["radii"],
                        scene_extent=extent,
                        grad_threshold=float(args.oscd_grad_threshold),
                    )
                    selected_indices: tuple[int, ...] = ()
                    importance_stats = {key: 0.0 for key in ("q05", "q50", "q95", "max", "mean")}
                    ratio_stats = dict(importance_stats)
                elif args.condition == "fastgs_gradient_only":
                    selected_indices = ()
                    importance_stats = {key: 0.0 for key in ("q05", "q50", "q95", "max", "mean")}
                    ratio_stats = dict(importance_stats)
                    density = apply_fastgs_change_densification(
                        change_bank,
                        package["radii"],
                        torch.full(
                            (change_bank.get_xyz.shape[0],),
                            float(args.importance_threshold) + 1.0,
                            device=change_bank.get_xyz.device,
                            dtype=change_bank.get_xyz.dtype,
                        ),
                        scene_extent=extent,
                        importance_threshold=float(args.importance_threshold),
                        grad_threshold=float(args.fastgs_grad_threshold),
                        grad_abs_threshold=float(args.fastgs_grad_abs_threshold),
                        dense_fraction=float(args.fastgs_dense_fraction),
                    )
                else:
                    if args.oracle_state_local_density_views:
                        selected_indices = oracle_state_local_view_indices(
                            records[: timestamp + 1],
                            timestamp,
                            int(args.k_views),
                            seed=int(args.seed),
                        )
                    else:
                        selected_indices = causal_random_view_indices(
                            timestamp,
                            int(args.k_views),
                            seed=int(args.seed),
                        )
                    if max(selected_indices) > timestamp:
                        raise RuntimeError("future density-score view accessed")
                    cross_state_count = sum(
                        records[index].segment_name != record.segment_name
                        for index in selected_indices
                    )
                    cross_state_density_view_access_count += cross_state_count
                    if args.oracle_state_local_density_views and cross_state_count:
                        raise RuntimeError("oracle density sample crossed a state boundary")
                    score = compute_soft_multiview_change_score(
                        processed_views,
                        selected_indices,
                        change_bank,
                        pipe,
                        background,
                        cue_scale=float(args.cue_scale),
                    )
                    importance_stats = quantile_summary(score.importance_score)
                    ratio_stats = quantile_summary(score.change_ratio[score.total_mass > 0])
                    importance_samples.append(sample_importance_for_plot(score.importance_score))
                    density = apply_fastgs_change_densification(
                        change_bank,
                        package["radii"],
                        score.importance_score,
                        scene_extent=extent,
                        importance_threshold=float(args.importance_threshold),
                        grad_threshold=float(args.fastgs_grad_threshold),
                        grad_abs_threshold=float(args.fastgs_grad_abs_threshold),
                        dense_fraction=float(args.fastgs_dense_fraction),
                    )
                lineage.apply_densification(density, timestamp)
                frame_clone += density.clone_count
                frame_split += density.split_count
                total_clones += density.clone_count
                total_splits += density.split_count
                event = {
                    "timestamp": timestamp,
                    "selected_view_indices": list(selected_indices),
                    "selected_view_segments": [
                        records[index].segment_name for index in selected_indices
                    ],
                    "selected_view_count": len(selected_indices),
                    "initial_count": density.initial_count,
                    "final_count": density.final_count,
                    "importance_candidate_count": (
                        int(density.masks.importance.sum().item())
                        if args.condition == "cue_vcd"
                        else 0
                    ),
                    "gradient_clone_candidate_count": int(density.masks.clone_gradient.sum().item()),
                    "gradient_split_candidate_count": int(density.masks.split_gradient.sum().item()),
                    "clone_count": density.clone_count,
                    "split_count": density.split_count,
                    "importance": importance_stats,
                    "change_ratio": ratio_stats,
                    "future_view_access_count": 0,
                    "cross_state_density_view_access_count": (
                        cross_state_count if args.condition == "cue_vcd" else 0
                    ),
                }
                density_events.append(event)
            global_update += 1

        with torch.no_grad():
            rendered = render_change(
                current_view, change_bank, pipe, background
            )["render"].mean(dim=0)
            prediction = (rendered > float(args.evaluation_threshold)).cpu().numpy()
        predictions.append(prediction)
        event = event or {
            "importance_candidate_count": 0,
            "gradient_clone_candidate_count": 0,
            "gradient_split_candidate_count": 0,
            "importance": {key: 0.0 for key in ("q05", "q50", "q95", "max", "mean")},
            "change_ratio": {key: 0.0 for key in ("q05", "q50", "q95", "max", "mean")},
        }
        frame_rows.append(
            {
                "timestamp": timestamp,
                "frame": record.name,
                "segment": record.segment_name,
                "condition": args.condition,
                "k_views": int(args.k_views) if args.condition == "cue_vcd" else 0,
                "gaussian_count": int(change_bank.get_xyz.shape[0]),
                "clone_count": frame_clone,
                "split_count": frame_split,
                "pruned_count": 0,
                "importance_candidate_count": event["importance_candidate_count"],
                "gradient_clone_candidate_count": event["gradient_clone_candidate_count"],
                "gradient_split_candidate_count": event["gradient_split_candidate_count"],
                "importance_q50": event["importance"]["q50"],
                "importance_q95": event["importance"]["q95"],
                "change_ratio_q50": event["change_ratio"]["q50"],
                "change_ratio_q95": event["change_ratio"]["q95"],
                "loss": last_loss,
                "predicted_positive_fraction": float(prediction.mean()),
                "frame_runtime_seconds": time.time() - frame_started,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "future_view_access_count": 0,
                "cross_state_density_view_access_count": event.get(
                    "cross_state_density_view_access_count", 0
                ),
            }
        )

    # Post-inference only: GT is first read inside this call.
    metrics = evaluate_after_inference(
        args.source_path, records, predictions, frame_rows
    )
    lineage_path = args.output_dir / "new_gaussian_lineage.npz"
    lineage_summary = lineage.save(
        lineage_path,
        final_timestamp=int(records[-1].global_index),
        initial_count=initial_count,
    )
    audit = topology_integrity(change_bank)
    immutable_hash_after = file_sha256(base_ply)
    if immutable_hash_after != immutable_hash_before:
        raise RuntimeError("immutable reference PLY changed")
    if not audit["passed"]:
        raise RuntimeError(f"final topology audit failed: {audit}")
    write_csv(args.output_dir / "frame_metrics.csv", frame_rows)
    with (args.output_dir / "density_events.jsonl").open("w", encoding="utf-8") as file:
        for event in density_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    plots = write_plots(args.output_dir, frame_rows, importance_samples, lineage_path)
    if args.save_final_ply:
        change_bank.save_ply(str(args.output_dir / "final_rchange.ply"))

    summary = {
        "schema_version": 1,
        "script": "experiments/run_ref_sc1_change_cue_density.py",
        "scope": SCOPE_LABELS[args.scope],
        "scope_key": args.scope,
        "condition": args.condition,
        "k_views": int(args.k_views) if args.condition == "cue_vcd" else None,
        "cue_contract": (
            "raw O-SCD pixel+SAM ref-inf cue in [0,1]; no >0.5 threshold"
            if args.condition == "cue_vcd"
            else "not used by this gradient-only control"
        ),
        "score_equation": (
            "importance_i = sum_j sum_p alpha_i(p) T_i(p) C_j(p) / K"
            if args.condition == "cue_vcd"
            else None
        ),
        "view_sampling": (
            (
                "current view plus K-1 random already-processed views from the same oracle state; no future views"
                if args.oracle_state_local_density_views
                else "current view plus K-1 random already-processed views; no future views"
            )
            if args.condition == "cue_vcd"
            else "not applicable"
        ),
        "fastgs_source_commit": FASTGS_SOURCE_COMMIT,
        "fastgs_preserved": [
            "multi-view sum/K aggregation",
            "gradient AND importance intersection",
            "small clone / large abs-gradient split",
            "optimizer-safe FastGS clone/split topology updates",
        ],
        "fastgs_changed": [
            "normalized RGB L1 threshold map -> soft alpha-T ref-inf change cue",
            "offline random train views -> causal already-processed ESCD views",
            "cue-based pruning disabled",
        ],
        "temporal_topology_integration": False,
        "temporal_topology_blocker": "TemporalGeometryChangeModel is fixed [N,S]; mutable density requires a separate residual bank",
        "gt_used_in_causal_loop": False,
        "gt_loaded_after_inference_only": True,
        "manual_boundaries_used": bool(args.oracle_state_local_density_views),
        "oracle_state_labels_used_for_density_sampling_only": bool(
            args.oracle_state_local_density_views
        ),
        "oracle_state_labels_used_for_training_replay": False,
        "initial_gaussian_count": initial_count,
        "final_gaussian_count": int(change_bank.get_xyz.shape[0]),
        "total_clones": int(total_clones),
        "total_splits": int(total_splits),
        "total_pruned": 0,
        "metrics": metrics,
        "lineage": lineage_summary,
        "topology_integrity": audit,
        "immutable_reference_ply_sha256": immutable_hash_before,
        "immutable_reference_unchanged": True,
        "future_view_access_count": 0,
        "cross_state_density_view_access_count": int(
            cross_state_density_view_access_count
        ),
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "plots": plots,
        "run_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "output_files": [
            "summary.json",
            "frame_metrics.csv",
            "density_events.jsonl",
            "new_gaussian_lineage.npz",
            *[Path(path).name for path in plots],
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fixed-cameras-json", type=Path, default=DEFAULT_FIXED_CAMERAS)
    parser.add_argument("--cue-cache-root", type=Path, default=DEFAULT_CUE_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("baseline", "fastgs_gradient_only", "cue_vcd"),
        required=True,
    )
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPE_MAX_FRAMES),
        default="scene_change1",
    )
    parser.add_argument("--oracle-state-local-density-views", action="store_true")
    parser.add_argument("--k-views", type=positive_int, default=10)
    parser.add_argument("--max-frames", type=positive_int, default=95)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--updates-per-frame", type=positive_int, default=16)
    parser.add_argument("--densify-update-index", type=int, default=4)
    parser.add_argument("--cue-scale", type=float, default=1.0)
    parser.add_argument("--importance-threshold", type=nonnegative_float, default=5.0)
    parser.add_argument("--oscd-grad-threshold", type=nonnegative_float, default=0.001)
    parser.add_argument("--fastgs-grad-threshold", type=nonnegative_float, default=0.0002)
    parser.add_argument("--fastgs-grad-abs-threshold", type=nonnegative_float, default=0.0012)
    parser.add_argument("--fastgs-dense-fraction", type=float, default=0.001)
    parser.add_argument("--percent-dense", type=float, default=0.01)
    parser.add_argument("--spatial-lr-scale", type=nonnegative_float, default=0.0)
    parser.add_argument("--dc-lr", type=nonnegative_float, default=0.0025)
    parser.add_argument("--xyz-lr", type=nonnegative_float, default=0.00016)
    parser.add_argument("--xyz-lr-final", type=nonnegative_float, default=0.0000016)
    parser.add_argument("--opacity-lr", type=nonnegative_float, default=0.025)
    parser.add_argument("--scaling-lr", type=nonnegative_float, default=0.005)
    parser.add_argument("--rotation-lr", type=nonnegative_float, default=0.001)
    parser.add_argument("--evaluation-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-final-ply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "condition": summary["condition"],
                "k": summary["k_views"],
                "mIoU": summary["metrics"]["mean_frame_iou"],
                "F1": summary["metrics"]["mean_frame_f1"],
                "initial_GS": summary["initial_gaussian_count"],
                "final_GS": summary["final_gaussian_count"],
                "clones": summary["total_clones"],
                "splits": summary["total_splits"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
