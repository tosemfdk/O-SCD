#!/usr/bin/env python3
"""Compare DA3Metric-Large and DA3-Small on immutable reference views.

Both models receive the same single reference image.  Their depth is aligned
to scene units with the same XFeat-to-COLMAP 3D camera-z anchors, then compared
against the immutable reference-GS rendered depth.  Metrics are reported both
with a per-view scale and with the first reference view's scale locked for all
subsequent views.  Dense holdout metrics exclude a neighbourhood around every
XFeat scale anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.analyze_da3_depth_prior_new_seeds import (
    _read_resized_rgb,
    _resize_tensor,
    render_reference_depth,
)
from experiments.build_causal_da3_seed_replay import (
    _resize_intrinsics_to_depth_grid,
    localization_depth_anchors,
)
from experiments.train_real_temporal_rchange import (
    BASE_PLY_REL,
    load_reference_frames,
    load_rgb_tensor,
)
from poses.feature_detector import Detector
from scene import GaussianModel
from scene.cameras import Camera
from temporal.depth_prior_new_seeding import (
    canonical_metric_depth,
    fit_metric_anchor_depth_scale,
)


DEFAULT_SOURCE = Path("data/Instance_1/scene_change1_2_3")
DEFAULT_OUTPUT = Path(
    "outputs/reference_da3metric_large_vs_small_xfeat_20260904"
)


@dataclass(frozen=True)
class ReferenceEvaluationView:
    name: str
    image: np.ndarray
    image_K: np.ndarray
    w2c: np.ndarray
    anchor_pixels_image: np.ndarray
    anchor_world: np.ndarray
    rendered_depth: np.ndarray
    rendered_alpha: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric-model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--small-model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--reference-views", type=int, default=16)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--xfeat-top-k", type=int, default=512)
    parser.add_argument("--reference-association-px", type=float, default=3.0)
    parser.add_argument("--reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--min-anchors", type=int, default=16)
    parser.add_argument("--min-render-alpha", type=float, default=0.50)
    parser.add_argument("--anchor-holdout-radius-px", type=float, default=8.0)
    args = parser.parse_args()
    for name in ("reference_views", "process_res", "xfeat_top_k", "min_anchors"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "resolution",
        "reference_association_px",
        "reprojection_error_px",
        "anchor_holdout_radius_px",
    ):
        if not math.isfinite(float(getattr(args, name))) or float(
            getattr(args, name)
        ) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0.0 < float(args.min_render_alpha) <= 1.0:
        parser.error("--min-render-alpha must lie in (0,1]")
    return args


def make_reference_camera(reference: Any, image: torch.Tensor, index: int) -> Camera:
    return Camera(
        colmap_id=reference.name,
        R=np.asarray(reference.Rt[:3, :3]).T,
        T=np.asarray(reference.Rt[:3, 3]),
        FoVx=float(reference.fovx),
        FoVy=float(reference.fovy),
        image=image,
        gt_alpha_mask=None,
        image_name=Path(reference.name).stem,
        uid=f"reference_depth_eval_{index:03d}",
    )


def render_evaluation_views(args: argparse.Namespace) -> list[ReferenceEvaluationView]:
    probe_path = next(
        path
        for path in sorted((args.source_path / "reference_scene" / "images").iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    probe = load_rgb_tensor(probe_path, float(args.resolution))
    image_height, image_width = int(probe.shape[1]), int(probe.shape[2])
    del probe
    detector = Detector(
        top_k=int(args.xfeat_top_k), width=image_width, height=image_height
    )
    references, image_K = load_reference_frames(
        args.source_path,
        float(args.resolution),
        int(args.reference_views),
        detector,
        float(args.reference_association_px),
        image_width,
        image_height,
    )
    del detector

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base.load_ply(str(args.source_path / BASE_PLY_REL))
    for parameter in (
        base._xyz,
        base._features_dc,
        base._features_rest,
        base._opacity,
        base._scaling,
        base._rotation,
    ):
        parameter.requires_grad_(False)
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    views: list[ReferenceEvaluationView] = []
    for index, reference in enumerate(references):
        image_tensor = load_rgb_tensor(Path(reference.image_path), float(args.resolution))
        camera = make_reference_camera(reference, image_tensor, index)
        w2c = np.asarray(reference.Rt, dtype=np.float32)
        depth, alpha = render_reference_depth(
            camera, base, torch.from_numpy(w2c), pipe, background
        )
        valid_anchor = (
            reference.desc_kpts.valid.detach().cpu().bool()
            & reference.has_pt3d.detach().cpu().bool()
        )
        views.append(
            ReferenceEvaluationView(
                name=reference.name,
                image=_read_resized_rgb(
                    Path(reference.image_path), image_height, image_width
                ),
                image_K=np.asarray(image_K, dtype=np.float32),
                w2c=w2c,
                anchor_pixels_image=reference.desc_kpts.kpts[valid_anchor]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                anchor_world=reference.pts3d[valid_anchor]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                rendered_depth=depth.numpy().astype(np.float32),
                rendered_alpha=alpha.numpy().astype(np.float32),
            )
        )
        print(
            f"[render {index + 1:02d}/{len(references):02d}] "
            f"{reference.name} anchors={int(valid_anchor.sum())}",
            flush=True,
        )
    del base
    torch.cuda.empty_cache()
    return views


def infer_depths(
    *,
    model_name: str,
    metric: bool,
    views: list[ReferenceEvaluationView],
    output_dir: Path,
    process_res: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from depth_anything_3.api import DepthAnything3

    cache_dir = output_dir / ("da3metric_cache" if metric else "da3small_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None
    predictions: list[tuple[np.ndarray, np.ndarray]] = []
    for index, view in enumerate(views):
        cache_path = cache_dir / f"{Path(view.name).stem}.npz"
        if cache_path.exists():
            with np.load(cache_path) as cache:
                predictions.append(
                    (
                        np.asarray(cache["depth"], dtype=np.float32),
                        np.asarray(cache["K"], dtype=np.float32),
                    )
                )
            continue
        if model is None:
            model = DepthAnything3.from_pretrained(model_name).cuda().eval()
        prediction = model.inference(
            [view.image],
            intrinsics=view.image_K[None],
            align_to_input_ext_scale=False,
            process_res=int(process_res),
            process_res_method="upper_bound_resize",
        )
        raw_depth = np.asarray(prediction.depth[-1], dtype=np.float32)
        depth_height, depth_width = raw_depth.shape
        image_height, image_width = view.image.shape[:2]
        depth_K = _resize_intrinsics_to_depth_grid(
            view.image_K,
            image_height=image_height,
            image_width=image_width,
            depth_height=depth_height,
            depth_width=depth_width,
        )
        depth = (
            canonical_metric_depth(
                torch.from_numpy(raw_depth), torch.from_numpy(depth_K)
            ).numpy()
            if metric
            else raw_depth
        ).astype(np.float32)
        np.savez_compressed(
            cache_path,
            depth=depth,
            raw_depth=raw_depth,
            K=depth_K,
            model=np.asarray(model_name),
        )
        predictions.append((depth, depth_K))
        print(
            f"[{Path(model_name).name} {index + 1:02d}/{len(views):02d}]",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    return predictions


def anchor_holdout_mask(
    shape: tuple[int, int], pixels_xy: torch.Tensor, radius_px: float
) -> torch.Tensor:
    mask = np.ones(shape, dtype=np.uint8)
    radius = max(1, int(round(float(radius_px))))
    for x, y in pixels_xy.detach().cpu().numpy():
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 0, -1)
    return torch.from_numpy(mask.astype(bool))


def depth_metrics(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: float
) -> dict[str, float | int]:
    aligned = predicted.detach().cpu().float() * float(scale)
    target = target.detach().cpu().float()
    valid = mask.detach().cpu().bool()
    valid &= torch.isfinite(aligned) & torch.isfinite(target)
    valid &= (aligned > 1.0e-8) & (target > 1.0e-8)
    if not bool(valid.any()):
        raise ValueError("depth evaluation mask has no valid pixels")
    prediction = aligned[valid]
    truth = target[valid]
    relative = (prediction - truth).abs() / truth.clamp_min(1.0e-8)
    ratio = torch.maximum(
        prediction / truth.clamp_min(1.0e-8),
        truth / prediction.clamp_min(1.0e-8),
    )
    log_error = torch.log(prediction) - torch.log(truth)
    return {
        "pixels": int(valid.sum().item()),
        "median_abs_rel": float(relative.median().item()),
        "mean_abs_rel": float(relative.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((prediction - truth) ** 2)).item()),
        "log_rmse": float(torch.sqrt(torch.mean(log_error**2)).item()),
        "delta_1_25": float((ratio < 1.25).float().mean().item()),
    }


def evaluate_model(
    *,
    label: str,
    views: list[ReferenceEvaluationView],
    predictions: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for view, (depth_array, depth_K) in zip(views, predictions):
        depth = torch.from_numpy(depth_array)
        height, width = depth.shape
        anchors = localization_depth_anchors(
            points2d=view.anchor_pixels_image,
            points3d=view.anchor_world,
            image_K=torch.from_numpy(view.image_K),
            depth_K=torch.from_numpy(depth_K),
            w2c=torch.from_numpy(view.w2c),
            depth_height=height,
            depth_width=width,
            max_reprojection_error_px=float(args.reprojection_error_px),
        )
        fit = fit_metric_anchor_depth_scale(
            depth,
            anchors.pixels_xy,
            anchors.camera_depth,
            min_samples=int(args.min_anchors),
        )
        numerator = _resize_tensor(
            torch.from_numpy(view.rendered_depth * view.rendered_alpha),
            height,
            width,
            "area",
        )
        alpha = _resize_tensor(
            torch.from_numpy(view.rendered_alpha), height, width, "area"
        )
        rendered_depth = numerator / alpha.clamp_min(1.0e-6)
        valid = alpha >= float(args.min_render_alpha)
        holdout = valid & anchor_holdout_mask(
            (height, width),
            anchors.pixels_xy,
            float(args.anchor_holdout_radius_px)
            * float(width)
            / float(view.image.shape[1]),
        )
        prepared.append(
            {
                "view": view,
                "depth": depth,
                "rendered_depth": rendered_depth,
                "valid": valid,
                "holdout": holdout,
                "anchors": anchors,
                "fit": fit,
            }
        )

    first_scale = float(prepared[0]["fit"].scale)
    bank_median_scale = float(
        np.median([float(item["fit"].scale) for item in prepared])
    )
    rows: list[dict[str, Any]] = []
    for item in prepared:
        fit = item["fit"]
        for policy, scale in (
            ("per_view_xfeat", float(fit.scale)),
            ("first_view_locked", first_scale),
            ("reference_bank_median_locked", bank_median_scale),
        ):
            all_metrics = depth_metrics(
                item["depth"], item["rendered_depth"], item["valid"], scale
            )
            holdout_metrics = depth_metrics(
                item["depth"], item["rendered_depth"], item["holdout"], scale
            )
            rows.append(
                {
                    "model": label,
                    "frame_name": item["view"].name,
                    "scale_policy": policy,
                    "applied_scale": scale,
                    "per_view_fitted_scale": float(fit.scale),
                    "scale_anchor_samples": int(fit.samples),
                    "scale_anchor_inliers": int(fit.inliers),
                    "scale_anchor_median_abs_rel": float(
                        fit.median_absolute_relative_error
                    ),
                    **{f"all_{key}": value for key, value in all_metrics.items()},
                    **{
                        f"holdout_{key}": value
                        for key, value in holdout_metrics.items()
                    },
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        summary[model] = {}
        model_rows = [row for row in rows if row["model"] == model]
        fitted_scales = np.asarray(
            [
                row["per_view_fitted_scale"]
                for row in model_rows
                if row["scale_policy"] == "per_view_xfeat"
            ],
            dtype=np.float64,
        )
        for policy in (
            "per_view_xfeat",
            "first_view_locked",
            "reference_bank_median_locked",
        ):
            selected = [row for row in model_rows if row["scale_policy"] == policy]
            summary[model][policy] = {
                "frames": len(selected),
                "mean_frame_all_median_abs_rel": float(
                    np.mean([row["all_median_abs_rel"] for row in selected])
                ),
                "median_frame_all_median_abs_rel": float(
                    np.median([row["all_median_abs_rel"] for row in selected])
                ),
                "mean_frame_holdout_median_abs_rel": float(
                    np.mean([row["holdout_median_abs_rel"] for row in selected])
                ),
                "median_frame_holdout_median_abs_rel": float(
                    np.median([row["holdout_median_abs_rel"] for row in selected])
                ),
                "mean_frame_holdout_mean_abs_rel": float(
                    np.mean([row["holdout_mean_abs_rel"] for row in selected])
                ),
                "mean_frame_holdout_delta_1_25": float(
                    np.mean([row["holdout_delta_1_25"] for row in selected])
                ),
            }
        summary[model]["scale_consistency"] = {
            "min": float(fitted_scales.min()),
            "median": float(np.median(fitted_scales)),
            "max": float(fitted_scales.max()),
            "mean": float(fitted_scales.mean()),
            "std": float(fitted_scales.std()),
            "coefficient_of_variation": float(
                fitted_scales.std() / fitted_scales.mean()
            ),
        }

    for policy in (
        "per_view_xfeat",
        "first_view_locked",
        "reference_bank_median_locked",
    ):
        metric = {
            row["frame_name"]: row
            for row in rows
            if row["model"] == "DA3Metric-Large"
            and row["scale_policy"] == policy
        }
        small = {
            row["frame_name"]: row
            for row in rows
            if row["model"] == "DA3-Small"
            and row["scale_policy"] == policy
        }
        shared = sorted(set(metric) & set(small))
        differences = np.asarray(
            [
                small[name]["holdout_median_abs_rel"]
                - metric[name]["holdout_median_abs_rel"]
                for name in shared
            ],
            dtype=np.float64,
        )
        summary[f"paired_{policy}"] = {
            "frames": len(shared),
            "metric_better": int((differences > 0.0).sum()),
            "small_better": int((differences < 0.0).sum()),
            "ties": int((np.abs(differences) < 1.0e-12).sum()),
            "median_abs_rel_advantage_positive_for_metric": float(
                np.median(differences)
            ),
        }
    return summary


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    policies = (
        "per_view_xfeat",
        "first_view_locked",
        "reference_bank_median_locked",
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = {"DA3Metric-Large": "#d95f02", "DA3-Small": "#1b9e77"}
    for axis, policy in zip(axes, policies):
        for model in ("DA3Metric-Large", "DA3-Small"):
            selected = [
                row
                for row in rows
                if row["model"] == model and row["scale_policy"] == policy
            ]
            axis.plot(
                np.arange(1, len(selected) + 1),
                [row["holdout_median_abs_rel"] for row in selected],
                marker="o",
                linewidth=1.8,
                label=model,
                color=colors[model],
            )
        axis.set_title(policy.replace("_", " "))
        axis.set_xlabel("uniform reference view index")
        axis.set_ylabel("dense holdout median AbsRel")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = render_evaluation_views(args)
    metric_predictions = infer_depths(
        model_name=args.metric_model,
        metric=True,
        views=views,
        output_dir=args.output_dir,
        process_res=int(args.process_res),
    )
    small_predictions = infer_depths(
        model_name=args.small_model,
        metric=False,
        views=views,
        output_dir=args.output_dir,
        process_res=int(args.process_res),
    )
    rows = [
        *evaluate_model(
            label="DA3Metric-Large",
            views=views,
            predictions=metric_predictions,
            args=args,
        ),
        *evaluate_model(
            label="DA3-Small",
            views=views,
            predictions=small_predictions,
            args=args,
        ),
    ]
    with (args.output_dir / "per_view_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment": "reference-view DA3Metric-Large versus DA3-Small",
        "contract": {
            "input": "same single immutable-reference image",
            "scale": "same XFeat-associated COLMAP 3D camera-z anchors",
            "target": "immutable reference-GS alpha-weighted rendered camera-z",
            "holdout": (
                "dense valid render pixels excluding XFeat anchor neighbourhoods"
            ),
            "metric_depth_conversion": "raw_depth * mean(fx,fy) / 300",
            "ground_truth_change_masks_used": False,
        },
        "configuration": {key: str(value) for key, value in vars(args).items()},
        "summary": summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    save_plot(rows, args.output_dir / "depth_error_comparison.png")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
