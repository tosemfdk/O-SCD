#!/usr/bin/env python3
"""Run local-cue O-SCD with element-wise previous-update replay.

The original online sampler is preserved.  An ordinary sampled view receives
the B local loss.  When the selected view is the newest frame t, the
same optimizer step also renders t-1 and averages the B losses from t and t-1.
The t-1 rendering receives one extra pixel-wise regularizer weighted by the
positive mask growth caused by t-1's earlier frame-update block.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSCD_REPO = REPO_ROOT.parent / "O-SCD"


def pop_experiment_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--oscd-repo", type=Path, default=DEFAULT_OSCD_REPO)
    parser.add_argument(
        "--fusion-variant",
        choices=("sum", "power_product"),
        default="power_product",
    )
    parser.add_argument("--power-cue-dir", type=Path)
    parser.add_argument("--cue-scale", type=float)
    parser.add_argument("--local-support-scale", type=float, default=2.0)
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    parser.add_argument("--growth-replay-weight", type=float, default=1.0)
    parser.add_argument(
        "--save-continuous-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-growth-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    known, remaining = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0], *remaining]
    return known


def load_fusion_helpers():
    module_path = REPO_ROOT / "temporal" / "fusion.py"
    spec = importlib.util.spec_from_file_location(
        "oscd_evolving_fusion_loss", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load loss module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_ssf_loss, module.compute_growth_replay_regularization


def model_output_path(argv: list[str]) -> Path:
    for index, argument in enumerate(argv[:-1]):
        if argument in {"-m", "--model_path"}:
            return Path(argv[index + 1])
    raise RuntimeError("The downstream O-SCD command requires -m/--model_path")


override = pop_experiment_args(sys.argv)
if override.cue_scale is None:
    override.cue_scale = 1.0 if override.fusion_variant == "sum" else 2.0
if override.fusion_variant == "power_product" and override.power_cue_dir is None:
    raise ValueError("power_product requires --power-cue-dir")
if override.cue_scale <= 0.0:
    raise ValueError("cue-scale must be positive")
if override.local_support_scale <= 0.0:
    raise ValueError("local-support-scale must be positive")
if override.regularization_weight < 0.0:
    raise ValueError("regularization-weight must be nonnegative")
if override.growth_replay_weight < 0.0:
    raise ValueError("growth-replay-weight must be nonnegative")

compute_ssf_loss, compute_growth_replay_regularization = load_fusion_helpers()
oscd_protocol_path = override.oscd_repo / "oscd_protocols.py"
if not oscd_protocol_path.is_file():
    raise FileNotFoundError(oscd_protocol_path)
sys.path.insert(0, str(override.oscd_repo))
import oscd_protocols as protocol  # noqa: E402


_load_original_sum_cue = protocol.load_cached_cue


def load_experiment_cue(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor:
    if override.fusion_variant == "sum":
        return _load_original_sum_cue(path, expected_shape).mul_(override.cue_scale)
    assert override.power_cue_dir is not None
    candidates = (
        override.power_cue_dir / path.name,
        override.power_cue_dir / "cues" / path.name,
    )
    cue_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if cue_path is None:
        raise FileNotFoundError(
            f"No product cue for {path.name} under {override.power_cue_dir}"
        )
    value = torch.load(cue_path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Invalid product cue payload: {cue_path}")
    value = value.float()
    if tuple(value.shape) != (1, *expected_shape):
        raise ValueError(
            f"Product cue shape mismatch for {path.name}: "
            f"{tuple(value.shape)} != {(1, *expected_shape)}"
        )
    return value.contiguous().cuda().mul_(override.cue_scale)


def render_local_b_loss(viewpoint, model, pipe, background):
    package = protocol.render_change(viewpoint, model, pipe, background)
    cue = viewpoint.candidate_map
    support = (cue / override.local_support_scale).clamp(0.0, 1.0)
    loss, parts = compute_ssf_loss(
        cue,
        package["render"],
        regularizer_offset=1.000000001,
        regularization_mode="local",
        local_support=support,
        regularization_weight=override.regularization_weight,
    )
    return loss, package, parts, support


def render_probability(view, model, pipe, background) -> torch.Tensor:
    rendered = protocol.render_change(view, model, pipe, background)["render"]
    return torch.sigmoid(rendered.mean(dim=0, keepdim=True))


def save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().float().cpu().contiguous(), temporary)
    os.replace(temporary, path)


def write_mask_and_score(path: Path, view, model, pipe, background) -> None:
    with torch.no_grad():
        rendered = protocol.render_change(view, model, pipe, background)["render"]
        score = rendered.mean(dim=0).clamp(0.0, 1.0)
        binary = score.gt(0.5).to(torch.uint8).mul_(255)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), binary.cpu().numpy()):
        raise OSError(f"Failed to write mask: {path}")
    if override.save_continuous_scores and path.parent.name == "online_at_arrival":
        output_root = path.parents[2]
        save_tensor_atomic(
            output_root
            / "continuous_scores"
            / "online_at_arrival"
            / f"{path.stem}.pt",
            score,
        )


audit: list[dict[str, float | str | int]] = []


def run_online_two_view_growth_replay(
    views,
    reference_centers,
    reference_ply,
    output_root,
    sh_degree,
    opt,
    pipe,
    seed,
    steps_per_frame,
) -> dict:
    protocol.seed_everything(seed)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    arrival_dir = output_root / "renders" / "online_at_arrival"
    final_dir = output_root / "renders" / "online_final_rerender"
    growth_dir = output_root / "previous_update_growth_maps"
    protocol.recreate_directory(arrival_dir)
    if override.save_growth_maps:
        protocol.recreate_directory(growth_dir)
    model = protocol.create_change_model(reference_ply, sh_degree, opt)
    initial_points = int(model.get_xyz.shape[0])
    total_steps = 0
    seen_views = []
    previous_view = None
    previous_growth_map = None
    progress = protocol.tqdm(views, desc="Online two-view growth-replay O-SCD")
    start = time.monotonic()

    for frame_index, current_view in enumerate(progress):
        seen_views.append(current_view)
        with torch.no_grad():
            pre_probability = render_probability(
                current_view, model, pipe, background
            )

        joint_steps = 0
        growth_replay_values: list[float] = []
        for local_step in range(steps_per_frame):
            total_steps += 1
            if np.random.rand() > 0.33:
                viewpoint = seen_views[random.randint(0, len(seen_views) - 1)]
            else:
                viewpoint = seen_views[-1]

            model.update_learning_rate(total_steps)
            selected_loss, package, _, _ = render_local_b_loss(
                viewpoint, model, pipe, background
            )
            loss = selected_loss

            if viewpoint is current_view and previous_view is not None:
                if previous_growth_map is None:
                    raise RuntimeError("previous growth map is missing")
                previous_loss, _, previous_parts, previous_support = (
                    render_local_b_loss(previous_view, model, pipe, background)
                )
                growth_replay = compute_growth_replay_regularization(
                    previous_support,
                    previous_parts["change_probability"],
                    previous_growth_map,
                )
                loss = 0.5 * (selected_loss + previous_loss)
                loss = loss + override.growth_replay_weight * growth_replay
                joint_steps += 1
                growth_replay_values.append(float(growth_replay.detach().item()))

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                protocol.update_densification_statistics(model, package)
                if local_step == 4:
                    grads = model.xyz_gradient_accum / model.denom
                    grads[grads.isnan()] = 0.0
                    model.tmp_radii = package["radii"]
                    extent = protocol.scene_extent(reference_centers, seen_views)
                    model.densify_and_clone(
                        grads, opt.densify_grad_threshold * 5, extent
                    )
                    model.densify_and_split(
                        grads, opt.densify_grad_threshold * 5, extent
                    )
                    model.tmp_radii = None
                    torch.cuda.empty_cache()

        with torch.no_grad():
            post_probability = render_probability(
                current_view, model, pipe, background
            )
            current_growth_map = (post_probability - pre_probability).clamp_min(0.0)
            growth_mean = float(current_growth_map.mean().item())
            growth_max = float(current_growth_map.max().item())
            growth_nonzero_fraction = float(
                current_growth_map.gt(0.0).float().mean().item()
            )
        if override.save_growth_maps:
            save_tensor_atomic(
                growth_dir / f"{current_view.image_name}.pt", current_growth_map
            )

        audit.append(
            {
                "frame_index": frame_index,
                "frame": current_view.image_name,
                "joint_latest_steps": joint_steps,
                "growth_replay_mean_on_joint_steps": (
                    float(np.mean(growth_replay_values))
                    if growth_replay_values
                    else 0.0
                ),
                "measured_growth_mean": growth_mean,
                "measured_growth_max": growth_max,
                "measured_growth_nonzero_fraction": growth_nonzero_fraction,
            }
        )
        previous_view = current_view
        previous_growth_map = current_growth_map.detach()

        progress.set_postfix(
            step=total_steps,
            loss=f"{loss.item():.4f}",
            joint=joint_steps,
            growth=f"{growth_mean:.6f}",
        )
        write_mask_and_score(
            arrival_dir / f"{current_view.image_name}.png",
            current_view,
            model,
            pipe,
            background,
        )

    progress.close()
    protocol.render_all_masks(final_dir, views, model, pipe, background)
    summary = {
        "steps_per_frame": steps_per_frame,
        "total_steps": total_steps,
        "initial_points": initial_points,
        "final_points": int(model.get_xyz.shape[0]),
        "runtime_seconds": time.monotonic() - start,
        "joint_latest_steps": int(sum(row["joint_latest_steps"] for row in audit)),
        "growth_map_mean": float(
            np.mean([row["measured_growth_mean"] for row in audit])
        ),
        "growth_map_max": float(
            max(row["measured_growth_max"] for row in audit)
        ),
    }
    del model
    protocol.gc.collect()
    torch.cuda.empty_cache()
    return summary


output_path = model_output_path(sys.argv)
protocol.load_cached_cue = load_experiment_cue
protocol.write_change_mask = write_mask_and_score
protocol.run_online_protocol = run_online_two_view_growth_replay
protocol.main()

metadata_path = output_path / "run_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["two_view_growth_replay_ablation"] = {
    "fusion_variant": override.fusion_variant,
    "training_cue": (
        "P+S"
        if override.fusion_variant == "sum"
        else "2*P^0.3*S"
    ),
    "cue_scale": override.cue_scale,
    "local_support": (
        f"support=clamp(training_cue/{override.local_support_scale},0,1)"
    ),
    "ordinary_sample_loss": (
        "B(v)=mean(C_v*(1-m_v))+mean((1-support_v)*m_v)"
    ),
    "latest_sample_loss": (
        "0.5*(B(t)+B(t-1)) + growth_replay_weight*"
        "mean(G_prev*(1-support_(t-1))*m_(t-1,current))"
    ),
    "growth_map": (
        "G_prev=relu(sigmoid(render_post_(t-1))-"
        "sigmoid(render_pre_(t-1))), detached in the t-1 camera coordinates"
    ),
    "activation_rule": (
        "Apply the two-view loss iff the original replay sampler selects the "
        "newest frame t; frame 0 uses B only"
    ),
    "regularization_weight": override.regularization_weight,
    "growth_replay_weight": override.growth_replay_weight,
    "controlled_constants": (
        "fixed poses, seed, optimizer, geometry parameters, online replay, "
        "densification, and the protocol's configured update budget"
    ),
}
metadata["continuous_score_export"] = {
    "enabled": override.save_continuous_scores,
    "path": str(output_path / "continuous_scores" / "online_at_arrival"),
    "value": "mean(render_change render, channel), before score > 0.5",
    "binary_rule": "score > 0.5",
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
(output_path / "growth_replay_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
