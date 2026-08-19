#!/usr/bin/env python3
"""Run a 2X-product O-SCD ablation with delayed global regularization.

Frame 0 uses the cue-local B regularizer. For every later frame, the scalar
regularizer weight is the positive rendered-mask growth caused by the previous
frame's complete optimization. The previous frame is rendered before and after
its update from the same camera, so camera motion does not enter the scalar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSCD_REPO = REPO_ROOT.parent / "O-SCD"


def pop_experiment_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--oscd-repo", type=Path, default=DEFAULT_OSCD_REPO)
    parser.add_argument("--power-cue-dir", type=Path, required=True)
    parser.add_argument("--cue-scale", type=float, default=2.0)
    parser.add_argument("--local-support-scale", type=float, default=2.0)
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    parser.add_argument(
        "--include-local-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep B local regularization while adding the delayed global term.",
    )
    parser.add_argument(
        "--save-continuous-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    known, remaining = parser.parse_known_args(argv[1:])
    argv[:] = [argv[0], *remaining]
    return known


def load_compute_ssf_loss():
    module_path = REPO_ROOT / "temporal" / "fusion.py"
    spec = importlib.util.spec_from_file_location(
        "oscd_evolving_fusion_loss", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load loss module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_ssf_loss


def model_output_path(argv: list[str]) -> Path:
    for index, argument in enumerate(argv[:-1]):
        if argument in {"-m", "--model_path"}:
            return Path(argv[index + 1])
    raise RuntimeError("The downstream O-SCD command requires -m/--model_path")


override = pop_experiment_args(sys.argv)
if override.cue_scale <= 0.0:
    raise ValueError("cue-scale must be positive")
if override.local_support_scale <= 0.0:
    raise ValueError("local-support-scale must be positive")
if override.regularization_weight < 0.0:
    raise ValueError("regularization-weight must be nonnegative")

compute_ssf_loss = load_compute_ssf_loss()
oscd_protocol_path = override.oscd_repo / "oscd_protocols.py"
if not oscd_protocol_path.is_file():
    raise FileNotFoundError(oscd_protocol_path)
sys.path.insert(0, str(override.oscd_repo))
import oscd_protocols as protocol  # noqa: E402


def load_product_cue(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor:
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


class DelayedGlobalState:
    def __init__(self) -> None:
        self.first_frame = True
        self.previous_growth = 0.0
        self.current_mode = "local_first_frame"
        self.current_frame = ""
        self.audit: list[dict[str, float | str | int]] = []


state = DelayedGlobalState()


def delayed_change_loss(viewpoint, gaussians_change, pipe, background):
    package = protocol.render_change(viewpoint, gaussians_change, pipe, background)
    cue = viewpoint.candidate_map
    if state.first_frame:
        local_support = (cue / override.local_support_scale).clamp(0.0, 1.0)
        loss, _ = compute_ssf_loss(
            cue,
            package["render"],
            regularizer_offset=1.000000001,
            regularization_mode="local",
            local_support=local_support,
            regularization_weight=override.regularization_weight,
        )
    else:
        local_support = None
        regularization_mode = "previous_update_global"
        if override.include_local_base:
            local_support = (cue / override.local_support_scale).clamp(0.0, 1.0)
            regularization_mode = "local_plus_previous_update_global"
        loss, _ = compute_ssf_loss(
            cue,
            package["render"],
            regularizer_offset=1.000000001,
            regularization_mode=regularization_mode,
            local_support=local_support,
            regularization_weight=override.regularization_weight,
            previous_growth=state.previous_growth,
        )
    return loss, package


def render_probability(view, model, pipe, background) -> torch.Tensor:
    rendered = protocol.render_change(view, model, pipe, background)["render"]
    return torch.sigmoid(rendered.mean(dim=0, keepdim=True))


def save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().float().cpu().contiguous(), temporary)
    os.replace(temporary, path)


def write_mask_and_score(path: Path, view, gaussians_change, pipe, background) -> None:
    with torch.no_grad():
        rendered = protocol.render_change(
            view, gaussians_change, pipe, background
        )["render"]
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


def run_online_previous_update_global(
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
    protocol.recreate_directory(arrival_dir)
    model = protocol.create_change_model(reference_ply, sh_degree, opt)
    initial_points = int(model.get_xyz.shape[0])
    total_steps = 0
    seen_views = []
    progress = protocol.tqdm(views, desc="Online previous-update global O-SCD")

    import time

    start = time.monotonic()
    for frame_index, view in enumerate(progress):
        seen_views.append(view)
        with torch.no_grad():
            pre_probability = render_probability(view, model, pipe, background)

        state.first_frame = frame_index == 0
        state.current_frame = view.image_name
        state.current_mode = (
            "local_first_frame"
            if state.first_frame
            else (
                "local_plus_previous_update_global"
                if override.include_local_base
                else "previous_update_global"
            )
        )

        for local_step in range(steps_per_frame):
            total_steps += 1
            if np.random.rand() > 0.33:
                viewpoint = seen_views[random.randint(0, len(seen_views) - 1)]
            else:
                viewpoint = seen_views[-1]
            model.update_learning_rate(total_steps)
            loss, package = delayed_change_loss(
                viewpoint, model, pipe, background
            )
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
            post_probability = render_probability(view, model, pipe, background)
            positive_growth = (post_probability - pre_probability).clamp_min(0.0)
            next_growth = float(positive_growth.mean().item())
        state.audit.append(
            {
                "frame_index": frame_index,
                "frame": view.image_name,
                "regularization_mode": state.current_mode,
                "applied_previous_growth": state.previous_growth,
                "measured_positive_growth": next_growth,
            }
        )
        state.previous_growth = next_growth
        progress.set_postfix(
            step=total_steps,
            loss=f"{loss.item():.4f}",
            growth=f"{next_growth:.6f}",
        )
        write_mask_and_score(
            arrival_dir / f"{view.image_name}.png",
            view,
            model,
            pipe,
            background,
        )

    progress.close()
    protocol.render_all_masks(final_dir, views, model, pipe, background)
    result = {
        "steps_per_frame": steps_per_frame,
        "total_steps": total_steps,
        "initial_points": initial_points,
        "final_points": int(model.get_xyz.shape[0]),
        "runtime_seconds": time.monotonic() - start,
        "previous_growth_mean": float(
            np.mean([row["measured_positive_growth"] for row in state.audit])
        ),
        "previous_growth_max": float(
            max(row["measured_positive_growth"] for row in state.audit)
        ),
    }
    del model
    protocol.gc.collect()
    torch.cuda.empty_cache()
    return result


output_path = model_output_path(sys.argv)
protocol.load_cached_cue = load_product_cue
protocol.change_loss = delayed_change_loss
protocol.write_change_mask = write_mask_and_score
protocol.run_online_protocol = run_online_previous_update_global
protocol.main()

metadata_path = output_path / "run_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["previous_update_global_ablation"] = {
    "training_cue": "2*P^0.3*S",
    "first_frame_regularization": "mean((1-X)*m)",
    "later_regularization": (
        "mean((1-X)*m)+g_prev*log(mean(m)^2+1.000000001)"
        if override.include_local_base
        else "g_prev*log(mean(m)^2+1.000000001)"
    ),
    "include_local_base": override.include_local_base,
    "g_prev": (
        "mean(relu(sigmoid(render_post_prev)-sigmoid(render_pre_prev))) "
        "on the previous frame camera"
    ),
    "regularization_weight": override.regularization_weight,
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
(output_path / "previous_growth_audit.json").write_text(
    json.dumps(state.audit, indent=2) + "\n", encoding="utf-8"
)
