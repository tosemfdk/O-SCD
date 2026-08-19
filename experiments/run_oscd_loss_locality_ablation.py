#!/usr/bin/env python3
"""Run the controlled sum/product by global/local O-SCD loss ablation.

This wrapper reuses the fixed-pose online protocol from the sibling O-SCD
checkout while keeping the ablation loss implementation in this repository.
It preserves the Part 10 optimizer, replay, densification, pose, and cached-cue
conditions and changes only cue fusion or regularization locality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSCD_REPO = REPO_ROOT.parent / "O-SCD"


def pop_ablation_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--oscd-repo", type=Path, default=DEFAULT_OSCD_REPO)
    parser.add_argument(
        "--fusion-variant",
        choices=("sum", "power_product"),
        required=True,
    )
    parser.add_argument(
        "--regularization-mode",
        choices=("global", "local"),
        required=True,
    )
    parser.add_argument("--regularization-weight", type=float, default=1.0)
    parser.add_argument(
        "--cue-scale",
        type=float,
        default=None,
        help="Defaults to 1 for sum and 2 for P^0.3*S.",
    )
    parser.add_argument(
        "--local-support-scale",
        type=float,
        default=2.0,
        help="C_hat = clamp(training_cue / this fixed scale, 0, 1).",
    )
    product_source = parser.add_mutually_exclusive_group()
    product_source.add_argument(
        "--power-cue-dir",
        type=Path,
        help="Directory containing product cue tensors directly or under cues/.",
    )
    product_source.add_argument(
        "--power-cue-npz",
        type=Path,
        help="NPZ with one '<frame stem>__power' array per frame.",
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


override = pop_ablation_args(sys.argv)
if override.regularization_weight < 0.0:
    raise ValueError("regularization-weight must be nonnegative")
if override.local_support_scale <= 0.0:
    raise ValueError("local-support-scale must be positive")
if override.cue_scale is None:
    override.cue_scale = 1.0 if override.fusion_variant == "sum" else 2.0
if override.cue_scale <= 0.0:
    raise ValueError("cue-scale must be positive")
if override.fusion_variant == "power_product" and not (
    override.power_cue_dir or override.power_cue_npz
):
    raise ValueError("power_product requires --power-cue-dir or --power-cue-npz")

compute_ssf_loss = load_compute_ssf_loss()
oscd_protocol_path = override.oscd_repo / "oscd_protocols.py"
if not oscd_protocol_path.is_file():
    raise FileNotFoundError(oscd_protocol_path)
sys.path.insert(0, str(override.oscd_repo))
import oscd_protocols as protocol  # noqa: E402


load_sum_cue = protocol.load_cached_cue
power_arrays = (
    np.load(override.power_cue_npz)
    if override.power_cue_npz is not None
    else None
)


def load_product_cue(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor:
    if override.power_cue_dir is not None:
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
    else:
        assert power_arrays is not None
        key = f"{path.stem}__power"
        if key not in power_arrays:
            raise KeyError(f"Missing product cue array: {key}")
        value = torch.from_numpy(
            np.asarray(power_arrays[key], dtype=np.float32)
        ).unsqueeze(0)
    if tuple(value.shape) != (1, *expected_shape):
        raise ValueError(
            f"Product cue shape mismatch for {path.name}: "
            f"{tuple(value.shape)} != {(1, *expected_shape)}"
        )
    return value.contiguous().cuda()


def load_fusion_cue(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor:
    if override.fusion_variant == "sum":
        cue = load_sum_cue(path, expected_shape)
    else:
        cue = load_product_cue(path, expected_shape)
    return cue.mul_(override.cue_scale)


def controlled_change_loss(viewpoint, gaussians_change, pipe, background):
    package = protocol.render_change(viewpoint, gaussians_change, pipe, background)
    cue = viewpoint.candidate_map
    local_support = None
    if override.regularization_mode == "local":
        local_support = (cue / override.local_support_scale).clamp(0.0, 1.0)
    loss, _ = compute_ssf_loss(
        cue,
        package["render"],
        regularizer_offset=1.000000001,
        regularization_mode=override.regularization_mode,
        local_support=local_support,
        regularization_weight=override.regularization_weight,
    )
    return loss, package


def save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().float().cpu().contiguous(), temporary)
    os.replace(temporary, path)


def write_mask_and_score(
    path: Path, view, gaussians_change, pipe, background
) -> None:
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


output_path = model_output_path(sys.argv)
protocol.load_cached_cue = load_fusion_cue
protocol.change_loss = controlled_change_loss
protocol.write_change_mask = write_mask_and_score
protocol.main()

metadata_path = output_path / "run_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
raw_training_cue = "P+S" if override.fusion_variant == "sum" else "P^0.3*S"
metadata["loss_locality_ablation"] = {
    "fusion_variant": override.fusion_variant,
    "raw_training_cue": raw_training_cue,
    "cue_scale": override.cue_scale,
    "regularization_mode": override.regularization_mode,
    "regularization_weight": override.regularization_weight,
    "global_regularization": "log(mean(m)^2 + 1.000000001)",
    "local_regularization": "mean((1-C_hat)*m)",
    "local_support": (
        f"C_hat=clamp(training_cue/{override.local_support_scale},0,1)"
        if override.regularization_mode == "local"
        else None
    ),
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
