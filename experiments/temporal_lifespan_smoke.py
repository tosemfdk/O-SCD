"""Synthetic CUDA smoke test for optional per-Gaussian lifespans."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from gaussian_renderer import render_change, render_change_temporal
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from temporal import TemporalChangeModel
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import RGB2SH


BOUNDARIES = (95.0, 199.0)
QUERIES = (
    (94.0, 0, "[0, 95)"),
    (95.0, 1, "[95, 199)"),
    (199.0, 2, "[199, inf)"),
)
CHANGE_BRIGHTNESS = 0.70
BACKGROUND_BRIGHTNESS = 0.08
STATE_VALID = (
    (True, False, False),  # g0: segment 0 only
    (True, True, False),  # g1: outdated after segment 1
    (True, True, True),  # g2: persists through every segment
    (False, True, True),  # g3: born in segment 1
    (False, False, True),  # g4: born in segment 2
)
BASE_FIELDS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def build_synthetic_temporal_scene(image_size: int = 160):
    """Create fixed geometry with manual segment lifespans and identical DC."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the synthetic renderer smoke test")

    device = torch.device("cuda")
    points = torch.tensor(
        [
            [-1.20, 0.0, 3.0],
            [-0.60, 0.0, 3.0],
            [0.00, 0.0, 3.0],
            [0.60, 0.0, 3.0],
            [1.20, 0.0, 3.0],
        ],
        device=device,
    )
    n_gaussians = points.shape[0]

    base = GaussianModel(sh_degree=3, active_sh_degree=0)
    base._xyz = nn.Parameter(points)
    change_rgb = torch.full((n_gaussians, 3), CHANGE_BRIGHTNESS, device=device)
    base._features_dc = nn.Parameter(RGB2SH(change_rgb).view(n_gaussians, 1, 3))
    base._features_rest = nn.Parameter(torch.zeros((n_gaussians, 15, 3), device=device))
    base._opacity = nn.Parameter(
        inverse_sigmoid(torch.full((n_gaussians, 1), 0.95, device=device))
    )
    base._scaling = nn.Parameter(
        torch.full((n_gaussians, 3), math.log(0.12), device=device)
    )
    rotations = torch.zeros((n_gaussians, 4), device=device)
    rotations[:, 0] = 1.0
    base._rotation = nn.Parameter(rotations)

    fov = math.radians(60.0)
    world_view = torch.eye(4, device=device)
    projection = getProjectionMatrix(0.01, 100.0, fov, fov).t().to(device)
    full_projection = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera = MiniCam(
        image_size,
        image_size,
        fov,
        fov,
        0.01,
        100.0,
        world_view,
        full_projection,
    )
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )
    background = torch.full((3,), BACKGROUND_BRIGHTNESS, device=device)

    temporal_model = TemporalChangeModel.from_gaussians(base, max_states=3)
    starts = torch.tensor([[0.0, *BOUNDARIES]], device=device).repeat(n_gaussians, 1)
    ends = torch.tensor([[*BOUNDARIES, float("inf")]], device=device).repeat(n_gaussians, 1)
    with torch.no_grad():
        temporal_model.state_start.copy_(starts)
        temporal_model.state_end.copy_(ends)
        temporal_model.state_valid.copy_(torch.tensor(STATE_VALID, device=device))
        temporal_model.state_change_dc.copy_(
            base._features_dc.detach()[:, None].repeat(1, temporal_model.max_states, 1, 1)
        )

    return temporal_model, camera, pipe, background


def _base_snapshot(base):
    return {name: getattr(base, name).detach().clone() for name in BASE_FIELDS}


def _base_is_unchanged(base, snapshot) -> bool:
    return all(torch.equal(getattr(base, name).detach(), value) for name, value in snapshot.items())


def _gaussian_pixels(model, camera) -> list[list[float]]:
    """Project the synthetic Gaussian centers to image coordinates."""
    xyz = model.base.get_xyz.detach()
    ndc_x = xyz[:, 0] / (xyz[:, 2] * math.tan(camera.FoVx * 0.5))
    ndc_y = xyz[:, 1] / (xyz[:, 2] * math.tan(camera.FoVy * 0.5))
    pixel_x = (ndc_x + 1.0) * 0.5 * (camera.image_width - 1)
    pixel_y = (1.0 - ndc_y) * 0.5 * (camera.image_height - 1)
    return torch.stack((pixel_x, pixel_y), dim=1).cpu().tolist()


def _roi_means(image: torch.Tensor, pixels: list[list[float]], radius: int = 3) -> list[float]:
    """Measure a small image-space neighborhood around every Gaussian center."""
    intensity = image.mean(dim=0)
    height, width = intensity.shape
    values = []
    for pixel_x, pixel_y in pixels:
        center_x = round(pixel_x)
        center_y = round(pixel_y)
        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)
        values.append(float(intensity[y0:y1, x0:x1].mean().item()))
    return values


def _optimizer_isolation(image_size: int) -> dict:
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size)
    base_before = _base_snapshot(model.base)
    states_before = model.state_change_dc.detach().clone()
    state_parameter_id = id(model.state_change_dc)
    state_shape = tuple(model.state_change_dc.shape)
    optimizer = torch.optim.SGD([model.state_change_dc], lr=0.1)

    optimizer.zero_grad(set_to_none=True)
    rendered = render_change_temporal(camera, model, pipe, background, timestamp=95.0)["render"]
    rendered.mean().backward()
    optimizer.step()

    pair_delta = (model.state_change_dc.detach() - states_before).abs().sum(dim=(2, 3))
    changed_pairs = pair_delta > 0
    expected_pairs = torch.zeros_like(changed_pairs)
    expected_pairs[:, 1] = model.state_valid[:, 1]
    return {
        "state_delta_l1": pair_delta.sum(dim=0).cpu().tolist(),
        "changed_pairs": changed_pairs.cpu().tolist(),
        "only_active_pairs_changed": torch.equal(changed_pairs, expected_pairs),
        "base_unchanged": _base_is_unchanged(model.base, base_before),
        "state_parameter_unchanged": id(model.state_change_dc) == state_parameter_id,
        "state_shape_unchanged": tuple(model.state_change_dc.shape) == state_shape,
    }


def run_temporal_lifespan_smoke(image_size: int = 160) -> dict:
    """Render three timestamps and measure state and gradient isolation."""
    model, camera, pipe, background = build_synthetic_temporal_scene(image_size)
    base_before = _base_snapshot(model.base)

    baseline = render_change(camera, model.base, pipe, background)["render"]
    baseline_override = render_change(
        camera,
        model.base,
        pipe,
        background,
        override_dc=model.base._features_dc,
        override_opacity=model.base.get_opacity,
    )["render"]
    gaussian_pixels = _gaussian_pixels(model, camera)
    background_mean = float(background.mean().item())
    frames = []
    for timestamp, expected_state, interval in QUERIES:
        model.zero_grad(set_to_none=True)
        rendered = render_change_temporal(
            camera,
            model,
            pipe,
            background,
            timestamp=timestamp,
        )["render"]
        rendered.mean().backward()

        pair_gradient_l1 = model.state_change_dc.grad.abs().sum(dim=(2, 3))
        state_gradient_l1 = pair_gradient_l1.sum(dim=0)
        active_indices = model.get_active_state_indices(timestamp)
        active_gaussians = active_indices >= 0
        expected_pairs = torch.zeros_like(pair_gradient_l1, dtype=torch.bool)
        expected_pairs[:, expected_state] = model.state_valid[:, expected_state]

        intensity = (rendered.mean(dim=0) - background_mean).clamp_min(0.0)
        columns = torch.arange(rendered.shape[-1], device=rendered.device)
        centroid_x = (intensity.sum(dim=0) * columns).sum() / intensity.sum()
        roi_means = _roi_means(rendered, gaussian_pixels)
        active_roi_ok = all(
            value > background_mean + 0.20
            for value, active in zip(roi_means, active_gaussians.tolist())
            if active
        )
        inactive_roi_ok = all(
            value < background_mean + 0.04
            for value, active in zip(roi_means, active_gaussians.tolist())
            if not active
        )
        frames.append(
            {
                "timestamp": timestamp,
                "interval": interval,
                "global_state": expected_state,
                "active_state_indices": active_indices.cpu().tolist(),
                "active_gaussians": active_gaussians.cpu().tolist(),
                "render_mean": float(rendered.mean().item()),
                "render_max": float(rendered.max().item()),
                "centroid_x": float(centroid_x.item()),
                "roi_means": roi_means,
                "opacity_isolated": active_roi_ok and inactive_roi_ok,
                "state_gradient_l1": state_gradient_l1.detach().cpu().tolist(),
                "gradient_pair_mask": (pair_gradient_l1 > 0).cpu().tolist(),
                "gradient_isolated": torch.equal(pair_gradient_l1 > 0, expected_pairs),
                "image": rendered.detach().cpu(),
            }
        )

    result = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "boundaries": list(BOUNDARIES),
        "interval_rule": "[start, end)",
        "background_mean": background_mean,
        "gaussian_pixels": gaussian_pixels,
        "base_gaussian_count": model.base.get_xyz.shape[0],
        "temporal_state_shape": list(model.state_change_dc.shape),
        "state_dc_identical": bool(
            torch.equal(
                model.state_change_dc[:, :1].expand_as(model.state_change_dc),
                model.state_change_dc,
            )
        ),
        "base_override_equivalence_max_error": float(
            (baseline - baseline_override).abs().max().item()
        ),
        "base_unchanged_after_backward": _base_is_unchanged(model.base, base_before),
        "frames": frames,
        "optimizer_step": _optimizer_isolation(image_size),
    }
    return result


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_visualization(result: dict, output_path: Path) -> None:
    """Save renders, lifespan masks, and pair-level gradient evidence."""
    source_size = result["frames"][0]["image"].shape[-1]
    image_size = max(source_size, 220)
    panel_width = 280
    margin = 24
    text_height = 190
    width = 3 * panel_width + 4 * margin
    height = image_size + text_height + 132
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(19)
    body_font = _font(13)
    small_font = _font(11)

    title = "Optional per-Gaussian lifespan: fixed topology, identical DC"
    draw.text((margin, 14), title, fill="black", font=title_font)
    draw.text(
        (margin, 40),
        f"{result['device']} | boundaries {result['boundaries']} "
        f"| intervals {result['interval_rule']}",
        fill="#333333",
        font=small_font,
    )

    for column, frame in enumerate(result["frames"]):
        panel_x = margin + column * (panel_width + margin)
        x = panel_x + (panel_width - image_size) // 2
        y = 66
        array = (
            frame["image"].clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
        )
        image = Image.fromarray(array).resize((image_size, image_size), Image.Resampling.BILINEAR)
        canvas.paste(image, (x, y))
        draw.rectangle((x - 1, y - 1, x + image_size, y + image_size), outline="#555555")
        scale = image_size / source_size
        for gaussian_id, ((pixel_x, pixel_y), active) in enumerate(
            zip(result["gaussian_pixels"], frame["active_gaussians"])
        ):
            center_x = x + pixel_x * scale
            center_y = y + pixel_y * scale
            color = "#00a651" if active else "#d64045"
            radius = 8
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                outline=color,
                width=2,
            )
            draw.text(
                (center_x - 6, center_y - 24),
                f"g{gaussian_id}",
                fill=color,
                font=small_font,
            )
        draw.text(
            (panel_x, y + image_size + 8),
            f"t={frame['timestamp']:g}  manual segment {frame['global_state']}",
            fill="black",
            font=body_font,
        )
        draw.text(
            (panel_x, y + image_size + 29),
            f"lifespan {frame['interval']}",
            fill="#333333",
            font=small_font,
        )
        draw.text(
            (panel_x, y + image_size + 47),
            f"active GS {list(map(int, frame['active_gaussians']))}",
            fill="#333333",
            font=small_font,
        )
        draw.text(
            (panel_x, y + image_size + 65),
            f"slot index {frame['active_state_indices']}",
            fill="#333333",
            font=small_font,
        )
        gradients = ", ".join(f"{value:.4f}" for value in frame["state_gradient_l1"])
        draw.text(
            (panel_x, y + image_size + 83),
            f"state grad L1 [{gradients}]",
            fill="#0b6b2e" if frame["gradient_isolated"] else "#b00020",
            font=small_font,
        )
        draw.text(
            (panel_x, y + image_size + 101),
            f"opacity isolated: {frame['opacity_isolated']}",
            fill="#0b6b2e" if frame["opacity_isolated"] else "#b00020",
            font=small_font,
        )
        draw.text(
            (panel_x, y + image_size + 119),
            f"signal centroid x={frame['centroid_x']:.2f}",
            fill="#333333",
            font=small_font,
        )
        pair_rows = [
            "".join("1" if value else "0" for value in row)
            for row in frame["gradient_pair_mask"]
        ]
        draw.text(
            (panel_x, y + image_size + 137),
            f"grad pairs g0..g4: {' '.join(pair_rows)}",
            fill="#333333",
            font=small_font,
        )

    footer_y = height - 71
    optimizer = result["optimizer_step"]
    draw.text(
        (margin, footer_y),
        "Green = active, red = unborn/outdated. All DC slots have the same value.",
        fill="black",
        font=body_font,
    )
    draw.text(
        (margin, footer_y + 23),
        "Appearance changes only through lifespan opacity gating; no Gaussian tensor is deleted.",
        fill="black",
        font=small_font,
    )
    draw.text(
        (margin, footer_y + 41),
        "Only active (Gaussian, slot) pairs changed after one optimizer step: "
        f"{optimizer['only_active_pairs_changed']}; base unchanged: {optimizer['base_unchanged']}.",
        fill="#0b6b2e",
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_metrics(result: dict, output_path: Path) -> None:
    """Save machine-readable results without image tensors."""
    metrics = {key: value for key, value in result.items() if key != "frames"}
    metrics["frames"] = [
        {key: value for key, value in frame.items() if key != "image"}
        for frame in result["frames"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("docs/static/images/temporal_lifespan_cuda_smoke.png"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/temporal-lifespan-smoke-results.json"),
    )
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    result = run_temporal_lifespan_smoke(args.image_size)
    save_visualization(result, args.image)
    save_metrics(result, args.metrics)
    print(f"Saved visualization: {args.image}")
    print(f"Saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
