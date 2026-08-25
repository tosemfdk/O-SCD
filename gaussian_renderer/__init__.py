#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettingsFastGS, GaussianRasterizerFastGS

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult=0.5, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettingsFastGS(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizerFastGS(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc._features_dc, pc._features_rest
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}



def _validate_override_color(pc: GaussianModel, override_color: torch.Tensor) -> None:
    if not isinstance(override_color, torch.Tensor):
        raise TypeError("override_color must be a tensor")
    expected_shape = (pc.get_xyz.shape[0], 3)
    if override_color.shape != expected_shape:
        raise ValueError(f"override_color must have shape {expected_shape}")
    if override_color.dtype != pc._features_dc.dtype:
        raise ValueError("override_color must match the base DC dtype")
    if override_color.device != pc._features_dc.device:
        raise ValueError("override_color must be on the same device as the base DC")

def _validate_override_dc(pc: GaussianModel, override_dc: torch.Tensor) -> None:
    if not isinstance(override_dc, torch.Tensor):
        raise TypeError("override_dc must be a tensor")
    expected_shape = (pc.get_xyz.shape[0], 1, 3)
    if override_dc.shape != expected_shape:
        raise ValueError(f"override_dc must have shape {expected_shape}")
    if override_dc.dtype != pc._features_dc.dtype:
        raise ValueError("override_dc must match the base DC dtype")
    if override_dc.device != pc._features_dc.device:
        raise ValueError("override_dc must be on the same device as the base DC")


def _validate_override_features_rest(
    pc: GaussianModel, override_features_rest: torch.Tensor
) -> None:
    if not isinstance(override_features_rest, torch.Tensor):
        raise TypeError("override_features_rest must be a tensor")
    if override_features_rest.shape != pc._features_rest.shape:
        raise ValueError(
            "override_features_rest must have shape "
            f"{tuple(pc._features_rest.shape)}"
        )
    if override_features_rest.dtype != pc._features_rest.dtype:
        raise ValueError("override_features_rest must match the base SH-rest dtype")
    if override_features_rest.device != pc._features_rest.device:
        raise ValueError("override_features_rest must share the base SH-rest device")


def _validate_override_opacity(pc: GaussianModel, override_opacity: torch.Tensor) -> None:
    if not isinstance(override_opacity, torch.Tensor):
        raise TypeError("override_opacity must be a tensor")
    expected_shape = (pc.get_xyz.shape[0], 1)
    if override_opacity.shape != expected_shape:
        raise ValueError(f"override_opacity must have shape {expected_shape}")
    if override_opacity.dtype != pc._opacity.dtype:
        raise ValueError("override_opacity must match the base opacity dtype")
    if override_opacity.device != pc._opacity.device:
        raise ValueError("override_opacity must be on the same device as the base opacity")


def _validate_override_geometry(
    name: str,
    value: torch.Tensor,
    base: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != base.shape:
        raise ValueError(f"{name} must have shape {tuple(base.shape)}")
    if value.dtype != base.dtype:
        raise ValueError(f"{name} must match the base geometry dtype")
    if value.device != base.device:
        raise ValueError(f"{name} must be on the same device as the base geometry")


def render_change(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    mult=0.5,
    scaling_modifier=1.0,
    override_color=None,
    get_flag=None,
    metric_map=None,
    override_dc=None,
    override_opacity=None,
    override_xyz=None,
    override_scaling=None,
    override_rotation=None,
    clamp_output=True,
    override_features_rest=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    if override_color is not None and override_dc is not None:
        raise ValueError("override_color and override_dc cannot be used together")
    if override_color is not None:
        _validate_override_color(pc, override_color)
    if override_dc is not None:
        _validate_override_dc(pc, override_dc)
    if override_features_rest is not None:
        _validate_override_features_rest(pc, override_features_rest)
    if override_opacity is not None:
        _validate_override_opacity(pc, override_opacity)
    if override_xyz is not None:
        _validate_override_geometry("override_xyz", override_xyz, pc.get_xyz)
    if override_scaling is not None:
        _validate_override_geometry(
            "override_scaling", override_scaling, pc.get_scaling
        )
    if override_rotation is not None:
        _validate_override_geometry(
            "override_rotation", override_rotation, pc.get_rotation
        )

    means3D = pc.get_xyz if override_xyz is None else override_xyz

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros(
        (means3D.shape[0], 4),
        dtype=means3D.dtype,
        requires_grad=True,
        device=means3D.device,
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettingsFastGS(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizerFastGS(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity if override_opacity is None else override_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    selected_scales = pc.get_scaling if override_scaling is None else override_scaling
    selected_rotations = (
        pc.get_rotation if override_rotation is None else override_rotation
    )
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.covariance_activation(
            selected_scales,
            scaling_modifier,
            selected_rotations,
        )
    else:
        scales = selected_scales
        rotations = selected_rotations

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    elif override_dc is not None:
        dc = override_dc
        shs = (
            pc._features_rest
            if override_features_rest is None
            else override_features_rest
        )
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        dc, shs = pc._features_dc, pc._features_rest

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if clamp_output:
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts}


def render_change_temporal(
    viewpoint_camera,
    temporal_model,
    pipe,
    background: torch.Tensor,
    timestamp: float | None = None,
):
    """Render the active state attributes without changing Gaussian topology."""
    if timestamp is None:
        timestamp = getattr(viewpoint_camera, "timestamp", None)
    if timestamp is None:
        raise ValueError("timestamp is required for temporal change rendering")

    if hasattr(temporal_model, "get_active_render_attributes"):
        attributes = temporal_model.get_active_render_attributes(timestamp)
        return render_change(
            viewpoint_camera,
            temporal_model.base,
            pipe,
            background,
            override_dc=attributes["dc"],
            override_features_rest=attributes.get("features_rest"),
            override_opacity=attributes["opacity"],
            override_xyz=attributes["xyz"],
            override_scaling=attributes["scaling"],
            override_rotation=attributes["rotation"],
        )

    active_dc, active_gaussians = temporal_model.get_active_change(timestamp)
    active_opacity = temporal_model.base.get_opacity * active_gaussians[:, None]
    return render_change(
        viewpoint_camera,
        temporal_model.base,
        pipe,
        background,
        override_dc=active_dc,
        override_opacity=active_opacity,
    )
